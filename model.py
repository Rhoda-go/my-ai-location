
import torch
import copy
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geom_nn
from pytorch_lightning import LightningModule
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, IterableDataset

from swap_env import SwapEnv
from utils import to_device
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse as sp

def collate_fn_ppo(batch):
    states, actions, logp_olds, v_olds, qvals, advs = zip(*batch)
    new_states = {}
    new_states["mask"] = torch.stack([state["mask"] for state in states])
    #print(f"批量后mask：形状={new_states['mask'].shape}，维度数={new_states['mask'].ndim}")
    new_states["tabu_table"] = torch.stack([state["tabu_table"] for state in states])
    #print(f"批量后tabu：形状={new_states['tabu_table'].shape}，维度数={new_states['tabu_table'].ndim}") 
    new_states["fac_data"] = torch_geometric.data.Batch.from_data_list(
        [state["fac_data"] for state in states]
    )
    actions = torch.stack(actions)
    logp_olds = torch.stack(logp_olds)
    v_olds = torch.as_tensor(v_olds, dtype=torch.float).unsqueeze(-1)
    qvals = torch.as_tensor(qvals, dtype=torch.float).unsqueeze(-1)
    advs = torch.as_tensor(advs, dtype=torch.float).unsqueeze(-1)
    return (new_states, actions, logp_olds, v_olds, qvals, advs)

class HierarchicalGraphTransformer(nn.Module):
    """
    分层图Transformer（无边特征版本）
    特点：
    1. 局部图卷积 + 全局Transformer
    2. 结构位置编码
    3. 虚拟全局节点
    4. 支持多种GNN层（GAT, GCN, GraphSAGE, TransformerConv等）
    """
    def __init__(self, c_in, c_hidden, c_out, num_layers=3, num_heads=4, 
                 layer_name="GATv2Conv", dropout=0.1, use_edge_attr=False, **kwargs):
        super().__init__()
        
        self.num_layers = num_layers
        self.c_hidden = c_hidden
        self.use_edge_attr = use_edge_attr
        
        # 输入嵌入
        self.input_embedding = nn.Linear(c_in, c_hidden)
        
        # 结构位置编码（拉普拉斯特征向量）
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            self.pos_encoder = nn.Linear(8, c_hidden)  # 使用前8个特征向量
        
        # 虚拟全局节点嵌入
        self.global_node = nn.Parameter(torch.randn(1, c_hidden))
        
        # 局部图卷积层（移除edge_dim参数）
        gnn_layer_class = getattr(geom_nn, layer_name)
        
        # 根据不同的GNN层类型，使用不同的参数
        self.local_convs = nn.ModuleList()
        for _ in range(num_layers):
            if layer_name in ["GATConv", "GATv2Conv"]:
                # GAT系列：
                self.local_convs.append(
                    gnn_layer_class(
                        in_channels=c_hidden,
                        out_channels=c_hidden,  
                        heads=num_heads,
                        concat=False,  # False时输出是c_hidden，True时是c_hidden*heads
                        dropout=dropout,
                        add_self_loops=True,
                        bias=True
                    )
                )
            elif layer_name == "TransformerConv":
                # TransformerConv：支持heads
                self.local_convs.append(
                    gnn_layer_class(
                        c_hidden, 
                        c_hidden, 
                        heads=num_heads, 
                        concat=False,
                        dropout=dropout
                    )
                )
            elif layer_name == "GCNConv":
                # GCN：不支持heads
                self.local_convs.append(
                    gnn_layer_class(c_hidden, c_hidden)
                )
            elif layer_name == "GraphConv":
                # GraphConv：不支持heads
                self.local_convs.append(
                    gnn_layer_class(c_hidden, c_hidden)
                )
            elif layer_name == "SAGEConv":
                # SAGE：不支持heads
                self.local_convs.append(
                    gnn_layer_class(c_hidden, c_hidden)
                )
            else:
                # 默认：尝试不带额外参数
                self.local_convs.append(
                    gnn_layer_class(c_hidden, c_hidden)
                )
        
        # 全局Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_hidden,
            nhead=num_heads,
            dim_feedforward=c_hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.global_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 层归一化
        self.local_norms = nn.ModuleList([
            nn.LayerNorm(c_hidden) for _ in range(num_layers)
        ])
        
        # Dropout层
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, c_out)
        )
        
    def compute_laplacian_pe(self, edge_index, num_nodes, k=8):
        """计算拉普拉斯位置编码"""
        from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
        import scipy.sparse as sp
        
        try:
            edge_index, edge_weight = get_laplacian(
                edge_index, 
                normalization='sym', 
                num_nodes=num_nodes
            )
            L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
            
            # 计算特征向量
            # 如果节点数小于k，调整k的值
            k = min(k, num_nodes - 2)
            if k < 1:
                return None
            
            eig_vals, eig_vecs = sp.linalg.eigsh(L, k=k, which='SM')
            pe = torch.from_numpy(eig_vecs).float()
            
            # 如果特征向量数量不足8，进行padding
            if pe.size(1) < 8:
                padding = torch.zeros(num_nodes, 8 - pe.size(1))
                pe = torch.cat([pe, padding], dim=1)
            
            return pe
        except Exception as e:
            print(f"Warning: Failed to compute Laplacian PE: {e}")
            return None
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Args:
            x: 节点特征 [num_nodes, c_in]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征（可选，不使用）
            batch: 批次索引 [num_nodes]
        
        Returns:
            输出特征 [num_nodes, c_out]
        """
        num_nodes = x.size(0)
        
        # 输入嵌入
        x = self.input_embedding(x)
        
        # 添加位置编码
        if self.use_pos_encoding:
            pe = self.compute_laplacian_pe(edge_index, num_nodes)
            if pe is not None:
                x = x + self.pos_encoder(pe.to(x.device))
        
        # 局部图卷积
        for i, (conv, norm, dropout) in enumerate(
            zip(self.local_convs, self.local_norms, self.dropouts)
        ):
            x_res = x
            
            # GNN传播（不使用edge_attr）
            if isinstance(conv, (geom_nn.GATConv, geom_nn.GATv2Conv, geom_nn.TransformerConv)):
                # 这些层可能支持edge_attr，但我们不传递
                x = conv(x, edge_index)
            else:
                # 其他层直接传递
                x = conv(x, edge_index)
            
            # 残差连接 + 归一化 + dropout
            x = norm(x + x_res)
            x = dropout(x)
        
        # 准备全局Transformer输入
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        
        batch_size = batch.max().item() + 1
        
        # 为每个图添加虚拟全局节点
        global_nodes = self.global_node.expand(batch_size, -1)
        
        # 按batch组织节点特征
        x_batched = []
        for i in range(batch_size):
            mask = (batch == i)
            graph_nodes = x[mask]
            # 添加全局节点
            graph_with_global = torch.cat([global_nodes[i:i+1], graph_nodes], dim=0)
            x_batched.append(graph_with_global)
        
        # Padding到相同长度
        max_len = max(t.size(0) for t in x_batched)
        x_padded = torch.zeros(batch_size, max_len, self.c_hidden, device=x.device)
        padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=x.device)
        
        for i, graph_x in enumerate(x_batched):
            length = graph_x.size(0)
            x_padded[i, :length] = graph_x
            padding_mask[i, :length] = False
        
        # 全局Transformer
        x_global = self.global_transformer(x_padded, src_key_padding_mask=padding_mask)
        
        # 提取节点特征（去除全局节点和padding）
        x_output = []
        for i in range(batch_size):
            mask = (batch == i)
            num_graph_nodes = mask.sum().item()
            x_output.append(x_global[i, 1:num_graph_nodes+1])  # 跳过全局节点
        
        x = torch.cat(x_output, dim=0)
        
        # 输出
        x = self.output_layer(x)
        
        return x



class GraphFeatureExtractor(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", **kwargs):
        super().__init__()

        gnn_layer = getattr(geom_nn, layer_name)

        '''
        根据ai修改的代码，增加1
        '''
        heads_supported_layers = ["GATConv", "GATv2Conv", "TransformerConv"]
        use_heads = layer_name in heads_supported_layers and "heads" in kwargs
        # 移除GraphConv不支持的参数（heads/edge_dim）
        if not use_heads:
            kwargs.pop("heads", None)
            kwargs.pop("edge_dim", None)
        # 取heads值（支持的层用配置的，不支持的层默认1）
        heads = kwargs.get("heads", 1) if use_heads else 1   #到这


        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels, out_channels, **kwargs),
                nn.ReLU(inplace=True),
            ]
            #in_channels = c_hidden * kwargs["heads"]

            '''
            根据ai修改的代码，修改上一行
            '''
            in_channels = c_hidden * heads if use_heads else c_hidden


        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            if isinstance(layer, geom_nn.GATv2Conv):
                x = layer(x, edge_index, edge_attr=edge_attr)
            elif isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index, edge_weight=edge_attr)
            else:
                x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2):
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU(inplace=True)]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class ActorCritic(nn.Module):
    def __init__(
        self, fac_c_in, c_hidden, c_out, num_layers, num_heads, layer_name, **kwargs
    ) -> None:
        super().__init__()
        '''
        根据ai修改的代码，增加3
        '''
        heads_supported_layers = ["GATConv", "GATv2Conv", "TransformerConv"]
        use_heads = layer_name in heads_supported_layers
        if use_heads:
        # 支持heads的层，默认补1
            if "heads" not in kwargs:
                kwargs["heads"] = 1
            emb_size = c_out * kwargs["heads"] * 2
        else:
            # 不支持heads的层，强制删除参数，emb_size按heads=1计算
            kwargs.pop("heads", None)
            emb_size = c_out * 2  # 等价于c_out * 1 * 2  #到这，还原下面三行


        # if "heads" not in kwargs:
        #     kwargs["heads"] = 1
        # emb_size = c_out * kwargs["heads"] * 2
        
        self.actor_gnn = HierarchicalGraphTransformer(
            fac_c_in, c_hidden, c_out, num_layers, num_heads, layer_name, **kwargs
        )
        
        self.critic_gnn = HierarchicalGraphTransformer(
            fac_c_in, c_hidden, c_out, num_layers, num_heads, layer_name, **kwargs
        )
        
        # self.actor_gnn = GraphFeatureExtractor(
        #     fac_c_in, c_hidden, c_out, num_layers, layer_name, **kwargs
        # )
        # self.critic_gnn = GraphFeatureExtractor(
        #     fac_c_in, c_hidden, c_out, num_layers, layer_name, **kwargs
        # )
        self.actor_prob = MLP(emb_size, c_hidden, 1, num_layers)
        self.att = nn.Linear(emb_size, emb_size, bias=False)


        self.critic = MLP(emb_size, c_hidden, 1, num_layers)

    def actor_forward(self, state, action1=None):
        #batch_fac, mask, tabu_table = state["fac_data"], state["mask"], state["tabu_table"]
        batch_fac, mask, tabu_table = state["fac_data"], state["mask"], state["tabu_table"]
        batch = batch_fac.batch
        if batch is None:
            batch = torch.zeros(
                batch_fac.num_nodes, dtype=torch.long, device=batch_fac.x.device
            )

        emb_fac = self.actor_gnn(batch_fac.x, batch_fac.edge_index, batch_fac.edge_attr)
        pooling = geom_nn.global_mean_pool(emb_fac, batch)
        emb_fac = torch.cat([emb_fac, pooling[batch]], dim=-1)

        act_scores1 = self.actor_prob(emb_fac).reshape(pooling.shape[0], -1)
        mask1 = torch.where(mask, -float("inf"), 0)
        logits1 = act_scores1 + mask1
        pi1 = Categorical(logits=logits1)
        if action1 is None:
            action1 = pi1.sample()

        if batch_fac.batch is not None:
            action1_inc = action1 + batch_fac.ptr[:-1]
        else:
            action1_inc = action1

        feat_act = torch.tanh(self.att(emb_fac[action1_inc]))
        act_scores2 = torch.matmul(emb_fac, feat_act.T)
        act_scores2 = act_scores2[torch.arange(act_scores2.shape[0]), batch]
        act_scores2 = act_scores2.reshape(pooling.shape[0], -1)
        
        # #过滤逻辑，但是这是单个样本的处理逻辑，需要改成批量操作的逻辑
        # mask_tabu=(tabu_table == 1)      # torch.bool 
        # mask2 = copy.deepcopy(mask)
        # print('mask2',len(mask2))
        # candidate_indices = torch.where(mask2 == 0)[0]  # selected location index
      
        # if len(candidate_indices) > 0:
        #     for idx in candidate_indices:
        #         tabu_row = mask_tabu[idx] 
        #         mask2 = mask2 & tabu_row  #true&true=true，true&false=false
        '''
        filtered by tabu_table
        '''
        mask_tabu = (tabu_table == 1)  # [batch, nodes, nodes]
        mask2 = mask.clone()  # [batch, nodes]
        
        batch_size = mask2.shape[0]
        
        for i in range(batch_size):
            sample_mask = mask2[i]  # [nodes]
            candidate_indices = torch.where(sample_mask == 0)[0] # candidate_indices for batch i（where mask=0）

            for idx in candidate_indices:
                tabu_row = mask_tabu[i, idx]  # [nodes]
                sample_mask = sample_mask & tabu_row #true&true=true，true&false=false
            
            # update
            mask2[i] = sample_mask
                
        logits_mask = torch.where(mask2, 0, -float("inf"))
        logits2 = act_scores2 + logits_mask
        pi2 = Categorical(logits=logits2)
        action2 = pi2.sample()

        logits = torch.stack([logits1, logits2], dim=1).squeeze(0)
        action = torch.stack([action1, action2], dim=1).squeeze(0)
        return Categorical(logits=logits), action

    def critic_forward(self, state):
        batch_fac = state["fac_data"]
        emb_fac = self.critic_gnn(
            batch_fac.x, batch_fac.edge_index, batch_fac.edge_attr
        )
        mean_pool = geom_nn.global_mean_pool(emb_fac, batch_fac.batch)
        max_pool = geom_nn.global_max_pool(emb_fac, batch_fac.batch)
        emb_global = torch.cat([mean_pool, max_pool], dim=-1) 
        score = self.critic(emb_global).squeeze(0)
        return score

    def get_log_prob(self, pi: Categorical, actions):
        return pi.log_prob(actions)

    def actor_loss(self, state, action, logp_old, qval, adv, clip_ratio):
        pi, _ = self.actor_forward(state)
        logp = self.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()

        with torch.no_grad():
            log_ratio = logp - logp_old
            approx_kl_div = (
                torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy().item()
            )

        clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_ratio).float()).item()

        entropy_loss = -pi.entropy().mean()
        return loss_actor, entropy_loss, approx_kl_div, clip_fraction

    def critic_loss(self, state, action, logp_old, v_old, qval, adv, clip_ratio):
        value = self.critic_forward(state)
        clip_value = v_old + torch.clamp(value - v_old, -clip_ratio, clip_ratio)
        v_max = torch.max((qval - value).pow(2), (qval - clip_value).pow(2))
        loss_critic = v_max.mean()
        return loss_critic


class ExperienceSourceDataset(IterableDataset):
    def __init__(self, generate_batch):
        self.generate_batch = generate_batch

    def __iter__(self):
        iterator = self.generate_batch()
        return iterator


class PPOLightning(LightningModule):
    def __init__(
        self,
        model_params: dict = None,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 5e-4,
        lr_gamma: float = 0.99,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
        clip_decay: float = 1,
        ent_weight: float = 0.01,
        critic_weight: float = 0.5,
        gradient_clip_val: float = None,
        mode: str = "train",
        data_path: str = "./data/train/",
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        if mode == "test":
            self.actor_critic = ActorCritic(**self.hparams.model_params)
            return

        self.automatic_optimization = False

        self.env = SwapEnv(data_path=data_path)

        self.actor_critic = ActorCritic(**self.hparams.model_params)

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []
        self.batch_v = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.state = self.env.reset()[0]

    def forward(self, x: torch.Tensor):
        pi, action = self.actor_critic.actor_forward(x)
        value = self.actor_critic.critic_forward(x)

        return pi, action, value

    def predict(self, x):
        return self.actor_critic(x)

    def discount_rewards(self, rewards, discount):
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards, values, last_value):
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [
            rews[i] + self.hparams.gamma * vals[i + 1] - vals[i]
            for i in range(len(rews) - 1)
        ]
        adv = self.discount_rewards(delta, self.hparams.gamma * self.hparams.lam)

        return adv

    def generate_trajectory_samples(self):
        for step in range(self.hparams.steps_per_epoch):
            to_device(self.state, self.device)

            with torch.no_grad():
                pi, action, value = self(self.state)
                log_prob = self.actor_critic.get_log_prob(pi, action)
                self.logger.log_metrics(
                    {"entropy/pi": pi.entropy().mean()}, self.global_step
                )

            next_state, reward, done, truncated, _ = self.env.step(
                action.squeeze().cpu().numpy()
            )

            self.episode_step += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)
            self.batch_v.append(value)

            self.ep_rewards.append(reward.item())
            self.ep_values.append(value.item())

            self.state = next_state

            epoch_end = step == (self.hparams.steps_per_epoch - 1)

            if epoch_end or done or truncated:
                if not done:
                    to_device(self.state, self.device)
                    with torch.no_grad():
                        _, _, value = self(self.state)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                self.batch_qvals += self.discount_rewards(
                    self.ep_rewards + [last_value], self.hparams.gamma
                )[:-1]
                self.batch_adv += self.calc_advantage(
                    self.ep_rewards, self.ep_values, last_value
                )
                self.epoch_rewards.append(sum(self.ep_rewards))
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = self.env.reset()[0]

            if epoch_end:
                train_data = zip(
                    self.batch_states,
                    self.batch_actions,
                    self.batch_logp,
                    self.batch_v,
                    self.batch_qvals,
                    self.batch_adv,
                )
                for state, action, logp_old, v_old, qval, adv in train_data:
                    yield state, action, logp_old, v_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_v.clear()
                self.batch_qvals.clear()

                self.avg_reward = sum(self.epoch_rewards) / self.hparams.steps_per_epoch

                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (
                    self.hparams.steps_per_epoch - steps_before_cutoff
                ) / nb_episodes

                self.epoch_rewards.clear()

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/avg_ep_reward": -1})

    def training_step(self, batch, batch_idx):
        state, action, old_logp, v_old, qval, adv = batch

        if self.hparams.batch_size > 1:
            adv = (adv - adv.mean()) / adv.std()

        self.log(
            "hp/avg_ep_reward",
            self.avg_ep_reward,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        (
            loss_actor,
            entropy_loss,
            approx_kl_div,
            clip_fraction,
        ) = self.actor_critic.actor_loss(
            state, action, old_logp, qval, adv, self.hparams.clip_ratio
        )

        loss_critic = self.actor_critic.critic_loss(
            state, action, old_logp, v_old, qval, adv, self.hparams.clip_ratio
        )
        loss = (
            self.hparams.ent_weight * entropy_loss
            + loss_actor
            + self.hparams.critic_weight * loss_critic
        )

        self.manual_backward(loss)
        if self.hparams.gradient_clip_val is not None:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.hparams.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx + 1 == self.hparams.steps_per_epoch // self.hparams.batch_size:
            scheduler.step()
            self.hparams.clip_ratio *= self.hparams.clip_decay

        self.log(
            "loss/loss_critic",
            loss_critic,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "loss/loss_actor",
            loss_actor,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "loss/loss_entropy",
            entropy_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_gamma
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, *args, **kwargs):
        for _ in range(self.hparams.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def _dataloader(self) -> DataLoader:
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn_ppo,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader()
