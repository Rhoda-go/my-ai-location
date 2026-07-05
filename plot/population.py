from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter


CSV_PATH = Path("case_data/selected_fences_summary_k100.csv")
OUTPUT_PATH = Path("plot/pmp_gurobi_no_filter.png")

# Selected site IDs are interpreted as 0-based candidate indices.
# Example: 0 means the first candidate point.
SELECTED_REP_RANKS =  [0, 2, 3, 5, 8, 11, 12, 38, 46, 49] # gurobi no filter
#SELECTED_REP_RANKS =  [2, 3, 5, 7, 8, 11, 27, 37, 38, 42] #gurobi
#SELECTED_REP_RANKS =  [27, 4, 7, 2, 0, 22, 43, 11, 3, 5] #ppo #[43  3 27  7  5  0  4  1 11 22]
# SELECTED_REP_RANKS =  [27, 5, 11, 2, 4, 22, 3, 0, 7, 43] #greedy 
# SELECTED_REP_RANKS =  [ 3, 84, 12, 35, 11, 24, 57, 37, 13,  0] #RANDOM
# SELECTED_REP_RANKS =  [11,  3, 45, 27,  2,  0,  5, 22,  7,  6] #GA
# SELECTED_REP_RANKS =  [27, 11,  7,  5, 22,  3,  2,  0,  4, 43] #LNS

#RELOCATION [71, 95, 69, 73, 94,  7, 91, 10,  3, 80]
# SELECTED_REP_RANKS =  [71, 95, 69, 73, 94,  7, 91, 10,  3, 80] #ORIGINAL
# SELECTED_REP_RANKS =  [71, 95, 24,  5, 27,  7,  0, 11,  3,  2] #PPO
# SELECTED_REP_RANKS =  [0, 2, 3, 5, 7, 11, 22, 27, 91, 95] #GUROBI
# SELECTED_REP_RANKS =[0, 2, 3, 5, 7, 11, 12, 38, 91, 95] # gurobi no filter


def configure_plot_fonts() -> None:
	# Build a cross-platform fallback chain for Chinese-capable sans-serif fonts.
	candidates = [
		"Noto Sans CJK SC",
		"Noto Sans CJK TC",
		"Noto Sans CJK JP",
		"WenQuanYi Micro Hei",
		"WenQuanYi Zen Hei",
		"SimHei",
		"Microsoft YaHei",
		"PingFang SC",
		"Arial Unicode MS",
	]
	available = {f.name for f in font_manager.fontManager.ttflist}
	selected = [name for name in candidates if name in available]

	# Always keep a final latin fallback to avoid missing font errors.
	if "DejaVu Sans" not in selected:
		selected.append("DejaVu Sans")

	plt.rcParams["font.family"] = "sans-serif"
	plt.rcParams["font.sans-serif"] = selected
	plt.rcParams["axes.unicode_minus"] = False


def load_data(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)

	required_columns = ["center_lon", "center_lat", "resident_population_2km", "rep_rank"]
	missing = [col for col in required_columns if col not in df.columns]
	if missing:
		raise ValueError(f"CSV 缺少必要列: {missing}")

	for col in required_columns:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	df = df.dropna(subset=required_columns).copy()
	df = df[df["resident_population_2km"] > 0]

	if df.empty:
		raise ValueError("清洗后无可绘制数据")

	return df


def build_heat_layer(
	lon: np.ndarray,
	lat: np.ndarray,
	pop: np.ndarray,
	grid_size: int = 700,
	radius_km: float = 4.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
	lon_min, lon_max = float(lon.min()), float(lon.max())
	lat_min, lat_max = float(lat.min()), float(lat.max())

	lon_margin = max((lon_max - lon_min) * 0.12, 0.01)
	lat_margin = max((lat_max - lat_min) * 0.12, 0.01)

	# Tighten the frame so the map uses more of the canvas.
	lon_margin *= 0.80
	lat_margin *= 0.80

	west = lon_min - lon_margin
	east = lon_max + lon_margin
	south = lat_min - lat_margin
	north = lat_max + lat_margin

	hist, _, _ = np.histogram2d(
		lon,
		lat,
		bins=grid_size,
		range=[[west, east], [south, north]],
		weights=pop,
	)

	# Spread each point's population over an approximately 2km neighborhood.
	mean_lat = float(np.mean(lat))
	deg_per_km_lat = 1.0 / 111.0
	deg_per_km_lon = 1.0 / max(111.0 * np.cos(np.deg2rad(mean_lat)), 1e-6)

	# Use 3-sigma ~= radius so influence mainly falls within radius_km.
	sigma_lon_deg = (radius_km * deg_per_km_lon) / 3.0
	sigma_lat_deg = (radius_km * deg_per_km_lat) / 3.0

	cell_lon_deg = (east - west) / grid_size
	cell_lat_deg = (north - south) / grid_size
	sigma_x = max(sigma_lon_deg / max(cell_lon_deg, 1e-9), 1.0)
	sigma_y = max(sigma_lat_deg / max(cell_lat_deg, 1e-9), 1.0)

	# Transpose to align x=lon, y=lat for imshow.
	heat = gaussian_filter(hist.T, sigma=(sigma_y, sigma_x))
	if np.nanmax(heat) > 0:
		heat = heat / np.nanmax(heat)
		heat = heat ** 0.7

	extent = (west, east, south, north)
	return heat, extent


def plot_population_map(
	csv_path: Path = CSV_PATH,
	output_path: Path = OUTPUT_PATH,
	selected_rep_ranks: list[int] | None = None,
) -> Path:
	configure_plot_fonts()
	selected_rep_ranks = selected_rep_ranks or SELECTED_REP_RANKS
	df = load_data(csv_path)
	# Keep a stable candidate ordering so index 0 always means "the first candidate".
	df = df.sort_values("rep_rank").reset_index(drop=True)

	lon = df["center_lon"].to_numpy(dtype=float)
	lat = df["center_lat"].to_numpy(dtype=float)
	pop = df["resident_population_2km"].to_numpy(dtype=float)

	heat, extent = build_heat_layer(lon, lat, pop)
	west, east, south, north = extent

	fig, ax = plt.subplots(figsize=(14, 12), dpi=200)
	ax.set_facecolor("#f7f6f2")

	heat_img = ax.imshow(
		heat,
		extent=extent,
		origin="lower",
		cmap="YlOrRd",
		alpha=0.80,
		interpolation="bilinear",
		zorder=1,
	)

	# 100 candidate points are shown on the map for spatial reference.
	ax.scatter(
		lon,
		lat,
		s=16,
		c="#1f78b4",
		edgecolors="white",
		linewidths=0.45,
		alpha=0.85,
		zorder=3,
		label="备选点(100)",
	)

	# Use 0-based indices for selected points.
	selected_indices = [int(v) for v in selected_rep_ranks]
	valid_indices = [idx for idx in selected_indices if 0 <= idx < len(df)]
	missing_indices = sorted(set(selected_indices) - set(valid_indices))
	if missing_indices:
		print(f"警告: 以下 selected index 超出范围 [0, {len(df)-1}]: {missing_indices}")

	selected_df = df.iloc[valid_indices].copy()
	selected_df["selected_idx"] = valid_indices

	ax.scatter(
		selected_df["center_lon"],
		selected_df["center_lat"],
		marker="D",
		s=135,
		c="#1067ca",
		edgecolors="black",
		linewidths=0.9,
		zorder=5,
		label="选址点",
	)

	for _, row in selected_df.iterrows():
		ax.text(
			row["center_lon"],
			row["center_lat"],
			f"{int(row['selected_idx'])}",
			fontsize=13,
			color="black",
			ha="left",
			va="bottom",
			zorder=6,
			bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.5},
		)

	# ax.set_title("洪山区人口热力图", fontsize=16, pad=10)
	ax.set_xlabel("经度", fontsize=22)
	ax.set_ylabel("纬度", fontsize=22)
	ax.tick_params(axis="both", labelsize=18)
	ax.grid(True, linestyle="--", alpha=0.25, zorder=0)
	ax.legend(loc="upper left", frameon=True, fontsize=20)

	cbar_heat = fig.colorbar(heat_img, ax=ax, fraction=0.035, pad=0.02)
	cbar_heat.set_label("人口热力程度", fontsize=22)
	cbar_heat.ax.tick_params(labelsize=18)

	ax.set_xlim(west, east)
	ax.set_ylim(south, north)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)

	return output_path


def main() -> None:
	output_file = plot_population_map()
	print(f"已生成: {output_file.resolve()}")


if __name__ == "__main__":
	main()
