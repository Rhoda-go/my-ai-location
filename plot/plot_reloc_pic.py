import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


SCENARIOS = ["A1B1", "A1B2", "A2B1", "A2B2"]


def configure_plot_fonts():
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


def load_reloc_pic_data(csv_path: Path):
    methods = []
    gap_data = []
    time_data = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows[2:]:
        if not row or not row[0].strip():
            continue

        method = row[0].strip()
        values = [cell.strip() for cell in row[1:]]

        gaps = []
        times = []
        for idx in range(0, len(values), 2):
            if idx + 1 >= len(values):
                break
            gap_str = values[idx]
            time_str = values[idx + 1]
            if not gap_str or not time_str:
                continue
            gaps.append(float(gap_str))
            times.append(float(time_str))

        if len(gaps) != len(SCENARIOS) or len(times) != len(SCENARIOS):
            raise ValueError(
                f"Row for {method} does not contain 4 Gap/T pairs: {row}"
            )

        methods.append(method)
        gap_data.append(gaps)
        time_data.append(times)

    return methods, np.array(gap_data), np.array(time_data)


def plot_gap_and_time(csv_path: str = "plot/reloc_pic.csv", output_path: str = "plot/reloc_pic.png"):
    csv_file = Path(csv_path)
    methods, gap_data, time_data = load_reloc_pic_data(csv_file)

    x = np.arange(len(SCENARIOS), dtype=float)
    method_count = len(methods)
    bar_width = 0.14 if method_count >= 5 else 0.18
    offsets = (np.arange(method_count) - (method_count - 1) / 2.0) * bar_width

    colors = [
        
        "#726F6F",
        "#b2aaaa",
        "#ff7f0e",
        "#1f77b4",
        "#d62728",
        # "#e377c2",   
        # "#bcbd22",
        # "#17becf",
        # "#9467bd",
        # "#2ca02c",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]

    configure_plot_fonts()

    fig = plt.figure(figsize=(12, 7.2))
    # Bottom axis (0-1) occupies one quarter of the height.
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_gap = fig.add_subplot(grid[:, 0])
    ax_time_top = fig.add_subplot(grid[0, 0], sharex=ax_gap)
    ax_time_bottom = fig.add_subplot(grid[1, 0], sharex=ax_gap)

    ax_time_top.patch.set_alpha(0.0)
    ax_time_bottom.patch.set_alpha(0.0)
    ax_time_top.spines["bottom"].set_visible(False)
    ax_time_bottom.spines["top"].set_visible(False)
    ax_time_top.tick_params(labelbottom=False)
    ax_time_bottom.xaxis.tick_bottom()

    low_break = 1.0
    high_start = low_break
    max_time = float(np.max(time_data)) if time_data.size else 1.0
    high_end = max(low_break + 1.0, max_time * 1.08)
    shifted_methods = {"Random", "LNS", "PPO"}
    time_shift = low_break / 8.0

    for idx, method in enumerate(methods):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        x_pos = x + offsets[idx]
        time_vals = time_data[idx]
        y_offset = time_shift if method in shifted_methods else 0.0

        # Draw bars on both axes with clipped heights so each bar stays continuous
        # across the broken y-axis instead of being partially hidden.
        bottom_part = np.minimum(time_vals, low_break)
        top_part = np.maximum(time_vals - low_break, 0.0)
        has_top_part = top_part > 0.0

        ax_gap.plot(
            x_pos,
            gap_data[idx],
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=2.0,
            label=method,
            zorder=3,
        )
        if np.any(has_top_part):
            ax_time_top.bar(
                x_pos[has_top_part],
                top_part[has_top_part],
                bottom=low_break + y_offset,
                width=bar_width * 0.92,
                color=color,
                alpha=0.9,
                edgecolor=color,
                linewidth=1.0,
                zorder=1,
            )
        ax_time_bottom.bar(
            x_pos,
            bottom_part,
            bottom=y_offset,
            width=bar_width * 0.92,
            color=color,
            alpha=0.9,
            edgecolor=color,
            linewidth=1.0,
            zorder=1,
        )

        for x_i, t_i in zip(x_pos, time_vals):
            if t_i > low_break:
                y_top = min(t_i + y_offset + max((high_end - high_start) * 0.01, 0.05), high_end - 0.08)
                ax_time_top.text(
                    x_i,
                    y_top,
                    f"{t_i:.2f} s",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color,
                    zorder=4,
                )
            else:
                y_bottom = min(t_i + y_offset + low_break * 0.03, low_break - 0.02 + y_offset)
                ax_time_bottom.text(
                    x_i,
                    y_bottom,
                    f"{t_i:.2f} s",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color,
                    zorder=4,
                )

    ax_gap.set_xticks(x)
    ax_gap.tick_params(labelbottom=False)
    ax_gap.set_ylabel("Gap (%)", fontsize=15)
    #ax_gap.set_title("Gap and Runtime Comparison Across Scenarios", fontsize=14, pad=14)

    ax_gap.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax_gap.set_axisbelow(True)

    max_gap = float(np.max(gap_data)) if gap_data.size else 1.0
    ax_gap.set_ylim(0, max(5.0, max_gap * 1.25))

    ax_time_bottom.set_ylim(0, low_break + time_shift)
    ax_time_top.set_ylim(high_start, high_end)
    ax_time_bottom.set_yticks([0.0, 0.5, 1.0])
    ax_time_bottom.set_ylabel("T (s)", fontsize=15)
    ax_time_top.yaxis.set_label_position("right")
    ax_time_bottom.yaxis.set_label_position("right")
    ax_time_top.yaxis.tick_right()
    ax_time_bottom.yaxis.tick_right()

    # Use the same saturated colors for bars and markers, but keep the bars slightly translucent
    # so line markers remain readable on top of them.
    for ax in (ax_time_top, ax_time_bottom):
        ax.grid(axis="y", linestyle="--", alpha=0.18, zorder=0)
        ax.set_axisbelow(True)

    ax_time_top.spines["right"].set_visible(True)
    ax_time_bottom.spines["right"].set_visible(True)
    ax_time_top.spines["left"].set_visible(False)
    ax_time_bottom.spines["left"].set_visible(False)

    d = 0.012
    kwargs = dict(color="k", clip_on=False, linewidth=1.0)
    ax_time_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_time_top.transAxes, **kwargs)
    ax_time_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_time_bottom.transAxes, **kwargs)

    ax_time_top.set_xticks([])
    ax_time_bottom.set_xticks(x)
    ax_time_bottom.set_xticklabels(SCENARIOS, fontsize=15)
    # fig.supxlabel("Scenarios", fontsize=18, y=0.03)

    handles, labels = ax_gap.get_legend_handles_labels()
    ax_gap.legend(handles, labels, loc="upper left", ncol=3, frameon=True, title="方法",fontsize=12, title_fontsize=13)

    fig.tight_layout()
    output_file = Path(output_path)
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved figure to: {output_file.resolve()}")


if __name__ == "__main__":
    plot_gap_and_time()