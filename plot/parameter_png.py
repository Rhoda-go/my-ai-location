import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_plot_fonts() -> None:
	# Build a cross-platform fallback chain for Chinese-capable sans-serif fonts.
	candidates = [
		'Noto Sans CJK SC',
		'Noto Sans CJK TC',
		'Noto Sans CJK JP',
		'WenQuanYi Micro Hei',
		'WenQuanYi Zen Hei',
		'SimHei',
		'Microsoft YaHei',
		'PingFang SC',
		'Arial Unicode MS',
	]
	available = {f.name for f in font_manager.fontManager.ttflist}
	selected = [name for name in candidates if name in available]

	# Always keep a final latin fallback to avoid missing font errors.
	if 'DejaVu Sans' not in selected:
		selected.append('DejaVu Sans')

	plt.rcParams['font.family'] = 'sans-serif'
	plt.rcParams['font.sans-serif'] = selected
	plt.rcParams['axes.unicode_minus'] = False


def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
	coef = np.exp(np.math.lgamma(a + b) - np.math.lgamma(a) - np.math.lgamma(b))
	return coef * np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
	coef = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
	return coef * np.exp(-0.5 * np.square((x - mu) / sigma))


def normalize_on_range(y: np.ndarray, x: np.ndarray) -> np.ndarray:
	area = np.trapz(y, x)
	if area <= 0:
		return y
	return y / area


def distribution_moments(x: np.ndarray, pdf: np.ndarray) -> tuple[float, float]:
	mean = float(np.trapz(x * pdf, x))
	var = float(np.trapz(np.square(x - mean) * pdf, x))
	return mean, var


def main() -> None:
	configure_plot_fonts()
	print(f"Using sans-serif fallback: {plt.rcParams['font.sans-serif']}")

	# Truncated intervals used in your code.
	alpha_low, alpha_high = 0.015, 0.86
	beta_low, beta_high = 0.05, 1.80

	x_alpha = np.linspace(alpha_low, alpha_high, 1000)
	x_beta = np.linspace(beta_low, beta_high, 1200)

	# Parameter set 1
	alpha_pdf_1 = normalize_on_range(beta_pdf(x_alpha, a=1.2, b=8.0), x_alpha)
	beta_pdf_1 = normalize_on_range(normal_pdf(x_beta, mu=0.32, sigma=0.31), x_beta)

	# Parameter set 2
	alpha_pdf_2 = normalize_on_range(beta_pdf(x_alpha, a=2.0, b=5.0), x_alpha)
	beta_pdf_2 = normalize_on_range(normal_pdf(x_beta, mu=0.25, sigma=0.20), x_beta)

	alpha_mean_1, alpha_var_1 = distribution_moments(x_alpha, alpha_pdf_1)
	alpha_mean_2, alpha_var_2 = distribution_moments(x_alpha, alpha_pdf_2)

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

	axes[0].plot(x_alpha, alpha_pdf_1, color="#2F5597", lw=2.2, label="A1: Beta分布(1.2, 8.0)")
	axes[0].plot(x_alpha, alpha_pdf_2, color="#4BACC6", lw=2.2, label="A2: Beta分布(2.0, 5.0)")
	# axes[0].set_title("alpha Distribution Curves (Truncated)")
	axes[0].set_xlabel("A")
	axes[0].set_ylabel("概率密度")
	axes[0].grid(alpha=0.2)
	axes[0].legend()
	axes[0].text(
		0.03,
		0.97,
		f"A1: mean={alpha_mean_1:.3f}, var={alpha_var_1:.4f}",
		transform=axes[0].transAxes,
		va="top",
		color="#2F5597",
	)
	axes[0].text(
		0.03,
		0.90,
		f"A2: mean={alpha_mean_2:.3f}, var={alpha_var_2:.4f}",
		transform=axes[0].transAxes,
		va="top",
		color="#4BACC6",
	)

	axes[1].plot(x_beta, beta_pdf_1, color="#C0504D", lw=2.2, label="B1: 正态分布(0.32, 0.31)")
	axes[1].plot(x_beta, beta_pdf_2, color="#DC7B3F", lw=2.2, label="B2: 正态分布(0.25, 0.20)")
	# axes[1].set_title("beta Distribution Curves (Truncated)")
	axes[1].set_xlabel("beta")
	axes[1].set_ylabel("概率密度")
	
	axes[1].grid(alpha=0.2)
	axes[1].legend()

	# fig.suptitle("alpha / beta Parameter Set Curve Comparison", fontsize=14)
	fig.tight_layout()

	output_file = "alpha_beta_distribution_curve_comparison.png"
	plt.savefig(output_file, dpi=180, bbox_inches="tight")
	plt.show()

	print(f"Saved figure to: {output_file}")
	print("Plotted truncated distribution curves without random sampling.")


if __name__ == "__main__":
	main()
