import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Standard normal PDF/CDF (array-safe)
# ----------------------------
def phi(z):
    z = np.asarray(z, dtype=np.float64)
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)

def Phi(z):
    z = np.asarray(z, dtype=np.float64)
    try:
        from scipy.special import ndtr  # preferred, stable
        return ndtr(z)
    except ImportError:
        import math
        erf_vec = np.vectorize(math.erf)  # fallback
        return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))

# ----------------------------
# Weights and probabilities in Z-domain
# ----------------------------
def p_sub(t0, t1):
    return 2.0 * (Phi(t1) - Phi(t0))

def w_zero(t0):
    return 2.0 * Phi(t0) - 1.0 - 2.0 * t0 * phi(t0)

def w_norm(t1):
    return 2.0 * (t1 * phi(t1) + (1.0 - Phi(t1)))

# ----------------------------
# QSNR models (baseline)
# ----------------------------
def qsnr_int(kappa, b, rho=1.5):
    kappa = np.asarray(kappa, dtype=np.float64)
    return 10.8 + 6.02 * (b - 1) - 20.0 * np.log10(rho) - 20.0 * np.log10(kappa)

def qsnr_fp(kappa, M, B, Qmax, rho=1.5):
    kappa = np.asarray(kappa, dtype=np.float64)

    # Coefficients from proof
    alpha_M = 1.0 / (24.0 * (2.0 ** (2 * M)))
    beta = (2.0 ** (2.0 * (1 - B - M))) / (12.0 * (Qmax ** 2))

    # Thresholds in Z-domain (T/σ)
    scale = (rho * kappa) / Qmax
    tau0 = scale * (2.0 ** (-B - M))  # T0/σ
    tau1 = scale * (2.0 ** (1 - B))   # TN/σ

    # Fractions
    psub  = p_sub(tau0, tau1)
    wzero = w_zero(tau0)
    wnorm = w_norm(tau1)

    # Relative MSE and QSNR
    err_power = alpha_M * wnorm + beta * (rho * kappa) ** 2 * psub + wzero
    return -10.0 * np.log10(err_power)

# ----------------------------
# QSNR models (NV corrections, k=16)
# ----------------------------
def qsnr_int_nv(kappa, b, k=16, rho=1.5):
    kappa = np.asarray(kappa, dtype=np.float64)
    # 原始相对 MSE（忽略 -1 与饱和边界影响）
    Qmax_approx = 2.0 ** (b - 1)  # 近似 (2^{b-1})
    R0 = (rho * kappa) ** 2 / (12.0 * (Qmax_approx ** 2))
    R_corr = R0 * (1.0 - 1.0 / k)
    return -10.0 * np.log10(R_corr)

def qsnr_fp_nv(kappa, M, B, Qmax, k=16, rho=1.5):
    kappa = np.asarray(kappa, dtype=np.float64)

    alpha_M = 1.0 / (24.0 * (2.0 ** (2 * M)))
    beta = (2.0 ** (2.0 * (1 - B - M))) / (12.0 * (Qmax ** 2))

    # Thresholds in Z-domain (T/σ)
    scale = (rho * kappa) / Qmax
    tau0 = scale * (2.0 ** (-B - M))  # T0/σ
    tau1 = scale * (2.0 ** (1 - B))   # TN/σ

    # Fractions
    psub  = p_sub(tau0, tau1)
    wzero = w_zero(tau0)
    wnorm = w_norm(tau1)

    # 正常区修正：wnorm_corr = max(0, wnorm - κ^2/k)
    wnorm_corr = np.maximum(0.0, wnorm - (kappa ** 2) / float(k))

    err_power = alpha_M * wnorm_corr + beta * (rho * kappa) ** 2 * psub + wzero
    return -10.0 * np.log10(err_power)

# ----------------------------
# Styling
# ----------------------------
bit_colors = {"8": "#4c72b0", "6": "#2ca02c", "4": "#d62728"}  # 8/6/4-bit base colors
line_styles = {"INT": "-", "FP": "--"}                         # scheme styles
markers     = {"INT": "o", "FP": "s"}                          # scheme markers

E4M3_PURPLE = "#9467bd"

def get_curve_color(kind, bits, scale):
    if bits == 4 and scale == "E4M3":
        return E4M3_PURPLE
    return bit_colors[str(bits)]
def get_display_label(kind, bits, scale):
    if scale == "UE8M0":
        if kind == "INT":
            return f"MXINT{bits}"
        else:
            return f"MXFP{bits}"
    elif scale == "E4M3":
        if kind == "INT" and bits == 4:
            return "NVINT4"
        elif kind == "FP" and bits == 4:
            return "NVFP4"
    return f"{kind}{bits} ({scale})"
# ----------------------------
# Formats and parameters
# ----------------------------
formats = [
    ("INT8",      "INT", 8, {"b": 8},                        1.5,  "UE8M0"),
    ("FP8 E4M3",  "FP",  8, {"M": 3, "B": 7, "Qmax": 448.0}, 1.5,  "UE8M0"),

    ("INT6",      "INT", 6, {"b": 6},                        1.5,  "UE8M0"),
    ("FP6 E2M3",  "FP",  6, {"M": 3, "B": 1, "Qmax": 7.5},   1.5,  "UE8M0"),

    ("INT4",      "INT", 4, {"b": 4},                        1.5,  "UE8M0"),
    ("FP4 E2M1",  "FP",  4, {"M": 1, "B": 1, "Qmax": 6.0},   1.5,  "UE8M0"),

    # NV 系列（E4M3 标签，rho 略小）
    ("INT4",      "INT", 4, {"b": 4},                        1.05, "E4M3"),
    ("FP4 E2M1",  "FP",  4, {"M": 1, "B": 1, "Qmax": 6.0},   1.05, "E4M3"),
]

# ----------------------------
# Sweep κ and compute
# ----------------------------
kappa = np.linspace(1.0, 12, 600)

curves = {}  # label -> y
pairs_by_group = {}  # (bits(str), scale) -> {"INT": (label, y), "FP": (label, y)}

for name, kind, bits, params, rho, scale in formats:
    if kind == "INT":
        if scale == "E4M3":
            y = qsnr_int_nv(kappa, b=params["b"], k=16, rho=rho)
        else:
            y = qsnr_int(kappa, b=params["b"], rho=rho)
    else:
        if scale == "E4M3":
            y = qsnr_fp_nv(kappa, M=params["M"], B=params["B"], Qmax=params["Qmax"], k=16, rho=rho)
        else:
            y = qsnr_fp(kappa, M=params["M"], B=params["B"], Qmax=params["Qmax"], rho=rho)

    disp_label = get_display_label(kind, bits, scale)
    curves[disp_label] = y

    gkey = (str(bits), scale)
    pairs_by_group.setdefault(gkey, {})
    pairs_by_group[gkey][kind] = (disp_label, y)

# ----------------------------
# Plot
# ----------------------------
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

fig, ax = plt.subplots(figsize=(10, 6), dpi=140)

line_handles = {}  # label -> Line2D
for name, kind, bits, params, rho, scale in formats:
    color = get_curve_color(kind, bits, scale)
    ls = line_styles[kind]
    mk = markers[kind]
    disp_label = get_display_label(kind, bits, scale)
    y = curves[disp_label]

    (line,) = ax.plot(
        kappa, y,
        label=disp_label,
        color=color,
        linestyle=ls,
        marker=mk,
        markevery=40,
        markersize=5.5,
        linewidth=2.2,
        alpha=0.95,
        zorder=3,
    )
    line_handles[disp_label] = line

row_order = [('8', 'UE8M0'), ('6', 'UE8M0'), ('4', 'UE8M0'), ('4', 'E4M3')]

left_col_labels  = []
right_col_labels = []

for b, sc in row_order:
    grp = pairs_by_group.get((b, sc), {})
    if "INT" in grp:
        left_col_labels.append(grp["INT"][0])
    if "FP" in grp:
        right_col_labels.append(grp["FP"][0])

ordered_labels = left_col_labels + right_col_labels
handles = [line_handles[n] for n in ordered_labels]

ax.legend(
    handles=handles,
    labels=ordered_labels,
    ncol=2,             
    loc='upper right',
    frameon=True,
    fontsize=16,
)

def find_intersections(x, y_a, y_b):
    d = y_a - y_b
    idx = np.where(d[:-1] * d[1:] <= 0)[0]  # 符号变化区间
    points = []
    for i in idx:
        x0, x1 = x[i], x[i+1]
        d0, d1 = d[i], d[i+1]
        if d1 == d0:
            continue
        xc = x0 - d0 * (x1 - x0) / (d1 - d0)
        yc = np.interp(xc, [x0, x1], [y_a[i], y_a[i+1]])
        points.append((xc, yc))
    return points

for (b, sc), grp in pairs_by_group.items():
    if "INT" in grp and "FP" in grp:
        name_int, y_int = grp["INT"]
        name_fp,  y_fp  = grp["FP"]
        pts = find_intersections(kappa, y_int, y_fp)
        if not pts:
            continue

        color_ann = get_curve_color("FP", int(b), sc)
        offset_map = {
            ('8', 'UE8M0'): (12, 12),
            ('6', 'UE8M0'): (12, 12),
            ('4', 'UE8M0'): (-70, -50),
            ('4', 'E4M3'): (-15, 13),
        }
        dx, dy = offset_map.get((b, sc), (10, 10))

        for (xc, yc) in pts:
            ax.scatter(xc, yc, color=color_ann, edgecolors='black', s=70, marker='X', zorder=6)
            ax.annotate(
                f'κ={xc:.2f}\n{yc:.2f} dB',
                xy=(xc, yc),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=16,
                color=color_ann,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color_ann, alpha=0.85),
                arrowprops=dict(arrowstyle='->', color=color_ann, lw=1.0, alpha=0.9),
                zorder=7
            )

rho_set = {entry[4] for entry in formats}
rho_text = f'{list(rho_set)[0]:.2f}' if len(rho_set) == 1 else 'varies'

ax.set_xlabel('κ (crest factor)', fontsize=16)
ax.set_ylabel('QSNR (dB)', fontsize=16)
ax.grid(True, which='both', linestyle='--', alpha=0.6)

# y 轴范围自适应留白
ymin = min(np.min(v) for v in curves.values())
ymax = max(np.max(v) for v in curves.values())
ax.set_ylim(ymin - 2, ymax + 2)

plt.tight_layout()
plt.savefig('./qsnr_vs_kappa.png', bbox_inches='tight', dpi=300)
print("QSNR vs κ plot saved as qsnr_vs_kappa.png")
plt.show()
