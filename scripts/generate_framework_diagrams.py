#!/usr/bin/env python
"""Generate publication-grade framework diagrams for CIVIC-SAFE.

Generates:
  1. fig_architecture.pdf/.png: End-to-End System Architecture (Full-width overview)
  2. fig_feedback_loop.pdf/.png: Causal Feedback Loop Dynamics vs. Latent Deflation
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(r"C:\Users\kamle\OneDrive\Desktop\PCC project")
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def draw_system_architecture():
    """Generate Figure 1: Comprehensive End-to-End System Architecture."""
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Colors (Academic modern palette)
    c_data = "#E8F0FE"       # Light blue
    c_data_border = "#1A73E8"
    c_model = "#FEF7E0"      # Light yellow/amber
    c_model_border = "#F9AB00"
    c_ens = "#E6F4EA"        # Light green
    c_ens_border = "#137333"
    c_cp = "#FCE8E6"         # Light red/pink
    c_cp_border = "#D93025"
    c_latent = "#F3E8FD"     # Light purple
    c_latent_border = "#9334E6"
    c_policy = "#E0F2F1"     # Light teal
    c_policy_border = "#00796B"

    # Stage 1: Input Data & Graph Topology
    b1 = patches.FancyBboxPatch((0.3, 0.8), 2.2, 4.8, boxstyle="round,pad=0.1",
                                facecolor=c_data, edgecolor=c_data_border, linewidth=1.5)
    ax.add_patch(b1)
    ax.text(1.4, 5.3, "STAGE 1: INPUTS\n& URBAN GRAPHS", ha="center", va="center", weight="bold", fontsize=10, color="#1A73E8")

    # Inner boxes for Stage 1
    ax.add_patch(patches.FancyBboxPatch((0.5, 3.8), 1.8, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_data_border, lw=0.8))
    ax.text(1.4, 4.5, "Weekly Crime Tensors", ha="center", va="center", weight="bold", fontsize=8.5)
    ax.text(1.4, 4.1, r"$\mathbf{Y} \in \mathbb{N}^{S \times T \times C}$" + "\n(Violent, Property, Drug)", ha="center", va="center", fontsize=7.5, color="#555")

    ax.add_patch(patches.FancyBboxPatch((0.5, 2.4), 1.8, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_data_border, lw=0.8))
    ax.text(1.4, 3.1, "Dual Adjacency", ha="center", weight="bold", fontsize=8.5)
    ax.text(1.4, 2.7, r"$\mathcal{E}_{\mathrm{queen}}$: Geographic Spill" + "\n" + r"$\mathcal{E}_{\mathrm{knn}}$: Demographic k-NN", ha="center", fontsize=7.5, color="#555")

    ax.add_patch(patches.FancyBboxPatch((0.5, 1.0), 1.8, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_data_border, lw=0.8))
    ax.text(1.4, 1.7, "Exogenous Features", ha="center", weight="bold", fontsize=8.5)
    ax.text(1.4, 1.3, "ACS Demographics,\nTrailing Anchors", ha="center", fontsize=7.5, color="#555")

    # Arrow 1 -> 2
    ax.annotate("", xy=(2.9, 3.2), xytext=(2.5, 3.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555"))

    # Stage 2: Deep Sequential Spatiotemporal Encoder
    b2 = patches.FancyBboxPatch((2.9, 0.8), 2.5, 4.8, boxstyle="round,pad=0.1",
                                facecolor=c_model, edgecolor=c_model_border, linewidth=1.5)
    ax.add_patch(b2)
    ax.text(4.15, 5.3, "STAGE 2: BACKBONE\n(SEQUENTIAL V1)", ha="center", va="center", weight="bold", fontsize=10, color="#B06000")

    ax.add_patch(patches.FancyBboxPatch((3.1, 3.8), 2.1, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_model_border, lw=0.8))
    ax.text(4.15, 4.5, "Dual-Graph GATv2", ha="center", weight="bold", fontsize=8.5)
    ax.text(4.15, 4.1, r"Spatial Attention: $\gamma_{ij}$" + "\n" + r"over $\mathcal{E}_{\mathrm{queen}} \cup \mathcal{E}_{\mathrm{knn}}$", ha="center", fontsize=7.5, color="#555")

    ax.add_patch(patches.FancyBboxPatch((3.1, 2.4), 2.1, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_model_border, lw=0.8))
    ax.text(4.15, 3.1, "Causal Transformer", ha="center", weight="bold", fontsize=8.5)
    ax.text(4.15, 2.7, "Staged Temporal Attention\nMasked Causal Dynamics", ha="center", fontsize=7.5, color="#555")

    ax.add_patch(patches.FancyBboxPatch((3.1, 1.0), 2.1, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_model_border, lw=0.8))
    ax.text(4.15, 1.7, "ZINB Output Head", ha="center", weight="bold", fontsize=8.5)
    ax.text(4.15, 1.3, r"$\pi$ (zero), $\mu$ (mean), $r$ (disp)" + "\n" + r"with $r$-floor regularization", ha="center", fontsize=7.5, color="#555")

    # Arrow 2 -> 3
    ax.annotate("", xy=(5.8, 3.2), xytext=(5.4, 3.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555"))

    # Stage 3: Regularized EMOS Ensembling
    b3 = patches.FancyBboxPatch((5.8, 1.2), 2.0, 4.0, boxstyle="round,pad=0.1",
                                facecolor=c_ens, edgecolor=c_ens_border, linewidth=1.5)
    ax.add_patch(b3)
    ax.text(6.8, 4.9, "STAGE 3: ENSEMBLE\n(5-SEED EMOS)", ha="center", va="center", weight="bold", fontsize=10, color="#137333")

    ax.add_patch(patches.FancyBboxPatch((6.0, 3.2), 1.6, 1.3, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_ens_border, lw=0.8))
    ax.text(6.8, 4.1, "5 Independent Seeds", ha="center", weight="bold", fontsize=8.5)
    ax.text(6.8, 3.6, "Seeds {42, 137, 256,\n512, 1024}", ha="center", fontsize=7.5, color="#555")

    ax.add_patch(patches.FancyBboxPatch((6.0, 1.6), 1.6, 1.3, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_ens_border, lw=0.8))
    ax.text(6.8, 2.5, "Entropy Regularized", ha="center", weight="bold", fontsize=8.5)
    ax.text(6.8, 2.0, r"$\min_{\mathbf{w}} \mathrm{CRPS} - \lambda H(\mathbf{w})$" + "\n" + r"CRPS: $3.36 \to 2.82$", ha="center", fontsize=7.5, color="#555")

    # Arrow 3 -> 4
    ax.annotate("", xy=(8.2, 3.2), xytext=(7.8, 3.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555"))

    # Stage 4: Conformal Calibration Layer
    b4 = patches.FancyBboxPatch((8.2, 0.8), 2.2, 4.8, boxstyle="round,pad=0.1",
                                facecolor=c_cp, edgecolor=c_cp_border, linewidth=1.5)
    ax.add_patch(b4)
    ax.text(9.3, 5.3, "STAGE 4: CONFORMAL\nCALIBRATION", ha="center", va="center", weight="bold", fontsize=10, color="#D93025")

    ax.add_patch(patches.FancyBboxPatch((8.4, 3.8), 1.8, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_cp_border, lw=0.8))
    ax.text(9.3, 4.5, "CQR Scores", ha="center", weight="bold", fontsize=8.5)
    ax.text(9.3, 4.1, r"$V_i = \max(q_{\alpha/2}-y, y-q_{1-\alpha/2})$" + "\nNon-conformity score", ha="center", fontsize=7.5, color="#555")

    ax.add_patch(patches.FancyBboxPatch((8.4, 2.4), 1.8, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_cp_border, lw=0.8))
    ax.text(9.3, 3.1, "10 CP Variants", ha="center", weight="bold", fontsize=8.5)
    ax.text(9.3, 2.7, "Mondrian, ECRC, Split,\nVariance-Scaled CP", ha="center", fontsize=7.5, color="#555")

    ax.add_patch(patches.FancyBboxPatch((8.4, 1.0), 1.8, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_cp_border, lw=0.8))
    ax.text(9.3, 1.7, "Pre-registered Gate", ha="center", weight="bold", fontsize=8.5)
    ax.text(9.3, 1.3, r"Disparity $\leq 0.03$" + "\n" + r"Coverage $\geq 90\%$", ha="center", fontsize=7.5, color="#555")

    # Arrow 4 -> 5
    ax.annotate("", xy=(10.8, 3.2), xytext=(10.4, 3.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555"))

    # Stage 5: Latent Deflation Operator (The Core Theoretical Novelty)
    b5 = patches.FancyBboxPatch((10.8, 0.8), 2.2, 4.8, boxstyle="round,pad=0.1",
                                facecolor=c_latent, edgecolor=c_latent_border, linewidth=1.5)
    ax.add_patch(b5)
    ax.text(11.9, 5.3, "STAGE 5: LATENT\nCORRECTION (NEW)", ha="center", va="center", weight="bold", fontsize=10, color="#9334E6")

    ax.add_patch(patches.FancyBboxPatch((11.0, 3.8), 1.8, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_latent_border, lw=0.8))
    ax.text(11.9, 4.5, "Fixed-Point Inversion", ha="center", weight="bold", fontsize=8.5)
    ax.text(11.9, 4.1, r"$\hat{\lambda}_s = \mu_s / (\mu_s / M)^\kappa$" + "\n" + r"Recovers true $\lambda$", ha="center", fontsize=7.5, color="#555")

    ax.add_patch(patches.FancyBboxPatch((11.0, 2.4), 1.8, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_latent_border, lw=0.8))
    ax.text(11.9, 3.1, "Principled Abstention", ha="center", weight="bold", fontsize=8.5)
    ax.text(11.9, 2.7, r"Abstains if $m_s \notin [\bar{m}^{-1}, \bar{m}]$" + "\n" + r"or $\kappa \geq 0.9$ (Thm 2)", ha="center", fontsize=7.5, color="#555")

    ax.add_patch(patches.FancyBboxPatch((11.0, 1.0), 1.8, 1.1, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_latent_border, lw=0.8))
    ax.text(11.9, 1.7, "Latent Guarantees", ha="center", weight="bold", fontsize=8.5)
    ax.text(11.9, 1.3, r"Latent Cov: $16\% \to 93\%$" + "\n" + r"on retained subset", ha="center", fontsize=7.5, color="#555")

    # Arrow 5 -> 6
    ax.annotate("", xy=(13.4, 3.2), xytext=(13.0, 3.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555"))

    # Stage 6: Downstream Deployments
    b6 = patches.FancyBboxPatch((13.4, 1.2), 1.3, 4.0, boxstyle="round,pad=0.1",
                                facecolor=c_policy, edgecolor=c_policy_border, linewidth=1.5)
    ax.add_patch(b6)
    ax.text(14.05, 4.9, "STAGE 6:\nPOLICY", ha="center", va="center", weight="bold", fontsize=10, color="#00796B")

    ax.add_patch(patches.FancyBboxPatch((13.55, 3.2), 1.0, 1.3, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_policy_border, lw=0.8))
    ax.text(14.05, 4.0, "OICC Dispatch", ha="center", weight="bold", fontsize=8.0)
    ax.text(14.05, 3.5, "98.9% hits\n-38% over-\nallocation", ha="center", fontsize=7.0, color="#555")

    ax.add_patch(patches.FancyBboxPatch((13.55, 1.6), 1.0, 1.3, boxstyle="round,pad=0.05", facecolor="white", edgecolor=c_policy_border, lw=0.8))
    ax.text(14.05, 2.4, "Safe Routing", ha="center", weight="bold", fontsize=8.0)
    ax.text(14.05, 1.9, "-76% exposure\ndisparity\ncut", ha="center", fontsize=7.0, color="#555")

    out_path = FIG_DIR / "fig_architecture.png"
    out_pdf = FIG_DIR / "fig_architecture.pdf"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path} and {out_pdf}")


def draw_feedback_loop():
    """Generate Figure 2: Causal Feedback Loop Dynamics vs. Latent Deflation Framework."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")

    # Panel A: The Runaway Feedback Loop (Ensign et al. / Standard Systems)
    ax1.text(5, 7.5, "(a) Uncorrected Observation-Biased Loop\n(Runaway Disparity Amplification)",
             ha="center", va="center", weight="bold", fontsize=11, color="#C0392B")

    # Nodes
    ax1.add_patch(patches.Circle((2, 5.5), 1.0, facecolor="#EBF5FB", edgecolor="#2980B9", lw=1.8))
    ax1.text(2, 5.5, "True Latent\nIncidence\n" + r"$\lambda_s$", ha="center", va="center", weight="bold", fontsize=9)

    ax1.add_patch(patches.Circle((5, 5.5), 1.0, facecolor="#FDEDEC", edgecolor="#C0392B", lw=1.8))
    ax1.text(5, 5.5, "Recorded\nCrime\n" + r"$y_s \sim \mathrm{Pois}(\lambda_s g)$", ha="center", va="center", weight="bold", fontsize=8.5)

    ax1.add_patch(patches.Circle((8, 5.5), 1.0, facecolor="#FEF9E7", edgecolor="#F39C12", lw=1.8))
    ax1.text(8, 5.5, "Predictive\nModel\n" + r"$\mu_s$", ha="center", va="center", weight="bold", fontsize=9)

    ax1.add_patch(patches.Circle((5, 2.0), 1.0, facecolor="#F4ECF7", edgecolor="#8E44AD", lw=1.8))
    ax1.text(5, 2.0, "Patrol\nAllocation\n" + r"$a_s = \phi(\mu_s)$", ha="center", va="center", weight="bold", fontsize=9)

    # Arrows
    ax1.annotate("", xy=(3.9, 5.5), xytext=(3.1, 5.5), arrowprops=dict(arrowstyle="->", lw=2, color="#2980B9"))
    ax1.text(3.5, 5.8, "Generates", ha="center", fontsize=8, color="#2980B9")

    ax1.annotate("", xy=(6.9, 5.5), xytext=(6.1, 5.5), arrowprops=dict(arrowstyle="->", lw=2, color="#C0392B"))
    ax1.text(6.5, 5.8, "Trains On", ha="center", fontsize=8, color="#C0392B")

    ax1.annotate("", xy=(5.8, 2.6), xytext=(7.4, 4.7), arrowprops=dict(arrowstyle="->", lw=2, color="#F39C12"))
    ax1.text(7.2, 3.3, r"Policy $\phi(\mu)$", ha="center", fontsize=8.5, color="#F39C12")

    ax1.annotate("", xy=(4.5, 4.5), xytext=(4.5, 3.1), arrowprops=dict(arrowstyle="->", lw=2.5, color="#C0392B", ls="--"))
    ax1.text(3.4, 3.8, "Discovered\nCrime $g(a_s)$\n" + r"$\kappa = \beta \rho$", ha="center", fontsize=8.5, weight="bold", color="#C0392B")

    res_box1 = patches.FancyBboxPatch((1.0, 0.2), 8.0, 1.1, boxstyle="round,pad=0.08",
                                     facecolor="#FDEDEC", edgecolor="#C0392B", lw=1.2)
    ax1.add_patch(res_box1)
    ax1.text(5, 0.75, r"Theorem 1: Recorded Disparity $\Delta_y = (\Delta_\lambda)^{\frac{1}{1-\kappa}}$ (Blowup at $\kappa=1$)" + "\n" +
                      r"Conformal coverage on latent: Collapses to $16.2\%$ (Confidently Wrong!)",
             ha="center", va="center", fontsize=8.5, color="#900C3F", weight="bold")

    # Panel B: CIVIC-SAFE Feedback-Corrected Latent Conformal Prediction
    ax2.text(5, 7.5, "(b) CIVIC-SAFE Deflation & Latent Conformal\n(Restoring Latent Reality & Bounding Bias)",
             ha="center", va="center", weight="bold", fontsize=11, color="#1E8449")

    # Nodes
    ax2.add_patch(patches.Circle((2, 5.5), 1.0, facecolor="#EBF5FB", edgecolor="#2980B9", lw=1.8))
    ax2.text(2, 5.5, "True Latent\nIncidence\n" + r"$\lambda_s$", ha="center", va="center", weight="bold", fontsize=9)

    ax2.add_patch(patches.Circle((5, 5.5), 1.0, facecolor="#FDEDEC", edgecolor="#C0392B", lw=1.8))
    ax2.text(5, 5.5, "Biased Record\n" + r"$y_s \sim \mathrm{Pois}(\lambda_s g)$", ha="center", va="center", weight="bold", fontsize=8.5)

    ax2.add_patch(patches.Circle((8, 5.5), 1.0, facecolor="#FEF9E7", edgecolor="#F39C12", lw=1.8))
    ax2.text(8, 5.5, "Record Forecaster\n" + r"$\mu_s \approx \lambda_s (\mu_s/M)^\kappa$", ha="center", va="center", weight="bold", fontsize=8.5)

    ax2.add_patch(patches.FancyBboxPatch((6.8, 3.0), 2.4, 1.4, boxstyle="round,pad=0.05", facecolor="#E8F8F5", edgecolor="#1E8449", lw=1.8))
    ax2.text(8.0, 4.0, "Deflation Filter (Thm 2)", ha="center", weight="bold", fontsize=8.5, color="#1E8449")
    ax2.text(8.0, 3.4, r"$\hat{\lambda}_s = \frac{\mu_s}{(\mu_s/M)^\kappa}$" + "\nExact Inverse", ha="center", fontsize=8, color="#111")

    ax2.add_patch(patches.FancyBboxPatch((3.8, 3.0), 2.4, 1.4, boxstyle="round,pad=0.05", facecolor="#FEF9E7", edgecolor="#D4AC0D", lw=1.8))
    ax2.text(5.0, 4.0, "Abstention Guard", ha="center", weight="bold", fontsize=8.5, color="#B7950B")
    ax2.text(5.0, 3.4, r"Abstain if $m_s \notin [\bar{m}^{-1}, \bar{m}]$" + "\n" + r"Bounds Error: $|\kappa-\hat{\kappa}|\frac{\log \bar{m}}{\hat{\kappa}}$", ha="center", fontsize=7.5, color="#555")

    ax2.add_patch(patches.FancyBboxPatch((0.8, 3.0), 2.4, 1.4, boxstyle="round,pad=0.05", facecolor="#EAFAF1", edgecolor="#27AE60", lw=1.8))
    ax2.text(2.0, 4.0, "Latent Interval", ha="center", weight="bold", fontsize=8.5, color="#1E8449")
    ax2.text(2.0, 3.4, r"$\widehat{C}_s = [Q^\mathrm{Pois}_{\alpha/2}(\hat{\lambda}), Q^\mathrm{Pois}_{1-\alpha/2}(\hat{\lambda})]$" + "\nGuaranteed Latent Coverage", ha="center", fontsize=7.5, color="#111")

    # Arrows in B
    ax2.annotate("", xy=(3.9, 5.5), xytext=(3.1, 5.5), arrowprops=dict(arrowstyle="->", lw=2, color="#2980B9"))
    ax2.annotate("", xy=(6.9, 5.5), xytext=(6.1, 5.5), arrowprops=dict(arrowstyle="->", lw=2, color="#C0392B"))
    ax2.annotate("", xy=(8.0, 4.5), xytext=(8.0, 5.0), arrowprops=dict(arrowstyle="->", lw=2, color="#1E8449"))
    ax2.annotate("", xy=(6.3, 3.7), xytext=(6.7, 3.7), arrowprops=dict(arrowstyle="->", lw=2, color="#1E8449"))
    ax2.annotate("", xy=(3.3, 3.7), xytext=(3.7, 3.7), arrowprops=dict(arrowstyle="->", lw=2, color="#27AE60"))
    ax2.annotate("", xy=(2.0, 4.9), xytext=(2.0, 4.5), arrowprops=dict(arrowstyle="->", lw=2, color="#27AE60", ls="--"))
    ax2.text(1.2, 4.7, "Valid For", ha="center", fontsize=7.5, color="#27AE60", weight="bold")

    res_box2 = patches.FancyBboxPatch((1.0, 0.2), 8.0, 1.1, boxstyle="round,pad=0.08",
                                     facecolor="#EAFAF1", edgecolor="#27AE60", lw=1.2)
    ax2.add_patch(res_box2)
    ax2.text(5, 0.75, r"Theorem 2: Exact recovery $\hat{\lambda}_s = \lambda_s$ identically." + "\n" +
                      r"Latent Coverage restored to $93.0\%$ (on retained 15% cells at $\kappa=0.85$)",
             ha="center", va="center", fontsize=8.5, color="#145A32", weight="bold")

    out_path = FIG_DIR / "fig_feedback_loop.png"
    out_pdf = FIG_DIR / "fig_feedback_loop.pdf"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path} and {out_pdf}")


if __name__ == "__main__":
    draw_system_architecture()
    draw_feedback_loop()
