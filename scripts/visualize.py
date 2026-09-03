import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # headless-safe (A100/servers without a display)
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Spatial Error (Residual) Distribution Map (diverging choropleth using GeoPandas)
def load_real_residuals(city="chicago"):
    """Mean per-area residual (prediction minus actual) from saved test predictions.

    Returns an array of length n_areas, ordered as the prediction tensors are:
    axis 0 is weeks, axis 1 is spatial units. Raises if the file is missing, so
    a failure surfaces instead of silently reverting to random numbers.
    """
    npz_path = os.path.join(
        BASE_DIR, "outputs", "conformal_evaluation", f"{city}_predictions.npz"
    )
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"No saved predictions at {npz_path}. Run "
            f"scripts/run_conformal_evaluation.py --data {city} first."
        )
    with np.load(npz_path) as data:
        actual = data["actual_violent"]          # (weeks, areas)
        predicted = data["point_prediction"]     # (weeks, areas)
    residual = (predicted - actual).mean(axis=0)
    print(
        f"  Loaded real residuals from {os.path.basename(npz_path)}: "
        f"{actual.shape[0]} weeks x {actual.shape[1]} areas, "
        f"mean residual {residual.mean():+.4f}, "
        f"range [{residual.min():+.3f}, {residual.max():+.3f}]"
    )
    return residual


def plot_spatial_error(gdf, city="chicago"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    residual = load_real_residuals(city)
    if len(residual) != len(gdf):
        raise ValueError(
            f"Residual count ({len(residual)}) does not match the number of "
            f"geometries ({len(gdf)}). The shapefile and the prediction tensor "
            "describe different spatial units, so the join would be wrong."
        )
    gdf['residual'] = residual

    # Diverging colormap centred at zero so over- and under-prediction read
    # symmetrically; without vcenter the midpoint drifts to the data mean.
    cmap = 'RdBu_r'
    vmax = float(np.abs(residual).max())
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    gdf.plot(column='residual', ax=ax, legend=True,
             cmap=cmap, norm=norm,
             legend_kwds={'label': "Mean residual (predicted − actual), violent",
                          'orientation': "horizontal", 'shrink': 0.7},
             edgecolor='black', linewidth=0.5)

    ax.set_title(
        f"Spatial Error Distribution — {city.title()} violent crime, 2023 test set",
        fontsize=15, pad=20,
    )
    ax.axis('off')
    
    out_path = os.path.join(OUTPUT_DIR, "spatial_error_map.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# 2. Target-Centric Spatial Attention Map (GATv2 incoming edge weights using NetworkX + GeoPandas)
def plot_target_centric_attention(gdf, attention_weights=None):
    """Plot incoming GATv2 attention toward one target area.

    Requires REAL attention weights. The previous version synthesised them as
    exp(-distance) decay and drew them under a title claiming GATv2 attention,
    which is a fabricated figure -- the same defect as the mock residuals in
    plot_spatial_error. Rather than render something misleading, this now
    refuses unless real weights are supplied.

    To produce it for the paper: extract per-edge attention coefficients from a
    trained checkpoint's GATv2 layer (they are returned when the layer is called
    with return_attention_weights=True), aggregate incoming weight per source
    area for the chosen target, save them alongside the predictions, and pass
    the resulting length-n_areas array as attention_weights.
    """
    if attention_weights is None:
        raise NotImplementedError(
            "plot_target_centric_attention needs real GATv2 attention weights. "
            "No attention weights are currently persisted by the evaluation "
            "pipeline, so this figure cannot be generated honestly. It is NOT "
            "referenced by the paper. See the docstring for how to export them."
        )

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Ensure it's projected so we can get centroids properly (using a standard Web Mercator for plotting purposes)
    if gdf.crs and not gdf.crs.is_projected:
        gdf_proj = gdf.to_crs(epsg=3857)
    else:
        gdf_proj = gdf.copy()

    centroids = gdf_proj.geometry.centroid

    # Pick a target region index (e.g. roughly center)
    target_idx = len(gdf) // 2
    target_centroid = centroids.iloc[target_idx]

    attn_weights = pd.Series(np.asarray(attention_weights, dtype=float))
    if len(attn_weights) != len(gdf):
        raise ValueError(
            f"attention_weights has length {len(attn_weights)} but there are "
            f"{len(gdf)} areas."
        )
    attn_weights = attn_weights / attn_weights.sum()

    gdf['attention'] = attn_weights.values
    
    # Plot base map with attention colors
    # We plot using original gdf so crs matches if we want axes to be lat/lon, but let's just plot proj
    gdf_proj['attention'] = attn_weights
    gdf_proj.plot(column='attention', ax=ax, cmap='YlOrRd', edgecolor='gray', linewidth=0.5, alpha=0.8,
                  legend=True, legend_kwds={'label': "Incoming Attention Weight", 'shrink': 0.7})
    
    # Overlay arrows representing incoming attention
    # For top K attention weights
    top_k = min(15, len(gdf))
    # Exclude self-attention for drawing edges
    attn_without_self = attn_weights.copy()
    attn_without_self.iloc[target_idx] = 0
    top_indices = np.argsort(attn_without_self)[-top_k:]
    
    # Draw edges
    G = nx.DiGraph()
    pos = {i: (centroids.iloc[i].x, centroids.iloc[i].y) for i in range(len(gdf))}
    
    edges = []
    weights = []
    target_pos = pos[target_idx]
    for i in top_indices:
        if i == target_idx or attn_weights.iloc[i] <= 0.01:
            continue
        # Drop degenerate edges. A zero-length FancyArrowPatch raises
        # StopIteration inside matplotlib's bezier clipping, and it does so at
        # RENDER time (savefig), not when the edge is added -- which is why a
        # try/except around draw_networkx_edges never caught it.
        dx = pos[i][0] - target_pos[0]
        dy = pos[i][1] - target_pos[1]
        if (dx * dx + dy * dy) ** 0.5 < 1.0:
            continue
        edges.append((i, target_idx))
        weights.append(attn_weights.iloc[i])

    G.add_edges_from(edges)

    # Normalize weights for drawing
    if weights:
        max_w = max(weights)
        edge_widths = [w / max_w * 4 for w in weights]

        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                               width=edge_widths, edge_color='blue', alpha=0.6,
                               arrows=True, arrowstyle='-|>', arrowsize=20)
                               
    # Highlight target node
    ax.scatter(target_centroid.x, target_centroid.y, 
               color='red', marker='*', s=300, label='Target Region', zorder=5, edgecolor='black')
               
    ax.set_title("Target-Centric Spatial Attention Map (GATv2)", fontsize=16, pad=20)
    ax.axis('off')
    ax.legend(loc='upper right')
    
    out_path = os.path.join(OUTPUT_DIR, "target_spatial_attention.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# 3. Temporal Attention Heatmap (Transformer query vs key attention across time lags)
def plot_temporal_attention(attn_matrix=None):
    """Heatmap of transformer temporal attention (query time vs key time).

    Requires a REAL attention matrix. The previous version built one from
    exp(-0.3*lag) plus a hand-added bump at lags 12 and 24, which is a drawing
    of the author's expectations rather than a measurement. Passing that off as
    a transformer attention map would be fabrication, so it now refuses.

    To produce it: capture the temporal-attention tensor from a trained
    checkpoint's transformer block, average over heads, batch and space, and
    pass the resulting (T, T) array.
    """
    import seaborn as sns  # lazy import: optional dep

    if attn_matrix is None:
        raise NotImplementedError(
            "plot_temporal_attention needs a real (T, T) attention matrix. The "
            "evaluation pipeline does not currently persist transformer "
            "attention, so this figure cannot be generated honestly. It is NOT "
            "referenced by the paper. See the docstring for how to export it."
        )

    attn_matrix = np.asarray(attn_matrix, dtype=float)
    if attn_matrix.ndim != 2 or attn_matrix.shape[0] != attn_matrix.shape[1]:
        raise ValueError(
            f"Expected a square (T, T) matrix, got shape {attn_matrix.shape}."
        )
    T = attn_matrix.shape[0]

    # Normalize rows
    row_sums = attn_matrix.sum(axis=1, keepdims=True)
    attn_matrix = np.where(row_sums > 0, attn_matrix / row_sums, 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn_matrix, cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Weight'},
                xticklabels=range(1, T+1), yticklabels=range(1, T+1))
    
    ax.set_xlabel("Key (Source Time Step)", fontsize=12)
    ax.set_ylabel("Query (Target Time Step)", fontsize=12)
    ax.set_title("Temporal Attention Heatmap (Transformer)", fontsize=16, pad=20)
    
    out_path = os.path.join(OUTPUT_DIR, "temporal_attention_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# 4. Bivariate Choropleth Map (Predictions vs. Ground Truth using a 2D color matrix)
def plot_bivariate_choropleth(gdf, city="chicago"):
    """Bivariate map of actual vs predicted violent counts, from real predictions.

    Previously this drew np.random.uniform ground truth against a noised copy of
    itself, which made the map a picture of nothing. Both series now come from
    the saved test predictions.
    """
    npz_path = os.path.join(
        BASE_DIR, "outputs", "conformal_evaluation", f"{city}_predictions.npz"
    )
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"No saved predictions at {npz_path}. Run "
            f"scripts/run_conformal_evaluation.py --data {city} first."
        )
    with np.load(npz_path) as data:
        actual = data["actual_violent"].mean(axis=0)
        predicted = data["point_prediction"].mean(axis=0)
    if len(actual) != len(gdf):
        raise ValueError(
            f"Prediction tensor has {len(actual)} areas but the shapefile has "
            f"{len(gdf)}; the join would be wrong."
        )
    gdf['ground_truth'] = actual
    gdf['prediction'] = predicted
    print(
        f"  Loaded real actual/predicted means for {len(gdf)} areas "
        f"(actual {actual.mean():.2f}, predicted {predicted.mean():.2f})"
    )
    
    # Categorize into 3 quantiles for a 3x3 bivariate map
    gdf['gt_quant'] = pd.qcut(gdf['ground_truth'], 3, labels=[0, 1, 2]).astype(int)
    gdf['pred_quant'] = pd.qcut(gdf['prediction'], 3, labels=[0, 1, 2]).astype(int)
    
    # Bivariate color palette (3x3 grid)
    bivariate_colors = {
        (0, 0): '#e8e8e8', (1, 0): '#ace4e4', (2, 0): '#5ac8c8',
        (0, 1): '#dfb0d6', (1, 1): '#a5add3', (2, 1): '#5698b9',
        (0, 2): '#be64ac', (1, 2): '#8c62aa', (2, 2): '#3b4994'
    }
    
    gdf['biv_color'] = gdf.apply(lambda row: bivariate_colors[(row['gt_quant'], row['pred_quant'])], axis=1)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(color=gdf['biv_color'], ax=ax, edgecolor='white', linewidth=0.5)
    ax.set_title("Bivariate Choropleth: Predictions vs. Ground Truth", fontsize=16, pad=20)
    ax.axis('off')
    
    # Add a legend
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_legend = inset_axes(ax, width="20%", height="20%", loc='lower right', borderpad=2)
    
    # Create 3x3 grid for legend
    legend_grid = np.zeros((3, 3, 3)) # RGB
    for i in range(3): # gt (x-axis)
        for j in range(3): # pred (y-axis)
            hex_color = bivariate_colors[(i, j)].lstrip('#')
            rgb = tuple(int(hex_color[k:k+2], 16)/255.0 for k in (0, 2, 4))
            legend_grid[j, i] = rgb
            
    ax_legend.imshow(legend_grid, origin='lower')
    # Center ticks
    ax_legend.set_xticks([0, 1, 2])
    ax_legend.set_yticks([0, 1, 2])
    ax_legend.set_xticklabels(['Low', 'Med', 'High'], fontsize=9)
    ax_legend.set_yticklabels(['Low', 'Med', 'High'], fontsize=9, rotation=90, va='center')
    
    ax_legend.set_xlabel("Ground Truth \u2192", fontsize=11, fontweight='bold')
    ax_legend.set_ylabel("Prediction \u2192", fontsize=11, fontweight='bold')
    
    # Remove tick marks but keep labels
    ax_legend.tick_params(axis='both', which='both', length=0)
    
    # Add grid lines to separate the legend boxes
    ax_legend.set_xticks([0.5, 1.5], minor=True)
    ax_legend.set_yticks([0.5, 1.5], minor=True)
    ax_legend.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    out_path = os.path.join(OUTPUT_DIR, "bivariate_choropleth.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def main():
    print("Loading geographic boundaries...")
    shapefile_dir = os.path.join(DATA_DIR, "raw", "shapefiles")
    
    chicago_path = os.path.join(shapefile_dir, "chicago_community_areas.geojson")
    nyc_path = os.path.join(shapefile_dir, "nyc_police_precincts.geojson")
    
    if os.path.exists(chicago_path):
        gdf = gpd.read_file(chicago_path)
        city = "chicago"
        print("Loaded Chicago Community Areas.")
    elif os.path.exists(nyc_path):
        gdf = gpd.read_file(nyc_path)
        city = "nyc"
        print("Loaded NYC Police Precincts.")
    else:
        raise FileNotFoundError(f"Could not find boundary geojson files in {shapefile_dir}")

    print("Generating Spatial Error Choropleth...")
    try:
        plot_spatial_error(gdf.copy(), city=city)
    except Exception as e:
        print(f"  SKIPPED: {e}")
    
    print("Generating Target-Centric GATv2 Attention Map...")
    try:
        plot_target_centric_attention(gdf.copy())
    except Exception as e:
        print(f"  SKIPPED: {e}")
    
    print("Generating Transformer Temporal Heatmap...")
    try:
        plot_temporal_attention()
    except Exception as e:
        print(f"  SKIPPED: {e}")
    
    print("Generating Bivariate Choropleth Map...")
    try:
        plot_bivariate_choropleth(gdf.copy(), city=city)
    except Exception as e:
        print(f"  SKIPPED: {e}")
    
    print("All plots generated successfully in outputs/figures/")

if __name__ == "__main__":
    main()
