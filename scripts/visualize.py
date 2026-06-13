import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
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
def plot_spatial_error(gdf):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Mock residuals: Standard normal distribution
    gdf['residual'] = np.random.randn(len(gdf))
    
    # Diverging colormap
    cmap = 'RdBu_r'
    
    # Plot
    gdf.plot(column='residual', ax=ax, legend=True,
             cmap=cmap, 
             legend_kwds={'label': "Prediction Residual", 'orientation': "horizontal", 'shrink': 0.7},
             edgecolor='black', linewidth=0.5)
    
    ax.set_title("Spatial Error (Residual) Distribution Map", fontsize=16, pad=20)
    ax.axis('off')
    
    out_path = os.path.join(OUTPUT_DIR, "spatial_error_map.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# 2. Target-Centric Spatial Attention Map (GATv2 incoming edge weights using NetworkX + GeoPandas)
def plot_target_centric_attention(gdf):
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
    
    # Create mock attention weights from all nodes to the target node
    distances = centroids.distance(target_centroid)
    # Exponential decay based on distance for realistic-looking attention
    max_dist = distances.max()
    if max_dist == 0:
        max_dist = 1
    attn_weights = np.exp(-distances / max_dist * 5)
    attn_weights[target_idx] = 1.0 # self-attention is usually high
    attn_weights = attn_weights / attn_weights.sum() # Normalize
    
    gdf['attention'] = attn_weights
    
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
    for i in top_indices:
        if i != target_idx and attn_weights.iloc[i] > 0.01:
            edges.append((i, target_idx))
            weights.append(attn_weights.iloc[i])
            
    G.add_edges_from(edges)
    
    # Normalize weights for drawing
    if weights:
        max_w = max(weights)
        edge_widths = [w / max_w * 4 for w in weights]
        
        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, 
                               width=edge_widths, edge_color='blue', alpha=0.6,
                               arrows=True, arrowstyle='-|>', arrowsize=20, connectionstyle='arc3,rad=0.2')
                               
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
def plot_temporal_attention():
    # Sequence length T (e.g., 24 time steps)
    T = 24
    # Create mock attention matrix [Query Time, Key Time]
    attn_matrix = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            if j <= i: # Causal masking or just looking at past
                # decay based on lag
                attn_matrix[i, j] = np.exp(-0.3 * (i - j))
                # Add some periodic component (e.g. lag 12, 24)
                if (i - j) > 0 and (i - j) % 12 == 0:
                    attn_matrix[i, j] += 0.4
                    
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
def plot_bivariate_choropleth(gdf):
    # Mock Ground Truth and Predictions
    gdf['ground_truth'] = np.random.uniform(0, 100, len(gdf))
    gdf['prediction'] = gdf['ground_truth'] * 0.8 + np.random.normal(0, 15, len(gdf))
    
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
        print("Loaded Chicago Community Areas.")
    elif os.path.exists(nyc_path):
        gdf = gpd.read_file(nyc_path)
        print("Loaded NYC Police Precincts.")
    else:
        raise FileNotFoundError(f"Could not find boundary geojson files in {shapefile_dir}")

    print("Generating Spatial Error Choropleth...")
    plot_spatial_error(gdf.copy())
    
    print("Generating Target-Centric GATv2 Attention Map...")
    plot_target_centric_attention(gdf.copy())
    
    print("Generating Transformer Temporal Heatmap...")
    plot_temporal_attention()
    
    print("Generating Bivariate Choropleth Map...")
    plot_bivariate_choropleth(gdf.copy())
    
    print("All plots generated successfully in outputs/figures/")

if __name__ == "__main__":
    main()
