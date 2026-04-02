"""
================================================================================
Smart Grid Graph Construction from Energy Consumption Data
================================================================================

STEP 1: Smart Grid Optimization Problem
-----------------------------------------
Grid partitioning divides a power network into balanced sub-grids to:
  • Minimize transmission losses across partition boundaries
  • Balance electrical load evenly across partitions
  • Enable decentralized control and fault isolation

Nodes → Power regions (PJM interconnection zones)
Edges → Transmission interconnections between regions
Edge weights → Load similarity/difference between regions

STEP 2: Convert Dataset → Graph
---------------------------------
Process:
  1. Load hourly consumption data for each PJM region
  2. Align time series to common date range
  3. Compute mean load per region (node weight)
  4. Compute pairwise Pearson correlation (edge weight)
  5. Construct weighted graph using NetworkX

The correlation captures load synchrony: highly correlated regions have
similar consumption patterns, meaning cutting edges between them has
lower transmission cost (they don't need to share power as much).
Conversely, cutting edges between anti-correlated regions is expensive
(they benefit from sharing power to smooth peaks).

Edge weight = 1 - |correlation| → lower weight = more similar = cheaper to cut
================================================================================
"""

import numpy as np
import pandas as pd
import networkx as nx
import os
from datetime import datetime


def load_regional_data(dataset_dir="dataset"):
    """
    Load hourly energy consumption data for all PJM regions.

    Each CSV has columns: [Datetime, {REGION}_MW]

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory.

    Returns
    -------
    regions : dict
        {region_name: pd.DataFrame with Datetime index and MW column}
    """
    region_files = {
        "AEP": "AEP_hourly.csv",
        "COMED": "COMED_hourly.csv",
        "DAYTON": "DAYTON_hourly.csv",
        "DEOK": "DEOK_hourly.csv",
        "DOM": "DOM_hourly.csv",
        "DUQ": "DUQ_hourly.csv",
        "EKPC": "EKPC_hourly.csv",
        "FE": "FE_hourly.csv",
        "NI": "NI_hourly.csv",
        "PJME": "PJME_hourly.csv",
        "PJMW": "PJMW_hourly.csv",
    }

    regions = {}
    for name, filename in region_files.items():
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df.columns = ["Datetime", "MW"]
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime").sort_index()
            df = df[~df.index.duplicated(keep="first")]
            regions[name] = df

    print(f"Loaded {len(regions)} PJM regions:")
    for name, df in regions.items():
        print(f"  {name:8s}  rows={len(df):>7d}  "
              f"mean={df['MW'].mean():>10.0f} MW  "
              f"range=[{df.index.min().date()} → {df.index.max().date()}]")

    return regions


def align_and_aggregate(regions, freq="D"):
    """
    Align all regions to a common time range and aggregate.

    Regions with non-overlapping date ranges are automatically excluded
    to ensure a valid common period exists.

    Parameters
    ----------
    regions : dict
        {region: DataFrame}
    freq : str
        Resampling frequency ('D' for daily, 'W' for weekly).

    Returns
    -------
    aligned : pd.DataFrame
        Columns = regions, Index = common datetime, Values = MW
    mean_loads : pd.Series
        Mean load per region.
    """
    # Iteratively remove regions that prevent a valid common range
    valid_regions = dict(regions)
    while len(valid_regions) > 2:
        start_dates = {n: df.index.min() for n, df in valid_regions.items()}
        end_dates = {n: df.index.max() for n, df in valid_regions.items()}
        common_start = max(start_dates.values())
        common_end = min(end_dates.values())

        if common_start < common_end:
            break  # Valid overlap found

        # Find and remove the region causing the smallest end date
        # (the one that ends earliest, preventing overlap)
        worst_name = min(end_dates, key=end_dates.get)
        print(f"  Excluding '{worst_name}' (ends {end_dates[worst_name].date()}, "
              f"no overlap with later regions)")
        del valid_regions[worst_name]

    start_dates = {n: df.index.min() for n, df in valid_regions.items()}
    end_dates = {n: df.index.max() for n, df in valid_regions.items()}
    common_start = max(start_dates.values())
    common_end = min(end_dates.values())

    print(f"\nCommon date range: {common_start.date()} to {common_end.date()}")
    print(f"Using {len(valid_regions)} regions: {sorted(valid_regions.keys())}")

    # Resample and align
    aligned = pd.DataFrame()
    for name, df in valid_regions.items():
        subset = df.loc[common_start:common_end]
        resampled = subset.resample(freq).mean()
        aligned[name] = resampled["MW"]

    aligned = aligned.dropna()
    mean_loads = aligned.mean()

    print(f"Aligned shape: {aligned.shape} ({freq} frequency)")
    print(f"\nMean loads (MW):")
    for name, load in mean_loads.items():
        print(f"  {name:8s}  {load:>10.0f} MW")

    return aligned, mean_loads


def compute_correlation_matrix(aligned_data):
    """
    Compute pairwise Pearson correlation between regions.

    High correlation → similar consumption patterns → low transmission need.
    Low correlation → complementary patterns → high transmission benefit.

    Parameters
    ----------
    aligned_data : pd.DataFrame

    Returns
    -------
    corr_matrix : pd.DataFrame
        Correlation matrix.
    """
    corr_matrix = aligned_data.corr()
    print(f"\nCorrelation matrix ({corr_matrix.shape}):")
    print(corr_matrix.round(3).to_string())
    return corr_matrix


def build_grid_graph(mean_loads, corr_matrix, edge_threshold=0.0):
    """
    Construct a weighted graph representing the power grid.

    Nodes:
      - Each node is a power region
      - Node attribute 'load' = mean MW consumption

    Edges:
      - Connect regions with |correlation| > threshold
      - Edge weight = 1 - |correlation|
        (lower weight → more similar → cheaper to cut)

    For QAOA graph partitioning, we want to minimize total edge cut
    weight, meaning we prefer to cut edges between similar regions
    (low cut cost) rather than complementary ones.

    Parameters
    ----------
    mean_loads : pd.Series
        Mean load per region.
    corr_matrix : pd.DataFrame
        Pairwise correlation matrix.
    edge_threshold : float
        Minimum |correlation| to create an edge.

    Returns
    -------
    G : nx.Graph
        Weighted power grid graph.
    """
    regions = list(mean_loads.index)
    n = len(regions)

    G = nx.Graph()

    # Add nodes with load attributes
    for region in regions:
        G.add_node(region, load=float(mean_loads[region]))

    # Add edges with weights
    for i in range(n):
        for j in range(i + 1, n):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > edge_threshold:
                # Weight: higher correlation → lower cut cost
                weight = round(1.0 - corr, 4)
                G.add_edge(regions[i], regions[j], weight=weight)

    print(f"\nGrid Graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"\n  Node loads:")
    for node in G.nodes():
        print(f"    {node:8s}  {G.nodes[node]['load']:>10.0f} MW")
    print(f"\n  Top 5 strongest edges (lowest cut cost):")
    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]["weight"])
    for u, v, d in edges_sorted[:5]:
        print(f"    {u:8s} ↔ {v:8s}  weight={d['weight']:.4f}  "
              f"(corr={1-d['weight']:.4f})")

    return G


def select_representative_nodes(G, n_nodes=8):
    """
    For QAOA simulation, select a manageable subset of nodes.

    If the graph has more nodes than n_nodes, we select the top
    n_nodes by load diversity to keep the problem interesting.

    Parameters
    ----------
    G : nx.Graph
        Full grid graph.
    n_nodes : int
        Target number of nodes for QAOA.

    Returns
    -------
    G_sub : nx.Graph
        Subgraph with n_nodes nodes.
    """
    if G.number_of_nodes() <= n_nodes:
        return G.copy()

    # Select by load diversity (spread of loads)
    nodes_by_load = sorted(G.nodes(), key=lambda n: G.nodes[n]["load"])
    # Take evenly spaced nodes across the load range
    indices = np.linspace(0, len(nodes_by_load) - 1, n_nodes, dtype=int)
    selected = [nodes_by_load[i] for i in indices]

    G_sub = G.subgraph(selected).copy()
    print(f"\nSelected {n_nodes} representative nodes: {selected}")
    return G_sub


def build_smart_grid(dataset_dir="dataset", n_nodes=8, freq="D"):
    """
    End-to-end: dataset → graph.

    Parameters
    ----------
    dataset_dir : str
    n_nodes : int
        Maximum nodes for QAOA simulation.
    freq : str
        Aggregation frequency.

    Returns
    -------
    G : nx.Graph
        Power grid graph.
    aligned_data : pd.DataFrame
        Aligned consumption data.
    mean_loads : pd.Series
    corr_matrix : pd.DataFrame
    """
    print("=" * 65)
    print("  STEP 1-2: SMART GRID GRAPH CONSTRUCTION")
    print("=" * 65)

    # Load data
    regions = load_regional_data(dataset_dir)

    # Align and aggregate
    aligned_data, mean_loads = align_and_aggregate(regions, freq=freq)

    # Compute correlations
    corr_matrix = compute_correlation_matrix(aligned_data)

    # Build graph
    G = build_grid_graph(mean_loads, corr_matrix, edge_threshold=0.3)

    # Select subset for QAOA
    G = select_representative_nodes(G, n_nodes=n_nodes)

    return G, aligned_data, mean_loads, corr_matrix
