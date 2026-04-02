import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.dataset import build_smart_grid
G, aligned_data, mean_loads, corr_matrix = build_smart_grid("dataset", n_nodes=8)
print(f"\nGraph nodes: {sorted(G.nodes())}")
print(f"Corr cols: {list(corr_matrix.columns)}")
print(f"Nodes in corr: {[n for n in sorted(G.nodes()) if n in corr_matrix.columns]}")
