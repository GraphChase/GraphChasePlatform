# GraphChase

## Generate Graph Structure

### Generate Grid Graph

`graph/generate_custom_graph.py` builds a grid graph from the input dimensions, optionally prunes axis edges and adds diagonals based on probabilities, then saves the result as a `.gpickle` with positional metadata (nodes are relabeled from 1 and store `(col, -row)` in `pos`).

Run the generator with:

```bash
python graph/generate_custom_graph.py \
  --grid_width 7 --grid_height 7 \
  --output_path graph/custom_graph/7_7_grid_graph.gpickle
```

Options:

- `--grid_width/--grid_height`: grid dimensions (columns × rows).
- `--axis_edge_prob` (optional): probability to keep horizontal/vertical edges (default 1.0).
- `--diag_edge_prob` (optional): probability to add diagonal edges (default 0.0).
- `--seed`: random seed (default 0).
- `--output_path`: output `.gpickle` path.

### Generate Any Graph

You can build non-grid graphs in two common ways.

**A. Start from an existing `.gpickle` and add weighted edges (recommended for weighted UNSG experiments)**

`graphchase/build_game_sample.py` provides two helpers:
- `build_symmetric_adjacency(base_graph, seed, weight_range)`: same weight in both directions for each edge.
- `build_asymmetric_adjacency(base_graph, seed, weight_range)`: independent weights for two directions.

Typical workflow:
1. Load a base graph from `.gpickle`.
2. Build an adjacency matrix with one of the helpers above.
3. Put `adjacency_matrix` and `node_ids` into `metadata`.
4. Build `GameSettings(..., metadata=metadata, use_weighted_graph=True)`.

Example:

```python
import pickle
from graphchase.build_game_sample import build_symmetric_adjacency
from graphchase.graph.game_settings import GameSettings

with open("graphchase/graph/custom_graph/times_square.gpickle", "rb") as fp:
    base_graph = pickle.load(fp)

adjacency_matrix, node_ids = build_symmetric_adjacency(
    base_graph=base_graph,
    seed=0,
    weight_range=(0.5, 2.0),
)

metadata = {
    "source_gpickle": "graphchase/graph/custom_graph/times_square.gpickle",
    "adjacency_matrix": adjacency_matrix,
    "node_ids": node_ids,
}

settings = GameSettings(
    graph=base_graph,
    attacker_init=[1],
    defender_init=[2, 3],
    exit_nodes=[4, 5],
    time_horizon=7,
    metadata=metadata,
    use_weighted_graph=True,
)
```

If you want directed/one-way style edge costs, replace `build_symmetric_adjacency` with `build_asymmetric_adjacency`.

**B. Download a real-world map, then convert node IDs to start from 1**

1. Download a road network and save it as `.gpickle`:

```bash
python graphchase/download_map.py \
  --place "Times Square, New York, USA" \
  --output-path "graphchase/graph/custom_graph/times_square_raw.gpickle" \
  --dist-m 2000
```

2. Relabel graph node IDs to `1..N` (GraphChase-friendly indexing):

```bash
python graphchase/relabel_graph_idx_from1.py \
  --graph_path "graphchase/graph/custom_graph/times_square_raw.gpickle"
```

3. Use the generated `.gpickle` directly in your config/arguments (for unweighted runs), or feed it into workflow A above to create weighted adjacency metadata.

## Game Settings

`graph/game_settings.py` defines the `GameSettings` dataclass, which wraps a built NetworkX graph and the scenario parameters used by `UNSGEnv`.

**Use a built graph**
- Load a `.gpickle` graph and pass it into `GameSettings` with the initial positions and horizon.
- Example:
  ```python
  import pickle
  from graphchase.graph.game_settings import GameSettings

  with open("graph/custom_graph/7_7_grid_graph.gpickle", "rb") as fp:
      base_graph = pickle.load(fp)

  settings = GameSettings(
      graph=base_graph,
      attacker_init=[25],
      defender_init=[9, 28, 44, 46],
      exit_nodes=[4, 22, 43, 49],
      time_horizon=7,
      metadata={"source_gpickle": "graph/custom_graph/7_7_grid_graph.gpickle"},
  )
  ```

**Configurable parameters**
- `graph`: the base NetworkX graph structure.
- `attacker_init`, `defender_init`, `exit_nodes`: node IDs for agent starts and exits.
- `time_horizon`: episode length.
- `metadata`: optional dictionary for graph provenance or overrides (see below).
- `edge_weights`, `neighbor_map`: optional caches; leave as `None` to auto-populate from the graph.

**Override edge weights on a predefined graph**
- Provide `metadata` with an `adjacency_matrix` and matching `node_ids` list to rebuild the graph with custom weights (see `build_game_sample.py`).
- Requirements:
  - `adjacency_matrix` is a Python list-of-lists (or tuple-of-tuples) of floats with shape `N x N` (not a NumPy array).
  - `node_ids` is a Python list (or tuple) of the graph nodes (from `sorted(base_graph.nodes())`), and its length `N` must match the matrix.
  - Each `adjacency_matrix[i][j]` is the edge weight between `node_ids[i]` and `node_ids[j]`; use `0.0` for no edge.
  - The matrix is symmetric for an undirected graph; the helper fills both `[i][j]` and `[j][i]`. For directed graphs, it can be asymmetric.
  - Both `adjacency_matrix` and `node_ids` are required together when overriding.
- Example:
  ```python
  from graphchase.build_game_sample import build_symmetric_adjacency

  adjacency, node_ids = build_symmetric_adjacency(base_graph, seed=0, weight_range=(0.5, 2.0))
  metadata = {
      "source_gpickle": "graph/custom_graph/7_7_grid_graph.gpickle",
      "adjacency_matrix": adjacency,
      "node_ids": node_ids,
  }
  settings = GameSettings(
      graph=base_graph,
      attacker_init=[25],
      defender_init=[9, 28, 44, 46],
      exit_nodes=[4, 22, 43, 49],
      time_horizon=7,
      metadata=metadata,
  )
  ```

## PretrainedPSRO (scripts/run_pretrain_psro.py)

`scripts/run_pretrain_psro.py` is the entry point for the PretrainedPSRO algorithm. The workflow has two stages: a defender pretraining phase followed by PSRO fine-tuning. The runner reads a YAML config (default `solver_cfgs/pretrain_psro_cfgs.yaml`) and accepts overrides via `--set key=value`.

**Enable graph embeddings**
- Set `graph_embeddings=true` to train embeddings (or provide `load_embeddings=path.pkl` to reuse saved ones).
- Example:
  ```bash
  python scripts/run_pretrain_psro.py --config solver_cfgs/pretrain_psro_cfgs.yaml --set graph_embeddings=true --set save_dir=graph/custom_graph/graph_embeddings
  ```
- Load existing embeddings (skip training):
  ```bash
  python scripts/run_pretrain_psro.py --config solver_cfgs/pretrain_psro_cfgs.yaml --set graph_embeddings=true --set load_embeddings=graph/custom_graph/graph_embeddings/line_embeddings.pkl
  ```
- Disable graph embeddings (use raw node info only):
  ```bash
  python scripts/run_pretrain_psro.py --config solver_cfgs/pretrain_psro_cfgs.yaml --set graph_embeddings=false
  ```

**Enable pretraining before PSRO**
- Set `pretrain_iterations > 0` and define task/episode counts; use `pretrain_model_path` to save the pretrained defender.
- Example:
  ```bash
  python scripts/run_pretrain_psro.py --config solver_cfgs/pretrain_psro_cfgs.yaml --set pretrain_iterations=50 --set pretrain_tasks=30 --set pretrain_episodes_per_task=20 --set pretrain_model_path=./experiments/graphchase_psro1/pretrain_model.pt
  ```
- Load an existing pretrained defender:
  ```bash
  python scripts/run_pretrain_psro.py --config solver_cfgs/pretrain_psro_cfgs.yaml --set load_pretrained_model=true --set pretrain_model_path=./experiments/graphchase_psro1/pretrain_model.pt
  ```

**Skip pretraining (PSRO-only)**
- Keep `pretrain_iterations=0` and `load_pretrained_model=false`. This is equivalent to running PSRO directly.
- Example:
  ```bash
  python scripts/run_pretrain_psro.py --config solver_cfgs/pretrain_psro_cfgs.yaml --set pretrain_iterations=0 --set load_pretrained_model=false
  ```


## Download Map Graphs
```python
python download_map.py --place "Singapore, Singapore" --output-path "singapore.gpickle" --dist-m 2000
python download_map.py --place "Mumbai, India" --output-path "mumbai.gpickle"
python download_map.py --place "Sydney Opera House, Sydney, Australia" --output-path "sydney_opera_house.gpickle"
python download_map.py --place "Times Square, New York, USA" --output-path "times_square.gpickle"
```
