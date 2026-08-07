# pip install osmnx networkx

import argparse
import os
import pickle
import networkx as nx
import osmnx as ox

def build_undirected_graph_near_place(
    place: str,
    dist_m: int = 1200,
    network_type: str = "drive",
    output_path: str = "sydney_opera_house_undirected.gpickle",
    consolidate_tolerance_m: float = 15.0,
):
    """
    Download OSM road network around `place`, build an undirected simple graph (nx.Graph),
    and set edge attribute 'weight' for GraphChase.

    - weight priority: travel_time (seconds) -> length (meters) -> 1.0
    - collapses parallel edges by keeping the smallest weight between each (u, v)
    - merges nearby nodes within consolidate_tolerance_m distance
    """

    # Optional: cache requests to avoid repeated downloads
    ox.settings.use_cache = True
    ox.settings.log_console = True

    # 1) Geocode place name to (lat, lon)
    center_point = ox.geocode(place)

    # 2) Download road network (OSMnx returns a directed MultiDiGraph by default)
    G = ox.graph_from_point(
        center_point,
        dist=dist_m,
        network_type=network_type,
        simplify=True,
    )

    if consolidate_tolerance_m > 0:
        G_projected = ox.project_graph(G)
        G_projected = ox.consolidate_intersections(G_projected, tolerance=consolidate_tolerance_m, rebuild_graph=True)
        G = ox.project_graph(G_projected, to_crs="EPSG:4326")

    # 3) Add edge speeds and travel times (travel_time is in seconds)
    #    If speed data is missing, OSMnx uses heuristics / defaults.
    G = ox.routing.add_edge_speeds(G, fallback=30)
    G = ox.routing.add_edge_travel_times(G)

    # 4) Convert to undirected (still MultiGraph with possible parallel edges)
    U_multi = ox.convert.to_undirected(G)

    # 5) Collapse into a simple undirected Graph, keeping the minimum weight per (u, v)
    U = nx.Graph()
    U.add_nodes_from(U_multi.nodes(data=True))

    # Store node positions under "pos" for downstream renderers.
    for node, data in U.nodes(data=True):
        if "x" in data and "y" in data:
            data["pos"] = (data["x"], data["y"])

    for u, v, data in U_multi.edges(data=True):
        if u == v:
            continue
        # Prefer travel_time; fallback to length; otherwise 1.0
        w = data.get("travel_time", None)
        if w is None:
            w = data.get("length", 1.0)
        w = float(w)

        if U.has_edge(u, v):
            # keep the edge with smaller weight
            if w < float(U[u][v].get("weight", float("inf"))):
                # update attributes with the better edge
                U[u][v].clear()
                U[u][v].update(data)
                U[u][v]["weight"] = w
        else:
            U.add_edge(u, v, **data)
            U[u][v]["weight"] = w

    # 6) Save as .gpickle via pickle (works with NetworkX 3+)
    with open(output_path, "wb") as f:
        pickle.dump(U, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved undirected graph to: {os.path.abspath(output_path)}")
    print(f"Nodes: {U.number_of_nodes()}, Edges: {U.number_of_edges()}")
    # Sanity check
    sample_u, sample_v = next(iter(U.edges()))
    print("Sample edge weight:", U[sample_u][sample_v].get("weight"))

    return U


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OSM map and save as undirected gpickle.")
    parser.add_argument("--place", required=True, help="Place name to geocode, e.g. 'Singapore, Singapore'.")
    parser.add_argument("--output-path", "--output_path", dest="output_path", required=True, help="Output .gpickle path.")
    parser.add_argument("--dist-m", "--dist_m", dest="dist_m", type=int, default=1000, help="Search radius in meters.")
    parser.add_argument("--consolidate-tolerance-m", type=float, default=20.0, help="Merge nodes within this distance (meters).")
    args = parser.parse_args()

    build_undirected_graph_near_place(
        place=args.place,
        dist_m=args.dist_m,
        network_type="drive",      # or "walk"
        output_path=args.output_path,
        consolidate_tolerance_m=args.consolidate_tolerance_m,
    )
