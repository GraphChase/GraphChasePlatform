from __future__ import annotations

import os
import pickle
import random
import logging

import networkx as nx

from graphchase.solver_cfgs.pretrain_psro_cfgs_template import parse_args
from graphchase.envs.unsg_env import UNSGEnv
from graphchase.graph.game_settings import GameSettings
from graphchase.utils import run_episode

logger = logging.getLogger(__name__)


def sample_action(env: UNSGEnv, agent_idx: int) -> int:
    start, end, _ = env.positions[agent_idx]
    if start == end:
        neighbors = env.settings.neighbor_map.get(start, set()) if env.settings.neighbor_map else set(env.settings.neighbors(start))
        choices = list(neighbors) + [0]
        return random.choice(choices) if choices else 0
    return random.choice([0, start, end])


def build_symmetric_adjacency(base_graph: nx.Graph, seed: int = 0, weight_range: tuple[float, float] = (0.5, 2.0)):
    rng = random.Random(seed)
    node_ids = sorted(base_graph.nodes())
    n = len(node_ids)
    adjacency = [[0.0 for _ in range(n)] for _ in range(n)]
    low, high = weight_range
    index = {node: i for i, node in enumerate(node_ids)}
    for u, v in base_graph.edges():
        w = round(rng.uniform(low, high), 1)
        i, j = index[u], index[v]
        adjacency[i][j] = w
        adjacency[j][i] = w
    return adjacency, node_ids


def build_asymmetric_adjacency(base_graph: nx.Graph, seed: int = 0, weight_range: tuple[float, float] = (0.5, 2.0)):
    rng = random.Random(seed)
    node_ids = sorted(base_graph.nodes())
    n = len(node_ids)
    adjacency = [[0.0 for _ in range(n)] for _ in range(n)]
    low, high = weight_range
    index = {node: i for i, node in enumerate(node_ids)}
    for u, v in base_graph.edges():
        w_uv = round(rng.uniform(low, high), 1)
        w_vu = round(rng.uniform(low, high), 1)
        i, j = index[u], index[v]
        adjacency[i][j] = w_uv
        adjacency[j][i] = w_vu
    return adjacency, node_ids


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.graph_gpickle_path is None:
        default_path = os.path.join("graphchase", "graph", "custom_graph", "7_7_grid_graph.gpickle")
        args.graph_gpickle_path = default_path
        logger.info(f"[Info] Using default gpickle: {default_path}")

    with open(args.graph_gpickle_path, "rb") as fp:
        base_graph = pickle.load(fp)
    metadata = dict(args.graph_metadata or {})
    metadata.setdefault("source_gpickle", args.graph_gpickle_path)        

    # adjacency_matrix, node_ids = build_symmetric_adjacency(
    #     base_graph=base_graph,
    #     seed=getattr(args, "seed", 0),
    #     weight_range=(0.5, 2.0),
    # )

    # adjacency_matrix, node_ids = build_asymmetric_adjacency(
    #     base_graph=base_graph,
    #     seed=getattr(args, "seed", 0),
    #     weight_range=(0.5, 2.0),
    # )
    # num_nodes = len(node_ids)
    # num_edges = sum(1 for i in range(num_nodes) for j in range(num_nodes) if adjacency_matrix[i][j] != 0)
    # print(f"[Graph] Overriding with asymmetric adjacency_matrix: nodes={num_nodes}, directed_edges={num_edges}")
    # metadata = dict(args.graph_metadata or {})
    # metadata.update(
    #     {
    #         "source_gpickle": args.graph_gpickle_path,
    #         "adjacency_matrix": adjacency_matrix,
    #         "node_ids": node_ids,
    #     }
    # )

    settings = GameSettings(
        graph=base_graph,
        attacker_init=list(args.attacker_init),
        defender_init=list(args.defender_init),
        exit_nodes=list(args.exit_nodes),
        time_horizon=args.time_horizon,
        metadata=metadata,
        use_weighted_graph=bool(getattr(args, "use_weighted_graph", False)),
    )

    env = UNSGEnv(settings, render_mode="rgb_array")

    def attacker_policy(obs: dict, info: dict) -> list[int]:
        decisions = info.get("decision_agents", list(range(env.num_agents)))
        actions = [0] * env.num_attackers
        for i in range(env.num_attackers):
            if i in decisions:
                actions[i] = sample_action(env, i)
        return actions

    def defender_policy(obs: dict, info: dict) -> list[int]:
        decisions = info.get("decision_agents", list(range(env.num_agents)))
        actions = [0] * env.num_defenders
        for i in range(env.num_defenders):
            agent_idx = env.num_attackers + i
            if agent_idx in decisions:
                actions[i] = sample_action(env, agent_idx)
        return actions

    video_root = os.path.join("unsg_add_adjacency_test")
    episodes_data = run_episode(
        env=env,
        num_episodes=1,
        attacker_policy_fn=attacker_policy,
        defender_policy_fn=defender_policy,
        video_root=video_root,
        save_video=True,
    )

    for ep_idx, ep in enumerate(episodes_data):
        logger.info(f"[Episode {ep_idx}] steps={len(ep['info']) - 1}")
        # ep['info'] 和 ep['obs'] 含起始状态；actions/reward 从第一步开始
        for t in range(len(ep["info"])):
            info_t = ep["info"][t]
            obs_t = ep["obs"][t]
            decision_agents = info_t.get("decision_agents", [])
            attacker_action = ep["attacker_action"][t] if t < len(ep["attacker_action"]) else None
            defender_action = ep["defender_action"][t] if t < len(ep["defender_action"]) else None
            reward_t = ep["reward"][t - 1] if t > 0 and t - 1 < len(ep["reward"]) else None
            logger.info(
                f"  t={t}: decisions={decision_agents}, "
                f"attacker_action={attacker_action}, defender_action={defender_action}, "
                f"reward={reward_t}, obs={obs_t}"
            )

    env.close()


if __name__ == "__main__":
    main()
