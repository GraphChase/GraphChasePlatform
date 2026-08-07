from __future__ import annotations

import argparse
import json
import os


def comma_separated_ints(value: str) -> list[int]:
    if value.strip() == "":
        return []
    return [int(item) for item in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grasper MAPPO runner config")
    parser.add_argument("--graph_gpickle_path", type=str, default="graphchase/graph/custom_graph/7_7_grid_graph.gpickle", help="Path to gpickle graph file")
    parser.add_argument("--attacker_init", type=comma_separated_ints, default=comma_separated_ints("25"), help="Comma-separated attacker start nodes")
    parser.add_argument("--defender_init", type=comma_separated_ints, default=comma_separated_ints("9, 28, 44, 46"), help="Comma-separated defender start nodes")
    parser.add_argument("--exit_nodes", type=comma_separated_ints, default=comma_separated_ints("4, 22, 43, 49"), help="Comma-separated exit nodes")
    parser.add_argument("--time_horizon", type=int, default=7)
    parser.add_argument("--graph_metadata", type=str, default=None, help="Optional JSON string for custom graph metadata")
    parser.add_argument("--use_weighted_graph", action="store_true", default=False, help="Use edge weights from gpickle graphs")
    parser.add_argument("--graph_type", type=str, default="Custom_Graph", help="Identifier string for logging paths")
    parser.add_argument("--sf_sw_node_num", type=int, default=300, help="node number for SF/SW/ER graphs")
    parser.add_argument("--seed_to_generate_graph", type=int, default=100, help="seed for graph generation")
    parser.add_argument("--small_world_k", type=int, default=10, help="k for small-world graph")

    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--use_cuda", action="store_true", default=True, help="Enable CUDA (default: on)")
    parser.add_argument("--no_cuda", action="store_false", dest="use_cuda", help="Disable CUDA even if available")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="./experiments/grasper_mappo")

    parser.add_argument("--action_type", type=str, default="exit_node", help="Attacker action type (exit_node or all_path)")
    parser.add_argument("--attacker_path_type", type=str, choices=["shortest", "simple"], default="shortest", help="Path generation strategy for attacker")
    parser.add_argument("--strategy_type", type=str, default="mix", help="Attacker strategy type for PSRO")

    parser.add_argument("--state_emb_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--node_feat_dim", type=int, default=3)
    parser.add_argument("--gnn_hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_output_dim", type=int, default=32)
    parser.add_argument("--gnn_num_layer", type=int, default=2)
    parser.add_argument("--gnn_dropout", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=128)

    parser.add_argument("--pre_pretrain_save_path", type=str, default="graphchase/results/grasper/pre_pretrain/sy_graph/")
    parser.add_argument("--pretrain_save_path", type=str, default="graphchase/results/grasper/pretrain")
    parser.add_argument("--h_init", type=str, default="unif")
    parser.add_argument("--use_mix", action="store_true", default=True)
    parser.add_argument("--use_augmentation", action="store_true", default=False)
    parser.add_argument("--use_node_emb", action="store_true", default=False)
    parser.add_argument("--use_emb_layer", action="store_true", default=False)
    parser.add_argument("--use_end_to_end", action="store_true", default=False)
    parser.add_argument("--use_act_supervisor", action="store_true", default=False)
    parser.add_argument("--act_sup_coef_min", type=float, default=0.01)
    parser.add_argument("--act_sup_coef_max", type=float, default=0.1)
    parser.add_argument("--act_sup_coef_decay", type=int, default=40000)
    parser.add_argument("--perfect_info", action="store_true", default=True)
    parser.add_argument("--max_time_horizon_for_state_emb", type=int, default=20)
    parser.add_argument("--max_time_horizon", type=int, default=20)

    parser.add_argument("--load_game_pool_file", action="store_true", default=False)
    parser.add_argument("--pool_size", type=int, default=20000)
    parser.add_argument("--game_pool_dir", type=str, default="graphchase/graph/grasper_game_pool")
    parser.add_argument("--edge_probability", type=float, default=1.0)
    parser.add_argument("--row", type=int, default=10, help="row for grid graph")
    parser.add_argument("--row_min", type=int, default=9, help="min row for grid graph")
    parser.add_argument("--row_max", type=int, default=12, help="max row for grid graph")
    parser.add_argument("--column", type=int, default=10, help="column for grid graph")
    parser.add_argument("--column_min", type=int, default=9, help="min column for grid graph")
    parser.add_argument("--column_max", type=int, default=12, help="max column for grid graph")    
    parser.add_argument("--min_num_defender", type=int, default=5)
    parser.add_argument("--max_num_defender", type=int, default=5)
    parser.add_argument("--min_num_exit", type=int, default=8)
    parser.add_argument("--max_num_exit", type=int, default=8)
    parser.add_argument("--min_time_horizon", type=int, default=6)
    parser.add_argument("--min_attacker_pth_len", type=int, default=6)
    parser.add_argument("--prob_of_obs_attacker", type=float, default=1.0)
    parser.add_argument("--differ_size", action="store_true", default=False)
    parser.add_argument("--use_cache", action="store_true", default=False)

    parser.add_argument("--num_iterations", type=int, default=20000000)
    parser.add_argument("--save_every", type=int, default=2000000)
    parser.add_argument("--num_games", type=int, default=5)
    parser.add_argument("--num_task", type=int, default=5)
    parser.add_argument("--num_sample", type=int, default=10)
    parser.add_argument("--train_num_per_ite", type=int, default=1)
    parser.add_argument("--update_every_n_episodes", type=int, default=-1)
    parser.add_argument("--checkpoint", type=int, default=0)

    parser.add_argument("--load_graph_emb_model", action="store_true", default=False)
    parser.add_argument("--graph_emb_model_path", type=str, default=None)
    parser.add_argument("--max_epoch", type=int, default=2000)

    parser.add_argument("--load_pretrain_model", action="store_true", default=False)
    parser.add_argument("--pretrain_model_iteration", type=int, default=20000000)

    parser.add_argument("--num_psro_iteration", type=int, default=25)
    parser.add_argument("--rollouts_per_attacker_action", type=int, default=10000)
    parser.add_argument("--train_defender_batches", type=int, default=630)
    parser.add_argument("--episodes_per_batch", type=int, default=8)
    parser.add_argument("--vec_envs", type=int, default=16)
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--reward_mode", type=str, default="utility")

    # MAPPO args
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
    parser.add_argument("--opti_eps", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--ppo_epoch", type=int, default=5)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=1.0)
    parser.add_argument("--use_max_grad_norm", action="store_true", default=False)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--use_clipped_value_loss", action="store_true", default=False)
    parser.add_argument("--use_huber_loss", action="store_true", default=False)
    parser.add_argument("--huber_delta", type=float, default=10.0)
    parser.add_argument("--use_popart", action="store_true", default=False)
    parser.add_argument("--use_valuenorm", action="store_true", default=False)
    parser.add_argument("--use_advnorm", action="store_false", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--use_gae", action="store_true", default=False)
    parser.add_argument("--gae_lambda", type=float, default=0.95)    

    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.graph_metadata:
        args.graph_metadata = json.loads(args.graph_metadata)
    else:
        args.graph_metadata = {}
    return args
