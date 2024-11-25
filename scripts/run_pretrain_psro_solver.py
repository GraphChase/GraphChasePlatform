import os
import sys
import torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.pretrain_psro_configs import parse_args
from common_utils import set_seeds
from agent.pretrain_psro.node_embedding import NodeEmbedding
from graph.graph_files.custom_graph import CustomGraph
from agent.pretrain_psro.ppo_defender_runner import PretrainPsroDefenderRunner
from agent.pretrain_psro.ppo_defender_policy import PPOAgent
from agent.pretrain_psro.path_evader_runner import PathEvaderRunner
from common_utils import directory_config, store_args
from solver.psro_solver import PSRO


def main(args=None):
    if args is None:
        args = parse_args()
    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.use_cuda else "cpu")
    set_seeds(args.seed)

    graph = CustomGraph(args.graph_id)
    if args.graph_type == 'GridGraph':
        from env.grid_env import GridEnv
        env = GridEnv(graph, render_mode="rgb_array")
    elif args.graph_type == 'AnyGraph':
        from env.any_graph_env import AnyGraphEnv
        env = AnyGraphEnv(graph, render_mode="rgb_array")

    args.save_path = directory_config(args.save_path)
    store_args(args, args.save_path)      
    
    n2v = NodeEmbedding(graph, args)
    defender_policy = PPOAgent(env, args)
    defender_runner = PretrainPsroDefenderRunner(env, defender_policy, args, n2v)
    evader_runner = PathEvaderRunner(env, args)

    if args.train_mode == "OnlyPretrain":
        defender_runner.pretrain(evader_runner)

    elif args.train_mode == "PretrainPsro":
        defender_runner.pretrain(evader_runner)
        solver = PSRO(env, args, evader_runner, defender_runner)
        solver.solve()

    elif args.train_mode == "OnlyPsro":
        solver = PSRO(env, args, evader_runner, defender_runner)
        solver.solve()
        
if __name__ == "__main__":
    main()