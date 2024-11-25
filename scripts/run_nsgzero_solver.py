import os
import sys
import torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.nsgzero_configs import parse_args
from common_utils import set_seeds
from agent.nsgzero.mcts_defender_policy import NsgzeroDefenderPolicy
from agent.nsgzero.mcts_defender_runner import NsgzeroDefenderRunner
from agent.nsgzero.nfsp_evader_runner import NFSPAttacker
from solver.nsgzero_solver import NsgzeroSolver
from graph.graph_files.custom_graph import CustomGraph


def main(args=None):
    if args is None:
        args = parse_args()
    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.use_cuda else "cpu")
    set_seeds(args.seed)

    graph = CustomGraph(args.graph_id)
    if args.graph_type == 'GridGraph':
        from env.grid_env import GridEnv
        env = GridEnv(graph, return_reward_mode='defender', return_legal_action=True, nextstate_as_action=True, render_mode="rgb_array")
    elif args.graph_type == 'AnyGraph':
        from env.any_graph_env import AnyGraphEnv
        env = AnyGraphEnv(graph, return_reward_mode='defender', return_legal_action=True, render_mode="rgb_array")   

    defender_policy = NsgzeroDefenderPolicy(env, args)
    defender_runner = NsgzeroDefenderRunner(env, defender_policy, args)
    Attacker = NFSPAttacker(env, args)

    solver = NsgzeroSolver(env, defender_runner, Attacker)
    solver.solve()

if __name__ == "__main__":
    main()