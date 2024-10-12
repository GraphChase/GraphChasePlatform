import os
import sys
import torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.nsgnfsp_configs import parse_args
from common_utils import set_seeds
from env.grid_env import GridEnv
import agent.nsg_nfsp.nsgnfsp_model as model
import agent.nsg_nfsp.nsgnfsp_defender_policy as policy
from agent.nsg_nfsp.nsgnfsp_defender_policy import CreateDefender
from agent.nsg_nfsp.nsgnfsp_attacker_policy import CreateAttacker
from agent.nsg_nfsp.nsgnfsp_defender_runner import NsgNfspDefenderRunner
from solver.nsgnfsp_solver import NsgNfspSolver
from graph.graph_files.custom_graph import CustomGraph


def main(args=None):
    if args is None:
        args = parse_args()
    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.use_cuda else "cpu")
    model.device = args.device
    policy.device = args.device
    set_seeds(args.seed)

    graph = CustomGraph(args.graph_id)
    env = GridEnv(graph, return_reward_mode='defender', return_legal_action=True, nextstate_as_action=True, render_mode="rgb_array")

    defender_policy = CreateDefender(graph, args)
    defender_runner = NsgNfspDefenderRunner(env, defender_policy, args)
    Attacker = CreateAttacker(graph, args)

    solver = NsgNfspSolver(env, defender_runner, Attacker)
    solver.solve()

if __name__ == "__main__":
    main()