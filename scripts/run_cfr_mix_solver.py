import os
import sys
import torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from graph.cfr_graph_wrapper import CfrmixGraph
from agent.cfr_mix.algorithm.mix_deep_probe_cfr import deep_mix_probe_cfr 
from configs.cfr_mix_configs import parse_args
from graph.graph_files.custom_graph import CustomGraph
from common_utils import directory_config, store_args

args = parse_args()

if torch.cuda.is_available() and args.use_cuda:
    torch.cuda.set_device(args.device_id)

graph = CustomGraph(args.graph_id)
game_graph = CfrmixGraph(row=graph.row, column=graph.column, initial_locations=graph.initial_locations)
agent_init_location = []
agent_init_location = graph.attacker_init + [tuple(graph.defender_init)]

save_path = directory_config(args.save_path)
store_args(args, save_path)  

regret_file_name = [f"{save_path}/5DeepMCFR4_mix_probe_cfr_regret_0_model.dat", f"{save_path}/5DeepMCFR4_mix_probe_cfr_regret_1_model.dat"]
strategy_file_name = f"{save_path}"+'/5DeepMCFR4_mix_probe_cfr_defender_strategy_model_{}.dat'

#run deep cfr + mix
deep_mix_probe_cfr(game_graph, agent_init_location, graph.time_horizon, args.network_dim, args.sample_number, args.action_number, args.attacker_regret_batch_size,
                       args.defender_regret_batch_size, args.defender_strategy_batch_size, args.train_epoch, args.attacker_regret_lr, args.defender_regret_lr, args.defender_strategy_lr,
                       regret_file_name, strategy_file_name, args.iteration)
