import os
import sys
import torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.grasper_configs import parse_args, get_mappo_config
from agent.grasper.grasper_mappo_policy import RHMAPPO
from agent.grasper.mappo_policy import RMAPPO
from common_utils import set_seeds
from agent.grasper.generate_training_set import get_training_set
from agent.grasper.utils import sample_env
from agent.grasper.grasper_mappo_defender_runner import GrasperDefenderRunner
from agent.pretrain_psro.path_evader_runner import PathEvaderRunner
from solver.psro_solver import PSRO
from common_utils import directory_config, store_args
from graph.graph_files.custom_graph import CustomGraph
from env.grid_env import GridEnv

def main(args=None):
    if args is None:
        args = parse_args()
    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.use_cuda else "cpu")
    set_seeds(args.seed)

    parser = get_mappo_config()
    mappo_args = parser.parse_known_args(sys.argv[1:])[0]

    # Get the graph and env that you want to fine-tuning
    graph = CustomGraph(args.graph_id)
    env = GridEnv(graph, render_mode="rgb_array")

    evader_runner = PathEvaderRunner(env, args)
    
    pretrain_policy = RHMAPPO(env, mappo_args, args)
    finetune_policy = RMAPPO(env, mappo_args, args)        
    defender_runner = GrasperDefenderRunner(pretrain_policy, finetune_policy, args, env)

    if args.running_mode != 'OnlyFinetuning':
        # Step 1: Generate the graphs that have the same shape with your aiming graph
        # Save the training set to args.save_path/data/related_files/game_pool/(graph shape)/...pik
        args.row = env._rows; args.column = env._colums
        get_training_set(args)
        if args.running_mode == 'GenerateTrainingGraph':
            print("Generating Training Graph Done.")
            sys.exit()
        print("Generating Training Graph Done.")

        # Step 2: According to the training set graph, pre pretrain graph model
        # Save the pre-pretrain model to args.save_path/data/pretrain_models/graph_learning/...pt
        defender_runner.pre_pretrain()
        if args.running_mode == 'PrePretrain':
            print("Pre-Pretraining Phase Done.")
            sys.exit()
        print("Pre-Pretraining Phase Done.")

        # Step 3: Run pretrain phase
        # Random choose envs from env_pool, and train pretrain model which is used to generate fine-tuning models' parameters
        # Save the pretrain model to args.save_path/data/pretrain_models/grasper_mappo/...pt
        defender_runner.pretrain(PathEvaderRunner)
        if args.running_mode == 'Pretrain':
            print("Pretraining Phase Done.")
            sys.exit()     
        print("Pretraining Phase Done.")

    # Step 4: Fine-tuning the model with PSRO
    # First, prepare for the fine-tuning, get pre-pretrain model and pretrain model
    defender_runner.finetuning_init(env)
    # Second, set fine-tuning phase's save_path
    args.save_path = directory_config(args.save_path)
    store_args(args, args.save_path) 
    # Finally, PSRO to solve
    solver = PSRO(env, args, evader_runner, defender_runner)
    solver.solve()
        
if __name__ == "__main__":
    main()