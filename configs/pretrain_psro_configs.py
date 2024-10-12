import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_id', type=int, default=int(0), 
                        help='use which graph')
    parser.add_argument('--seed', type=int, default=int(5), 
                        help='training seed')
    parser.add_argument("--no_cuda", action="store_false", dest="use_cuda", 
                        help="Do not use CUDA even if available")
    parser.add_argument("--device_id", type=int, default=5, 
                        help="use cuda id")
    parser.add_argument('--save_path', type=str, default='./experiments/pretrain_psro')
    parser.add_argument("--train_mode", type=str, choices=["OnlyPretrain", "PretrainPsro", "OnlyPsro"], default="PretrainPsro", help="Use which train mode")
    
    # Pretrain
    parser.add_argument('--defender_runner_type', type=str, default="ppo", help='ppo')
    parser.add_argument('--num_pretrain_iteration', type=int, default=int(200), 
                        help='iteration number of pretraining')
    parser.add_argument('--rollout_evader_episodes', type=int, default=int(20), 
                        help='each evader policy is evaluated how many times')    
    parser.add_argument('--save_interval', type=int, default=int(10), 
                        help='after how many iterations save the model')
    
    # Graph Model parameters
    parser.add_argument("--use_node_embedding", type=bool, default=True, help="Whether to use node embedding")
    parser.add_argument("--use_past_history", type=bool, default=False, help="Whether to use evader's past trajectory")
    parser.add_argument("--emb_size", type=int, default=16, help="Embedding size for LINE")
    parser.add_argument("--node_embedding_method", type=str, default="line_plus_information", help="Method for node embedding") # 
    parser.add_argument("--load_node_embedding_model", type=str, default=None, help="Path to load the node embedding model")
    parser.add_argument("--load_node_information_file", type=str, default=None, help="Path to load the node information file")
    parser.add_argument("--load_information_proximity_matrix", type=str, default=None, help="Path to load the information proximity matrix")
    parser.add_argument("--node_information_type", type=str, default="all", help="Type of node information to load")
    parser.add_argument("--node_information_normalize", type=bool, default=True, help="Whether to normalize node information")
    parser.add_argument("--information_similarity", type=str, default="cosine", help="Type of information similarity to use")
    parser.add_argument("--line_batch_size", type=int, default=32, help="Batch size for LINE")
    parser.add_argument("--line_epochs", type=int, default=5, help="Number of epochs for LINE")
    parser.add_argument("--line_order", type=str, choices=["first", "second", "all"], default="all", help="Order for LINE")
    
    # Evader
    parser.add_argument('--attacker_type', choices=['all_path', 'exit_node'], default='exit_node',
                        help='Type of attacker: all_path or exit_node')
    parser.add_argument('--strategy_type', choices=['mix', 'greedy'], default='mix',
                        help='Strategy type: mix or greedy')
    parser.add_argument('--evader_oracle_size', type=int, default=int(30), 
                        help='generate how many random strategy')  
    
    # PPO
    parser.add_argument('--ppo_hidden_size', type=int, default=128,
                        help='Hidden size of the PPO network')
    parser.add_argument('--ppo_actor_lr', type=float, default= 1e-3,
                        help='Learning rate for the PPO actor network')
    parser.add_argument('--ppo_critic_lr', type=float, default= 3e-3,
                        help='Learning rate for the PPO critic network')
    parser.add_argument('--ppo_gamma', type=float, default=1.,
                        help='Discount factor for PPO rewards')
    parser.add_argument('--ppo_batch_size', type=int, default=32,
                        help='Batch size for PPO training')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='The PPO clipping parameter')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Maximum gradient norm for gradient clipping in PPO')
    parser.add_argument('--ppo_epochs', type=int, default=10,
                        help='Number of epochs to train PPO on a single batch of trajectories')
    parser.add_argument('--load_defender_model', default=None,
                        help='Path to a pre-trained defender model (default: %(default)s)')
    parser.add_argument('--pretrain_model_checkpoint', type=int, default=1,
                        help='the prefix of loading pretrained models')    
    
    # PSRO
    parser.add_argument('--eval_episodes', type=int, default=int(1e3),
                        help='Number of evaluation episodes')    
    parser.add_argument('--num_psro_iteration', type=int, default=int(20), 
                        help='iteration number of psro')
    parser.add_argument('--train_evader_number', type=int, default=int(1e4), 
                        help='train evader number to get best response')
    parser.add_argument('--train_pursuer_number', type=int, default=int(5e3),
                        help='train pursuer number to get best response')
    return parser.parse_args()