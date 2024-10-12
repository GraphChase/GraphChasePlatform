
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./experiments/grasper')
    parser.add_argument('--save_folder', default=None)
    parser.add_argument("--no_cuda", action="store_false", dest="use_cuda", 
                        help="Do not use CUDA even if available")
    parser.add_argument("--device_id", type=int, default=0, 
                        help="use cuda id")      
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument('--base_rl', type=str, default='grasper_mappo', help='mappo, grasper_mappo')
    parser.add_argument('--graph_id', type=int, default=int(0), 
                        help='use which graph')    

    # Evader
    parser.add_argument('--attacker_type', choices=['all_path', 'exit_node'], default='exit_node',
                        help='Type of attacker: all_path or exit_node')
    parser.add_argument('--strategy_type', choices=['mix', 'greedy'], default='mix',
                        help='Strategy type: mix or greedy')    

    # Graph generating parameters
    parser.add_argument('--graph_type', type=str, default='Grid_Graph', help='Grid_Graph, SG_Graph, SY_Graph, SF_Graph')
    parser.add_argument('--row', type=int, default=7, help='row for grid map')
    parser.add_argument('--column', type=int, default=7, help='column for grid map')
    parser.add_argument('--edge_probability', type=float, default=0.8, help='edge_probability')
    parser.add_argument('--min_time_horizon', type=int, default=6, help='min time horizon')
    parser.add_argument('--max_time_horizon', type=int, default=10, help='max time horizon')
    parser.add_argument('--num_defender', type=int, default=4, help='number of pursuers')
    parser.add_argument('--num_exit', type=int, default=4, help='number of exit nodes')
    parser.add_argument('--pool_size', type=int, default=1000, help='number of games in the pool')
    parser.add_argument('--min_evader_pth_len', type=int, default=6, help='column for grid map')
    parser.add_argument('--sf_sw_node_num', type=int, default=300, help='node number of sf or sw map')

    # Pre-pretrain parameters
    parser.add_argument('--node_feat_dim', type=int, default=3, help='feature dims of a node in GNN')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128, help='hidden dim of GNN')
    parser.add_argument('--gnn_output_dim', type=int, default=32, help='output dim of GNN')
    parser.add_argument('--gnn_num_layer', type=int, default=2, help='hidden dim of GNN')
    parser.add_argument('--gnn_dropout', type=float, default=0.5, help='dropout rate of GNN')
    parser.add_argument("--graph_pretrain_batch_size", type=int, default=32, help="Batch size for graph pretraining")
    parser.add_argument("--graph_pretrain_lr", type=float, default=0.00015, help="Learning rate for graph pretraining")
    parser.add_argument("--graph_pretrain_weight_decay", type=float, default=1e-5, help="Weight decay for graph pretraining")
    parser.add_argument("--graph_pretrain_max_epoch", type=int, default=200, help="Maximum epochs for graph pretraining")

    # Pretrain parameters
    parser.add_argument('--load_graph_emb_model', action='store_true', help='whether load graph emb model')
    parser.add_argument('--load_pretrain_model', action='store_true', help='whether load pretrained model')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_games', type=int, default=5, help='number of envs in a PPO/MAPPO pretrain iteration')
    parser.add_argument('--num_task', type=int, default=5, help='number of opponent policies of a given game')
    parser.add_argument('--num_sample', type=int, default=10, help='number of plays of a given opponent policy')
    parser.add_argument('--num_iterations', type=int, default=2000, help='number of PPO/MAPPO pretrain iterations')
    parser.add_argument('--use_act_supervisor', action='store_true', default=False, help='whether use HMP')
    parser.add_argument('--use_emb_layer', action='store_true', default=False, help='whether use observation embedding layer')
    parser.add_argument('--use_augmentation', action='store_true', default=False, help='concat. game config embed with state')
    parser.add_argument('--no_end_to_end', action='store_true', dest="use_end_to_end", help='end-to-end training of whole network architecture')    
    parser.add_argument('--row_max_for_state_emb', type=int, default=20, help='row for grid map')
    parser.add_argument('--column_max_for_state_emb', type=int, default=20, help='row for grid map')
    parser.add_argument('--max_time_horizon_for_state_emb', type=int, default=20, help='max time horizon for  state emb')
    parser.add_argument('--hypernet_z_dim', type=int, default=128, help='z dim of hyper net')
    parser.add_argument('--hypernet_dynamic_hidden_dim', type=int, default=128, help='hidden dim of hyper net')
    parser.add_argument('--hypernet_hidden_dim', type=int, default=128, help='hidden dim of hyper net')
    parser.add_argument('--head_init_method', choices=['uniform', 'xavier_uniform', 'kaiming_uniform'], default='kaiming_uniform',
                        help='HyperNet init method')
    parser.add_argument('--state_emb_dim', type=int, default=16, help='state embedding dims')
    parser.add_argument('--checkpoint', type=int, default=0, help='checkpoint')
    parser.add_argument('--save_every', type=int, default=2000, help='save pretrain models every x iterations')

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

def get_mappo_config():

    parser = argparse.ArgumentParser(description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default='mappo', choices=["rmappo", "mappo"])
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int, default=1, help="Number of torch threads for training")

    # env parameters
    parser.add_argument("--use_obs_instead_of_state", action='store_true', default=False, help="Whether to use global state or concatenated obs")
    # network parameters
    parser.add_argument("--share_policy", action='store_false', default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false', default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true', default=False, help="Whether to use stacked_frames")
    parser.add_argument("--layer_N", type=int, default=1, help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false', default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_true', default=False, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_true', default=False, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_true', default=False, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")
    
    parser.add_argument('--act_sup_coef_min', type=float, default=0.01, help='min coef of action regularization')
    parser.add_argument('--act_sup_coef_max', type=float, default=0.1, help='max coef of action regularization')
    parser.add_argument('--act_sup_coef_decay', type=int, default=40000, help='number of episodes to decay action regularization')     

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true', default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false', default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10, help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=3e-4, help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=10, help='number of ppo epochs (default: 15)') # 
    parser.add_argument("--use_clipped_value_loss", action='store_true', default=False, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float, default=1.0, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm", action='store_true', default=False, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_advnorm", action='store_false', default=True, help='use normalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--use_gae", action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument("--gae_lambda", type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_true', default=False, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks", action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",  action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true', default=False, help='use a linear schedule on the learning rate')

    return parser