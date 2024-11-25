import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_id', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./experiments/nsg_nfsp')
    parser.add_argument('--save_folder', default=None)
    parser.add_argument("--no_cuda", action="store_false", dest="use_cuda", 
                        help="Do not use CUDA even if available")
    parser.add_argument("--device_id", type=int, default=2, 
                        help="use cuda id")      
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--graph_type", type=str, choices=["GridGraph", "AnyGraph"], default="GridGraph", help="The type of graph structure")
    ############### Policy Network ##############
    parser.add_argument('--embedding_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--relevant_v_size', type=int, default=64)
    parser.add_argument('--if_naivedrrn', type=bool, default=False)
    ############### Replay Buffer ##############
    parser.add_argument('--br_buffer_capacity', type=int, default=int(5e5))
    parser.add_argument('--avg_buffer_capacity', type=int, default=int(1e7))
    ############### Agent ##############
    parser.add_argument('--br_lr', type=float, default=0.0001)
    parser.add_argument('--avg_lr', type=float, default=0.0001)
    parser.add_argument('--d_expl', type=float, default=0.0)
    parser.add_argument('--a_expl', type=float, default=0.1)
    parser.add_argument('--br_prob', type=float, default=0.1)
    parser.add_argument('--seq_mode', type=str, default='cnn')
    parser.add_argument('--pre_embedding_path', type=str, default=None)  
    parser.add_argument('--br_warmup_path', type=str, default=None)  
    parser.add_argument('--defender_rl_mode', type=str, default='drrn')
    parser.add_argument('--defender_sl_mode', type=str, default='drrn')
    parser.add_argument('--attacker_mode', type=str, default='bandit')

    ############### Train and Eval ##############
    parser.add_argument('--br_idx', type=int, default=0)
    parser.add_argument('--max_episodes', type=int, default=int(5e6))
    parser.add_argument('--train_br_freq', type=int, default=4)
    parser.add_argument('--train_avg_freq', type=int, default=32)
    parser.add_argument('--check_freq', type=int, default=int(1e5))
    parser.add_argument('--check_from', type=int, default=int(1e5))
    parser.add_argument('--display_freq', type=int, default=int(1e3))
    parser.add_argument('--min_to_train', type=int, default=1000)
    parser.add_argument('--br_batch_size', type=int, default=128)
    parser.add_argument('--avg_batch_size', type=int, default=256)
    parser.add_argument('--exact_br', type=bool, default=False)
    args = parser.parse_args()
    return args