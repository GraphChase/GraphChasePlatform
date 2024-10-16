import argparse

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--graph_id', type=int, default=0)
    argparser.add_argument('--save_path', type=str, default='./experiments/nsg_nfsp')
    argparser.add_argument('--save_folder', default=None)
    argparser.add_argument("--no_cuda", action="store_false", dest="use_cuda", 
                        help="Do not use CUDA even if available")
    argparser.add_argument("--device_id", type=int, default=2, 
                        help="use cuda id")      
    argparser.add_argument("--seed", type=int, default=777)     
    ############### Policy Network ##############
    argparser.add_argument('--embedding_size', type=int, default=32)
    argparser.add_argument('--hidden_size', type=int, default=64)
    argparser.add_argument('--relevant_v_size', type=int, default=64)
    argparser.add_argument('--if_naivedrrn', type=bool, default=False)
    ############### Replay Buffer ##############
    argparser.add_argument('--br_buffer_capacity', type=int, default=int(5e5))
    argparser.add_argument('--avg_buffer_capacity', type=int, default=int(1e7))
    ############### Agent ##############
    argparser.add_argument('--br_lr', type=float, default=0.0001)
    argparser.add_argument('--avg_lr', type=float, default=0.0001)
    argparser.add_argument('--d_expl', type=float, default=0.0)
    argparser.add_argument('--a_expl', type=float, default=0.1)
    argparser.add_argument('--br_prob', type=float, default=0.1)
    argparser.add_argument('--seq_mode', type=str, default='cnn')
    argparser.add_argument('--pre_embedding_path', type=str, default=None)  
    argparser.add_argument('--br_warmup_path', type=str, default=None)  
    argparser.add_argument('--defender_rl_mode', type=str, default='drrn')
    argparser.add_argument('--defender_sl_mode', type=str, default='drrn')
    argparser.add_argument('--attacker_mode', type=str, default='bandit')
    ############### Environment ##############
    # argparser.add_argument('--time_horizon', type=int, default=9)
    ############### Train and Eval ##############
    argparser.add_argument('--br_idx', type=int, default=0)
    argparser.add_argument('--max_episodes', type=int, default=int(5e6))
    argparser.add_argument('--train_br_freq', type=int, default=4)
    argparser.add_argument('--train_avg_freq', type=int, default=32)
    argparser.add_argument('--check_freq', type=int, default=int(1e5))
    argparser.add_argument('--check_from', type=int, default=int(1e5))
    argparser.add_argument('--display_freq', type=int, default=int(1e3))
    argparser.add_argument('--min_to_train', type=int, default=1000)
    argparser.add_argument('--br_batch_size', type=int, default=128)
    argparser.add_argument('--avg_batch_size', type=int, default=256)
    argparser.add_argument('--exact_br', type=bool, default=False)
    args = argparser.parse_args()
    return args