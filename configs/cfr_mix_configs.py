import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='CFR-MIX Parameters')

    parser.add_argument('--graph_id', type=int, default=int(0), 
                        help='use which graph')   
    parser.add_argument('--save_path', type=str, default='./experiments/cfr_mix')
    
    # Network and sample parameters
    parser.add_argument('--network_dim', type=int, default=32,
                        help='Dimension of the network')
    parser.add_argument('--sample_number', type=int, default=20,
                        help='Number of samples')
    parser.add_argument('--action_number', type=int, default=1000,
                        help='Number of actions')
    parser.add_argument('--train_epoch', type=int, default=200,
                        help='Number of training epochs')

    # Batch sizes
    parser.add_argument('--attacker_regret_batch_size', type=int, default=32,
                        help='Batch size for attacker regret')
    parser.add_argument('--defender_regret_batch_size', type=int, default=512,
                        help='Batch size for defender regret')
    parser.add_argument('--defender_strategy_batch_size', type=int, default=32,
                        help='Batch size for defender strategy')

    # Learning rates
    parser.add_argument('--attacker_regret_lr', type=float, default=0.0015,
                        help='Learning rate for attacker regret')
    parser.add_argument('--defender_regret_lr', type=float, default=0.0015,
                        help='Learning rate for defender regret')
    parser.add_argument('--defender_strategy_lr', type=float, default=0.0015,
                        help='Learning rate for defender strategy')

    # Iteration
    parser.add_argument('--iteration', type=int, default=10000,
                        help='Number of iterations')

    return parser.parse_args()