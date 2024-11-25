import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Graph Chase Game Parameters")

    # General settings
    parser.add_argument("--graph_id", type=int, default=0, help="Graph ID")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--device_id", type=int, default=2, help="CUDA device ID")
    parser.add_argument("--no_cuda", action="store_false", dest="use_cuda", 
                        help="Do not use CUDA even if available")    
    parser.add_argument("--no_tensorboard", action="store_false", dest="use_tensorboard",
                        help="Use TensorBoard for logging")
    parser.add_argument("--no_save_model", action="store_false", dest="save_model",
                        help="Save the trained model")
    parser.add_argument('--save_path', type=str, default='./experiments/nsgzero')
    parser.add_argument("--graph_type", type=str, choices=["GridGraph", "AnyGraph"], default="GridGraph", help="The type of graph structure")

    # Training parameters
    parser.add_argument("--max_episodes", type=int, default=100000, help="Maximum number of episodes")
    parser.add_argument("--embedding_dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--train_every", type=int, default=16, help="Train every n episodes")
    parser.add_argument("--train_from", type=int, default=128, help="Start training after n episodes")
    parser.add_argument("--test_every", type=int, default=500, help="Test every n episodes")
    parser.add_argument("--test_nepisodes", type=int, default=30, help="Number of episodes for testing")
    parser.add_argument("--save_every", type=int, default=500, help="Save model every n episodes")
    parser.add_argument("--log_every", type=int, default=500, help="Log every n episodes")

    # MCTS parameters
    parser.add_argument("--num_sims", type=int, default=15, help="Number of MCTS simulations (at least 2)")
    parser.add_argument("--bias", type=float, default=0.5, help="MCTS bias")
    parser.add_argument("--cpuct", type=float, default=0.3, help="MCTS exploration constant (larger than 0)")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature for action selection")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")

    # Other parameters
    parser.add_argument("--att_type", type=str, default="nfsp", help="Attention type")
    parser.add_argument("--ban_capacity", type=int, default=500, help="Ban capacity")
    parser.add_argument("--cache_capacity", type=int, default=20, help="Cache capacity")
    parser.add_argument("--br_rate", type=float, default=0.2, help="Best response rate")

    return parser.parse_args()
