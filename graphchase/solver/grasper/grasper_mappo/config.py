import os
import argparse
import os.path as osp


def get_config():

    parser = argparse.ArgumentParser(description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default='mappo', choices=["rmappo", "mappo"])
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    # parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    # parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int, default=1, help="Number of torch threads for training")
    # parser.add_argument("--n_rollout_threads", type=int, default=20, help="Number of parallel environments for training rollouts")
    # parser.add_argument("--n_eval_rollout_threads", type=int, default=20, help="Number of parallel environments for evaluating rollouts")
    # parser.add_argument("--n_render_rollout_threads", type=int, default=1, help="Number of parallel environments for rendering rollouts")
    # parser.add_argument("--num_env_steps", type=int, default=10e6, help='Number of environment steps to train (default: 10e6)')
    # parser.add_argument("--user_name", type=str, default='marl',help="[for wandb usage], to specify user's name for simply collecting training data.")
    # parser.add_argument("--use_wandb", action='store_false', default=False, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    # parser.add_argument("--env_name", type=str, default='MyEnv', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true', default=False, help="Whether to use global state or concatenated obs")

    # parser.add_argument("--episodes", type=int, default=10, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false', default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false', default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true', default=False, help="Whether to use stacked_frames")
    # parser.add_argument("--hidden_size", type=int, default=64, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=1, help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false', default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_true', default=False, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_true', default=False, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_true', default=False, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")

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
    parser.add_argument("--ppo_epoch", type=int, default=1, help='number of ppo epochs (default: 15)')
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


def get_mtl_model_results_dir(game, action_type, args, iteration):
    base_dir = args.pretrain_save_path
    graph_type = getattr(game._graph, "type", "Custom_Graph")
    edge_probability = getattr(game._graph, "edge_probability", getattr(args, "edge_probability", 1.0))
    differ_size = bool(getattr(args, "differ_size", False))
    prob_of_obs_attacker = getattr(args, "prob_of_obs_attacker", 1.0)
    if graph_type == "Grid_Graph":
        if differ_size:
            save_path = os.path.join(
                base_dir,
                "graph_ds_probability_{}_poa_{}".format(edge_probability, prob_of_obs_attacker),
                "pretrain_model",
            )
        else:
            save_path = os.path.join(
                base_dir,
                "graph_{}_probability_{}_poa_{}".format(args.column*args.row, edge_probability, prob_of_obs_attacker),
                "pretrain_model",
            )
    elif graph_type == "Map_Graph":
        save_path = os.path.join(base_dir, f"map_graph_probability_{edge_probability}", "pretrain_model")
    elif graph_type == "SY_Graph":
        save_path = os.path.join(base_dir, "sy_graph", "pretrain_model")
    elif graph_type == "SF_Graph":
        save_path = os.path.join(base_dir, f"sf_graph_{game._graph.total_node_number}", "pretrain_model")
    elif graph_type == "SW_Graph":
        save_path = os.path.join(
            base_dir,
            f"sw_graph_{game._graph.total_node_number}_"
            f"k{getattr(args, 'small_world_k', 0)}_prob{getattr(args, 'edge_probability', edge_probability)}",
            "pretrain_model",
        )
    elif graph_type == "ER_Graph":
        save_path = os.path.join(
            base_dir,
            f"er_graph_{game._graph.total_node_number}_"
            f"prob{getattr(args, 'edge_probability', edge_probability)}",
            "pretrain_model",
        )
    else:
        save_path = os.path.join(base_dir, "custom_graph", "pretrain_model")
    if args.use_end_to_end:
        save_path += "/use_e2e"
    else:
        save_path += "/not_e2e"
    if not osp.exists(save_path):
        os.makedirs(save_path)

    location = "num_gtst{}_{}_{}_{}_iter{}_bsize{}_node_feat{}_gnn{}_{}_{}_dnum{}_{}_enum{}_{}_T{}_{}_mep{}{}".format(
        args.num_games,
        args.num_task,
        args.num_sample,
        args.train_num_per_ite,
        iteration,
        args.batch_size,
        args.node_feat_dim,
        args.gnn_num_layer,
        args.gnn_hidden_dim,
        args.gnn_output_dim,
        args.min_num_defender,
        args.max_num_defender,
        args.min_num_exit,
        args.max_num_exit,
        args.min_time_horizon,
        args.max_time_horizon,
        args.min_attacker_pth_len,
        "" if args.update_every_n_episodes > 0 else "_une{}".format(args.update_every_n_episodes),
    )
    # location = "num_gtst{}_{}_{}_{}_iter{}_bsize{}_node_feat{}_dnum{}_{}_enum{}_{}_T{}_{}_mep{}" \
    #     .format(args.num_games, args.num_task, args.num_sample, args.train_num_per_ite, iteration, args.batch_size,
    #             args.node_feat_dim, args.min_num_defender, args.max_num_defender, args.min_num_exit, args.max_num_exit,
    #             args.min_time_horizon, args.max_time_horizon, args.min_attacker_pth_len)

    if game.use_past_history:
        location += "_use_history"

    if game._graph.use_node_embedding:
        if game._graph.node_embedding_method == "line":
            location += "_use_{}_order_{}".format(game._graph.node_embedding_method, game._graph.embedding_order)
        else:
            location += "_use_{}_order_{}_information_type_{}_normalized_{}_proximity_{}".format(
            game._graph.node_embedding_method, game._graph.embedding_order, game._graph.node_information_type,
                    game._graph.node_information_normalize, game._graph.similarity)

    if args.use_end_to_end:
        if args.use_emb_layer:
            location += "_use_el"
        if args.use_augmentation:
            location += "_aug"
        if args.use_cache:
            location += "_cache"
        if args.load_game_pool_file:
            location += f"_gp{args.pool_size}"
    else:
        if args.use_emb_layer:
            location += "_use_el"
            if args.use_augmentation:
                location += "_aug"
            if args.load_graph_emb_model:
                location += "_load_gem"
            if args.use_cache:
                location += "_cache"
            if args.load_game_pool_file:
                location += f"_gp{args.pool_size}"
        if args.use_node_emb:
            location += "_use_ne"
            if args.use_augmentation:
                location += "_aug"
            if args.load_graph_emb_model:
                location += "_load_gem"
            if args.use_cache:
                location += "_cache"
            if args.load_game_pool_file:
                location += f"_gp{args.pool_size}"

    location += "_pi{}".format(int(args.perfect_info))

    if args.use_act_supervisor:
        location += "_as1_{}_{}_{}".format(args.act_sup_coef_max, args.act_sup_coef_min, args.act_sup_coef_decay)

    return osp.join(save_path, location)


def get_train_utility_results_dir(game, action_type, args):
    base_dir = args.pretrain_save_path
    graph_type = getattr(game._graph, "type", "Custom_Graph")
    edge_probability = getattr(game._graph, "edge_probability", getattr(args, "edge_probability", 1.0))
    differ_size = bool(getattr(args, "differ_size", False))
    prob_of_obs_attacker = getattr(args, "prob_of_obs_attacker", 1.0)
    if graph_type == "Grid_Graph":
        if differ_size:
            save_path = os.path.join(
                base_dir,
                "graph_ds_probability_{}_poa_{}".format(edge_probability, prob_of_obs_attacker),
                "utility_record",
            )
        else:
            save_path = os.path.join(
                base_dir,
                "graph_{}_probability_{}_poa_{}".format(game._graph.total_node_number, edge_probability, prob_of_obs_attacker),
                "utility_record",
            )
    elif graph_type == "Map_Graph":
        save_path = os.path.join(base_dir, f"map_graph_probability_{edge_probability}", "utility_record")
    elif graph_type == "SY_Graph":
        save_path = os.path.join(base_dir, "sy_graph", "utility_record")
    elif graph_type == "SF_Graph":
        save_path = os.path.join(base_dir, f"sf_graph_{game._graph.total_node_number}", "utility_record")
    elif graph_type == "SW_Graph":
        save_path = os.path.join(
            base_dir,
            f"sw_graph_{game._graph.total_node_number}_"
            f"k{getattr(args, 'small_world_k', 0)}_prob{getattr(args, 'edge_probability', edge_probability)}",
            "utility_record",
        )
    elif graph_type == "ER_Graph":
        save_path = os.path.join(
            base_dir,
            f"er_graph_{game._graph.total_node_number}_"
            f"prob{getattr(args, 'edge_probability', edge_probability)}",
            "utility_record",
        )
    else:
        save_path = os.path.join(base_dir, "custom_graph", "utility_record")
    if args.use_end_to_end:
        save_path += "/use_e2e"
    else:
        save_path += "/not_e2e"
    if not osp.exists(save_path):
        os.makedirs(save_path)

    location = "seed{}_num_gtst{}_{}_{}_{}_bsize{}_node_feat{}_gnn{}_{}_{}_dnum{}_{}_enum{}_{}_T{}_{}_mep{}{}".format(
        args.seed,
        args.num_games,
        args.num_task,
        args.num_sample,
        args.train_num_per_ite,
        args.batch_size,
        args.node_feat_dim,
        args.gnn_num_layer,
        args.gnn_hidden_dim,
        args.gnn_output_dim,
        args.min_num_defender,
        args.max_num_defender,
        args.min_num_exit,
        args.max_num_exit,
        args.min_time_horizon,
        args.max_time_horizon,
        args.min_attacker_pth_len,
        "" if args.update_every_n_episodes > 0 else "_une{}".format(args.update_every_n_episodes),
    )
    # location = "seed{}_num_gtst{}_{}_{}_{}_bsize{}_node_feat{}_dnum{}_{}_enum{}_{}_T{}_{}_mep{}" \
    #     .format(args.seed, args.num_games, args.num_task, args.num_sample,
    #             args.train_num_per_ite, args.batch_size, args.node_feat_dim,
    #             args.min_num_defender, args.max_num_defender, args.min_num_exit,
    #             args.max_num_exit, args.min_time_horizon, args.max_time_horizon, args.min_attacker_pth_len)

    if game.use_past_history:
        location += "_use_history"

    if game._graph.use_node_embedding:
        if game._graph.node_embedding_method == "line":
            location += "_use_{}_order_{}".format(game._graph.node_embedding_method, game._graph.embedding_order)
        else:
            location += "_use_{}_order_{}_information_type_{}_normalized_{}_proximity_{}" \
                .format(game._graph.node_embedding_method, game._graph.embedding_order,
                        game._graph.node_information_type,
                        game._graph.node_information_normalize, game._graph.similarity)

    if args.use_end_to_end:
        if args.use_emb_layer:
            location += "_use_el"
        if args.use_augmentation:
            location += "_aug"
        if args.use_cache:
            location += "_cache"
        if args.load_game_pool_file:
            location += f"_gp{args.pool_size}"
    else:
        if args.use_emb_layer:
            location += "_use_el"
            if args.use_augmentation:
                location += "_aug"
            if args.load_graph_emb_model:
                location += "_load_gem"
            if args.use_cache:
                location += "_cache"
            if args.load_game_pool_file:
                location += f"_gp{args.pool_size}"
        if args.use_node_emb:
            location += "_use_ne"
            if args.use_augmentation:
                location += "_aug"
            if args.load_graph_emb_model:
                location += "_load_gem"
            if args.use_cache:
                location += "_cache"
            if args.load_game_pool_file:
                location += f"_gp{args.pool_size}"

    location += "_pi{}".format(int(args.perfect_info))

    if args.use_act_supervisor:
        location += "_as1_{}_{}_{}".format(args.act_sup_coef_max, args.act_sup_coef_min, args.act_sup_coef_decay)

    location += "_reward.pik"

    return osp.join(save_path, location)


def get_runs_dir(game, action_type, args):
    graph_type = getattr(game._graph, "type", "Custom_Graph")
    edge_probability = getattr(game._graph, "edge_probability", getattr(args, "edge_probability", 1.0))
    differ_size = bool(getattr(args, "differ_size", False))
    prob_of_obs_attacker = getattr(args, "prob_of_obs_attacker", 1.0)
    if graph_type == "Grid_Graph":
        if differ_size:
            location = "grasper_mappo_graph_ds_probability_{}_poa_{}".format(edge_probability, prob_of_obs_attacker)
        else:
            location = "grasper_mappo_graph_{}_probability_{}_poa_{}".format(
                game._graph.total_node_number, edge_probability, prob_of_obs_attacker
            )
    elif graph_type == "Map_Graph":
        location = f"grasper_mappo_map_graph_probability_{edge_probability}"
    elif graph_type == "SY_Graph":
        location = "grasper_mappo_sy_graph"
    elif graph_type == "SF_Graph":
        location = f"grasper_mappo_sf_graph_{game._graph.total_node_number}"
    elif graph_type == "SW_Graph":
        location = (
            f"grasper_mappo_sw_graph_{game._graph.total_node_number}_k{getattr(args, 'small_world_k', 0)}_"
            f"prob{getattr(args, 'edge_probability', edge_probability)}"
        )
    elif graph_type == "ER_Graph":
        location = f"grasper_mappo_er_graph_{game._graph.total_node_number}_prob{getattr(args, 'edge_probability', edge_probability)}"
    else:
        location = "grasper_mappo_custom_graph"
    location += "_seed{}_num_gtst{}_{}_{}_{}_bsize{}_node_feat{}_gnn{}_{}_{}_dnum{}_{}_enum{}_{}_T{}_{}_mep{}{}" \
        .format(args.seed, args.num_games, args.num_task, args.num_sample, args.train_num_per_ite, args.batch_size, args.node_feat_dim,
                args.gnn_num_layer, args.gnn_hidden_dim, args.gnn_output_dim, args.min_num_defender, args.max_num_defender,
                args.min_num_exit, args.max_num_exit, args.min_time_horizon, args.max_time_horizon, args.min_attacker_pth_len,
                '' if args.update_every_n_episodes > 0 else '_une{}'.format(args.update_every_n_episodes))
    # location += "_seed{}_num_gtst{}_{}_{}_{}_bsize{}_node_feat{}_dnum{}_{}_enum{}_{}_T{}_{}_mep{}" \
    #     .format(args.seed, args.num_games, args.num_task, args.num_sample, args.train_num_per_ite, args.batch_size,
    #             args.node_feat_dim, args.min_num_defender, args.max_num_defender, args.min_num_exit, args.max_num_exit,
    #             args.min_time_horizon, args.max_time_horizon, args.min_attacker_pth_len)
    if game.use_past_history:
        location += "_use_history"

    if game._graph.use_node_embedding:
        if game._graph.node_embedding_method == "line":
            location += "_use_{}_order_{}".format(game._graph.node_embedding_method, game._graph.embedding_order)
        else:
            location += "_use_{}_order_{}_information_type_{}_normalized_{}_proximity_{}".format(
            game._graph.node_embedding_method, game._graph.embedding_order, game._graph.node_information_type,
                    game._graph.node_information_normalize, game._graph.similarity)

    if args.use_end_to_end:
        location += "_use_e2e"
        if args.use_emb_layer:
            location += "_use_el"
        if args.use_augmentation:
            location += "_aug"
        if args.use_cache:
            location += "_cache"
        if args.load_game_pool_file:
            location += f"_gp{args.pool_size}"
    else:
        location += "_not_e2e"
        if args.use_emb_layer:
            location += "_use_el"
            if args.use_augmentation:
                location += "_aug"
            if args.load_graph_emb_model:
                location += "_load_gem"
            if args.use_cache:
                location += "_cache"
            if args.load_game_pool_file:
                location += f"_gp{args.pool_size}"
        if args.use_node_emb:
            location += "_use_ne"
            if args.use_augmentation:
                location += "_aug"
            if args.load_graph_emb_model:
                location += "_load_gem"
            if args.use_cache:
                location += "_cache"
            if args.load_game_pool_file:
                location += f"_gp{args.pool_size}"

    location += "_pi{}".format(int(args.perfect_info))

    if args.use_act_supervisor:
        location += "_as1_{}_{}_{}".format(args.act_sup_coef_max, args.act_sup_coef_min, args.act_sup_coef_decay)

    return location
