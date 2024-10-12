import os
from agent.grasper.utils import sample_env
import pickle

def get_training_set(args):

    if args.graph_type == 'Grid_Graph':
        save_path = 'data/related_files/game_pool/grid_{}_probability_{}'.format(args.row * args.column, args.edge_probability)
    elif args.graph_type == 'SG_Graph':
        save_path = 'data/related_files/game_pool/sg_graph_probability_{}'.format(args.edge_probability)
    elif args.graph_type == 'SY_Graph':
        save_path = 'data/related_files/game_pool/sy_graph'
    elif args.graph_type == 'SF_Graph':
        save_path = 'data/related_files/game_pool/sf_graph_{}'.format(args.sf_sw_node_num)
    else:
        raise ValueError(f'Unrecognized graph type {args.graph_type}')
    save_path = os.path.join(args.save_path, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, 'game_pool_size{}_dnum{}_enum{}_T{}_{}_mep{}.pik'
                         .format(args.pool_size, args.num_defender, args.num_exit, args.min_time_horizon,
                                 args.max_time_horizon, args.min_evader_pth_len))
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print('Game pool is already generated.')
        pass
    else:
        print('Generate game pool ...')
        game_pool = []
        game_str_list = []
        for game_idx in range(args.pool_size):
            while True:
                game = sample_env(args)
                if game.condition_to_str() not in game_str_list:
                    print('Generate game: ', game_idx)
                    game_pool.append(game)
                    game_str_list.append(game.condition_to_str())
                    break
        pickle.dump({'game_pool': game_pool}, open(file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)