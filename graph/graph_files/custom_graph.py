from graph.grid_graph import GridGraph
from graph.any_graph import AnyGraph

def CustomGraph(graph_id, time_horizon=None):
    if graph_id == 0:
        # 7 * 7, caught_prob 1
        column = 7; row = 7
        if time_horizon is None:
            time_horizon = 6
        defender_init = [4, 22, 28, 46]
        attacker_init = [25]
        exits = [1, 7, 43, 49]

        graph = GridGraph(defender_init, attacker_init, exits, time_horizon, row, column)
    elif graph_id == 1:
        # 7 * 7, caught_prob 0.5
        column = 7; row = 7
        if time_horizon is None:
            time_horizon = 6
        defender_init = [9, 28, 44, 46]
        attacker_init = [25]
        exits = [4, 22, 43, 49]

        graph = GridGraph(defender_init, attacker_init, exits, time_horizon, row, column)
    elif graph_id == 2:
        # 5 * 5, caught_prob 1
        column = 5; row = 5
        if time_horizon is None:
            time_horizon = 4
        defender_init = [3, 11, 15, 23]
        attacker_init = [13]
        exits = [1, 5, 21, 25]

        graph = GridGraph(defender_init, attacker_init, exits, time_horizon, row, column)
    elif graph_id == 3:
        # 5 * 5, caught_prob 0.6
        column = 5; row = 5
        if time_horizon is None:
            time_horizon = 4
        defender_init = [7, 15, 22, 23]
        attacker_init = [13]
        exits = [3, 11, 21, 25]

        graph = GridGraph(defender_init, attacker_init, exits, time_horizon, row, column)

    elif graph_id == 4:
        sg_map_path = f"/home/shuxin_zhuang/workspace/GraphChase/graph/graph_files/sg.gpickle"
        defender_init = [120, 60, 8, 11]
        attacker_init = [340]
        exits = [10,201,200,213,290,320,157,159,115,50]
        if time_horizon is None:
            time_horizon = 15

        graph = AnyGraph(defender_init, attacker_init, exits, time_horizon, sg_map_path)

    elif graph_id == 5:
        column = 15; row = 15
        if time_horizon is None:
            time_horizon = 15
        defender_init = [53, 117, 173, 109]
        attacker_init = [113]
        exits = [61,151,217,223,180,75,12,6,225,31]

        graph = GridGraph(defender_init, attacker_init, exits, time_horizon, row, column)

    elif graph_id == 6:
        manhattan_map_path = f"/home/shuxin_zhuang/workspace/GraphChase/graph/graph_files/manhattan.gpickle"
        defender_init = [66, 408, 124,185, 328, 467]
        attacker_init = [48]
        exits = [27, 72, 111, 138, 309, 239, 244, 551, 429, 356]
        if time_horizon is None:
            time_horizon = 30
        graph = AnyGraph(defender_init, attacker_init, exits, time_horizon, manhattan_map_path)
     
    return graph