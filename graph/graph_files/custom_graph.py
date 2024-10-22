from graph.grid_graph import GridGraph

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

    return graph