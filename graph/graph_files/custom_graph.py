from graph.grid_graph import GridGraph

def CustomGraph(graph_id):
    if graph_id == 0:
        column = 7; row = 7
        time_horizon = 6
        defender_init = [4, 22, 28, 46]
        attacker_init = [25]
        exits = [1, 7, 43, 49]

        graph = GridGraph(defender_init, attacker_init, exits, time_horizon, row, column)

    return graph