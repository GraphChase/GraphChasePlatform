from graph.base_graph import BaseGraph
import networkx as nx
import random
import numpy as np

class GridGraph(BaseGraph):
    def __init__(self, 
                 defender_position:list,
                 attacker_position:list,
                 exit_position:list,
                 time_horizon:int,                 
                 rows:int, 
                 cols:int, 
                 sides_exist_prob:float=1.0,
                 diagonal_exist_prob:float=0.0):
        super().__init__(defender_position, attacker_position, exit_position, time_horizon)

        self.row = rows
        self.column = cols
        self.sides_exist_prob = sides_exist_prob
        self.diagonal_exist_prob = diagonal_exist_prob
        self.size = [self.row, self.column]

        self.build_graph()

    def build_graph(self):
        """
        Constructs a 2D grid graph and modifies it by randomly removing and adding edges 
        based on defined probabilities. 

        After executing this method, the following attributes are set:
        - self.graph: networkx
        - self.num_nodes: is the number of nodes in the graph.
        - self.adjlist: dict, records the adjacency list.
        - self.degree: calculates the maximum possible number of actions.
        """

        g = nx.grid_2d_graph(self.row, self.column)
        g = nx.convert_node_labels_to_integers(g, first_label=1)

        map_adjlist = nx.to_dict_of_lists(g)
        intra_nodes = [i for i in g.nodes() if len(map_adjlist[i]) == 4]

        for node in g.nodes():
            g.add_edge(node, node)
        to_remove = []
        for e in g.edges():
            if random.random() >= self.sides_exist_prob:
                to_remove.append(e)
        g.remove_edges_from(to_remove)

        def other_nodes(node, n=self.column):
            return node-n-1, node-n+1, node+n-1, node+n+1
        add_edges = []
        for node in intra_nodes:
            for other_node in other_nodes(node):
                if random.random() <= self.diagonal_exist_prob:
                    add_edges.append((node, other_node))
        g.add_edges_from(add_edges)

        map_adjlist = nx.to_dict_of_lists(g)
        max_actions = 0
        for node in map_adjlist:
            map_adjlist[node].sort()
            if len(map_adjlist[node]) > max_actions:
                max_actions = len(map_adjlist[node])

        self.num_nodes = len(map_adjlist)
        self.adjlist = map_adjlist        
        self.max_actions = pow(max_actions, self.num_defender)
        self.degree=max_actions
        self.graph = g

    def get_shortest_path(self, length) -> tuple[list, dict]:
        """
        Caculate the valid evader path in the custom graph

        Parameters:
        - length: (lenght + 1) is the max length of the evader path
        """
        path = []
        path_list = {}
        for j in range(len(self.exits)):
            path_temp = []
            if nx.has_path(self.graph, source=self.attacker_init[0], target=self.exits[j]):

                shortest_path = nx.all_shortest_paths(self.graph, source=self.attacker_init[0], target=self.exits[j])

                for p in shortest_path:
                    if len(p) > length + 1 or len(path_temp) >= 200:
                        break
                    else:
                        if list(set(p) & set(self.exits)) == [self.exits[j]]:
                            path_temp.append(p)

            path_list[self.exits[j]] = path_temp
            path += path_temp
        return path, path_list
    
    # return adjacent matrix and node information of each node
    def get_graph_info(self, node_feat_dim, return_adj=False, normalize_adj=True):
        MAX_DEGREES = 30
        dgre = np.array(self.graph.degree())[:,1]
        dgre[dgre > MAX_DEGREES] = MAX_DEGREES
        dgre_one_hot = np.eye(MAX_DEGREES + 1)[dgre]
        feat = np.zeros((self.num_nodes, node_feat_dim))
        for node in self.exits:
            feat[node - 1, 0] = 1
        for node in self.initial_locations[0]:
            feat[node - 1, 1] += 1
        for node in self.initial_locations[1]:
            feat[node - 1, 2] += 1
        feat = np.concatenate((feat, dgre_one_hot), axis=1)
        if return_adj:
            if normalize_adj:
                return feat, norm_adj(np.asarray(nx.to_numpy_array(self.graph)))
            else:
                return feat, np.asarray(nx.to_numpy_array(self.graph))
        else:
            return feat
        
    def get_demonstration(self, obs:np.ndarray, evader_pos:int):
        """
        For grasper_mappo, get demenstrated defender's actions. According current defender's position and evader position
        calculate the shortest path, and get defender's action(up, down, left, right)
        input:
            obs: [evader_pos, defender_pos, time, id] * defender_num
        """
        num_agent = obs.shape[0]
        # pths = [nx.shortest_path(self.graph, source=obs[i][i+1], target=evader_pos) for i in range(num_agent)]
        pths = []
        for i in range(num_agent):
            source = obs[i][i + 1]
            target = evader_pos
            try:
                path = nx.shortest_path(self.graph, source=source, target=target)
            except nx.NetworkXNoPath:
                path = [source]
            pths.append(path)        
        act_probs = [np.zeros(5,) for _ in range(num_agent)]
        for i in range(num_agent):
            if len(pths[i]) == 1:   # stay
                act_probs[i][0] = 1.0
            else:
                curr_node, next_node = pths[i][0], pths[i][1]
                if curr_node - next_node > 1:   # up
                    act_probs[i][1] = 1.0
                elif curr_node - next_node < -1:    # down
                    act_probs[i][2] = 1.0
                elif curr_node - next_node == 1:    # left
                    act_probs[i][3] = 1.0
                elif curr_node - next_node == -1:   # right
                    act_probs[i][4] = 1.0
        return np.array(act_probs)        
        
def norm_adj(adj):
    adj += np.eye(adj.shape[0])
    degr = np.array(adj.sum(1))
    degr = np.diag(np.power(degr, -0.5))
    return degr.dot(adj).dot(degr)