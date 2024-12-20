from abc import ABC, abstractmethod
import networkx as nx

class BaseGraph(ABC):
    def __init__(self, 
                 defender_position:list,
                 attacker_position:list,
                 exit_position:list,
                 time_horizon:int):
        """
        Initializes an instance of the BaseGraph class. This constructor takes the following parameters:

        - defender_position: A list representing the initial positions of the defender(s).
        - attacker_position: A list representing the initial positions of the attacker(s).
        - exit_position: A list representing the positions of exits.
        - time_horizon: An integer representing the time limit for the game or simulation.

        Upon initialization, the following attributes are set:
        - self.graph: The graph structure, initially set to None. It will be generated by build_graph() method.
        - self.defender_init: Stores the initial positions of the defenders.
        - self.attacker_init: Stores the initial positions of the attackers.
        - self.exits: Stores the positions of the exits.
        - self.time_horizon: Defines the time limit for the game.
        - self.num_defender: The number of defenders.
        - self.intial_locations: Stores all the postions of defenders, attackers and exits
        """        
        self.graph: nx.graph = None
        self.defender_init = defender_position
        self.attacker_init = attacker_position
        self.exits = exit_position
        self.initial_locations = [self.attacker_init] + [self.defender_init] + [self.exits]

        self.time_horizon = time_horizon
        self.num_defender = len(self.defender_init)       

    @abstractmethod
    def build_graph(self):
        """
        Abstract method that must be implemented by subclasses.
        This method should build the graph structure.
        """
        pass

    @abstractmethod
    def build_changestate_legal_action(self):
        """
        Abstract method that must be implemented by subclasses.
        This method should build the graph structure.
        """
        pass

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