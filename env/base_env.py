import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from copy import deepcopy
import networkx as nx
from graph.base_graph import BaseGraph


class BaseGame(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, 
                 Graph: BaseGraph, 
                 render_mode = None,
                 ) -> None:
        super().__init__()
        """
        Initialize the game environment.

        This method sets up the initial state of the game, including:
        - The graph structure of the game world
        - Time horizon for the game
        - Initial locations of evaders and defenders
        - Number of evaders and defenders
        - Exit locations
        - Whether to use next state as action

        Parameters:
        - Graph (BaseGraph): The graph representing the game world structure
        - render_mode (str, optional): The mode for rendering the game. Defaults to None.
        - nextstate_as_action (bool): Flag to determine if next state should be used as action. Defaults to False.

        The method initializes various attributes of the game environment, including:
        - self.nx_graph: The NetworkX graph object representing the game world
        - self.graph: The BaseGraph object containing game-specific information
        - self.time_horizon: The total time steps for the game
        - self._initial_location: A deep copy of the initial locations for all entities
        - self.evader_num: The number of evaders in the game
        - self.defender_num: The number of defenders in the game
        - self._evader_initial_locations: A numpy array of initial evader locations
        - self._defender_initial_locations: A numpy array of initial defender locations
        - self._exit_locations: A numpy array of exit locations
        - self.nextstate_as_action: Flag indicating whether to use next state as action

        This initialization sets up the fundamental structure and parameters for the game environment.
        """
        # game initial definition
        self.nx_graph = Graph.graph
        self.graph = Graph
        self.time_horizon = Graph.time_horizon
        self._initial_location = deepcopy(Graph.initial_locations)
        self.evader_num = len(self._initial_location[0])
        self.defender_num = len(self._initial_location[1])
        self._evader_initial_locations = deepcopy(np.array(self._initial_location[0]))
        self._defender_initial_locations = deepcopy(np.array(self._initial_location[1]))
        self._exit_locations = deepcopy(np.array(self._initial_location[2]))


# class NSGGridGraphGame(GridGraphGame):
#     def __init__(self, graph, time_horizon, render_mode=None, nextstate_as_action=False):
#         super().__init__(graph, time_horizon, render_mode, nextstate_as_action)
    
#     def step(self, action):
#         """When using parallel env, only return defender's reward.
#             Because NSG series algorithms need legal actions, so add legal actions information to info
#         """

#         observation, reward, terminated, truncated, info = super().step(action)
        
#         return observation, reward[1], terminated, truncated, info         
    
#     def _get_info(self, ):

#         info = super()._get_info()
#         defender_legal_act, attacker_legal_act = self.graph.legal_action(False, info["evader_history"], info["defender_history"][-1])
#         info["defender_legal_actions"] = defender_legal_act
#         info["evader_legal_actions"] = attacker_legal_act

#         return info
    
#     def next_pos_2_actions(self, next_position:tuple, current_position:tuple) -> list:
#         """Used for MCTS algorithm, since the method only use postion instead of actions, but environment needs actions to step
        
#         """
#         env_actions = []
#         for next_pos, curr_pos in zip(next_position, current_position):
#             if curr_pos - next_pos > 1:   # up
#                 env_actions.append(1)
#             elif curr_pos - next_pos < -1:    # down
#                 env_actions.append(2)
#             elif curr_pos - next_pos == 1:    # left
#                 env_actions.append(3)
#             elif curr_pos - next_pos == -1:   # right
#                 env_actions.append(4)
#             elif curr_pos == next_pos:
#                 env_actions.append(0)

#         return env_actions