
from env.base_env import BaseGame
from graph.base_graph import BaseGraph
from gymnasium import spaces
import numpy as np
import pygame
from typing import Literal


class AnyGraphEnv(BaseGame):
    def __init__(self,
                 Graph: BaseGraph, 
                 return_reward_mode: Literal["both", "defender", "evader"] = "both",
                 return_legal_action: bool  = False,
                 nextstate_as_action = True,
                 render_mode = None,
                ) -> None:
        super().__init__(Graph, render_mode)
        """
        Initialize a GridEnv instance, which is a specialized environment for grid-based games.

        Parameters:
        - Graph (BaseGraph): The graph structure representing the game environment. It should be an instance of BaseGraph or its subclasses.
        
        - nextstate_as_action (bool, optional): If True, the next state is treated as an action. Default is False.
        
        - return_reward_mode (Literal["both", "defender", "evader"], optional): Specifies which player's rewards should be returned.
          - "both": Return rewards for both defender and evader: [evader_reward, defender_reward].
          - "defender": Only return defenders' rewards.
          - "evader": Only return evader's rewards.
          Default is "both".
        
        - return_legal_action (bool, optional): If True, the environment will return legal actions along with other information. Default is False.
        
        - render_mode (str, optional): Specifies the rendering mode for the environment.

        """        

        # Validate return_reward_mode
        if return_reward_mode not in ["both", "defender", "evader"]:
            raise ValueError("return_reward_mode must be 'both', 'defender', or 'evader'")
        self.return_reward_mode = return_reward_mode
        self.nextstate_as_action = nextstate_as_action
        self.return_legal_action = return_legal_action

        self.time_steps = 0

        self.observation_space = spaces.MultiDiscrete([Graph.num_nodes for _ in range(self.evader_num + self.defender_num)], 
                                                       start = [1 for _ in range(self.evader_num + self.defender_num)])
        self.action_space = spaces.MultiDiscrete([5 for _ in range(self.evader_num + self.defender_num)], seed=7)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.window_size = 512
          

    def step(self, actions: np.ndarray,) -> tuple[np.ndarray, list, bool, bool, dict]:
        """
        Parameters:
        evader_actions (np.ndarray): The actions of the evaders 0-4.
        defender_actions (np.ndarray): The actions of the defenders.
        """
        current_states = self._get_obs()

        next_states = actions

        actions_feasible = np.array([True if self.nx_graph.has_edge(current_states[idx], next_state) else  False for idx, next_state in enumerate(next_states)])
        next_states = np.where(actions_feasible, next_states, current_states)
        self.time_steps += 1

        self._evader_current_locations = next_states[:self.evader_num]
        self._defender_current_locations = next_states[self.evader_num:]

        observation = self._get_obs()
        info  = self._get_info()

        terminated = self._is_terminal()
        reward = self._get_rewards(terminated)

        if terminated:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["defender_r"] = reward[1]
            info["episode"]["l"] = len(info["evader_history"])
            info["episode"]["evader_captured"] = True if info["evader_history"][-1] in info["defender_history"][-1] else False

        # if self.render_mode == "human":
        #     self._render_frame()

        if self.return_reward_mode == "defender":
            reward = reward[1]
        elif self.return_reward_mode == "evader":
            reward = reward[0]

        return observation, reward, terminated, False, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._evader_current_locations = self._evader_initial_locations
        self._defender_current_locations = self._defender_initial_locations
        self.time_steps = 0
        self.info = {
            "evader_history": [],
            "defender_history": [],
        }        

        # if self.render_mode == "human":
        #     self._render_frame()

        observation = self._get_obs()
        info  = self._get_info()

        return observation, info


    def _get_obs(self,):
        return np.concatenate((self._evader_current_locations, self._defender_current_locations))
    
    def _get_info(self, ):
        self.info["evader_history"].append(self._evader_current_locations.item())
        self.info["defender_history"].append(self._defender_current_locations.tolist())

        if self.return_legal_action:
            defender_legal_act, attacker_legal_act = self.get_legal_action(False, False, self.info["evader_history"], self.info["defender_history"][-1])
            self.info["defender_legal_actions"] = defender_legal_act
            self.info["evader_legal_actions"] = attacker_legal_act

        return self.info

    def get_legal_action(self, done=False, combinational=False, attacker_his=None, defender_position=None):
        if done:
            attacker_legal_act=[0]
            if combinational:
                defender_legal_act=[(0,)*self.defender_num] #[(0, 0, ..)]
            else:
                defender_legal_act=[[0]]*self.defender_num #[[0],[0],..]
        else:
            assert attacker_his is not None and defender_position is not None, "attacker_his and defender_pos cannot be empty"

            attacker_legal_act = self.graph.adjlist[attacker_his[-1]]
            if combinational:
                assert False, "legal_action combinational is True have not been deal"
            else:
                defender_legal_act=[]
                for i in range(self.defender_num):
                    defender_legal_act.append(self.graph.adjlist[defender_position[i]])
        return defender_legal_act, attacker_legal_act       

    def _is_terminal(self, attacker_his=None, defender_position=None):
        if attacker_his is None and defender_position is None:
            if self.time_steps == self.time_horizon:
                return True
            
            teminal_flag = np.ones((self.evader_num,)) # 1 means not done, 0 means done
            for idx, attacker_location in enumerate(self._evader_current_locations):
                if attacker_location in self._defender_current_locations or attacker_location in self._exit_locations:
                    teminal_flag[idx] = 0

            if np.sum(teminal_flag) == 0:
                return True
            
            return False
        else:
            # assert len(defender_his)==len(attacker_his), "defender and attacker history must have the same lenght"
            assert len(attacker_his) <= self.time_horizon+1, "attacker history's length should be smaller than max time_horizon"
            if (len(attacker_his)==self.time_horizon+1) or \
                (attacker_his[-1] in defender_position) or \
                (attacker_his[-1] in self._exit_locations.tolist()):
                return True
            else:
                return False            

    def _get_rewards(self, terminal, attacker_his=None, defender_position=None):
        if attacker_his is None and defender_position is None:
            if terminal:
                # TODO: multi-evaders

                if self._evader_current_locations in self._defender_current_locations:
                    return [-1., 1.]
                if self._evader_current_locations in self._exit_locations:
                    return [1., -1.]
                else:
                    return [-1., 1.]
            return [0., 0.]  
        else:
            if terminal:
                if attacker_his[-1] in defender_position:
                    return [-1., 1.]
                if attacker_his[-1] in self._exit_locations:
                    return [1., -1.]
                else:
                    return [-1., 1.]
            return [0., 0.]               

    def render(self,):
        assert False, "To be done"
        if self.render_mode == "rgb_array":
            return self._render_frame()     

    def close(self,):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self,):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))       

        # First we draw the grid
        for j in range(self._rows-1):
            for i in range(self._colums-1):
                pygame.draw.line(canvas, 
                                 (0, 0, 0), 
                                 self.point_positions[j*self._colums+i], 
                                 self.point_positions[j*self._colums+i+1], 
                                 1)
                pygame.draw.line(canvas, 
                                 (0, 0, 0), 
                                 self.point_positions[j*self._colums+i], 
                                 self.point_positions[(j+1)*self._colums+i], 
                                 1)                
        for i in range(self._colums-1):
            pygame.draw.line(canvas, 
                                (0, 0, 0), 
                                self.point_positions[(self._colums-1)*self._colums+i], 
                                self.point_positions[(self._colums-1)*self._colums+i+1], 
                                1)
        for j in range(self._rows-1):
                pygame.draw.line(canvas, 
                                 (0, 0, 0), 
                                 self.point_positions[(j+1)*self._colums-1], 
                                 self.point_positions[(j+2)*self._colums-1], 
                                 1)  

        # Now we draw the exits position
        for exit_pos in self._exit_locations:
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (self.point_positions[exit_pos-1][0], self.point_positions[exit_pos-1][1]),
                8,
            )         

        # Now we draw the defenders position
        for defender_pos in self._defender_current_locations:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (self.point_positions[defender_pos-1][0], self.point_positions[defender_pos-1][1]),
                6,
            )    

        # Now we draw the evaders position
        for evader_pos in self._evader_current_locations:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (self.point_positions[evader_pos-1][0], self.point_positions[evader_pos-1][1]),
                4,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )                   
        
    def condition_to_str(self):
        return f"T{self.time_horizon}_loc{self._initial_location[:-1]}_exit{self._exit_locations.tolist()}"
    
    # def next_pos_2_actions(self, next_position:tuple, current_position:tuple) -> list:
    #     """Used for MCTS algorithm, since the method only use postion instead of actions, but environment needs actions to step
        
    #     """
    #     env_actions = []
    #     for next_pos, curr_pos in zip(next_position, current_position):
    #         if curr_pos - next_pos > 1:   # up
    #             env_actions.append(1)
    #         elif curr_pos - next_pos < -1:    # down
    #             env_actions.append(2)
    #         elif curr_pos - next_pos == 1:    # left
    #             env_actions.append(3)
    #         elif curr_pos - next_pos == -1:   # right
    #             env_actions.append(4)
    #         elif curr_pos == next_pos:
    #             env_actions.append(0)

    #     return env_actions    