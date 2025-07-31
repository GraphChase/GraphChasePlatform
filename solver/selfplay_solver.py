import numpy as np
import torch
import time
import os
import copy


class SelfPlay(object):
    def __init__(self, 
                 env, 
                 args, 
                 evader_runner,
                 pursuer_runner,):
        
        self._num_players = 2
        self.env = env
        self.args = args

        self.evader_runner = copy.deepcopy(evader_runner)
        self.pursuer_runner = copy.deepcopy(pursuer_runner)
        
        # Store the latest trained policies
        self.latest_evader = None
        self.latest_pursuer = None
        
        # Track which player to train next (0: evader, 1: pursuer)
        self.current_training_player = 0

        self.save_path = os.path.join(args.save_path, "selfplay_model")

    def _evaluate(self, evader, defender):
        """Evaluate the performance between evader and defender"""
        total_rewards = np.zeros((self._num_players,))

        for _ in range(self.args.eval_episodes):

            if self.env.nextstate_as_action:
                _, evader_actions = evader.get_action()
                evader_actions = evader_actions[1:]
            else:
                evader_actions, _ = evader.get_action()

            terminated = False
            observation, info = self.env.reset()
            t = 0

            while not terminated:
                evader_act = evader_actions[t]
                with torch.no_grad():
                    env_actions = defender.get_env_actions(observation, t)
                actions = np.array(env_actions)

                if self.env.nextstate_as_action:
                    for i in range(1, len(observation)):
                        actions[i-1] = self.env.graph.change_state[observation[i] - 1][actions[i-1]]               

                actions = np.insert(actions, 0, evader_act)
                observation, reward, terminated, truncated, info = self.env.step(actions)
                t += 1

                if terminated or truncated:
                    total_rewards += np.array(reward)
        
        return total_rewards / self.args.eval_episodes

    def _get_latest_runner(self, player_id):
        """Get a copy of the latest trained runner for further training"""
        if player_id == 0:
            if self.latest_evader is not None:
                return copy.deepcopy(self.latest_evader)
            else:
                return copy.deepcopy(self.evader_runner)
        if player_id == 1:
            if self.latest_pursuer is not None:
                return copy.deepcopy(self.latest_pursuer)
            else:
                return copy.deepcopy(self.pursuer_runner)

    def solve(self):
        """Main self-play training loop"""
        self.init()
        
        time_list = []
        performance_list = []
        start_time = time.time()
        
        for i in range(self.args.num_psro_iteration):
            print(f"Self-play iteration {i+1}/{self.args.num_psro_iteration}")
            
            # Perform one round of self-play training
            performance = self.selfplay_iteration()
            performance_list.append(performance)
            
            # Save the latest model
            if self.current_training_player == 1:  # Just trained pursuer
                self.latest_pursuer.save(self.save_path + "_pursuer", i+1)
            else:  # Just trained evader
                # Save evader if it has a save method, otherwise skip
                if hasattr(self.latest_evader, 'save'):
                    self.latest_evader.save(self.save_path + "_evader", i+1)
            
            # Alternate training player for next iteration
            self.current_training_player = 1 - self.current_training_player
            
            time_list.append(time.time() - start_time)
            print(f"Iteration {i+1} done, performance: {performance}, total time: {time_list[-1]:.2f}s")
            
        print(f"Self-play training completed! Total runtime: {time_list[-1]:.2f}s")
        return performance_list

    def selfplay_iteration(self):
        """Perform one iteration of self-play training"""
        
        if self.current_training_player == 0:
            # Train evader against latest pursuer
            print("Training evader against latest pursuer...")
            start_time = time.time()
            
            # Get latest evader runner for training (not original)
            training_evader = self._get_latest_runner(0)
            
            # Train evader against the latest pursuer (fixed opponent)
            rewards = training_evader.train([self.latest_pursuer], np.array([1.0]), self.args.train_evader_number)
            
            # Update latest evader
            self.latest_evader = training_evader
            
            # Evaluate performance
            performance = self._evaluate(self.latest_evader, self.latest_pursuer)
            
            print(f"Evader training completed in {time.time() - start_time:.2f}s")
            print(f"Evader reward: {performance[0]:.4f}, Pursuer reward: {performance[1]:.4f}")
            
            return performance[0]  # Return evader's performance
            
        else:
            # Train pursuer against latest evader
            print("Training pursuer against latest evader...")
            start_time = time.time()
            
            # Get latest pursuer runner for training (not original)
            training_pursuer = self._get_latest_runner(1)
            
            # Train pursuer against the latest evader (fixed opponent)
            train_infos = training_pursuer.train([self.latest_evader], np.array([1.0]))
            
            # Update latest pursuer
            self.latest_pursuer = training_pursuer
            
            # Evaluate performance
            performance = self._evaluate(self.latest_evader, self.latest_pursuer)
            
            print(f"Pursuer training completed in {time.time() - start_time:.2f}s")
            print(f"Evader reward: {performance[0]:.4f}, Pursuer reward: {performance[1]:.4f}")
            
            return performance[1]  # Return pursuer's performance

    def init(self):
        """Initialize self-play with initial policies"""
        print("Initializing self-play...")
        
        # Set original policies as the initial latest policies
        self.latest_evader = copy.deepcopy(self.evader_runner)
        self.latest_pursuer = copy.deepcopy(self.pursuer_runner)
        
        # Evaluate initial performance
        initial_performance = self._evaluate(self.latest_evader, self.latest_pursuer)
        print(f"Initial performance - Evader: {initial_performance[0]:.4f}, Pursuer: {initial_performance[1]:.4f}")
        
        # Save initial models
        self.latest_pursuer.save(self.save_path + "_pursuer", 0)
        if hasattr(self.latest_evader, 'save'):
            self.latest_evader.save(self.save_path + "_evader", 0)
