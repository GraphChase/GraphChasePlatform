from agent.base_runner import BaseRunner
from agent.nsg_nfsp.utils import evaluate, Transition, Sample, decay_expl
import time
import copy
from itertools import product
import numpy as np
from os.path import join
from common_utils import directory_config, store_args

class NsgNfspDefenderRunner(BaseRunner):
    def __init__(self, env, policy, args):
        super().__init__()
        
        self.args = args
        self.save_path = directory_config(self.args.save_path)
        store_args(self.args, self.save_path)

        self.env = env
        self.policy = policy

    def train(self, Attacker):

        if self.args.exact_br:
            pass
            # defender_utility = exact_br.Defender_Utility(Defender)
            # Utilities = [(0, defender_utility)]
            # print('Before training, Defender utility is {:.4f}'.format(
            #     defender_utility))
        else:
            RunningExpl = []
        
        start_time = time.time()
        for episode in range(1, self.args.max_episodes + 1):
            observation, info = self.env.reset()
            terminated = False

            if self.args.attacker_mode == 'bandit':
                self.policy.sample_mode(exlp_prob=decay_expl(self.args.d_expl, episode, 5e6))
                # sample mode and set exit to commit
                Action = Attacker.sample_mode(exlp_prob=self.args.a_expl)
                AttackerReturn = 0

            else:
                self.policy.sample_mode()
                Attacker.sample_mode()                

            while not terminated:
                defender_obs = [copy.deepcopy(info["evader_history"]), 
                                copy.deepcopy(tuple(info["defender_history"][-1]))]
                attacker_obs = [copy.deepcopy(info["evader_history"]), 
                                copy.deepcopy(tuple(info["defender_history"][0]))]
                
                def_current_legal_action = list(product(*info["defender_legal_actions"]))
                att_current_legal_action = info["evader_legal_actions"]

                defender_a = self.policy.select_action(
                    [defender_obs], [def_current_legal_action], is_evaluation=False)
                attacker_a = Attacker.select_action(
                    [attacker_obs], [att_current_legal_action], is_evaluation=False)
                
                env_action = np.array(list(attacker_a + defender_a[0])) # shape: (defender_num +1, )
                observation, reward, terminated, truncated, info = self.env.step(env_action)

                def_next_obs = [copy.deepcopy(info["evader_history"]), 
                                copy.deepcopy(tuple(info["defender_history"][-1]))]
                att_next_obs = [copy.deepcopy(info["evader_history"]), 
                                copy.deepcopy(tuple(info["defender_history"][0]))]
                
                def_reward = reward
                att_reward = -reward

                if self.policy.is_br == True:
                    self.policy.avg_buffer.add(
                        Sample(defender_obs, defender_a))
                    
                if self.args.attacker_mode == 'bandit':
                    if Attacker.is_expl == False:
                        self.policy.br_buffer.add(Transition(defender_obs, defender_a,
                                                        def_reward, def_next_obs, terminated))
                    AttackerReturn += att_reward
                else:
                    self.policy.br_buffer.add(Transition(defender_obs, defender_a,
                                                    def_reward, def_next_obs, terminated))
                    Attacker.br_buffer.add(Transition(attacker_obs, attacker_a,
                                                    att_reward, att_next_obs, terminated))  
                    if Attacker.is_br == True:
                        Attacker.avg_buffer.add(
                            Sample(attacker_obs, attacker_a))                     
                                     
            if self.args.attacker_mode == 'bandit' and self.policy.is_expl == False:
                Attacker.BrAgent.update(Action, AttackerReturn)

            if episode % self.args.train_br_freq == 0:
                if len(self.policy.br_buffer) > self.args.min_to_train:
                    transitions = self.policy.br_buffer.sample(self.args.br_batch_size)
                    self.policy.learning_br_net(transitions)
                if self.args.attacker_mode == 'bandit' == False and len(Attacker.br_buffer) > self.args.min_to_train:
                    transitions = Attacker.br_buffer.sample(self.args.br_batch_size)
                    Attacker.learning_br_net(transitions)

            if episode % self.args.train_avg_freq == 0:
                if len(self.policy.avg_buffer) > self.args.min_to_train:
                    samples = self.policy.avg_buffer.sample(self.args.avg_batch_size)
                    self.policy.learning_avg_net(samples)
                if self.args.attacker_mode == 'bandit' == False and len(Attacker.avg_buffer) > self.args.min_to_train:
                    samples = Attacker.avg_buffer.sample(self.args.avg_batch_size)
                    Attacker.learning_avg_net(samples)

            if self.args.attacker_mode == 'bandit' and episode % (self.args.display_freq * 10) == 0:
                Attacker.update_N_a()
                Attacker.save_model(join(self.save_path, 'ATTACKER'), episode)

            if episode % self.args.check_freq == 0 and episode >= self.args.check_from:
                EPS = episode/(time.time()-start_time)
                REMAIN_TIME = (self.args.max_episodes-episode) / 60. / EPS
                print('Episode : {} , Store Model! EPS: {: .2f}, Time left: {: .2f} min.'.format(
                    episode, EPS, REMAIN_TIME))
                self.policy.save_model(join(self.save_path, 'DEFENDER'), episode)

            if episode % self.args.display_freq == 0:
                EPS = episode/(time.time()-start_time)
                REMAIN_TIME = (self.args.max_episodes-episode) / 60. / EPS
                if self.args.exact_br:
                    pass
                    # defender_utility = exact_br.Defender_Utility(Defender)
                    # Utilities.append((episode, defender_utility))
                    # print('Episode : {}, Defender Utility: {:.4f}, EPS: {:.2f}, Time left: {:.2f} min'.format(
                    #     episode, defender_utility, EPS, REMAIN_TIME))
                    # np.save(join(save_path, 'DEFENDER_UTILITY.npy'), Utilities)
                else:
                    self.policy.set_mode('br')
                    Attacker.set_mode('avg')
                    attacker_avg_return, std_return = evaluate(
                        self.env, self.policy, Attacker, 0, 300)
                    log = 'BR Defender return : {}\n'.format(attacker_avg_return)
                    self.policy.set_mode('avg')
                    Attacker.set_mode('br')
                    defender_avg_return, std_return = evaluate(
                        self.env, self.policy, Attacker, 1, 300)
                    log += 'BR Attacker return : {}\n'.format(defender_avg_return)
                    log += 'Episode : {} , EPS: {: .2f}, Time left: {: .2f} min.\n\n'.format(
                        episode, EPS, REMAIN_TIME)
                    with open(join(self.save_path, 'log.txt'), 'a') as f:
                        f.write(log)
                    print(log)
                    NashConv = attacker_avg_return+defender_avg_return
                    RunningExpl.append(
                        (episode, NashConv, attacker_avg_return, defender_avg_return))
                    np.save(join(self.save_path, 'RunningExpl.npy'), RunningExpl)