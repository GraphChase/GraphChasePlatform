import torch
from agent.grasper.utils import set_pretrain_model_path
import numpy as np
import time
import dgl
from torch import optim as optim
from agent.grasper.grasper_mappo_policy import RHMAPPO
from agent.grasper.mappo_policy import RMAPPO
from agent.grasper.utils import sample_env, get_dgl_graph
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from agent.grasper.graph_model import PreModel
from agent.pretrain_psro.path_evader_runner import PathEvaderRunner

from tensorboardX import SummaryWriter
import os
import pickle
from datetime import datetime


class GrasperDefenderRunner(object):

    def __init__(self, pretrain_policy, finetune_policy, args, env=None):
        
        self.pretrain_agent = pretrain_policy
        self.ft_agent = finetune_policy
        self.args = args
        self.graph_emb_model = None
        self.env = sample_env(args) if env is None else env

    def pre_pretrain(self, ):
        # save_path = self.args.save_path
        save_path = os.path.join(self.args.save_path, f"data/pretrain_models/graph_learning")
        if not os.path.exists(save_path):
            os.makedirs(save_path)        

        def collate_fn(batch):
            graphs = [x for x in batch]
            batch_g = dgl.batch(graphs)
            return batch_g  
        
        def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
            opt_lower = opt.lower()

            parameters = model.parameters()
            opt_args = dict(lr=lr, weight_decay=weight_decay)

            opt_split = opt_lower.split("_")
            opt_lower = opt_split[-1]
            if opt_lower == "adam":
                optimizer = optim.Adam(parameters, **opt_args)
            elif opt_lower == "adamw":
                optimizer = optim.AdamW(parameters, **opt_args)
            elif opt_lower == "adadelta":
                optimizer = optim.Adadelta(parameters, **opt_args)
            elif opt_lower == "radam":
                optimizer = optim.RAdam(parameters, **opt_args)
            elif opt_lower == "sgd":
                opt_args["momentum"] = 0.9
                return optim.SGD(parameters, **opt_args)
            else:
                assert False and "Invalid optimizer"

            return optimizer                 

        start_time = time.time()
        if self.args.graph_type == 'Grid_Graph':
            pth = f'data/related_files/game_pool/grid_{self.args.row * self.args.column}_probability_{self.args.edge_probability}'
        elif self.args.graph_type == 'SG_Graph':
            pth = f'data/related_files/game_pool/sg_graph_probability_{self.args.edge_probability}'
        elif self.args.graph_type == 'SY_Graph':
            pth = f'data/related_files/game_pool/sy_graph'
        elif self.args.graph_type == 'SF_Graph':
            pth = f'data/related_files/game_pool/sf_graph_{self.args.sf_sw_node_num}'
        else:
            raise ValueError(f"Unknown graph type: {self.args.graph_type}.")
        file_path = os.path.join(self.args.save_path, pth, 
                                 f'game_pool_size{self.args.pool_size}_dnum{self.args.num_defender}_enum{self.args.num_exit}_'
                                f'T{self.args.min_time_horizon}_{self.args.max_time_horizon}_mep{self.args.min_evader_pth_len}.pik')
        print('Load game pool ...')
        game_pool = pickle.load(open(file_path, 'rb'))['game_pool']
        graphs = []
        for game in game_pool:
            hg = get_dgl_graph(game.graph, self.args.node_feat_dim)
            graphs.append(hg)
        game_pool_str = f"_gp{self.args.pool_size}"

        node_feat_dim = hg.ndata['attr'].shape[1]
        print(f"******** # Num Graphs: {len(graphs)}, # Num Feat: {node_feat_dim} ********")
        writer = SummaryWriter(log_dir=save_path,
                               comment=f'graph_feat_learning_type_{self.args.graph_type}_ep{self.args.edge_probability}{game_pool_str}_'
                                    f'layer{self.args.gnn_num_layer}_hidden{self.args.gnn_hidden_dim}_out{self.args.gnn_output_dim}_dnum{self.args.num_defender}_'
                                    f'enum{self.args.num_exit}_mep{self.args.min_evader_pth_len}')
        
        train_idx = torch.arange(len(graphs))
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn, 
                                       batch_size=self.args.graph_pretrain_batch_size, pin_memory=True)
        
        model = PreModel(node_feat_dim, self.args.gnn_hidden_dim, self.args.gnn_output_dim, self.args.gnn_num_layer, self.args.gnn_dropout)
        model.to(self.args.device)

        optimizer = create_optimizer("adam", model, self.args.graph_pretrain_lr, self.args.graph_pretrain_weight_decay)
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / self.args.graph_pretrain_max_epoch)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

        train_loss = []
        for epoch in range(self.args.graph_pretrain_max_epoch):
            model.train()
            loss_list = []
            for batch_g in train_loader:
                batch_g = batch_g.to(self.args.device)
                feat = batch_g.ndata["attr"]
                model.train()
                loss, loss_dict = model(batch_g, feat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            scheduler.step()
            mean_loss = np.mean(loss_list)
            train_loss.append(mean_loss)
            print(f"Epoch {epoch + 1} | train_loss: {mean_loss:.4f}")
            writer.add_scalar('gnn_train_loss', mean_loss, epoch + 1)
            if (epoch + 1) % 200 == 0:
                model.save(save_path + f"/checkpoint_epoch{epoch + 1}_type_{self.args.graph_type}_ep{self.args.edge_probability}{game_pool_str}_layer{self.args.gnn_num_layer}_"
                                    f"hidden{self.args.gnn_hidden_dim}_out{self.args.gnn_output_dim}_dnum{self.args.num_defender}_enum{self.args.num_exit}_"
                                    f"mep{self.args.min_evader_pth_len}.pt")
        end_time = time.time()
        train_time = end_time - start_time
        self.graph_emb_model = model
        pickle.dump({'train_time': train_time, 'train_loss': train_loss},
                    open(save_path + f'/train_record_type_{self.args.graph_type}_ep{self.args.edge_probability}{game_pool_str}_layer{self.args.gnn_num_layer}_'
                                    f'hidden{self.args.gnn_hidden_dim}_out{self.args.gnn_output_dim}_dnum{self.args.num_defender}_enum{self.args.num_exit}_'
                                    f'mep{self.args.min_evader_pth_len}.pik', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)        
       
    def pretrain(self, evader_runner_cls:PathEvaderRunner):

        save_path = os.path.join(self.args.save_path, f"data/pretrain_models/pretrain_agent")
        writer = SummaryWriter(f"{save_path}/pretrain_loss")
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)         

        env_pool = self.load_env_pool()
        
        # Set graph_embedding_model
        env = np.random.choice(env_pool)
        self.load_pre_pretrain_model(env)

        # Set pretrain model
        # if self.args.load_pretrain_model: 
        #     print("Load hyper model {}/{}".format(self.args.actor_model, self.args.critic_model))
        #     self.pretrain_agent.policy.actor.load_state_dict(torch.load(self.args.actor_model))
        #     self.pretrain_agent.policy.critic.load_state_dict(torch.load(self.args.critic_model))    

        setup_str = f"End2End: {self.args.use_end_to_end}, Emb Layer: {self.args.use_emb_layer}, Augment: {self.args.use_augmentation}, Act Supervisor: {self.args.use_act_supervisor}"       

        start_time = time.time()
        iter_cnt = 0
        aloss_list, vloss_list = [], []
        time_list = [] # Record each env's training time
        reward_list = [] # Record each env's defender's average reward
            
        update_game_freq = self.args.num_task * self.args.num_sample
        min_update_episodes = self.args.num_games * update_game_freq
        min_update_eps = min_update_episodes if min_update_episodes > 1 else self.args.batch_size
        start_ = datetime.now().replace(microsecond=0)

        for _ in range(self.args.num_iterations // (update_game_freq)):
            
            # 每 update_game_freq 次更换一个 env
            self.env = np.random.choice(env_pool)
            evader_runner = evader_runner_cls(self.env, self.args)

            adaption_reward = 0
            episode_count = 0

            if self.args.use_end_to_end: # TODO
                pass
            else:
                pooled_node_embs, Ts = self.get_GEmbs_TEmbs(self.env)              

            for _ in range(self.args.num_task):
                evader_policy = evader_runner._get_strategy()

                for _ in range(self.args.num_sample):
                    iter_cnt += 1
                    episode_count += 1
                    evader_actions, evader_path = evader_runner.get_action(evader_policy)

                    # sample to collect data, rollout
                    episode_reward = self.run_one_episode(evader_actions, evader_path, pooled_node_embs, Ts, self.args.use_act_supervisor)
                    adaption_reward += episode_reward

                    # Start Training                                                              
                    if iter_cnt >= min_update_episodes and iter_cnt % min_update_eps == 0:  
                        self.pretrain_agent.policy.actor.train()
                        self.pretrain_agent.policy.critic.train()
                        train_infos = self.pretrain_agent.train()
                        aloss_list.append(train_infos['value_loss'])
                        vloss_list.append(train_infos['policy_loss'])

                    if iter_cnt % self.args.save_every == 0:
                        path = set_pretrain_model_path(self.args, iter_cnt)
                        self.pretrain_agent.save(path)
                        
                    print_every = update_game_freq * 10 if update_game_freq > 1 else 1000
                    if iter_cnt % print_every == 0 or iter_cnt == self.args.num_iterations:
                        print('{}, {}, Iteration: {}/{}, Train Reward: {:.6f}, MEP: {}, Time: {:.6f}'
                            .format(self.args.base_rl, setup_str, iter_cnt, self.args.num_iterations, adaption_reward / episode_count,
                                    self.args.min_evader_pth_len, time.time() - start_time))                        
                
                # Record one env defender_reward
                if writer is not None:
                    writer.add_scalar('defender_reward', adaption_reward / episode_count, iter_cnt)
                time_list.append(time.time() - start_time)
                reward_list.append(adaption_reward / episode_count)

        print("============================================================================================")
        end_ = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_)
        print("Finished training at (GMT) : ", end_)
        print("Total training time  : ", end_ - start_)
        print("============================================================================================")            

    def load_emb_layer(self):
        self.ft_agent.policy.actor.init_emb_layer(
            self.pretrain_agent.policy.actor.base.node_idx_emb_layer.state_dict(),
            self.pretrain_agent.policy.actor.base.time_idx_emb_layer.state_dict(),
            self.pretrain_agent.policy.actor.base.agent_id_emb_layer.state_dict())
        self.ft_agent.policy.critic.init_emb_layer(
            self.pretrain_agent.policy.critic.base.node_idx_emb_layer.state_dict(),
            self.pretrain_agent.policy.critic.base.time_idx_emb_layer.state_dict())                   

    def train(self, evader_policy_list:list, meta_strategy:np.ndarray,
              train_num_per_ite=10) -> tuple[np.ndarray, list, list]:
        """
        Fine-tuning RL policy
        """
        # self.writer = SummaryWriter(f"{self.args.save_root}/finetuning_phase")       

        for _ in range(self.args.train_pursuer_number):
            
            evader_strategy_num = len(evader_policy_list[0].path_selections)
            strategy = np.zeros((len(evader_policy_list), evader_strategy_num))

            for i, runner in enumerate(evader_policy_list):
                strategy[i] = runner._get_strategy()
            strategy = (strategy * meta_strategy[:, np.newaxis]).sum(axis=0)
            strategy /= np.sum(strategy)

            evader_runner = evader_policy_list[0]
            evader_policy = strategy

            evader_actions, evader_path = evader_runner.get_action(evader_policy)

            # sample to collect data, rollout
            self.ft_agent.policy.actor.eval()
            self.ft_agent.policy.critic.eval()            
            episode_reward = self.run_one_episode_ft(evader_actions, evader_path, self.pooled_node_embs, self.args.use_act_supervisor)
            # adaption_reward += episode_reward

            # Start Training               
            self.ft_agent.prep_training()
            train_infos, trained = self.ft_agent.train()
            if trained:
                self.ft_agent.buffer.after_update()            
        
        return train_infos
        
    def run_one_episode(self, evader_actions, evader_path, pooled_node_embs, Ts, use_act_supervisor=True):

        episode_reward, episode_length = 0, 0
        terminated = False
        observation, info =  self.env.reset()
        shared_obs = np.tile(np.append(observation, episode_length), (self.env.defender_num, 1)) # shape: (n_agent, evader_num+defender_num+1) include t
        obs = np.hstack((shared_obs, np.arange(self.env.defender_num).reshape(-1,1))) # shape: (n_agent, evader_num+defender_num+2) include t and id
        if use_act_supervisor:
            demonstration_distribs = self.env.graph.get_demonstration(obs, evader_path[-1])
        else:
            demonstration_distribs = None
        
        while not terminated:
            evader_act = evader_actions[episode_length]
            
            values, actions, action_log_probs, env_actions = self.collect(pooled_node_embs, Ts, shared_obs, obs)
            # Add evader's action
            env_actions = np.insert(env_actions, 0, evader_act) # shape: (defender_num +1, )

            observation, reward, terminated, truncated, info = self.env.step(env_actions)
            episode_length += 1

            episode_reward += reward[1]
            
            rewards = np.array([reward[1] for _ in range(self.env.defender_num)]).reshape(-1, 1)
            dones = np.array([terminated for _ in range(self.env.defender_num)])

            data = pooled_node_embs, Ts, shared_obs, obs, rewards, dones, values, actions, action_log_probs, demonstration_distribs
            self.insert2buffer(data)

            shared_obs_next = np.tile(np.append(observation, episode_length), (self.env.defender_num, 1)) # shape: (n_agent, evader_num+defender_num+1) include t
            obs_next = np.hstack((shared_obs_next, np.arange(self.env.defender_num).reshape(-1,1)))

            if use_act_supervisor:
                demonstration_distribs_next = self.env.graph.get_demonstration(obs_next, evader_path[-1])
            else:
                demonstration_distribs_next = None
            
            shared_obs, obs = shared_obs_next.copy(), obs_next.copy()
            if demonstration_distribs_next is not None:
                demonstration_distribs = demonstration_distribs_next.copy()            
        
        self.pretrain_agent.buffer.episode_length.append(episode_length)
        self.compute(pooled_node_embs, Ts, shared_obs)
        return episode_reward
    
    def run_one_episode_ft(self, evader_actions, evader_path, pooled_node_emb, use_act_supervisor=True):
        episode_reward, episode_length = 0, 0
        terminated = False
        observation, info =  self.env.reset()
        shared_obs = np.tile(np.append(observation, episode_length), (self.env.defender_num, 1)) # shape: (n_agent, evader_num+defender_num+1) include t
        obs = np.hstack((shared_obs, np.arange(self.env.defender_num).reshape(-1,1))) # shape: (n_agent, evader_num+defender_num+2) include t and id                
        
        if use_act_supervisor:
            demonstration_distribs = self.env.graph.get_demonstration(obs, evader_path[-1])
        else:
            demonstration_distribs = None
        
        if not self.args.use_augmentation:
            pooled_node_emb = None

        while not terminated:
            evader_act = evader_actions[episode_length]
            values, actions, action_log_probs, actions_env = self.collect_ft(shared_obs, obs, pooled_node_emb)
            # Add evader's action
            env_actions = np.insert(actions_env, 0, evader_act) # shape: (defender_num +1, )

            observation, reward, terminated, truncated, info = self.env.step(env_actions)
            episode_length += 1

            episode_reward += reward[1]
            rewards = np.array([reward[1] for _ in range(self.env.defender_num)]).reshape(-1, 1)
            dones = np.array([terminated for _ in range(self.env.defender_num)])            
            data = shared_obs, obs, rewards, dones, values, actions, action_log_probs, demonstration_distribs, pooled_node_emb
            self.insert2buffer_ft(data)

            shared_obs_next = np.tile(np.append(observation, episode_length), (self.env.defender_num, 1)) # shape: (n_agent, evader_num+defender_num+1) include t
            obs_next = np.hstack((shared_obs_next, np.arange(self.env.defender_num).reshape(-1,1)))

            if use_act_supervisor:
                demonstration_distribs_next = self.env.graph.get_demonstration(obs_next, evader_path[-1])
            else:
                demonstration_distribs_next = None
            
            shared_obs, obs = shared_obs_next.copy(), obs_next.copy()
            if demonstration_distribs_next is not None:
                demonstration_distribs = demonstration_distribs_next.copy()                     

        self.ft_agent.buffer.episode_length.append(episode_length)
        self.compute_ft(shared_obs, pooled_node_emb)
        return episode_reward 
    
    @torch.no_grad()
    def collect(self, pooled_node_embs, Ts, shared_obs, obs):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.args.device)
            obs = torch.LongTensor(obs).to(self.args.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.args.device)
            obs = torch.FloatTensor(obs).to(self.args.device)
        Ts = torch.FloatTensor(Ts).to(self.args.device)
        pooled_node_embs = torch.FloatTensor(pooled_node_embs).to(self.args.device)
        value, action, action_log_prob = self.pretrain_agent.policy.get_actions(pooled_node_embs, Ts, shared_obs, obs, batch=True)
        values = value.detach().cpu().numpy()
        actions = action.detach().cpu().numpy()
        action_log_probs = action_log_prob.detach().cpu().numpy()
        actions_env = np.reshape(actions, self.env.defender_num)
        return values, actions, action_log_probs, actions_env
    
    @torch.no_grad()
    def collect_ft(self, shared_obs, obs, pooled_node_emb):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.args.device)
            obs = torch.LongTensor(obs).to(self.args.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.args.device)
            obs = torch.FloatTensor(obs).to(self.args.device)
        if self.args.use_augmentation:
            pooled_node_emb = torch.FloatTensor(pooled_node_emb).to(self.args.device)
        value, action, action_log_prob = self.ft_agent.policy.get_actions(shared_obs, obs, pooled_node_emb, batch=True)
        values = value.detach().cpu().numpy()
        actions = action.detach().cpu().numpy()
        action_log_probs = action_log_prob.detach().cpu().numpy()
        actions_env = np.reshape(actions, self.args.num_defender)
        return values, actions, action_log_probs, actions_env    

    @torch.no_grad()
    def compute(self, pooled_node_embs, Ts, shared_obs):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.args.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.args.device)
        Ts = torch.FloatTensor(Ts).to(self.args.device)
        pooled_node_embs = torch.FloatTensor(pooled_node_embs).to(self.args.device)
        next_values = self.pretrain_agent.policy.get_values(pooled_node_embs, Ts, shared_obs, batch=True)
        next_values = next_values.detach().cpu().numpy()
        self.pretrain_agent.buffer.compute_returns(next_values, self.pretrain_agent.value_normalizer)

    @torch.no_grad()
    def compute_ft(self, shared_obs, pooled_node_emb):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.args.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.args.device)
        next_values = self.ft_agent.policy.get_values(shared_obs, pooled_node_emb, batch=True)
        next_values = next_values.detach().cpu().numpy()
        self.ft_agent.buffer.compute_returns(next_values, self.ft_agent.value_normalizer)        
 
    def insert2buffer(self, data):
        pooled_node_embs, Ts, shared_obs, obs, rewards, dones, values, actions, action_log_probs, demonstration_distribs = data
        masks = np.ones((self.env.defender_num, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        self.pretrain_agent.buffer.insert(pooled_node_embs, Ts, shared_obs, obs, actions, action_log_probs, values, rewards, masks, demonstration_distribs)

    def insert2buffer_ft(self, data):
        shared_obs, obs, rewards, dones, values, actions, action_log_probs, demonstration_distribs, pooled_node_emb = data
        masks = np.ones((self.args.num_defender, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        self.ft_agent.buffer.insert(shared_obs, obs, actions, action_log_probs, values, rewards, masks, demonstration_distribs, pooled_node_emb)    

    def load_env_pool(self,):
        if self.args.graph_type == 'Grid_Graph':
            save_path = 'data/related_files/game_pool/grid_{}_probability_{}'.format(self.args.row * self.args.column, self.args.edge_probability)
        elif self.args.graph_type == 'SG_Graph':
            save_path = 'data/related_files/game_pool/sg_graph_probability_{}'.format(self.args.edge_probability)
        elif self.args.graph_type == 'SY_Graph':
            save_path = 'data/related_files/game_pool/sy_graph'
        elif self.args.graph_type == 'SF_Graph':
            save_path = 'data/related_files/game_pool/sf_graph_{}'.format(self.args.sf_sw_node_num)
        else:
            raise ValueError('Unrecognized graph type.')
        save_path = os.path.join(self.args.save_path, save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, 'game_pool_size{}_dnum{}_enum{}_T{}_{}_mep{}.pik'
                            .format(self.args.pool_size, self.args.num_defender, self.args.num_exit, self.args.min_time_horizon,
                                    self.args.max_time_horizon, self.args.min_evader_pth_len))
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"Load game pool from {file_path}...")
            env_pool = pickle.load(open(file_path, 'rb'))['game_pool']
        else:
            raise ValueError('Game pool does not exist.')
        return env_pool   
    
    def get_env_actions(self, observation, time_step):
        """
        For PSRO, return actions in env e.g. [0,1,2,3,4] = [Up, Down, ..]
        """ 
        shared_obs = np.tile(np.append(observation, time_step), (self.env.defender_num, 1)) # shape: (n_agent, evader_num+defender_num+1) include t
        obs = np.hstack((shared_obs, np.arange(self.env.defender_num).reshape(-1,1))) # shape: (n_agent, evader_num+defender_num+2) include t and id        
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.args.device)
            obs = torch.LongTensor(obs).to(self.args.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.args.device)
            obs = torch.FloatTensor(obs).to(self.args.device)

        pooled_node_embs, Ts = self.get_GEmbs_TEmbs(self.env)
        if self.args.use_augmentation:
            pooled_node_emb = torch.FloatTensor(pooled_node_emb).to(self.args.device)

        with torch.no_grad():
            actions = self.ft_agent.policy.act(obs, pooled_node_embs, batch=True)
        actions = actions.detach().cpu().numpy()
        actions_env = np.reshape(actions, self.args.num_defender)            
        return actions_env

    @torch.no_grad()
    def get_GEmbs_TEmbs(self, env):
        hg = get_dgl_graph(env.graph, self.args.node_feat_dim)
        hg = hg.to(self.args.device)
        feat = hg.ndata["attr"]
        _, pooled_node_emb = self.graph_emb_model.embed(hg, feat)
        pooled_node_emb = pooled_node_emb.cpu().numpy()
        pooled_node_embs = np.array([pooled_node_emb for _ in range(env.defender_num)])

        T_emb = [0] * self.args.max_time_horizon_for_state_emb
        T_emb[env.time_horizon] = 1
        Ts = np.array([T_emb for _ in range(env.defender_num)])

        return pooled_node_embs, Ts

    def finetuning_init(self, env=None):
        if env is not None:
            self.env = env
        # Set graph_embedding_model
        self.load_pre_pretrain_model(self.env)

        # Set pretrain model
        if self.args.load_pretrain_model: 
            pretrain_model = set_pretrain_model_path(self.args, self.args.num_iterations)
            actor_model = pretrain_model + '_actor.pt'
            critic_model = pretrain_model + '_critic.pt'
            print("Load hyper model {}/{}".format(actor_model, critic_model))
            self.pretrain_agent.policy.actor.load_state_dict(torch.load(actor_model))
            self.pretrain_agent.policy.critic.load_state_dict(torch.load(critic_model)) 

        # Initialize finetuning model parameters
        if self.args.use_end_to_end:
            hgs = get_dgl_graph(self.env)
            hgs_batch = dgl.batch([hgs])
            hgs_batch = hgs_batch.to(self.args.device)
            T = np.array([0] * self.args.max_time_horizon_for_state_emb)
            T[self.env.time_horizon] = 1            
            Ts = torch.FloatTensor(np.array([T])).to(self.args.device)
            wa, ba, _, pooled_node_emb = self.pretrain_agent.policy.actor.base.get_weight(hgs_batch, Ts)
            self.ft_agent.policy.actor.init_paras(wa, ba)
            self.pooled_node_emb = pooled_node_emb.numpy()
            wc, bc, _, _ = self.pretrain_agent.policy.critic.base.get_weight(hgs_batch, Ts)
            self.ft_agent.policy.critic.init_paras(wc, bc)
        else:
            self.pooled_node_embs, Ts = self.get_GEmbs_TEmbs(self.env)
            graph_embs = torch.FloatTensor(self.pooled_node_embs[0]).to(self.args.device)
            Ts = torch.FloatTensor(Ts[0]).to(self.args.device)
            wa, ba = self.pretrain_agent.policy.actor.base.get_weight(graph_embs.unsqueeze(0), Ts.unsqueeze(0))
            self.ft_agent.policy.actor.init_paras(wa, ba)
            wc, bc = self.pretrain_agent.policy.critic.base.get_weight(graph_embs.unsqueeze(0), Ts.unsqueeze(0))
            self.ft_agent.policy.critic.init_paras(wc, bc)
        if self.args.use_emb_layer and self.args.load_pretrain_model:
            self.load_emb_layer() 

    def load_pretrain_net_checkpoint(self, pretrain_actor_path, pretrain_critic_path):
        self.pretrain_agent.policy.actor.load_state_dict(torch.load(pretrain_actor_path))
        self.pretrain_agent.policy.critic.load_state_dict(torch.load(pretrain_critic_path))

    def load_pre_pretrain_model(self, env):
        """
        self.graph_emb_model: Set pre_pretrain_model
        """
        feat = env.graph.get_graph_info(self.args.node_feat_dim)
        if self.args.use_end_to_end:
            self.args.use_emb_layer = True
            hg = get_dgl_graph(env)
            hgs = [hg for _ in range(env.defender_num)]
        else:
            if self.args.load_graph_emb_model:
                self.graph_emb_model = PreModel(feat.shape[1], self.args.gnn_hidden_dim, self.args.gnn_output_dim, self.args.gnn_num_layer, self.args.gnn_dropout)
                self.graph_emb_model.to(self.args.device)            
                pretrain_graph_model_file = f"data/pretrain_models/graph_learning/checkpoint_epoch{self.args.graph_pretrain_max_epoch}_type_{self.args.graph_type}_" \
                                            f"ep{self.args.edge_probability}_gp{self.args.pool_size}_layer{self.args.gnn_num_layer}_" \
                                            f"hidden{self.args.gnn_hidden_dim}_out{self.args.gnn_output_dim}_dnum{self.args.num_defender}_" \
                                            f"enum{self.args.num_exit}_mep{self.args.min_evader_pth_len}.pt"
                pretrain_graph_model_file = os.path.join(self.args.save_path, pretrain_graph_model_file)
                self.graph_emb_model.load(torch.load(pretrain_graph_model_file))
                print(f"Load pretrained graph model from {pretrain_graph_model_file}...")
            else:
                assert self.graph_emb_model is not None, "graph_model is None, please load graph model file or run pre_pretrain"        

    def save(self, save_folder, prefix=None):
        os.makedirs(save_folder, exist_ok=True)
        actor_name = 'actor.pt'
        critic_name = 'critic.pt'
        if prefix:
            actor_name = f"{str(prefix)}_{actor_name}"
            critic_name = f"{str(prefix)}_{critic_name}"
        torch.save(self.ft_agent.policy.actor.state_dict(), os.path.join(save_folder, actor_name))
        torch.save(self.ft_agent.policy.critic.state_dict(), os.path.join(save_folder, critic_name))