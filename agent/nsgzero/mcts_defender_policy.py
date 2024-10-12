import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-8

class policy_net(nn.Module):
    def __init__(self, game, args):
        super(policy_net, self).__init__()
        self.args=args
        self.game=game
        self.num_nodes=game.graph.num_nodes
        self.time_horizon=game.time_horizon
        self.num_defender=game.defender_num
        self.obsn_embedding = nn.Embedding(
            self.num_nodes+1, args.embedding_dim, padding_idx=0)
        self.obs_f1 = nn.Linear((self.num_defender+1)*args.embedding_dim+self.time_horizon+1, args.hidden_dim)
        self.obs_f2=nn.Linear(args.hidden_dim, args.hidden_dim)
    
        self.actn_embedding = nn.Embedding(
            self.num_nodes+1, args.embedding_dim, padding_idx=0)
        self.act_f1=nn.Linear(args.embedding_dim, args.hidden_dim)
        self.act_f2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.init_weights()
        self.to(args.device)

    def forward(self, obs, legal_act):
        with torch.no_grad():
            #copy_obs=copy.deepcopy(obs)
            attacker_his, defender_position = copy.deepcopy(obs)
            if isinstance(defender_position, tuple):
                defender_position=list(defender_position)
            attacker_position=attacker_his[-1]

            defender_position.append(attacker_position)
            obs = torch.LongTensor(defender_position).to(self.args.device)
            obs=self.obsn_embedding(obs).flatten()

            t=F.one_hot(torch.tensor(len(attacker_his)-1), num_classes=self.time_horizon+1).to(self.args.device).float()
            obs=torch.cat([obs,t])
            obs=self.obs_f1(obs)
            obs=self.obs_f2(F.relu(obs)) # [hidden_dim]

            act = torch.LongTensor(legal_act).to(self.args.device)
            act=self.actn_embedding(act) # [n_legal_act, embedding_dim]
            act=self.act_f1(act)
            act = self.act_f2(F.relu(act))  # [n_legal_act, hidden_dim]

            logits=torch.matmul(act, obs)
            return F.softmax(logits, dim=-1)

    def batch_forward(self, def_pos, att_pos, time, legal_act):
        batch_size = def_pos.shape[0]
        num_player = legal_act.shape[1]
        obs = torch.cat([def_pos, att_pos], dim=-1).long()
        obs = self.obsn_embedding(obs).view(batch_size, -1)
        t = F.one_hot(time,
                      num_classes=self.time_horizon+1).float()
        obs = torch.cat([obs, t], dim=-1)
        obs = self.obs_f1(obs)
        obs = self.obs_f2(F.relu(obs))  # [batch, hidden_dim]
        obs = obs.unsqueeze(1).repeat(1, num_player, 1)
    
        act=self.actn_embedding(legal_act.long()) # [batch, num_player, max_act, embedding_dim]
        act=self.act_f1(act)
        act = self.act_f2(F.relu(act))  # [batch, num_player, max_act, hidden_dim]
        
        logits = torch.bmm(act.view(batch_size*num_player, -1, self.args.hidden_dim),
                           obs.view(batch_size*num_player, self.args.hidden_dim).unsqueeze(-1))
        return logits.squeeze()  # [batch*num_player, max_act]

    def init_weights(self):
        for param in self.parameters():
            nn.init.normal_(param, 0.0, 0.1)

class value_net(nn.Module):
    def __init__(self, game, args):
        super(value_net, self).__init__()
        self.args=args
        self.game=game
        self.num_nodes = game.graph.num_nodes
        self.time_horizon = game.time_horizon
        self.num_defender = game.defender_num
        self.obsn_embedding = nn.Embedding(
            self.num_nodes+1, args.embedding_dim, padding_idx=0)
        self.obs_f1 = nn.Linear(
            (self.num_defender+1)*args.embedding_dim+self.time_horizon+1, args.hidden_dim)
        self.obs_f2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.obs_f3=nn.Linear(args.hidden_dim, 1)

        self.to(args.device)

    def forward(self, obs):
        with torch.no_grad():
            attacker_his, defender_position = obs
            assert defender_position, tuple
            defender_position = list(defender_position)
            attacker_position = attacker_his[-1]

            defender_position.append(attacker_position)
            obs = torch.LongTensor(defender_position).to(self.args.device)
            obs = self.obsn_embedding(obs).flatten()

            t = F.one_hot(torch.tensor(len(attacker_his)-1),
                        num_classes=self.time_horizon+1).to(self.args.device).float()
            obs = torch.cat([obs, t])
            obs = self.obs_f1(obs)
            obs = self.obs_f2(F.relu(obs))  # [batch, hidden_dim]
            obs = self.obs_f3(F.relu(obs))
            return torch.sigmoid(obs)

    def batch_forward(self, def_pos, att_pos, time):
        batch_size=def_pos.shape[0]
        obs = torch.cat([def_pos, att_pos], dim=-1).long()
        obs = self.obsn_embedding(obs).view(batch_size,-1)
        t = F.one_hot(time,
                      num_classes=self.time_horizon+1).float()
        obs = torch.cat([obs, t], dim=-1)
        obs = self.obs_f1(obs)
        obs = self.obs_f2(F.relu(obs))  # [hidden_dim]
        logis = self.obs_f3(F.relu(obs))
        return logis # [batch, 1]

class pr_net(nn.Module):
    def __init__(self,game, args):
        super(pr_net, self).__init__()
        self.args=args
        self.game=game
        self.policy_net=policy_net(game, args)
        self.value_net=value_net(game, args)

    def state_value(self, obs):
        return self.value_net(obs).item()

    def prior_pol(self, obs, legal_act): # output should be a distribution over legal_act
        return self.policy_net(obs, legal_act).cpu().numpy()

class dy_net(nn.Module):
    def __init__(self,game, args):
        super(dy_net, self).__init__()
        self.args=args
        self.game=game
        self.policy_net=policy_net(game, args)

    def predict(self, obs, legal_act):
        probs = self.policy_net(obs, legal_act).cpu().numpy()
        return np.random.choice(legal_act, p=probs) 

class MCTS:
    def __init__(self, game, dy_net: dy_net, pr_net: pr_net, args):
        self.game=game
        self.dy_net=dy_net
        self.pr_net=pr_net
        self.args=args
        self.create_func()
        
        self.num_defender=game.defender_num
        self.Ns={}
       
        self.Ls = [{}]  # legal actions at s for defender and attacker
        self.Ps=[]
        self.Qsa=[]
        self.Nsa=[]
        
        for i in range(self.num_defender):
            self.Ls.append({})
            self.Ps.append({})
            self.Qsa.append({})
            self.Nsa.append({})

    def act_prob(self, obs, defender_legal_act, temp=1):
        for _ in range(self.args.num_sims):
            self.search(obs)
        str_obs=str(obs)

        counts = []
        probs = []
        for i in range(self.num_defender):
            counts.append([self.Nsa[i][(str_obs, str(a))] if (str_obs, str(
                a)) in self.Nsa[i] else 0 for a in defender_legal_act[i]])
            if temp==0:
                bestAs=np.array(np.argwhere(counts[i]==np.max(counts[i]))).flatten()
                bestA=np.random.choice(bestAs)
                p=[0]*len(counts[i])
                p[bestA]=1
                probs.append(p)
            else:
                counts[i] = [x ** (1. / temp) for x in counts[i]]
                counts_sum = float(sum(counts[i]))
                probs.append([x / counts_sum for x in counts[i]])
        return probs
    
    def act_prior_prob(self, obs, defender_legal_act):
        probs=[]
        for i in range(self.num_defender):
            probs.append(self.pr_net.prior_pol(obs, defender_legal_act[i]))
        return(probs)

    def search(self, obs):
        str_obs=str(obs)
        attacker_his, defender_position=obs

        if self._is_end(attacker_his, defender_position):
            return self._reward(True, attacker_his, defender_position)[1] # reward for defender

        if str_obs not in self.Ns: # leaf node
            self.Ns[str_obs]=0      
            v = self.pr_net.state_value(obs)    
            defender_legal_act, attacker_legal_act = self._legal_action(self._is_end(attacker_his, defender_position),
                                                                        False, attacker_his, defender_position)
            self.Ls[-1][str_obs]=attacker_legal_act
            for i in range(self.num_defender):
                self.Ls[i][str_obs]=defender_legal_act[i]
                self.Ps[i][str_obs] = self.pr_net.prior_pol(
                    obs, defender_legal_act[i])
            return v

        best_act=[]
        for i in range(self.num_defender):
            best_value = -float('inf')
            cur_best_act=0
            defender_legal_act = self.Ls[i][str_obs]
            prior_pol=self.Ps[i][str_obs]
            for idx in range(len(defender_legal_act)):
                a=defender_legal_act[idx]
                str_a=str(a)
                if (str_obs, str_a) in self.Qsa[i]:
                    u = (self.Qsa[i][(str_obs, str_a)]-self.args.bias) + self.args.cpuct*prior_pol[idx] * math.sqrt(self.Ns[str_obs]) / (
                        1 + self.Nsa[i][(str_obs, str_a)])
                else:
                    u = self.args.cpuct*prior_pol[idx] * math.sqrt(self.Ns[str_obs]+eps)
                if u>best_value:
                    best_value=u
                    cur_best_act=a
            assert not cur_best_act==0
            best_act.append(cur_best_act)

        best_act=tuple(best_act)
        attacker_legal_act=self.Ls[-1][str_obs]
        attacker_act=self.dy_net.predict(obs, attacker_legal_act)

        next_attack_his=copy.deepcopy(attacker_his)
        next_attack_his.append(attacker_act)

        next_obs=(next_attack_his,best_act)
        v=self.search(next_obs)
        
        for i in range(self.num_defender):
            a = best_act[i]
            str_a=str(a)
            if (str_obs, str_a) in self.Qsa[i]:
                self.Qsa[i][(str_obs, str_a)] = (self.Nsa[i][(str_obs, str_a)]*self.Qsa[i][(str_obs, str_a)]+v)/(self.Nsa[i][(str_obs, str_a)]+1)
                self.Nsa[i][(str_obs, str_a)]+=1
            else:
                self.Qsa[i][(str_obs, str_a)]=v
                self.Nsa[i][(str_obs, str_a)]=1
        self.Ns[str_obs]+=1
        return v

    def create_func(self):
        self._is_end=self.game._is_terminal
        self._reward=self.game._get_rewards
        self._legal_action=self.game.get_legal_action

class NsgzeroDefenderPolicy(object):
    def __init__(self, game, args):
        super().__init__()

        self.game = game
        self.time_horizon = game.time_horizon
        self.args = args
        self.num_defender = game.defender_num
        self.dy_net = dy_net(game, args)
        self.pr_net = pr_net(game, args)
        self.mcts = MCTS(self.game, self.dy_net, self.pr_net, self.args)

        self.buffer = []
        self.total_traj = 0
        self.buffer_size = args.buffer_size
        self.v_opt = torch.optim.Adam(
            self.pr_net.value_net.parameters(), lr=self.args.lr*10)
        self.def_pre_opt = torch.optim.Adam(
            self.pr_net.policy_net.parameters(), lr=self.args.lr)
        self.att_pre_opt = torch.optim.Adam(
            self.dy_net.policy_net.parameters(), lr=self.args.lr)
        #self.att_pre_opt = torch.optim.RMSprop(self.dy_net.policy_net.parameters(), lr=self.args.lr)

    def add_trajectory(self, trajectory):
        if self.total_traj < self.buffer_size:
            self.buffer.append(trajectory)
            self.total_traj += 1
        else:
            idx = self.total_traj % self.buffer_size
            self.buffer[idx] = trajectory
            self.total_traj += 1

    def select_act(self, obs, defender_legal_act, prior=False, temp=1):
        self.mcts = MCTS(self.game, self.dy_net, self.pr_net, self.args)
        copy_obs = copy.deepcopy(obs)
        
        if prior:
            probs = self.act_prior_prob(copy_obs, defender_legal_act)
        else:
            probs = self.act_prob(copy_obs, defender_legal_act, temp)
        
        act = []
        for i in range(self.num_defender):
            act.append(np.random.choice(defender_legal_act[i], p=probs[i]))
        act = tuple(act)
        return act

    def train_select_act(self, obs, defender_legal_act, prior=False):
        self.mcts = MCTS(self.game, self.dy_net, self.pr_net, self.args)
        copy_obs = copy.deepcopy(obs)
    
        if prior:
            probs = self.act_prior_prob(copy_obs, defender_legal_act)
        else:
            probs = self.act_prob(copy_obs, defender_legal_act, self.args.temp)
        act = []
        act_idx = []

        for i in range(self.num_defender):
            index = np.arange(len(defender_legal_act[i]))
            index = np.random.choice(index, p=probs[i])
            act_idx.append(index)
            act.append(defender_legal_act[i][index])
        act = tuple(act)
        act_idx = tuple(act_idx)
        return act, act_idx

    def act_prob(self, obs, defender_legal_act, temp=1):
        return self.mcts.act_prob(obs, defender_legal_act, temp)

    def act_prior_prob(self, obs, defender_legal_act):
        return self.mcts.act_prior_prob(obs, defender_legal_act)

    def learn(self, batch_size=64):
        assert len(self.buffer) >= batch_size
        data = np.random.choice(self.buffer, batch_size, replace=False)
        defender_his = []
        attacker_his = []
        defender_his_idx = []
        attacker_his_idx = []
        defender_legal_act = []
        attacker_legal_act = []
        ret = []
        mask = []
        for trajectory in data:
            defender_his.append(trajectory["defender_his"])
            attacker_his.append(trajectory["attacker_his"])
            defender_his_idx.append(trajectory["defender_his_idx"])
            attacker_his_idx.append(trajectory["attacker_his_idx"])
            defender_legal_act.append(trajectory["defender_legal_act"])
            attacker_legal_act.append(trajectory["attacker_legal_act"])
            ret.append(trajectory["return"])
            mask.append(trajectory["mask"])
        defender_his = torch.tensor(
            np.stack(defender_his)).to(self.args.device)
        attacker_his = torch.tensor(
            np.stack(attacker_his)).to(self.args.device)
        defender_his_idx = torch.tensor(
            np.stack(defender_his_idx)).to(self.args.device)
        attacker_his_idx = torch.tensor(
            np.stack(attacker_his_idx)).to(self.args.device)
        defender_legal_act = torch.tensor(
            np.stack(defender_legal_act)).to(self.args.device)
        attacker_legal_act = torch.tensor(
            np.stack(attacker_legal_act)).to(self.args.device)
        ret = torch.tensor(np.stack(ret)).to(self.args.device)
        mask = torch.tensor(np.stack(mask)).to(self.args.device)
        length = mask.sum(dim=1)
        
        v_loss = 0
        def_pre_loss = 0
        att_pre_loss = 0
        for t in range(0, self.time_horizon+1):
            def_pos = defender_his[:, t]  # [batch, num_defender]
            att_pos = attacker_his[:, t]  # [batch, 1]
            time = torch.tensor(t).to(self.args.device).repeat(
                batch_size).detach()  # [batch]
            pre_v = self.pr_net.value_net.batch_forward(def_pos, att_pos, time)
            lable_v = ret[:, t]*(self.args.gamma**(torch.maximum(length-t,torch.zeros_like(length))))
            #lable_v = ret[:, t]
            # v_loss += (mask[:, t] * ((pre_v-lable_v)**2))
            v_loss += mask[:, t]*(-lable_v*torch.log(torch.sigmoid(pre_v)+eps)-(1-lable_v)
                        * torch.log(1-torch.sigmoid(pre_v)+eps))
            # v_loss += (mask[:, t]*F.binary_cross_entropy_with_logits(pre_v, lable_v,  reduction="none"))
                                                                     #weight=torch.tensor([1-self.args.bias]).to(self.args.device), reduction="none"))
            if t==self.time_horizon:
                break

            def_pos = defender_his[:, t]  # [batch, num_defender]
            att_pos = attacker_his[:, t]  # [batch, 1]
            time = torch.tensor(t).to(self.args.device).repeat(
                batch_size).detach()  # [batch]
            def_leg_act = defender_legal_act[:, t+1]
            pre_def_act = self.pr_net.policy_net.batch_forward(
                def_pos, att_pos, time, def_leg_act)
            lable_def_act = defender_his_idx[:, t+1].flatten().long()
            def_mask = mask[:, t+1].repeat(1, self.num_defender).flatten()
            def_pre_loss += (def_mask*F.cross_entropy(pre_def_act,
                                                      lable_def_act, reduction="none"))

            att_leg_act = attacker_legal_act[:, t+1]
            pre_att_act = self.dy_net.policy_net.batch_forward(
                def_pos, att_pos, time, att_leg_act)
            lable_att_act = attacker_his_idx[:, t+1].flatten().long()
            att_pre_loss += (mask[:, t+1].flatten()*F.cross_entropy(
                pre_att_act, lable_att_act, reduction="none"))

        v_loss = (v_loss.mean()/length.mean())
        def_pre_loss = (def_pre_loss.mean() /
                        length.mean()/self.num_defender)
        att_pre_loss = (att_pre_loss.mean()/length.mean())

        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pr_net.value_net.parameters(), 1)
        self.v_opt.step()

        def_pre_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pr_net.policy_net.parameters(), 1)
        self.def_pre_opt.step()

        att_pre_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dy_net.policy_net.parameters(), 1)
        self.att_pre_opt.step()

        return v_loss, def_pre_loss, att_pre_loss

