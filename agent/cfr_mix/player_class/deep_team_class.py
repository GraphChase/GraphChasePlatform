import copy
import random
from abc import ABC
import numpy as np
import torch
import torch.utils.data as Data


class AgentNet(torch.nn.Module, ABC):
    def __init__(self, input_dim, hidden_dim, hidden_dim_2, output_dim):
        super(AgentNet, self).__init__()
        self.state_linear = torch.nn.Linear(input_dim, hidden_dim)
        self.action_linear = torch.nn.Linear(1, hidden_dim_2)
        self.observation_linear = torch.nn.Linear(1, hidden_dim_2)
        self.linear_hidden = torch.nn.Linear(hidden_dim + 2 * hidden_dim_2, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, o, action):
        x = torch.tanh(self.state_linear(x))
        o = torch.tanh(self.observation_linear(o))
        a = torch.tanh(self.action_linear(action))
        x = torch.cat((x, o, a), dim=1)
        x = torch.tanh(self.linear_hidden(x))
        x = torch.abs(self.out(x))
        return x


class DefenderRegretMixNet(torch.nn.Module):
    def __init__(self, obs_input_dim, player_number, hidden_dim):
        super(DefenderRegretMixNet, self).__init__()
        self.n_player = player_number
        self.agent_model = AgentNet(obs_input_dim, hidden_dim, 4, 1)

    def forward(self, obs_1_list, obs_2_list, action_list):
        agent_regret_list = []
        for player in range(self.n_player):
            obs_1 = obs_1_list[:, player, :]
            obs_2 = obs_2_list[:, player, :]
            action = action_list[:, player, :]
            agent_regret = self.agent_model(obs_1, obs_2, action)
            agent_regret_list.append(agent_regret)

        total_regret_list = agent_regret_list[0]
        for agent_regret in agent_regret_list[1:]:
            total_regret_list = total_regret_list * agent_regret

        return total_regret_list


class AgentStrategyNet(torch.nn.Module, ABC):
    def __init__(self, input_dim, hidden_dim, hidden_dim_2, output_dim):
        super(AgentStrategyNet, self).__init__()
        self.state_linear = torch.nn.Linear(input_dim, hidden_dim)
        self.action_linear = torch.nn.Linear(1, hidden_dim_2)
        self.observation_linear = torch.nn.Linear(1, hidden_dim_2)
        self.linear_hidden = torch.nn.Linear(hidden_dim + 2 * hidden_dim_2, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, o, action):
        x = torch.tanh(self.state_linear(x))
        o = torch.tanh(self.observation_linear(o))
        a = torch.tanh(self.action_linear(action))
        x = torch.cat((x, o, a), dim=1)
        x = torch.tanh(self.linear_hidden(x))
        x = torch.abs(self.out(x))
        return x


class DefenderGroup(object):
    def __init__(self, time_horizon, player_number, hidden_dim, reservoir=False, argumentation=False):
        self.player_id = 1
        self.input_dim = time_horizon * (player_number + 1)
        self.time_horizon = time_horizon
        self.player_number = player_number
        self.regret_training_data = []
        self.strategy_training_data = []
        self.argumentation_flag = argumentation
        self.reservoir_flag = reservoir
        self.regret_model = DefenderRegretMixNet(self.input_dim, player_number, hidden_dim)
        self.strategy_model = AgentStrategyNet(self.input_dim, hidden_dim, 4, 1)

        if self.reservoir_flag:
            self.memory_size = 10000
            self.regret_data_number = 0
            self.strategy_data_number = 0
        if torch.cuda.is_available():
            self.regret_model = torch.nn.DataParallel(self.regret_model)
            self.regret_model = self.regret_model.cuda()
            self.strategy_model = torch.nn.DataParallel(self.strategy_model)
            self.strategy_model = self.strategy_model.cuda()

    def info_to_state(self, info):
        info_embedding_1 = copy.deepcopy(info.key[0])
        info_embedding_1 = np.pad(info_embedding_1, (0, self.time_horizon - len(info_embedding_1)))
        info_embedding_2 = copy.deepcopy(info.key[1])
        info_embedding_2 = np.array(info_embedding_2).reshape(len(info_embedding_2[0]) * len(info_embedding_2))
        info_embedding = np.concatenate((info_embedding_1, info_embedding_2), axis=0)
        state = np.pad(info_embedding, (0, self.input_dim - len(info_embedding)))
        return state

    def reservoir_record(self, type, data):
        if type == 'regret':
            if len(self.regret_training_data) < self.memory_size:
                self.regret_training_data.append(data)
            else:
                m = random.randint(1, self.regret_data_number)
                if m <= self.memory_size:
                    self.regret_training_data[m - 1] = data
            self.regret_data_number += 1
        else:
            if len(self.strategy_training_data) < self.memory_size:
                self.strategy_training_data.append(data)
            else:
                m = random.randint(1, self.strategy_data_number)
                if m <= self.memory_size:
                    self.strategy_training_data[m - 1] = data
            self.strategy_data_number += 1

    def regret_memory_add(self, info, available_action, regret):
        state = self.info_to_state(info)

        current_location = info.key[1][-1]
        obs_1_list = []
        obs_2_list = []
        for j, l in enumerate(current_location):
            temp_1 = list(copy.deepcopy(state))
            temp_2 = [l]
            obs_1_list.append(temp_1)
            obs_2_list.append(temp_2)

        for idx, action in enumerate(available_action):
            action_list = []
            for i, a in enumerate(action):
                action_list.append([a])
            if self.reservoir_flag:
                self.reservoir_record('regret', [np.array(obs_1_list), np.array(obs_2_list), action_list, [regret[idx]]])
            else:
                self.regret_training_data.append([np.array(obs_1_list), np.array(obs_2_list), action_list, [regret[idx]]])

            if self.argumentation_flag:
                temp = random.random() - 0.5
                obs_1_list_temp, obs_2_list_temp, action_list_temp = [], [], []
                for i in range(len(obs_1_list)):
                    obs_1_list_temp.append([t + temp for t in obs_1_list[i]])
                    obs_2_list.append([t + temp for t in obs_2_list[i]])
                    action_list.append([a + temp for a in action_list[i]])
                if self.reservoir_flag:
                    self.reservoir_record('regret', [np.array(obs_1_list_temp), np.array(obs_2_list_temp), action_list_temp, [regret[idx]]])
                else:
                    self.regret_training_data.append([np.array(obs_1_list_temp), np.array(obs_2_list_temp), action_list_temp, [regret[idx]]])

    def strategy_memory_add(self, info, agent_location, agent_avail_action, strategy):
        x = self.info_to_state(info)
        for i, action in enumerate(agent_avail_action):
            if self.reservoir_flag:
                self.reservoir_record('strategy', [x, np.array([agent_location]), [action], [strategy[i]]])
            else:
                self.strategy_training_data.append([x, np.array([agent_location]), [action], [strategy[i]]])

    def get_strategy(self, info, agent_location, agent_avail_action, time):
        if time == 1:
            strategy = []
            for i in range(len(agent_location)):
                strategy.append(np.zeros(len(agent_avail_action[i])) + 1. / float(len(agent_avail_action[i])))
        else:
            s = self.info_to_state(info)
            o, state, ac = [], [], []
            for i, action in enumerate(agent_avail_action):
                for a in action:
                    state.append(s)
                    o.append([agent_location[i]])
                    ac.append([a])

            if torch.cuda.is_available():
                state = torch.tensor(np.array(state)).cuda().float()
                obs = torch.tensor(np.array(o)).cuda().float()
                action = torch.tensor(np.array(ac)).cuda().float()
                prediction = self.regret_model.module.agent_model(state, obs, action).cpu().squeeze(1).detach().numpy()
            else:
                state = torch.tensor(np.array(state)).float()
                obs = torch.tensor(np.array(o)).float()
                action = torch.tensor(np.array(ac)).float()
                prediction = self.regret_model.agent_model(state, obs, action).squeeze(1).detach().numpy()
            strategy = self.regret_to_strategy(prediction, agent_avail_action)
        return strategy

    def get_average_strategy(self, info, agent_location, agent_avail_action_set):
        s = self.info_to_state(info)
        o, state, ac = [], [], []
        for i, action in enumerate(agent_avail_action_set):
            for a in action:
                state.append(s)
                o.append([agent_location[i]])
                ac.append([a])

        if torch.cuda.is_available():
            state = torch.tensor(np.array(state)).cuda().float()
            obs = torch.tensor(np.array(o)).cuda().float()
            action = torch.tensor(np.array(ac)).cuda().float()
            prediction = self.strategy_model.module(state, obs, action).cpu().squeeze(1).detach().numpy()
        else:
            state = torch.tensor(np.array(state)).float()
            obs = torch.tensor(np.array(o)).float()
            action = torch.tensor(np.array(ac)).float()
            prediction = self.strategy_model(state, obs, action).squeeze(1).detach().numpy()
        strategy = self.regret_to_strategy(prediction, agent_avail_action_set)
        return strategy

    def regret_to_strategy(self, regret, avail_actions):
        regret_list = np.array(regret)
        strategy_list = []
        index = 0
        for i, _ in enumerate(avail_actions):
            strategy = regret_list[index:len(avail_actions[i])+index]
            index += len(avail_actions[i])
            strategy = np.where(strategy > 0, strategy, 0)
            total = float(sum(strategy))
            if total > 0:
                strategy_list.append(strategy / total)
            else:
                max_index = list(strategy).index(max(strategy))
                strategy = np.zeros(len(avail_actions[i]))
                strategy[max_index] = 1
                strategy_list.append(strategy)
                # strategy_list.append(np.zeros(len(avail_actions[i])) + 1. / float(len(avail_actions[i])))
        return strategy_list

    def train_regret_network(self, lr, time, train_epoch=2000, batch_size=64):
        optimizer = torch.optim.Adam(self.regret_model.parameters(), lr)
        criterion = torch.nn.MSELoss()
        print("number of regret training data", len(self.regret_training_data))
        train_obs_1 = np.array([i[0] for i in self.regret_training_data[:]])
        train_obs_2 = np.array([i[1] for i in self.regret_training_data[:]])
        train_action = np.array([i[2] for i in self.regret_training_data[:]])
        train_regret = np.array([i[3] for i in self.regret_training_data[:]])

        if torch.cuda.is_available():
            train_obs_1 = torch.tensor(train_obs_1).cuda().float()
            train_obs_2 = torch.tensor(train_obs_2).cuda().float()
            train_action = torch.tensor(train_action).cuda().float()
            train_regret = torch.tensor(train_regret).cuda().float()
            criterion = criterion.cuda()
        else:
            train_obs_1 = torch.tensor(train_obs_1).float()
            train_obs_2 = torch.tensor(train_obs_2).float()
            train_action = torch.tensor(train_action).float()
            train_regret = torch.tensor(train_regret).float()

        torch_dataset = Data.TensorDataset(train_obs_1, train_obs_2, train_action, train_regret)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(train_epoch):
            epoch_total_loss = 0
            for step, (batch_obs_1, batch_obs_2, batch_action, batch_y) in enumerate(loader):
                predict_y = self.regret_model(batch_obs_1, batch_obs_2, batch_action)
                batch_loss = criterion(predict_y, batch_y)
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.regret_model.parameters(), 1)
                optimizer.step()
                epoch_total_loss += batch_loss.data

            if epoch == 0 or (epoch + 1) % 1000 == 0:
                print("Training defender mixed regret model, ", "Epoch", epoch, ", Training epoch loss", epoch_total_loss)

    def train_strategy_network(self, lr, time, train_epoch=2000, batch_size=32):
        optimizer = torch.optim.Adam(self.strategy_model.parameters(), lr)
        criterion = torch.nn.MSELoss()

        print("number of strategy training data", len(self.strategy_training_data))
        train_x1 = np.array([i[0] for i in self.strategy_training_data[:]])
        train_x2 = np.array([i[1] for i in self.strategy_training_data[:]])
        train_action = np.array([i[2] for i in self.strategy_training_data[:]])
        train_y = np.array([i[3] for i in self.strategy_training_data[:]])

        if torch.cuda.is_available():
            train_x1 = torch.tensor(train_x1).cuda().float()
            train_x2 = torch.tensor(train_x2).cuda().float()
            train_action = torch.tensor(train_action).cuda().float()
            train_y = torch.tensor(train_y).cuda().float()
            criterion = criterion.cuda()
        else:
            train_x1 = torch.tensor(train_x1).float()
            train_x2 = torch.tensor(train_x2).float()
            train_action = torch.tensor(train_action).float()
            train_y = torch.tensor(train_y).float()

        torch_dataset = Data.TensorDataset(train_x1, train_x2, train_action, train_y)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(train_epoch):
            epoch_total_loss = 0
            for step, (batch_x1, batch_x2, batch_action, batch_y) in enumerate(loader):
                predict_y = self.strategy_model(batch_x1, batch_x2, batch_action)
                batch_loss = criterion(predict_y, batch_y)
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.strategy_model.parameters(), 1)
                optimizer.step()
                epoch_total_loss += batch_loss.data
            if epoch == 0 or (epoch + 1) % 1000 == 0:
                print("Training defender strategy model, ", "Epoch", epoch, ", Training epoch loss", epoch_total_loss)
