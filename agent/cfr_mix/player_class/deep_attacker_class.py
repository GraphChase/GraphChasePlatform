import copy
import random
from abc import ABC
import numpy as np
import torch
import torch.utils.data as Data


class RegretNet(torch.nn.Module, ABC):
    def __init__(self, input_dim, input_dim_2, hidden_dim, hidden_dim_2, output_dim):
        super(RegretNet, self).__init__()
        self.state_linear = torch.nn.Linear(input_dim, hidden_dim)
        self.action_linear = torch.nn.Linear(input_dim_2, hidden_dim_2)
        self.linear_hidden = torch.nn.Linear(hidden_dim + hidden_dim_2, hidden_dim)
        # self.linear_hidden_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, action):
        x = torch.tanh(self.state_linear(x))
        action = torch.tanh(self.action_linear(action))
        x = torch.cat((x, action), dim=1)
        x = torch.tanh(self.linear_hidden(x))
        # x = torch.tanh(self.linear_hidden_2(x))
        x = torch.abs(self.out(x))
        return x


class StrategyNet(torch.nn.Module, ABC):
    def __init__(self, input_dim, input_dim_2, hidden_dim, hidden_dim_2, output_dim):
        super(StrategyNet, self).__init__()
        self.state_linear = torch.nn.Linear(input_dim, hidden_dim)
        self.action_linear = torch.nn.Linear(input_dim_2, hidden_dim_2)
        self.linear_hidden = torch.nn.Linear(hidden_dim+hidden_dim_2, hidden_dim)
        # self.linear_hidden_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, action):
        x = torch.tanh(self.state_linear(x))
        action = torch.tanh(self.action_linear(action))
        x = torch.cat((x, action), dim=1)
        x = torch.tanh(self.linear_hidden(x))
        # x = torch.tanh(self.linear_hidden_2(x))
        x = torch.abs(self.out(x))
        return x


class Attacker(object):
    def __init__(self, time_horizon, hidden_dim, reservoir=False, argumentation=False):
        self.player_id = 0
        self.input_dim = time_horizon
        self.regret_training_data = []
        self.strategy_training_data = []
        self.argumentation_flag = argumentation
        self.reservoir_flag = reservoir
        self.regret_model = RegretNet(self.input_dim, 1, hidden_dim, 4, 1)
        self.strategy_model = StrategyNet(self.input_dim, 1, hidden_dim, 4, 1)

        if self.reservoir_flag:
            self.memory_size = 10000
            self.regret_data_number = 0
            self.strategy_data_number = 0
        if torch.cuda.is_available():
            # self.regret_model = torch.nn.DataParallel(self.regret_model)
            self.regret_model = self.regret_model.cuda()
            # self.strategy_model = torch.nn.DataParallel(self.strategy_model)
            self.strategy_model = self.strategy_model.cuda()

    def info_to_state(self, info):
        state = copy.deepcopy(info.key)
        state = np.pad(state, (0, self.input_dim - len(state)))
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
        x = self.info_to_state(info)

        for i, action in enumerate(available_action):
            if self.reservoir_flag:
                self.reservoir_record('regret', [x, [action], [regret[i]]])
            else:
                self.regret_training_data.append([x, [action], [regret[i]]])

            if self.argumentation_flag:
                temp = random.random() - 0.5
                x_temp = [i + temp for i in list(x)]
                if self.reservoir_flag:
                    self.reservoir_record('regret', [np.array(x_temp), [action + temp], [regret[i]]])
                else:
                    self.regret_training_data.append([np.array(x_temp), [action + temp], [regret[i]]])

    def strategy_memory_add(self, info, available_action, strategy):
        x = self.info_to_state(info)
        for i, action in enumerate(available_action):
            if self.reservoir_flag:
                self.reservoir_record('strategy', [x, [action], [strategy[i]]])
            else:
                self.strategy_training_data.append([x, [action], [strategy[i]]])

    def get_strategy(self, info, time):
        if time == 1:
            strategy = np.zeros(info.action_number) + 1./float(info.action_number)
        else:
            x = self.info_to_state(info)
            state = [x] * info.action_number
            action_list = np.array(info.available_actions).reshape((-1, 1))
            if torch.cuda.is_available():
                state = torch.from_numpy(np.array(state)).float().cuda()
                action_list = torch.Tensor(action_list).cuda()
                # prediction = self.regret_model.module(state, action_list).cpu().squeeze(1).detach().numpy()
                prediction = self.regret_model(state, action_list).cpu().squeeze(1).detach().numpy()
            else:
                state = torch.Tensor(state)
                action_list = torch.Tensor(action_list)
                prediction = self.regret_model(state, action_list).squeeze(1).detach().numpy()
            strategy = regret_to_strategy(prediction, info.action_number)
        return strategy

    def get_average_strategy(self, info):
        x = self.info_to_state(info)
        state = [x] * info.action_number
        action_list = np.array(info.available_actions).reshape((-1, 1))
        if torch.cuda.is_available():
            state = torch.from_numpy(np.array(state)).float().cuda()
            action_list = torch.Tensor(action_list).cuda()
            # prediction = self.strategy_model.module(state, action_list).cpu().squeeze(1).detach().numpy()
            prediction = self.strategy_model(state, action_list).cpu().squeeze(1).detach().numpy()
        else:
            state = torch.Tensor(state)
            action_list = torch.Tensor(action_list)
            prediction = self.strategy_model(state, action_list).squeeze(1).detach().numpy()
        strategy = regret_to_strategy(prediction, info.action_number)
        return strategy

    def train_regret_network(self, lr, time, train_epoch=2000, batch_size=64):
        print("number of attacker's regret training data", len(self.regret_training_data))
        train_x = np.array([i[0] for i in self.regret_training_data[:]])
        train_action = np.array([i[1] for i in self.regret_training_data[:]])
        train_y = np.array([i[2] for i in self.regret_training_data[:]])

        optimizer = torch.optim.SGD(self.regret_model.parameters(), lr)
        criterion = torch.nn.MSELoss()

        if torch.cuda.is_available():
            train_x = torch.tensor(train_x).cuda().float()
            train_action = torch.tensor(train_action).cuda().float()
            train_y = torch.tensor(train_y).cuda().float()
            criterion = criterion.cuda()
        else:
            train_x = torch.tensor(train_x).float()
            train_action = torch.tensor(train_action).float()
            train_y = torch.tensor(train_y).float()

        torch_dataset = Data.TensorDataset(train_x, train_action, train_y)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(train_epoch):
            epoch_total_loss = 0

            for step, (batch_x, batch_action, batch_y) in enumerate(loader):
                predict_y = self.regret_model(batch_x, batch_action)
                batch_loss = criterion(predict_y, batch_y)
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.regret_model.parameters(), 1)
                optimizer.step()
                epoch_total_loss += batch_loss.data

            if epoch == 0 or (epoch + 1) % 1000 == 0:
                print("Training attacker regret model, ", "Epoch", epoch, ", Training epoch loss", epoch_total_loss)

    def train_strategy_network(self, lr, time, train_epoch=2000, batch_size=64):
        optimizer = torch.optim.SGD(self.strategy_model.parameters(), lr)
        criterion = torch.nn.MSELoss()
        print("number of attacker's strategy training data", len(self.strategy_training_data))
        train_x = np.array([i[0] for i in self.strategy_training_data[:]])
        train_action = np.array([i[1] for i in self.strategy_training_data[:]])
        train_y = np.array([i[2] for i in self.strategy_training_data[:]])

        if torch.cuda.is_available():
            train_x = torch.tensor(train_x).cuda().float()
            train_action = torch.tensor(train_action).cuda().float()
            train_y = torch.tensor(train_y).cuda().float()
            criterion = criterion.cuda()
        else:
            train_x = torch.tensor(train_x).float()
            train_action = torch.tensor(train_action).float()
            train_y = torch.tensor(train_y).float()

        torch_dataset = Data.TensorDataset(train_x, train_action, train_y)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
        for epoch in range(train_epoch):
            epoch_total_loss = 0
            for step, (batch_x, batch_action, batch_y) in enumerate(loader):
                predict_y = self.strategy_model(batch_x, batch_action)
                batch_loss = criterion(predict_y, batch_y)
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.strategy_model.parameters(), 1)
                optimizer.step()
                epoch_total_loss += batch_loss.data

            if epoch == 0 or (epoch + 1) % 1000 == 0:
                print("Training attacker strategy model, ", "Epoch", epoch, ", Training epoch loss", epoch_total_loss)


def regret_to_strategy(regret, action_number):
    strategy = np.array(regret)
    strategy = np.where(strategy > 0, strategy, 0)
    total = float(sum(strategy))
    if total > 0:
        strategy = strategy / total
    else:
        max_index = list(strategy).index(max(strategy))
        strategy = np.zeros(action_number)
        strategy[max_index] = 1
        # strategy = np.zeros(action_number) + 1. / float(action_number)
    return strategy
