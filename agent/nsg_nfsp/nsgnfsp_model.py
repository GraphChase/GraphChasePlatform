import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import dgl
from dgl.nn.pytorch import GATConv, GraphConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFENDER_CURRENT = 0
ATTACKER_CURRENT = 0.5
ATTACKER_VISITED = 0.1
EXIT = 1.0

class SeqEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, Map):
        super(SeqEncoder, self).__init__()
        self.num_nodes = in_dim+1
        self.fc1 = nn.Linear(self.num_nodes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.Map = Map

    def forward(self, attacker_history):
        batch_size = len(attacker_history)
        n_feature = torch.zeros((batch_size, self.num_nodes)).to(device)
        for i in range(batch_size):
            n_feature[i][self.Map.exits] = EXIT
            n_feature[i][attacker_history[i][:-1]] = ATTACKER_VISITED
            n_feature[i][attacker_history[i][-1]] = ATTACKER_CURRENT
        n_feature = F.relu(self.fc1(n_feature))
        out_feature = F.relu(self.fc2(n_feature))
        return out_feature


class StateEncoder(nn.Module):
    def __init__(self, num_nodes, time_horizon, embedding_size=16,
                 hidden_size=32, relevant_v_size=64, num_defender=None, seq_mode='mlp', Map=None, node_embedding=None):
        super(StateEncoder, self).__init__()

        assert seq_mode in ['mlp', 'gcn', 'gat', 'gru', 'cnn']
        if seq_mode == 'mlp':
            self.seq_encoder = SeqEncoder(
                num_nodes, hidden_size, hidden_size, Map)
        elif seq_mode == 'gcn':
            self.seq_encoder = Encoder_GCN(
                16, hidden_size, Map, if_attention=False)
        elif seq_mode == 'gat':
            self.seq_encoder = Encoder_GCN(
                16, hidden_size, Map, if_attention=True)
        elif seq_mode == 'cnn':
            self.seq_encoder = Gated_CNN(
                num_nodes, time_horizon, embedding_size, hidden_size, [3], True, node_embedding=node_embedding)
        elif seq_mode == 'gru':
            self.seq_encoder = Encoder_GRU(
                num_nodes, embedding_size, hidden_size, node_embedding=node_embedding)

        self.fc_t_1 = nn.Linear(1, 8)
        self.fc_t_2 = nn.Linear(8, 8)
        if node_embedding:
            self.embedding_p=node_embedding
        else:
            self.embedding_p = nn.Embedding(
                num_nodes+1, embedding_size, padding_idx=0)
        if num_defender:
            self.fc_p_1 = nn.Linear(embedding_size*num_defender, hidden_size)
            self.fc_p_2 = nn.Linear(hidden_size, hidden_size)
            self.fc1 = nn.Linear(hidden_size*2+8, relevant_v_size)
        else:
            self.fc1 = nn.Linear(hidden_size+8, relevant_v_size)

        self.num_defender = num_defender
        self.time_horizon = time_horizon

    def forward(self, states, actions=None):
        attacker_history, position = zip(*states)
        norm_t = [[(len(h)-1)/self.time_horizon] for h in attacker_history]

        if self.num_defender:
            if self.num_defender > 1:
                assert len(position[0]) == self.num_defender
            elif self.num_defender == 1:
                assert isinstance(position[0], int)

        h_feature = F.relu(self.seq_encoder(attacker_history))

        norm_t = torch.Tensor(norm_t).to(device)
        t_feature = F.relu(self.fc_t_1(norm_t))
        t_feature = F.relu(self.fc_t_2(t_feature))

        if self.num_defender:
            position = torch.LongTensor(position).to(device)
            p_feature = self.embedding_p(position)
            if self.num_defender > 1:
                p_feature = torch.flatten(p_feature, start_dim=1)

            p_feature = F.relu(self.fc_p_1(p_feature))
            p_feature = F.relu(self.fc_p_2(p_feature))
            if h_feature.dim() == 1:
                h_feature.unsqueeze_(0)
            feature = torch.cat((h_feature, p_feature, t_feature), dim=1)
        else:
            if h_feature.dim() == 1:
                h_feature.unsqueeze_(0)
            feature = torch.cat((h_feature, t_feature), dim=1)
        output = self.fc1(feature)
        return output  # shape: (batch_size, relevant_v_sie)
    
class Encoder_GCN(nn.Module):
    def __init__(self, hidden_dim, out_dim, Map, if_attention=True):
        super(Encoder_GCN, self).__init__()
        self.Map = Map
        self.graph = dgl.DGLGraph(Map.graph)
        self.if_attention = if_attention
        if self.if_attention:
            self.conv1 = GATConv(1, hidden_dim, 1)
            self.conv2 = GATConv(hidden_dim, out_dim, 1)
        else:
            self.conv1 = GraphConv(1, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, attacker_history):
        batch_size = len(attacker_history)
        graph = dgl.batch([self.graph]*batch_size)
        n_feature = torch.zeros((batch_size, self.Map.num_nodes)).to(device)
        for i in range(batch_size):
            n_feature[i][self.Map.exits] = EXIT
            n_feature[i][attacker_history[i][:-1]] = ATTACKER_VISITED
            n_feature[i][attacker_history[i][-1]] = ATTACKER_CURRENT

        n_feature = F.relu(self.conv1(graph, n_feature.view(-1, 1)))
        n_feature = F.relu(self.conv2(graph, n_feature))
     

        graph.ndata['h'] = n_feature
        out_feature = dgl.mean_nodes(graph, 'h')
        #out_feature = n_feature
        if self.if_attention:
            out_feature.squeeze_(1)
        return out_feature


class Gated_CNN(nn.Module):
    def __init__(self, num_nodes, time_horizon, embedding_size=32, num_kernels=64, kernel_size=[3], if_gate=True,node_embedding=None):
        # num_kernels is the dimension of features
        super(Gated_CNN, self).__init__()
        self.max_length = time_horizon+1  # to guide padding
        self.time_indicator = [[1 for idx in range(self.max_length)]]
        self.if_gate = if_gate
        if node_embedding:
            self.embedding=node_embedding
        else:
            self.embedding = nn.Embedding(
                num_nodes+1, embedding_size, padding_idx=0)  # +1 is for padding

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size]
        self.kernel_size = kernel_size
        self.conv = torch.nn.ModuleList()
        self.gate = torch.nn.ModuleList()
        for i in range(len(kernel_size)):
            self.conv.append(
                nn.Conv1d(embedding_size, num_kernels, kernel_size[i]))
            self.gate.append(
                nn.Conv1d(embedding_size, num_kernels, kernel_size[i]))
        self.to(device)
        # self.fc=nn.Linear(embedding_size,np.sum(num_kernels))

    def forward(self, inputs):
        # inputs:[[a,b,c,...],[c,d,a,...]]
        if not isinstance(inputs, list):
            inputs = list(inputs)
        inputs = inputs+self.time_indicator
        inputs = [torch.LongTensor(k).to(device) for k in inputs]
        inputs = pad_sequence(inputs, batch_first=True,
                              padding_value=0).detach()[:-1]
        inputs.requires_grad = False

        assert inputs.size(
            1) == self.max_length, 'max input sequence length must less than time horizon.'
        inputs = self.embedding(inputs).permute(0, 2, 1)

        outputs = []
        for i in range(len(self.kernel_size)):
            x = self.conv[i](inputs)
            if self.if_gate:
                gate = self.gate[i](inputs)
                outputs.append(x*F.sigmoid(gate))
            else:
                outputs.append(x)
        x = torch.cat(outputs, dim=2)
        x = F.max_pool1d(x, x.shape[-1])
        x = torch.squeeze(x)
        # x is the feature vector for the input sequence
        del inputs
        return x


class Encoder_GRU(nn.Module):
    def __init__(self, num_nodes, embedding_size, hidden_size, node_embedding=None):
        super(Encoder_GRU, self).__init__()
        if node_embedding:
            self.embedding=node_embedding
        else:
            self.embedding = nn.Embedding(
                num_nodes+1, embedding_size, padding_idx=0)  # +1 is for padding
        self.encoder = nn.GRU(embedding_size, hidden_size)
        self.to(device)

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = list(inputs)
        if not hasattr(self, '_flattened'):
            self.encoder.flatten_parameters()
            setattr(self, '_flattened', True)
        lengths = torch.tensor([len(n) for n in inputs],
                               dtype=torch.long, device=device)

        inputs = [torch.LongTensor(k).to(device) for k in inputs]
        inputs = pad_sequence(inputs, batch_first=True,
                              padding_value=0).detach()
        inputs.requires_grad = False

        inputs = self.embedding(inputs).permute(1, 0, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths, enforce_sorted=False)
        out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        idx = (lengths-1).view(-1, 1).expand(len(lengths),
                                             out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze()
        return out

class DRRN(nn.Module):
    def __init__(self, num_nodes, time_horizon, embedding_size=16,
                 hidden_size=32, relevant_v_size=64, num_defender=None, naive=False, seq_mode='mlp', out_mode='rl', Map=None, pre_embedding_path=None):
        super(DRRN, self).__init__()
        if pre_embedding_path:
            weight = torch.FloatTensor(np.load(pre_embedding_path))
            assert weight.size(1)==embedding_size
            self.embedding = nn.Embedding.from_pretrained(
                weight, freeze=True, padding_idx=0)
            self.state_encoder = StateEncoder(num_nodes, time_horizon, embedding_size,
                                              hidden_size, relevant_v_size, num_defender, seq_mode=seq_mode, Map=Map,node_embedding=self.embedding)
            #self.embedding_a = self.embedding
        else:
            self.state_encoder = StateEncoder(num_nodes, time_horizon, embedding_size,
                                            hidden_size, relevant_v_size, num_defender, seq_mode=seq_mode, Map=Map, node_embedding=None)
            # self.embedding_a=self.state_encoder.embedding_p
        self.embedding_a = nn.Embedding(
            num_nodes+1, embedding_size, padding_idx=0)
        if num_defender:
            self.fc_a_1 = nn.Linear(embedding_size*num_defender, hidden_size)
            self.fc_a_2 = nn.Linear(hidden_size, relevant_v_size)
        else:
            self.fc_a_1 = nn.Linear(embedding_size, hidden_size)
            self.fc_a_2 = nn.Linear(hidden_size, relevant_v_size)
        if not naive:
            self.fc1 = nn.Linear(relevant_v_size*2, relevant_v_size)
            self.fc2 = nn.Linear(relevant_v_size, 1)

        self.num_nodes = num_nodes
        self.num_defender = num_defender
        self.naive = naive
        assert out_mode in ['sl', 'rl']
        self.out_mode = out_mode

        self.init_weights()
        self.to(device)

    def forward(self, states, actions):
        s_feature = self.state_encoder(states)
        a_in = [torch.LongTensor(k).to(device) for k in actions]
        a_in = pad_sequence(a_in, batch_first=True,
                            padding_value=0).detach()
        a_in.requires_grad = False  # shape:(batch, max_num_actions)

        a_feature = self.embedding_a(a_in)
        if self.num_defender > 1:
            a_feature = torch.flatten(a_feature, start_dim=2)
        a_feature = F.relu(self.fc_a_1(a_feature))
        # shape:(batch, max_num_actions, relevant_v_size)
        a_feature = self.fc_a_2(a_feature)
        if not self.naive:
            s_feature = s_feature.unsqueeze(1).repeat(1, a_feature.size(1), 1)

            # shape(batch, max_num_actions, full_features)
            feature = torch.cat((s_feature, a_feature), dim=2)
            output = F.relu(self.fc1(feature))
            output = self.fc2(output).squeeze(2)
        else:
            output = torch.matmul(a_feature, s_feature.unsqueeze(
                2)).squeeze(2)

        if output.size(0) > 1:
            mask = [torch.zeros(len(k)).to(device) for k in actions]
            mask = pad_sequence(mask, batch_first=True,
                                padding_value=-1e9).detach()
            mask.requires_grad = False
            output += mask
        if self.out_mode == 'sl':
            # delete tensor
            del s_feature, a_in, a_feature
            if not self.naive:
                del feature
            if output.size(0) > 1:
                del mask

            return output.squeeze(0)
        elif self.out_mode == 'rl':
            optimal_q, idx = torch.max(output, dim=1)
            optimal_a = a_in[torch.arange(a_in.size(0)), idx]

            # delete tensor
            del s_feature, a_in, a_feature
            if not self.naive:
                del feature
            if output.size(0) > 1:
                del mask       
                     
            return optimal_a, optimal_q, output.squeeze(0)
        else:
            ValueError

    def init_weights(self):
        for param in self.parameters():
            nn.init.normal_(param, 0.0, 0.1)

class AA_MA(nn.Module):
    def __init__(self, max_actions, num_nodes, time_horizon, embedding_size=16,
                 hidden_size=32, relevant_v_size=64, num_defender=None, seq_mode='mlp', Map=None, pre_embedding_path=None):
        super(AA_MA, self).__init__()
        if pre_embedding_path:
            weight = torch.FloatTensor(np.load(pre_embedding_path))
            assert weight.size(1)==embedding_size
            self.embedding = nn.Embedding.from_pretrained(
                weight, freeze=True, padding_idx=0)
            self.state_encoder = StateEncoder(num_nodes, time_horizon, embedding_size,
                                              hidden_size, relevant_v_size, num_defender, seq_mode=seq_mode, Map=Map,node_embedding=self.embedding)
        else:
            self.state_encoder = StateEncoder(num_nodes, time_horizon, embedding_size,
                                            hidden_size, relevant_v_size, num_defender, seq_mode=seq_mode, Map=Map,node_embedding=None)
        self.fc = nn.Linear(relevant_v_size, max_actions)

        self.num_nodes = num_nodes
        self.num_defender = num_defender
        self.time_horizon = time_horizon

        self.init_weights()
        self.to(device)

    def forward(self, states, actions=None):
        feature = F.relu(self.state_encoder(states))
        output = self.fc(feature)
        return output.squeeze(0)

    def init_weights(self):
        for param in self.parameters():
            nn.init.normal_(param, 0.0, 0.1)

class Defender_MA_DQN(nn.Module):
    def __init__(self, max_actions, num_nodes, time_horizon, embedding_size=32, num_kernels=64, kernel_size=[3, 8],
                 hidden_size=64, relevant_v_size=64, if_gate=True, if_gru=False, num_defender=1):
        super(Defender_MA_DQN, self).__init__()
        if not if_gru:
            self.seq_encoder = Gated_CNN(
                num_nodes, time_horizon, embedding_size, num_kernels, kernel_size, if_gate)
        else:
            self.seq_encoder = Encoder_GRU(
                num_nodes, embedding_size, num_kernels)
        self.embedding_p = self.seq_encoder.embedding
        # nn.Embedding(
        #     num_nodes, embedding_size_p, padding_idx=0)
        self.fc_p_1 = nn.Linear(embedding_size*num_defender, hidden_size)
        self.fc_p_2 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(num_kernels+hidden_size, relevant_v_size)
        self.fc2 = nn.Linear(relevant_v_size, max_actions)

        self.init_weights()
        self.to(device)
        self.num_defender = num_defender
        self.num_nodes = num_nodes

    def forward(self, states, actions=None):
        attacker_history, position = zip(*states)
        if self.num_defender > 1:
            assert len(position[0]) == self.num_defender
        elif self.num_defender == 1:
            assert isinstance(position[0], int)

        position = torch.LongTensor(position).to(device)

        # shape: (batch, num_kernels)
        h_feature = self.seq_encoder(attacker_history)
        #p_feature = self.embedding_p(position)
        p_feature = self.embedding_p(position)
        if self.num_defender > 1:
            p_feature = torch.flatten(p_feature, start_dim=1)
        # shape: (batch, hidden_size)
        p_feature = F.relu(self.fc_p_1(p_feature))
        p_feature = F.relu(self.fc_p_2(p_feature))
        if h_feature.dim() == 2:
            feature = torch.cat((h_feature, p_feature), dim=1)
        else:
            feature = torch.cat((h_feature, p_feature.squeeze()), dim=0)

        feature = F.relu(self.fc1(feature))
        q_val = self.fc2(feature)
        return q_val

    def init_weights(self):
        for param in self.parameters():
            nn.init.normal_(param, 0.0, 0.1)

class Attacker_MA_DQN(nn.Module):
    def __init__(self, max_actions, num_nodes, time_horizon, embedding_size=32, num_kernels=64, kernel_size=[3, 8],
                 hidden_size=64, relevant_v_size=64, if_gate=True, if_gru=False):
        super(Attacker_MA_DQN, self).__init__()
        if not if_gru:
            self.seq_encoder = Gated_CNN(
                num_nodes, time_horizon, embedding_size, num_kernels, kernel_size, if_gate)
        else:
            self.seq_encoder = Encoder_GRU(
                num_nodes, embedding_size, num_kernels)
        self.fc1 = nn.Linear(num_kernels, relevant_v_size)
        self.fc2 = nn.Linear(relevant_v_size, max_actions)
        self.to(device)
        self.num_defender = None

    def forward(self, states, actions=None):
        feature = self.seq_encoder(states)
        feature = F.relu(self.fc1(feature))
        q_val = self.fc2(feature)
        return q_val