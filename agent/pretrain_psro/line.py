import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias

def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

def line_loss(y_true, y_pred):
    return -torch.mean(torch.log(torch.sigmoid(y_true * y_pred)))

def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx

class NodeEmbeddingModel(nn.Module):
    def __init__(self, numNodes: int, embedding_size: int, order='first'):
        super(NodeEmbeddingModel, self).__init__()
        
        self.order = order
        self.first_emb = nn.Embedding(numNodes, embedding_size)
        self.second_emb = nn.Embedding(numNodes, embedding_size)
        self.context_emb = nn.Embedding(numNodes, embedding_size)
    
    def forward(self, v_i, v_j):
        v_i_emb = self.first_emb(v_i)
        v_j_emb = self.first_emb(v_j)
        
        v_i_emb_second = self.second_emb(v_i)
        v_j_context_emb = self.context_emb(v_j)
        
        first_order = torch.sum(v_i_emb * v_j_emb, dim=-1, keepdim=True)
        second_order = torch.sum(v_i_emb_second * v_j_context_emb, dim=-1, keepdim=True)
        
        if self.order == 'first':
            return first_order
        elif self.order == 'second':
            return second_order # shape: [batch_size, 1]
        else:
            return torch.cat((first_order, second_order), dim=-1) # shape: [batch_size, 2]



class LINE(object):
    def __init__(self, graph, information_proximity_matrix, embedding_size, batch_size, epochs,
                 negative_ratio=5, order="all", device="cuda") -> None:
        
        self.graph = graph
        self.information_proximity_matrix = information_proximity_matrix
        self.emb_size = embedding_size
        self.epochs = epochs
        self.negative_ratio = negative_ratio
        self.batch_size = batch_size
        self.order = order
        self.device = device

        self.samples_per_epoch = self.graph.number_of_edges() * (1+self.negative_ratio)
        self.steps_per_epoch = ((self.samples_per_epoch - 1) // self.batch_size + 1)

        self.n2v = NodeEmbeddingModel(self.graph.number_of_nodes(), self.emb_size, self.order).to(self.device)
        self.n2v_optimizer = torch.optim.Adam(self.n2v.parameters())

        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        # list, dict

        # get sampling table for nodes and edges
        self._gen_sampling_table()


    def train(self, ) -> dict:
        for i in range(self.epochs):
            iter_loss = self.train_iteration()
            if self.order == 'all':
                print(f"Epoch {i+1}/{self.epochs}, "
                    f"loss: {np.mean(np.array(iter_loss['first_order'])) + np.mean(np.array(iter_loss['second_order']))}, "
                    f"first_order_loss: {np.mean(np.array(iter_loss['first_order']))}, "
                    f"second_order_loss: {np.mean(np.array(iter_loss['second_order']))}")
            elif self.order == 'first':
                print(f"Epoch {i+1}/{self.epochs}, "
                    f"first order loss: {np.mean(np.array(iter_loss['first_order']))}") 
            elif self.order == 'second':
                print(f"Epoch {i+1}/{self.epochs}, "
                    f"second order loss: {np.mean(np.array(iter_loss['second_order']))}")                            
        return self._get_embeddings()

    def train_iteration(self,):
        self.n2v.train()
        loss_iteration = {"first_order":[], "second_order":[]}
        step_cnt = 0
        for h_nodes, t_nodes, sign_nodes in self._batch_iter(self.node2idx):
            step_cnt += 1
            predictions = self.n2v(h_nodes.to(self.device),
                                   t_nodes.to(self.device))
            # loss = line_loss(sign_nodes.to(self.device), predictions)
            if self.order == 'all':
                first_order_loss = line_loss(sign_nodes[:,0].to(self.device), predictions[:,0])
                second_order_loss = line_loss(sign_nodes[:,1].to(self.device), predictions[:,1])
                loss = first_order_loss + second_order_loss
            elif self.order == 'first':
                first_order_loss = line_loss(sign_nodes.to(self.device), predictions[:,0])
                loss = first_order_loss
            elif self.order == 'second':
                second_order_loss = line_loss(sign_nodes.to(self.device), predictions[:,0])
                loss = second_order_loss

            self.n2v_optimizer.zero_grad()
            loss.backward()
            self.n2v_optimizer.step()

            # loss_iteration.append(loss.detach().cpu().item())
            if self.order == 'all':
                loss_iteration["first_order"].append(first_order_loss.detach().cpu().item())
                loss_iteration["second_order"].append(second_order_loss.detach().cpu().item())
            elif self.order == 'first':
                loss_iteration["first_order"].append(first_order_loss.detach().cpu().item())
            elif self.order == 'second':
                loss_iteration["second_order"].append(second_order_loss.detach().cpu().item())

            if step_cnt == self.steps_per_epoch:
                break

        return loss_iteration

    def _gen_sampling_table(self):
        # create sampling table for vertex
        power = 0.75
        numNodes = self.graph.number_of_nodes()
        node_degree = np.zeros((numNodes,))  # out degree

        # create sampling table for edge
        numEdges = self.graph.number_of_edges()
        edge_weight = np.zeros((numEdges, ))
        
        node2idx = self.node2idx
        for idx, edge in enumerate(self.graph.edges()):
            # calculate sampling nodes prob
            node_degree[node2idx[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight', 1.0)
            # calculate sampling edges prob
            edge_weight[idx] = self.graph[edge[0]][edge[1]].get('weight', 1.0)

        # get nodes sampling table
        total_sum = np.sum(np.power(node_degree, power))
        norm_prob = np.power(node_degree, power) / total_sum
        self.node_accept, self.node_alias = create_alias_table(norm_prob)
        
        # get edges sampling table
        total_sum = np.sum(edge_weight)
        norm_prob = edge_weight * numEdges / total_sum
        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)  


    def _batch_iter(self, node2idx):

        edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph.edges()]

        data_size = self.graph.number_of_edges()
        # shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffle_indices = np.arange(data_size)
        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        count = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, data_size)
        while True:
            if mod == 0:
                h = []
                t = []
                w = []
                for i in range(start_index, end_index):
                    if random.random() >= self.edge_accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                    w.append(self.information_proximity_matrix[cur_h][cur_t])
                sign = np.ones(len(h))
                sign = np.array([sign[i] + w[i] for i in range(len(h))])
                # sign = np.array([w[i] for i in range(len(h))])
            else:
                sign = np.ones(len(h))*-1
                t = []
                w = []
                for i in range(len(h)):
                    t.append(alias_sample(self.node_accept, self.node_alias))
                    w.append(self.information_proximity_matrix[h[i]][t[i]])
                # sign = np.array([sign[i] + w[i] for i in range(len(h))])

            if self.order == 'all':
                # yield ([np.array(h), np.array(t)], [sign, sign])
                yield (torch.tensor(np.array(h)),
                       torch.tensor(np.array(t)),
                       torch.cat((torch.tensor(sign).view(len(h), 1), torch.tensor(sign).view(len(h), 1)), dim=-1)
                       )
            else:
                # yield ([np.array(h), np.array(t)], [sign])
                yield (torch.tensor(np.array(h)).view(len(h), 1),
                       torch.tensor(np.array(t)).view(len(h), 1),
                       torch.tensor(sign).view(len(h), 1)
                       )

            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

            if start_index >= data_size:
                count += 1
                mod = 0
                h = []
                # shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffle_indices = np.arange(data_size)
                start_index = 0
                end_index = min(start_index + self.batch_size, data_size)

    def _get_embeddings(self,):
        self._embeddings = {}
        
        if self.order == 'first':
            embeddings = next(self.n2v.first_emb.parameters()).detach().cpu().numpy()
        elif self.order == 'second':
            embeddings = next(self.n2v.second_emb.parameters()).detach().cpu().numpy()
        else:
            embeddings = np.hstack((next(self.n2v.first_emb.parameters()).detach().cpu().numpy(),
                                    next(self.n2v.second_emb.parameters()).detach().cpu().numpy()))
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding

        return self._embeddings