import torch.nn as nn
import torch
import numpy as np
import random
import networkx as nx

def update_epsilon(init_epsilon, finial_epsilon, cur_step, min_step, max_step):
    if max_step == 0:
        return 0.0
    elif cur_step <= min_step:
        return init_epsilon
    elif cur_step > max_step:
        return finial_epsilon
    else:
        inter = (init_epsilon - finial_epsilon) / (max_step - min_step)
        return init_epsilon - inter * (cur_step - min_step)

def weights_init_uniform(m):
    """initializing weights"""
    a = -1
    b = 1
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.uniform_(a, b)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(a, b)
        m.bias.data.fill_(0)
    elif classname.find('Embedding') != -1:
        m.weight.data.uniform_(a, b)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

def init_weights_kaiming_uniform(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

def mean_weights(m):
    if type(m) == nn.Linear:
        print('w ', torch.mean(m.weight), ' b ', torch.mean(m.bias))

def save_model(model, path):
    """save trained model parameters"""
    torch.save(model.state_dict(), path)

def load_model(model, path, avoid=None):
    """load trained model parameters"""
    state_dict = dict(torch.load(path))
    if avoid is not None:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith(avoid)}
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(state_dict)

def to_Cuda(tensor, cuda, cuda_id=None):
    """convert a tensor to a cuda tensor"""
    if cuda_id is None:
        return tensor.cuda() if cuda else tensor
    else:
        return tensor.cuda(cuda_id) if cuda else tensor

def sample_unseen_task(min_N, max_N, tasks=[], num_of_unseen=1, method='uniform', lam=1):
    sampled = [False] * (max_N - min_N)
    unseen_task = []
    for i in range(num_of_unseen):
        if method == 'uniform':
            while True:
                a_num = np.random.randint(min_N, max_N - 1)
                if not sampled[a_num] and a_num not in tasks:
                    break
        elif method == 'poisson':
            while True:
                a_num = np.random.poisson(lam, size=1)
                if min_N <= a_num <= max_N - 1 and not sampled[a_num] and a_num not in tasks:
                    break
        elif method == 'exponent':
            while True:
                a_num = int(np.round(np.random.exponential(lam, size=1)))
                if min_N <= a_num <= max_N - 1 and not sampled[a_num] and a_num not in tasks:
                    break
        else:
            raise ValueError(f"Unknown env sample method: {method}")
        unseen_task.append(a_num)
        sampled[a_num] = True

    return unseen_task

def l1_reg_loss(model):
    reg_loss = None
    for param in model.parameters():
        if reg_loss is None:
            reg_loss = torch.sum(torch.abs(param))
        else:
            reg_loss += torch.sum(torch.abs(param))
    return reg_loss

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, max_encode=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / max_encode ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sin_pos_embed_from_grid(embed_dim, pos, max_encode=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim, dtype=np.float32)
    omega /= embed_dim
    omega = 1.0 / max_encode ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    # print(out)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    # emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb_cos


def ids_2dto1d(i, j, M, N):
    """
    convert (i,j) in a M by N matrix to index in M*N list. (row wise)
    matrix: [[1,2,3], [4, 5, 6]]
    list: [0, 1, 2, 3, 4, 5, 6]
    index start from 0
    """
    assert 0 <= i < M and 0 <= j < N
    index = int(i * N + j)
    return index


def ids_1dto2d(ids, M, N):
    """ inverse of ids_2dto1d(i, j, M, N)
        index start from 0
    """
    i = ids // N
    j = ids - N * i
    return i, j

def norm_adj(adj):
    adj += np.eye(adj.shape[0])
    degr = np.array(adj.sum(1))
    degr = np.diag(np.power(degr, -0.5))
    return degr.dot(adj).dot(degr)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def _emb_from_nodes(node_emb, nodes, node_to_idx):
    emb_dim = node_emb.shape[1]
    embeddings = []
    for node in nodes:
        if node == 0:
            embeddings.append(np.zeros(emb_dim, dtype=node_emb.dtype))
            continue
        idx = node_to_idx.get(int(node))
        if idx is None:
            embeddings.append(np.zeros(emb_dim, dtype=node_emb.dtype))
        else:
            embeddings.append(node_emb[idx])
    return np.concatenate(embeddings, axis=0)


def shared_obs_query(node_emb, shared_obs, max_time_horizon, t, node_to_idx):
    num_agent = shared_obs.shape[0]
    shared_node_embs = np.concatenate(
        [_emb_from_nodes(node_emb, shared_obs[i][:-1], node_to_idx).reshape(1, -1) for i in range(num_agent)],
        axis=0,
    )
    t_one_hot = np.zeros((num_agent, max_time_horizon))
    if 0 <= t < max_time_horizon:
        t_one_hot[:, t] = 1
    return np.concatenate([shared_node_embs, t_one_hot], axis=1)


def obs_query(node_emb, obs, max_time_horizon, t, node_to_idx):
    num_agent = obs.shape[0]
    node_embs = np.concatenate(
        [_emb_from_nodes(node_emb, obs[i][:-2], node_to_idx).reshape(1, -1) for i in range(num_agent)], axis=0
    )
    t_one_hot = np.zeros((num_agent, max_time_horizon))
    if 0 <= t < max_time_horizon:
        t_one_hot[:, t] = 1
    return np.concatenate([node_embs, t_one_hot, np.eye(num_agent)[np.arange(num_agent)]], axis=1)


def get_demonstration(obs, game, exit_node):
    num_agent = obs.shape[0]
    act_probs = [np.zeros(game.defender_mix_action) for _ in range(num_agent)]
    graph = game._graph.graph
    for i in range(num_agent):
        defender_node = int(obs[i][1])
        if defender_node == 0 or exit_node == 0:
            act_probs[i][0] = 1.0
            continue
        try:
            path = nx.shortest_path(graph, source=defender_node, target=exit_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            act_probs[i][0] = 1.0
            continue
        if len(path) <= 1:
            act_probs[i][0] = 1.0
            continue
        next_node = path[1]
        legal_actions = game._graph.legal_actions(defender_node)
        if next_node in legal_actions:
            action_idx = legal_actions.index(next_node)
        else:
            action_idx = 0
        if action_idx < len(act_probs[i]):
            act_probs[i][action_idx] = 1.0
        else:
            act_probs[i][0] = 1.0
    return np.array(act_probs)
