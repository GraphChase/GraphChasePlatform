import dgl
import torch
import numpy as np
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os.path as osp
import os
import time
import logging
from datetime import datetime
import pickle

from graphchase.graph.gnn_graph import generate_game_pool, load_game_pool
from graphchase.solver.grasper.grasper_game import GrasperGame
from .encoder import PreModel
from ..utils.graph_learning_utils import (
    create_optimizer,
    set_random_seed,
    get_dgl_graph
)

def collate_fn(batch):
    graphs = [x for x in batch]
    batch_g = dgl.batch(graphs)
    return batch_g


logger = logging.getLogger(__name__)

def build_pretrain_graphs(args):
    if args.load_game_pool_file:
        settings_list = load_game_pool(args)
        game_pool_str = f"_gp{args.pool_size}"
    else:
        settings_list = generate_game_pool(args)
        game_pool_str = ""
    differ_size_str = "_ds" if args.graph_type == "Grid_Graph" and args.differ_size else ""
    graphs = []
    for settings in settings_list:
        game = GrasperGame(settings, args, action_type=args.action_type, compute_path=False)
        graphs.append(get_dgl_graph(game))
    if not graphs:
        raise ValueError("No graphs available for pretraining")
    return graphs, game_pool_str, differ_size_str


def graph_pretrain(args, graphs, game_pool_str, differ_size_str):
    save_path = args.pre_pretrain_save_path
    if not osp.exists(save_path):
        os.makedirs(save_path)

    lr = 0.00015
    weight_decay = 1e-5
    max_epoch = args.max_epoch

    start_ = datetime.now().replace(microsecond=0)
    start_time = time.time()

    node_feat_dim = graphs[0].ndata["attr"].shape[1]
    print(f"******** # Num Graphs: {len(graphs)}, # Num Feat: {node_feat_dim} ********")
    logger.info(
        "Start pretrain: graph_type=%s%s ep=%s%s layer=%s hidden=%s out=%s dnum=%s_%s enum=%s_%s mep=%s",
        args.graph_type,
        differ_size_str,
        args.edge_probability,
        game_pool_str,
        args.gnn_num_layer,
        args.gnn_hidden_dim,
        args.gnn_output_dim,
        args.min_num_defender,
        args.max_num_defender,
        args.min_num_exit,
        args.max_num_exit,
        args.min_attacker_pth_len,
    )

    train_idx = torch.arange(len(graphs))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn, batch_size=128, pin_memory=True)

    set_random_seed(args.seed)
    model = PreModel(node_feat_dim, args.gnn_hidden_dim, args.gnn_output_dim, args.gnn_num_layer, args.gnn_dropout)
    model.to(args.device)

    optimizer = create_optimizer("adam", model, lr, weight_decay)
    scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

    epoch_iter = tqdm(range(max_epoch))
    train_loss = []
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch_g in train_loader:
            batch_g = batch_g.to(args.device)
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
        epoch_iter.set_description(f"Epoch {epoch + 1} | train_loss: {mean_loss:.4f}")
        logger.info("Epoch %s | train_loss: %.4f", epoch + 1, mean_loss)
        if (epoch + 1) % 200 == 0:
            model.save(save_path + f"/checkpoint_epoch{epoch + 1}_type_{args.graph_type}{differ_size_str}_ep{args.edge_probability}{game_pool_str}_layer{args.gnn_num_layer}_"
                                   f"hidden{args.gnn_hidden_dim}_out{args.gnn_output_dim}_dnum{args.min_num_defender}_{args.max_num_defender}_enum{args.min_num_exit}_"
                                   f"{args.max_num_exit}_mep{args.min_attacker_pth_len}.pt")
    end_time = time.time()
    train_time = end_time - start_time
    pickle.dump({'train_time': train_time, 'train_loss': train_loss},
                open(save_path + f'/train_record_type_{args.graph_type}{differ_size_str}_ep{args.edge_probability}{game_pool_str}_layer{args.gnn_num_layer}_'
                                 f'hidden{args.gnn_hidden_dim}_out{args.gnn_output_dim}_dnum{args.min_num_defender}_{args.max_num_defender}_enum{args.min_num_exit}_'
                                 f'{args.max_num_exit}_mep{args.min_attacker_pth_len}.pik', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print("============================================================================================")
    end_ = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_)
    print("Finished training at (GMT) : ", end_)
    print("Total training time  : ", end_ - start_)
    print("============================================================================================")
    return
