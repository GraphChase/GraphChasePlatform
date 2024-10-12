import torch
import logging
from agent.cfr_mix.player_class.attacker_class import Attacker
from agent.cfr_mix.player_class.defender_class import Defender
from agent.cfr_mix.traverse_methods.deep_probe_cfr import probe_traverse_tree
from agent.cfr_mix.evaluation.double_worst_case_utility import evaluation_deep_cfr

def double_deep_probe_cfr(game_graph, init_location, time_horizon, network_dim, sample_number, action_number, attacker_regret_batch_size,
                       defender_regret_batch_size, defender_strategy_batch_size, train_epoch, attacker_regret_lr, defender_regret_lr, defender_strategy_lr,
                       regret_file_name, strategy_file_name, iteration):

    player_list = [Attacker(time_horizon=time_horizon, hidden_dim=network_dim),
               Defender(time_horizon=time_horizon, player_number=len(init_location[1]), hidden_dim=network_dim,)]
    exploitability = []
    for i in range(iteration):
        logging.info("Iteration time:{}, sample start!".format(i))
        for t in range(sample_number):
            probe_traverse_tree(game_graph, init_location, 0, player_list, time_horizon, action_number, i + 1)
            probe_traverse_tree(game_graph, init_location, 1, player_list, time_horizon, action_number, i + 1)

        logging.info("Iteration time:{}, sample {} times complete!".format(i, sample_number))
        player_list[0].train_regret_network(attacker_regret_lr, i + 1, train_epoch, attacker_regret_batch_size)
        player_list[1].train_regret_network(defender_regret_lr, i + 1, train_epoch, defender_regret_batch_size)
        player_list[1].train_strategy_network(defender_strategy_lr, i + 1, train_epoch, defender_strategy_batch_size)

        torch.save(player_list[0].regret_model, regret_file_name[0])
        torch.save(player_list[1].regret_model, regret_file_name[1])
        logging.info("Iteration time:{}, train network complete!")

        if i % 10 == 0:
            torch.save(player_list[1].strategy_model, strategy_file_name.format(i))
            ex = evaluation_deep_cfr(time_horizon, game_graph, init_location, network_dim, strategy_file_name, [i])
            exploitability.append(ex[0])
            logging.info("Iteration time:{}, worse case utility:{}".format(i, exploitability))

