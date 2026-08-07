import numpy as np


def train_execute_episode(game, defender, attacker):
    time_horizon = game.time_horizon
    max_act = game.graph.degree
    num_defender = game.num_defender

    trajectory = {}
    trajectory["defender_his"] = np.zeros([time_horizon+1, num_defender])
    trajectory["attacker_his"] = np.zeros([time_horizon+1, 1])
    trajectory["defender_his_idx"] = np.zeros([time_horizon+1, num_defender])
    trajectory["attacker_his_idx"] = np.zeros([time_horizon+1, 1])
    trajectory["defender_legal_act"] = np.zeros(
        [time_horizon+1, num_defender, max_act])
    trajectory["attacker_legal_act"] = np.zeros([time_horizon+1, 1, max_act])
    trajectory["return"] = np.zeros([time_horizon+1, 1])
    trajectory["mask"] = np.zeros([time_horizon+1, 1])

    t = 0
    game.reset()
    defender.reset()
    attacker.reset()
    trajectory["defender_his"][t] = game.current_state.defender_his[-1]
    trajectory["attacker_his"][t] = game.current_state.attacker_his[-1]
    defender_legal_act, attacker_legal_act=game.current_state.legal_action(False)
    for i in range(num_defender):
        length = len(defender_legal_act[i])
        trajectory["defender_legal_act"][t][i][:length] = defender_legal_act[i]
    length = len(attacker_legal_act)
    trajectory["attacker_legal_act"][t][0][:length]=attacker_legal_act
    trajectory["mask"][t] = 1

    while not game.current_state.is_end():
        defender_obs, attacker_obs = game.current_state.obs()
        defender_act, defender_act_idx, defender_legal_act, attacker_legal_act = defender.train_select_act(
            defender_obs, prior=False)
        attacker_act, attacker_act_idx = attacker.train_select_act(attacker_obs)
        game.step(defender_act, attacker_act)
        t += 1

        trajectory["defender_his"][t] = game.current_state.defender_his[-1]
        trajectory["attacker_his"][t] = game.current_state.attacker_his[-1]
        trajectory["defender_his_idx"][t] = defender_act_idx
        trajectory["attacker_his_idx"][t] = attacker_act_idx
        for i in range(num_defender):
            length = len(defender_legal_act[i])
            trajectory["defender_legal_act"][t][i][:length]=defender_legal_act[i]
        length = len(attacker_legal_act)
        trajectory["attacker_legal_act"][t][0][:length] = attacker_legal_act
        trajectory["mask"][t]=1
    ret = game.current_state.reward()[0]
    trajectory["return"][:t+1]=ret
    if attacker.require_update:
        return trajectory, (attacker.selected_exit, -ret, attacker.is_br)
    return trajectory


def test_execute_episode(game, defender, attacker, prior=False, temp=1):
    game.reset()
    defender.reset()
    attacker.reset(train=False)
    while not game.current_state.is_end():
        defender_obs, attacker_obs = game.current_state.obs()
        defender_act = defender.select_act(defender_obs, prior=prior, temp=temp)
        attacker_act = attacker.select_act(attacker_obs)
        game.step(defender_act, attacker_act)
    return game.current_state.reward()[0]

def worker(remote, parent_remote, train_execute_episode, test_execute_episode, game, defender, attacker):
    while True:
        cmd=remote.recv()
        if cmd == 'train_epi':
            if attacker.require_update:
                act_est, N_acts=remote.recv()
                attacker.synch(act_est, N_acts)
                trajectory, a_v = train_execute_episode(game, defender, attacker)
                remote.send(trajectory)
                remote.send(a_v)
            else:
                trajectory=train_execute_episode(game, defender, attacker)
                remote.send(trajectory)
        if cmd == 'test_epi':
            reward=test_execute_episode(game, defender, attacker, prior=False, temp=1)
            remote.send(reward)
        elif cmd == 'close':
            remote.close()
            break
