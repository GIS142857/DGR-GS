# coding:utf-8

from Utils import *
import torch
import copy
from torch.autograd import Variable
# from tensorboard_logger import Logger
from tensorboardX import SummaryWriter
import os

from collections import defaultdict
from heapq import *
import math
import random
import networkx as nx
from node import Packet

# import matplotlib.pyplot as plt

# logger = Logger(logdir="./logs", flush_secs=10)
writer1 = SummaryWriter(log_dir='log999_11')
writer2 = SummaryWriter(log_dir='log999_22')

# writer8_1 = SummaryWriter(log_dir='logn_881')
# writer8_2 = SummaryWriter(log_dir='logn_882')

file = open("q_value.txt", "w")
# f1 = open('reward_1.txt', 'w')
# f2 = open("reward_2.txt", "w")
# f3 = open("reward_3.txt", "w")

Inf = 100000


class learnAgent:
    def __init__(self, flow_num, graph, device, episodes, tau_N, sourcess, dess, slots, deadlines, dmax_set, DDQN=False,
                 kappa=1):
        self.c = 1
        self.flow_num = int(flow_num)
        self.node_list = graph.node_list
        self.state_dim = 4
        self.graph = graph
        self.env = ENV(self.state_dim, graph)
        self.episodes = episodes
        self.tau_N = tau_N
        self.double_q_learning = DDQN
        self.slots = slots
        self.deadlines = deadlines
        self.dmax_set = dmax_set
        self.maxPackets = 10
        self.slot_time = 30

        N = graph.node_num
        # self.S1 = sourcess[0]
        # self.S2 = sourcess[1]
        # self.S3 = sourcess[2]
        # self.D1 = dess[0]
        # self.D2 = dess[1]
        # self.D3 = dess[2]
        self.sources = sourcess
        self.dess = dess

        self.D_list1 = []
        print(self.node_list[self.sources[0]].all_worst_to_des)

        for d in self.node_list[self.sources[0]].all_worst_to_des:
            self.D_list1.append(int(d.tolist()[0]))

        self.D_list2 = []
        for d in self.node_list[self.sources[1]].all_worst_to_des:
            self.D_list2.append(int(d.tolist()[0]))

        self.D_list3 = []
        for d in self.node_list[self.sources[2]].all_worst_to_des:
            self.D_list3.append(int(d.tolist()[0]))

        # print('d_list1', self.D_list1)
        # print('d_list2', self.D_list2)

        self.max_reward = 10000
        self.batch_size = 32
        self.device = device
        self.loss_history = []
        self.loss_num1 = 0
        self.reward_history1 = []
        self.reward_len1 = 0
        self.loss_num2 = 0
        self.reward_history2 = []
        self.reward_len2 = 0
        self.loss_num_8_1 = []
        self.loss_num_8_2 = []

        self.pass_weight = np.ones((N, N))
        self.penaty = 1000

        taus = torch.arange(
            0, tau_N + 1, device=device, dtype=torch.float32) / tau_N
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, tau_N)

        self.N = N
        self.kappa = kappa

    def get_min_q(self, action, next_state):
        q_values = self.node_list[action].agent.model.get_q_values(next_state)
        return min(q_values)

    def update_all_q_values_node(self, node):
        """
        Update all Q-values in the replay-memory of the node.

        When states and Q-values are added to the replay-memory, the
        Q-values have been estimated by the Neural Network. But we now
        have more data available that we can use to improve the estimated
        Q-values, because we now know which actions were taken and the
        observed rewards. We sweep backwards through the entire replay-memory
        to use the observed data to improve the estimated Q-values.
        """

        # Copy old Q-values so we can print their statistics later.
        # Note that the contents of the arrays are copied.
        self.graph.node_list[node].agent.replay_memory.q_values_old[:] = self.graph.node_list[
                                                                             node].agent.replay_memory.q_values[:]

        # Process the replay-memory backwards and update the Q-values.
        # This loop could be implemented entirely in NumPy for higher speed,
        # but it is probably only a small fraction of the overall time usage,
        # and it is much easier to understand when implemented like this.
        num_used = self.graph.node_list[node].agent.replay_memory.num_used
        for k in reversed(range(num_used)):
            # Get the data for the k'th state in the replay-memory.
            replay_memory = self.graph.node_list[node].agent.replay_memory
            next_state = replay_memory.next_state[k]
            action = replay_memory.actions[k]
            reward = replay_memory.rewards[k]
            end_life = replay_memory.end_life[k]

            # Calculate the Q-value for the action that was taken in this state.
            if end_life:
                # If the agent lost a life or it was game over / end of episode,
                # then the value of taking the given action is just the reward
                # that was observed in this single step. This is because the
                # Q-value is defined as the discounted value of all future game
                # steps in a single life of the agent. When the life has ended,
                # there will be no future steps.
                action_value = reward
            else:
                # Otherwise the value of taking the action is the reward that
                # we have observed plus the discounted value of future rewards
                # from continuing the game. We use the estimated Q-values for
                # the following state and take the maximum, because we will
                # generally take the action that has the highest Q-value.
                # valid_q = self.q_values[k + 1][valid_actions]
                # Reference [1] equation in algorithm 1
                max_q = self.get_min_q(action, next_state)
                action_value = reward + replay_memory.discount_factor * max_q

            # Error of the Q-value that was estimated using the Neural Network.
            replay_memory.estimation_errors[k] = abs(action_value - replay_memory.q_values[k, action])

            # Update the Q-value with the better estimate.
            replay_memory.q_values[k, action] = action_value

    def calculate_action_num(self, node, child, child_list):
        for i in range(len(child_list)):
            if child == child_list[i]:
                self.graph.node_list[node].action_num[i] += 1
                return i

    def get_action_id(self, child, child_list):
        for i in range(len(child_list)):
            if child == child_list[i]:
                return i

    def get_action(self, q_values, iteration, state, node, normal):
        q_values = torch.squeeze(q_values, 0)
        print('node get_action', node)
        # print('state', state)
        # print('normal', normal)
        D = state[0] * normal[0]
        k = int(state[1] * normal[1] - 1)
        des = state[3] * normal[3]
        # print('des', des)
        # h = int(state[2] * normal[2])
        num, child_list = self.graph.get_nb_num(node)
        # print('child_list', child_list)
        c_list = []
        q = []
        ucb_list = []
        sum = 0
        for i in range(num):
            # print('node', node, 'child', child_list[i], 'worst', self.graph.G[node][child_list[i]]['worst'] + self.graph.to_des_worst[child_list[i]], 'D', D)
            nb = child_list[i]
            key = str(nb) + ',' + str(k)  # + ',' + str(h)
            # print('key', key)
            # print('nb_delay', self.node_list[node].nb_delay)
            nb_delay = max(self.node_list[node].nb_delay[key])

            nb_to_des_worst = 0
            num, _ = self.graph.get_nb_num(nb)
            if num > 0 and nb < des - 1:
                # key2 = str(k) + ',' + str(h)
                # nb_to_des_worst = self.node_list[nb].worst_to_des[key2]
                if nx.has_path(self.graph.G, nb, des-1):
                    nb_to_des_worst = self.get_short_path(nb, des-1, k)
                else:
                    nb_to_des_worst = 10000
                #nb_to_des_worst = self.get_short_path(nb, des - 1, k)
            elif num == 0 and nb < des - 1:
                nb_to_des_worst = self.max_reward
            elif nb > des - 1:
                nb_to_des_worst = self.max_reward
            # print('D', D)
            # print('nb', nb)
            # print('nb_d', nb_delay,'nb_to_des', nb_to_des_worst)
            # print('nb_d+nb_to_des', nb_delay + nb_to_des_worst, 'D', D, '-', nb_delay + nb_to_des_worst - D)
            if (nb_delay + nb_to_des_worst - D) <= 1e-15:
                c_list.append(child_list[i])
                # print('q_values', q_values, 'i', i)
                q.append(q_values[i].detach().cpu())
                sum += 1
        #         C = -1
        #         if self.graph.node_list[node].num > 0:
        #             C = self.c * math.sqrt((self.graph.node_list[node].action_num[i]/self.graph.node_list[node].num))
        #             # print('action_num', self.graph.node_list[node].action_num[i])
        #             # print('node_num', self.graph.node_list[node].num)
        #             # print('C', C)
        #         ucb = q_values[i] + C
        #         print('q_values', q_values[i],'c', C)
        #         ucb_list.append(ucb)
        # # print('c_list', c_list)
        # # print('child_list', child_list)
        # # print('q', q)
        # # print('ucb_list', ucb_list)
        # if len(c_list) == 0:
        #     return -1, -1
        # else:
        #     id = np.argmin(ucb_list)
        #     action_index = self.calculate_action_num(node, c_list[id], child_list)
        #     #print('action', c_list[id], 'index', action_index)
        #     return c_list[id], action_index

        # print('sum', sum)
        if sum == 0:
            return -1, -1

        eps = self.node_list[node].agent.epsilon_greedy.get_epsilon(iteration, True)
        if node == 0:
           print('eps', eps, 'iter', iteration)
        rand = np.random.random()
        # print('rand', rand, 'eps', eps)
        # print('c_list', c_list, 'q', q)
        if rand <= eps:
            # print('random')
            # print('sum', sum)
            rand_a = random.randint(1, sum)
            # rand_a = self.get_random_a(node, c_list)
            index = self.get_action_id(c_list[rand_a - 1], child_list)
            #print('random_action', c_list[rand_a-1])
            return c_list[rand_a - 1], index
        else:
            # print('q', q)
            id = np.argmin(q)
            # print('q_a', q)
            # print('id', id)
            index = self.get_action_id(c_list[id], child_list)
            #print('best_action', c_list[id])
            return c_list[id], index

    def get_action_max(self, q_values, state, node, normal):
        #print('node get_max_action', node)
        # print('state', state)
        D = state[0] * normal[0]
        k = int(state[1] * normal[1] - 1)
        des = state[3] * normal[3]
        # h = int(state[2] * normal[2])
        num, child_list = self.graph.get_nb_num(node)
        c_list = []
        q = []
        sum = 0
        for i in range(num):
            # print('node', node, 'child', child_list[i], 'worst', self.graph.G[node][child_list[i]]['worst'] + self.graph.to_des_worst[child_list[i]], 'D', D)
            nb = child_list[i]
            key = str(nb) + ',' + str(k)  # + ',' + str(h)
            # print('key', key)
            # print('nb_delay', self.node_list[node].nb_delay)
            nb_delay = max(self.node_list[node].nb_delay[key])
            multi = 1  # self.graph.pass_matrix[node][nb]
            if multi > 0:
                nb_delay = multi * nb_delay
            nb_to_des_worst = 0
            num, _ = self.graph.get_nb_num(nb)
            if num > 0 and nb < des - 1:
                # print('111')
                # key2 = str(k) + ',' + str(h)
                # nb_to_des_worst = self.node_list[nb].worst_to_des[key2]
                if nx.has_path(self.graph.G, nb, des-1):
                    nb_to_des_worst = self.get_short_path(nb, des-1, k)
                else:
                    nb_to_des_worst = 10000
                #nb_to_des_worst = self.get_short_path(nb, des - 1, k)
                # print('nb_to_des_worst', nb_to_des_worst)
            elif num == 0 and nb < des - 1:
                # print('22')
                nb_to_des_worst = self.max_reward
            elif nb > des - 1:
                # print('33')
                nb_to_des_worst = self.max_reward
            # print('nb', nb, 'nb_delay', nb_delay, 'des', des-1, 'nb_to_des', nb_to_des_worst)
            # print('num', num)
            # print('D', D)
            # print('nb_d+nb_to_des', nb_delay + nb_to_des_worst, 'D', D, '-', nb_delay + nb_to_des_worst - D)
            if (nb_delay + nb_to_des_worst - D) <= 1e-15:
                c_list.append(child_list[i])
                q_values = torch.squeeze(q_values, 0)
                # print('q_values', q_values)
                q.append(q_values[i].detach().cpu())
                sum += 1
        if len(c_list) == 0:
            return -1, -1

        #print('c_list', c_list, 'q', q)
        id = np.argmin(q)
        # if Test:
        #     file.write('node' + str(node) + ' childs ' + str(child_list))
        #     file.write('\n')
        #     file.write('c_list: ' + str(c_list))
        #     file.write('\n')
        #     file.write('q_values: ' + str(q))
        #     file.write('\n')
        # file.write('q: ' + str(q))
        # file.write('\n')
        # file.write('id: ' + str(id))
        # file.write('\n')
        # print('q_a', q_a)
        # print('id', id)
        #print('action', c_list[id])
        return c_list[id], id

    def get_tar_q_values(self, next_states, actions, rewards, done, size):
        # print('get_tar_q_values')
        # print('rewards', rewards)
        batch_size = size
        if done[0]:
            tar_q = rewards[0]
        else:
            next_node = actions[0]
            next_q = min(self.graph.node_list[next_node].agent.model.net(next_states[0]))
            # print('next_q', next_q)
            # q_list = next_q.detach().numpy().tolist()
            # print('q_list', q_list, 'reward', rewards[i])
            tar_q = rewards[0] + next_q
        tar_q = tar_q.unsqueeze(0)
        # print('tar_q', tar_q)
        for i in range(1, batch_size):
            if done[i]:
                tar_q = torch.cat((tar_q, rewards[i].unsqueeze(0)), 0)
            else:
                next_node = actions[i]
                next_q = min(self.graph.node_list[next_node].agent.model.net(next_states[i]))
                # print('next_state', next_states[i])
                # print('next_q', next_q)
                temp = rewards[i] + next_q
                # print('temp', temp, 'untemp', temp.unsqueeze(0))
                # print('tar_q', tar_q, 'unq', tar_q.unsqueeze(0))
                tar_q = torch.cat((tar_q, temp.unsqueeze(0)), 0)
                # print('tar_q', tar_q)
                # print('q_list', q_list, 'reward', rewards[i])
        # print('q_values', q_values)
        # var_qs = torch.Tensor(q_values)
        # print('tar_q', tar_q)
        return tar_q

    def get_action_q(self, pred_q, action_index, size):
        mask = pred_q.ge(100000)
        for i in range(size):
            mask[i][action_index[i]] = 1
        # print('pred_q', pred_q)
        # print('mask', mask)
        # print('select_q', torch.masked_select(pred_q, mask))
        # q.values
        return torch.masked_select(pred_q, mask)

    def get_samples_for_fisher(self, node):
        a = self.graph.node_list[node].agent.replay_memory.num_used
        b = self.graph.node_list[node].agent.model.max_fisher_len
        c = a if a < b else b

        state_batch, action_batch, action_index, reward_batch, next_states, done = self.graph.node_list[
            node].agent.replay_memory.random_batch(c)
        var_s = torch.tensor(state_batch, dtype=torch.float)
        var_next_s = torch.tensor(next_states, dtype=torch.float)
        var_rewards = torch.tensor(reward_batch, dtype=torch.float)
        # print('var_rewards', var_rewards)
        # print('state_batch', state_batch)
        # print('var_s', variablex)

        # print('reward_batch', reward_batch)
        # print('next_state_batch', next_states)
        # print('done', done)
        self.graph.node_list[node].agent.model.opt.zero_grad()
        var_s = Variable(var_s)
        var_next_s = Variable(var_next_s)
        var_s = var_s.to(self.device)
        var_next_s = var_next_s.to(self.device)
        var_rewards = var_rewards.to(self.device)
        pred_q = self.graph.node_list[node].agent.model.net(var_s)
        # print('pred_q___', pred_q)
        pred_q_a = self.get_action_q(pred_q, action_index, c)
        # print('pred_q_a____', pred_q_a)
        # print('b', b)
        q_values_batch = self.get_tar_q_values(var_next_s, action_batch, var_rewards,
                                               done, c)  ### next_states, actions, rewards, done

        # print('pred_q_a', pred_q_a)
        # print('q_values_batch', q_values_batch)

        return pred_q_a, q_values_batch

    def evaluate_quantile_at_action(self, batch_size, s_quantiles, actions):
        # print('batch_size', batch_size)
        # print('s_quantiles_', s_quantiles, 'shape', s_quantiles.shape)
        # #actions = actions.unsqueeze(0)
        # print('actions', actions, 'shape', actions.shape)
        assert s_quantiles.shape[0] == actions.shape[0]

        # batch_size = s_quantiles.shape[0]
        N = s_quantiles.shape[1]

        # Expand actions into (batch_size, N, 1).
        # action_index = actions[..., None].expand(batch_size, N, 1)
        # print('actions', actions)
        # #actions.reshape()
        # s_size = actions.shape
        # action = torch.randint(0, 1, (s_size[0], 1))
        # print('action', action)
        # print('batch_size', batch_size)
        # print('repeat', actions.repeat(N, 1))
        # print('reshape', actions.repeat(N, 1).reshape(batch_size*N, 1))
        a_index = torch.randint(0, 1, (batch_size, N, 1))
        for i in range(batch_size):
            for j in range(N):
                a_index[i][j][0] = actions[i]
        # print('a_index', a_index)
        # action_index = action.expand(batch_size, N, 1)
        # print('action_index', a_index, 'shape', a_index.shape)

        # Calculate quantile values at specified actions.
        a_index = a_index.to(self.device)
        sa_quantiles = torch.gather(s_quantiles, 2, a_index)

        # print('sa_qu', sa_quantiles)

        return sa_quantiles

    def calculate_huber_loss(self, td_errors, kappa=1.0):
        return torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa))

    def calculate_quantile_huber_loss(self, td_errors, taus, kappa=1.0, weights=None):
        assert not taus.requires_grad
        batch_size, N, N_dash = td_errors.shape

        # Calculate huber loss element-wisely.
        element_wise_huber_loss = self.calculate_huber_loss(td_errors, kappa)
        assert element_wise_huber_loss.shape == (
            batch_size, N, N_dash)

        # print('element_wise_huber_loss', element_wise_huber_loss)

        # Calculate quantile huber loss element-wisely.
        # print('taus', taus[..., None])
        # print('td_errors.detach()', td_errors.detach())
        # print('td_errors', (td_errors.detach() < 0).float())
        element_wise_quantile_huber_loss = torch.abs(
            taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
        assert element_wise_quantile_huber_loss.shape == (
            batch_size, N, N_dash)
        # print('element_wise_quantile_huber_loss', element_wise_quantile_huber_loss)

        # Quantile huber loss.
        batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
            dim=1).mean(dim=1, keepdim=True)
        # print('batch_quantile_huber_loss', batch_quantile_huber_loss)
        assert batch_quantile_huber_loss.shape == (batch_size, 1)

        if weights is not None:
            quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
        else:
            quantile_huber_loss = batch_quantile_huber_loss.mean()

        return quantile_huber_loss

    def calculate_loss(self, batch_size, samples, node,
                       flow):  ### batch_size, states, actions, rewards, next_states, dones
        # Calculate quantile values of current states and actions at taus.
        states = []
        actions = []
        # print('node loss', node)
        # print('sanples', samples)
        for i in range(batch_size):
            sample = samples[i]
            # print('sample', sample)
            # print('len_sample', len(sample))
            states.append(sample[0][0])
            actions.append(sample[0][2])
        # print('states', states)
        # print('samples', samples)
        # print('actions', actions)
        var_s = torch.tensor(states, dtype=torch.float)
        var_s = var_s.to(self.device)
        var_action = torch.tensor(actions, dtype=torch.int64)
        var_action = var_action.to(self.device)

        # print('node', node)
        # print('var_s', var_s)

        current_sa_quantiles = self.evaluate_quantile_at_action(batch_size,
                                                                self.graph.node_list[node].agent.model.forward(
                                                                    batch_size, var_s, flow), var_action)
        # print('current_sa_quantiles', current_sa_quantiles, 'shape', current_sa_quantiles.shape)
        # print('batch_size', batch_size, 'N', self.tau_N, current_sa_quantiles.shape == (batch_size, self.tau_N, 1))
        assert current_sa_quantiles.shape == (batch_size, self.tau_N, 1)

        with torch.no_grad():
            # print('next_states_qu')
            # Calculate Q values of next states.
            if not self.double_q_learning:
                # Sample the noise of online network to decorrelate between
                # the action selection and the quantile calculation.
                # self.online_net.sample_noise()
                target_qu = []
                for i in range(batch_size):
                    sample = samples[i]
                    next_state_list = []
                    action_list = []
                    rewards = []
                    dones = []
                    for j in range(len(sample)):
                        ex = sample[j]
                        next_state_list.append(ex[4])
                        action_list.append(ex[1])
                        rewards.append(ex[3])
                        dones.append(ex[5])
                    # print('samples', samples)
                    # print('rewards', rewards)
                    # print('dones', dones)
                    var_next_s = torch.tensor(next_state_list, dtype=torch.float)
                    var_next_s = var_next_s.to(self.device)
                    # print('var_next_s', var_next_s)
                    next_node = action_list[0]
                    # print('next_node', next_node)
                    # print('node', node)
                    if next_node == self.dess[flow]:
                        unit = [0.0]
                        next_qu = []
                        for m in range(self.tau_N):
                            next_qu.append(unit)
                        next_qu = torch.tensor(next_qu).unsqueeze(0)
                        # print('next_qu', next_qu)
                        next_sa = next_qu
                    else:
                        next_q = self.graph.node_list[next_node].agent.model.calculate_q(1, var_next_s[0], flow)
                        # print('next_q', next_q)
                        # Calculate greedy actions.
                        next_actions = torch.argmin(next_q, dim=1, keepdim=True)
                        # print('next_actions', next_actions)
                        assert next_actions.shape == (1, 1)

                        # print('len', len(sample), 'next_s_list', next_state_list)
                        next_qu = self.graph.node_list[next_node].agent.model.forward(1, var_next_s[0], flow)
                        next_sa = self.evaluate_quantile_at_action(1, next_qu, next_actions)
                        # print('next_qu', next_qu)
                        # print('shape', next_qu.shape)

                    for l in range(1, len(sample)):
                        next_node = action_list[l]
                        if next_node == self.dess[flow]:
                            unit = [0.0]
                            next_qu_ = []
                            for m in range(self.tau_N):
                                next_qu_.append(unit)
                            next_qu_ = torch.tensor(next_qu_).unsqueeze(0)
                            next_qu = torch.cat((next_qu, next_qu_), 0)
                            next_sa_ = next_qu
                            next_sa = torch.cat((next_sa_, next_sa), 0)
                        else:
                            next_q = self.graph.node_list[next_node].agent.model.calculate_q(1, var_next_s[l], flow)
                            # Calculate greedy actions.
                            next_actions = torch.argmin(next_q, dim=1, keepdim=True)
                            # print('next_actions', next_actions)
                            assert next_actions.shape == (1, 1)

                            # print('len', len(sample), 'next_s_list', next_state_list)
                            next_qu_ = self.graph.node_list[next_node].agent.model.forward(1, var_next_s[l], flow)
                            # print('next_qu_', next_qu_)
                            # print('shape', next_qu_.shape)
                            next_qu = torch.cat((next_qu, next_qu_), 0)
                            next_sa_ = self.evaluate_quantile_at_action(1, next_qu_, next_actions)
                            next_sa = torch.cat((next_sa_, next_sa), 0)

                    # next_sa = self.evaluate_quantile_at_action(len(sample), next_qu, next_actions)
                    # print('next_sa', next_sa)
                    # next_sa.append(avg(next_sa_quantiles))
                    target_sa_quantiles = next_sa
                    for k in range(len(sample)):
                        for i in range(next_sa.shape[1]):
                            # print('i', i, target_sa_quantiles[k][0][i])
                            target_sa_quantiles[k][i][0] = rewards[k] + (1.0 - dones[k]) * next_sa[k][i][0]
                    # print('len_sample', len(sample))
                    # print('target_sa_quantiles', target_sa_quantiles, 'shape', target_sa_quantiles.shape)
                    if len(sample) == 1:
                        avg_target_qu = target_sa_quantiles[0]
                    else:
                        avg_target_qu = target_sa_quantiles.mean(0)
                    # print('avg_target_qu',  avg_target_qu)
                    avg_target_qu = avg_target_qu.cpu()
                    target_list = avg_target_qu.numpy().tolist()
                    # print('target_list', target_list)
                    target_qu.append(target_list)

            # print('target_qu___', target_qu)
            target_qu = torch.tensor(target_qu)
            # print('target_qu_tensor', target_qu, 'shape', target_qu.shape)
            target_qu = target_qu.transpose(1, 2)
            target_qu = target_qu.to(self.device)
            # print('target_qu', target_qu, 'shape', target_qu.shape)
            # Calculate quantile values of next states and actions at tau_hats.
            # next_sa = next_sa.transpose(1, 2)
            # assert next_sa.shape == (batch_size, 1, self.tau_N)
            #
            # # Calculate target quantile values.
            # # print('rewards', rewards)
            # # print('dones', dones)
            # #print('current_qu', current_sa_quantiles)
            # target_sa_quantiles = next_sa
            # #print('next_sa_quantiles', next_sa_quantiles)
            # # print('next_qu_shape', next_sa_quantiles.shape)
            # for k in range(batch_size):
            #     for i in range(next_sa.shape[2]):
            #         #print('i', i, target_sa_quantiles[k][0][i])
            #         target_sa_quantiles[k][0][i] = rewards[k] + (1.0 - dones[k]) * next_sa[k][0][i]
            # # target_sa_quantiles = rewards[..., None] + (
            # #     1.0 - dones[..., None]) * next_sa_quantiles
            # print('target_qu', target_qu, 'shape', target_qu.shape)
            assert target_qu.shape == (batch_size, 1, self.tau_N)
        # print('target_qu', target_qu)
        # print('next_sa_quantiles', next_sa_quantiles)
        # print('current_sa_quantiles', current_sa_quantiles)
        td_errors = target_qu - current_sa_quantiles
        assert td_errors.shape == (batch_size, self.tau_N, self.tau_N)
        # print('td_errors', td_errors)

        quantile_huber_loss = self.calculate_quantile_huber_loss(
            td_errors, self.tau_hats, self.kappa)

        # print('quantile_huber_loss', quantile_huber_loss)

        return quantile_huber_loss, \
               td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)

    def optimize_adam(self, node, learning_rate, flow):  ### node, learning_rate, flow
        batch_size = self.batch_size
        epochs = 1
        size = self.batch_size
        if self.graph.node_list[node].agent.replay_memory.num_used > batch_size:
            epochs = int(self.graph.node_list[node].agent.replay_memory.num_used / batch_size)
            epochs = min(1, epochs)
        else:
            size = self.graph.node_list[node].agent.replay_memory.num_used

        print('node optimize', node, 'size', size)
        print('num_used', self.graph.node_list[node].agent.replay_memory.num_used)
        # #self.graph.node_list[node].agent.replay_memory.print_statistics()
        # epochs = 1
        # self.graph.node_list[node].agent.replay_memory.prepare_sampling_prob(batch_size=batch_size)

        # Buffer for storing the loss-values of the most recent batches.
        sum_loss = 0
        sum_train = 0
        # self.graph.node_list[node].agent.model.set_lr(learning_rate)
        # print('lr', self.graph.node_list[node].agent.model.lr)
        # self.graph.node_list[node].agent.replay_memory.print_statistics()

        for t in range(epochs):
            # state_batch, action_batch, action_index, reward_batch, next_states, done = self.graph.node_list[
            #     node].agent.replay_memory.random_batch(size)
            samples = self.graph.node_list[node].agent.replay_memory.random_batch(size)
            # print('samples', samples)

            # var_s = torch.tensor(state_batch, dtype=torch.float)
            # var_next_s = torch.tensor(next_states, dtype=torch.float)
            # var_rewards = torch.tensor(reward_batch, dtype=torch.float)
            # var_action = torch.tensor(action_batch, dtype=torch.int64)
            # var_action_index = torch.tensor(action_index, dtype=torch.int64)
            # # var_rewards = var_rewards.cuda()
            #
            # # print('state_batch', state_batch)
            # # print('action_batch', action_batch)
            # # print('var_rewards', var_rewards)
            # # # # print('var_s', variablex)
            # # #
            # # print('reward_batch', reward_batch)
            # # print('next_state_batch', next_states)
            # # print('done', done)
            # var_s = Variable(var_s)
            # var_next_s = Variable(var_next_s)
            # var_s = var_s.to(self.device)
            # # print('var_s', var_s)
            # var_next_s = var_next_s.to(self.device)
            # var_rewards = var_rewards.to(self.device)
            # pred_q = self.graph.node_list[node].agent.model.net(var_s)
            # print('pred_q___', pred_q)
            # pred_q_a = self.get_action_q(pred_q, action_index, size)
            # print('pred_q_a____', pred_q_a)
            # print('b', b)
            # q_values_batch = self.get_tar_q_values(var_next_s, action_batch, var_rewards,
            #                                       done, size)  ### next_states, actions, rewards, done
            # q_values_batch = q_values_batch.cuda()
            # print('pred_q_a', pred_q_a)
            # print('q_values_batch', q_values_batch)

            # objective_loss = self.graph.node_list[node].agent.model.loss_func(pred_q_a, q_values_batch)

            quantile_loss, errors = self.calculate_loss(
                size, samples, node, flow)
            assert errors.shape == (size, 1)

            loss = quantile_loss

            if node == 0:
                sum_loss += loss.data

            if loss < 0.00001:
                continue

            sum_train += 1
            # loss = Variable(loss, requires_grad=True)
            # criterion = torch.nn.MSELoss()
            # print('loss', criterion(pred_q_a, q_values_batch))
            print('loss', loss)

            # sum_loss += objective_loss
            # self.graph.node_list[node].agent.model.set_lr(learning_rate)
            opt = self.graph.node_list[node].agent.model.opt1
            if flow == 1:
                opt = self.graph.node_list[node].agent.model.opt2
            elif flow == 2:
                opt = self.graph.node_list[node].agent.model.opt3
            opt.zero_grad()

            # Manual
            # ewc_loss = self.graph.node_list[node].agent.model.update_ewc_loss(50, cuda=torch.cuda.is_available())
            # loss = objective_loss #+ ewc_loss
            # print('loss', loss)

            loss.backward()
            # # Clip norms of gradients to stebilize training.
            # if grad_cliping:
            #     for net in networks:
            #         torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)

            # if node == 0:
            #     i = 0
            #     for p in self.graph.node_list[node].agent.model.net.parameters():
            #         if i > 0:
            #             break
            #         print('bef_p', p)
            #         #print('grad', p.grad)
            #         # print('p_old', self.graph.node_list[node].agent.model.stored_variable_list[i])
            #         i += 1

            # self.graph.node_list[node].agent.model.opt.zero_grad()
            # loss.backward()
            # print('grad', self.graph.node_list[node].agent.model.params.grad.data)

            # print('lr', self.graph.node_list[node].agent.model.lr)
            # if node == 0:
            #     i = 0
            #     for p in self.graph.node_list[node].agent.model.net.parameters():
            #         if i > 0:
            #             break
            #         print('grad', p.grad)
            #         #print('after_p', p)
            #         # print('p_old', self.graph.node_list[node].agent.model.stored_variable_list[i])
            #         i += 1

            opt.step()

        if node == 0:
            avg_loss = sum_loss / epochs
            self.loss_history.append(avg_loss)
            if flow == 0:
                self.loss_num1 += 1
                writer1.add_scalar('loss', avg_loss, self.loss_num1)
            elif flow == 1:
                self.loss_num2 += 1
                writer2.add_scalar('loss', avg_loss, self.loss_num2)

        if sum_train > 0:
            return True
        else:
            return False

    def train(self, nodes, num_states, flow):
        batch_size = self.batch_size
        sum_flag = 0
        for i in range(len(nodes) - 1):  # nodes_list
            node = nodes[i]
            # How much of the replay-memory should be used.
            # print('node train', node)
            use_fraction = self.graph.node_list[node].agent.replay_fraction.get_value(iteration=num_states)
            replay_memory = self.graph.node_list[node].agent.replay_memory
            # When the replay-memory is sufficiently full.
            # print('replay_memory_num_used', replay_memory.num_used)
            # print('node_train', node, 'memory_len', replay_memory.num_used)
            if replay_memory.is_full() or replay_memory.num_used >= 1:  # batch_size:
                learning_rate = self.graph.node_list[node].agent.learning_rate_control.get_value(iteration=num_states)
                loss_limit = self.graph.node_list[node].agent.loss_limit_control.get_value(iteration=num_states)

                flag = self.optimize_adam(node, learning_rate, flow)
                if flag:
                    sum_flag += 1
            else:
                sum_flag += 1
        if sum_flag > 0:  ### one node is trained
            return True
        else:
            return False

    def dijkstra_raw(self, edges, from_node, to_node):
        # print('from', from_node)
        # print('to', to_node)
        g = defaultdict(list)
        for l, r, c in edges:
            g[l].append((c, r))
        q, seen = [(0, from_node, ())], set()
        # print('q', q)
        while q:
            (cost, v1, path) = heappop(q)
            # print('qq', q)
            # print('v1', v1)
            # print('seen', seen)
            if v1 not in seen:
                seen.add(v1)
                path = (v1, path)
                if v1 == to_node:
                    # print('cost', cost)
                    # print('path', path)
                    return cost, path
                for c, v2 in g.get(v1, ()):
                    if v2 not in seen:
                        heappush(q, (cost + c, v2, path))
        return 10000, None

    def dijkstra_(self, edges, from_node, to_node):
        # print('from_node',from_node)
        # print('to', to_node)
        len_shortest_path = -1
        ret_path = []
        # exist = nx.has_path(self.graph, from_node, to_node)
        # print('exist', exist)
        length, path_queue = self.dijkstra_raw(edges, from_node, to_node)
        if path_queue == None:
            return length, None
        if len(path_queue) > 0:
            len_shortest_path = length  ## 1. Get the length firstly;
            ## 2. Decompose the path_queue, to get the passing nodes in the shortest path.
            left = path_queue[0]
            ret_path.append(left)  ## 2.1 Record the destination node firstly;
            right = path_queue[1]
            while len(right) > 0:
                left = right[0]
                ret_path.append(left)  ## 2.2 Record other nodes, till the source-node.
                right = right[1]
            ret_path.reverse()  ## 3. Reverse the list finally, to make it be normal sequence.
        return len_shortest_path, ret_path

    def dijstra(self, adj, src, dst, n):
        # print('dijstra', 'src', src, 'des', dst)
        dist = [Inf] * n
        dist[src] = 0
        book = [0] * n  # 记录已经确定的顶点
        # 每次找到起点到该点的最短途径
        u = src
        for _ in range(n - 1):  # 找n-1次
            # print('u', u)
            if u == self.graph.node_num - 1 or u == None:
                break
            book[u] = 1  # 已经确定
            # 更新距离并记录最小距离的结点
            next_u, minVal = None, float('inf')
            for v in range(u + 1, n):  # w
                w = adj[u][v]
                # print('u', u, 'v', v, 'w', w)
                if w == Inf:  # 结点u和v之间没有边
                    continue
                # print('book_v', book[v], 'dis1',  dist[u] + w, 'dis2', dist[v])
                if not book[v] and dist[u] + w < dist[v]:  # 判断结点是否已经确定了，
                    dist[v] = dist[u] + w
                    if dist[v] < minVal:
                        next_u, minVal = v, dist[v]
            # 开始下一轮遍历
            u = next_u
        # print('short path', dist[dst])
        return dist[dst]

    def pop_list(self, next_list):
        l = len(next_list)
        node = next_list[l - 1]
        temp = []
        for i in range(l - 1):
            temp.append(next_list[i])
        return node, temp

    def push(self, child, next_list):
        for i in range(len(next_list)):
            if child == next_list[i]:
                return next_list
        next_list.append(child)
        return next_list

    def get_short_path(self, nb, des, k):
        # print('get_short_path', nb, h)
        M = 100000
        adj_weight = np.ones((self.graph.node_num, self.graph.node_num)) * M
        edges = []
        next_list = []
        pre_node = nb
        next_list.append(pre_node)
        while len(next_list) > 0:
            # print('node', node)
            node, next_list = self.pop_list(next_list)
            # print('node', node)
            if node == des:
                continue
            num, child_list = self.graph.get_nb_num(node)
            # print('childs', child_list)
            for i in range(num):
                child = child_list[i]
                # print('child', child)
                key = str(child) + ',' + str(k)
                # print('key', key)
                nb_delay = max(self.node_list[node].nb_delay[key])
                adj_weight[node][child] = nb_delay
                next_list = self.push(child, next_list)
            # h += 1
        # print('pass_weight', self.graph.pass_matrix)
        # print('adj_weight', adj_weight)
        for i in range(len(adj_weight)):
            for j in range(len(adj_weight[0])):
                if i != j and adj_weight[i][j] != M:
                    edges.append((i, j, adj_weight[i][j]))

        # shrot_dis = self.dijstra(adj_weight, nb, self.graph.node_num - 1, self.graph.node_num)
        short_2, path = self.dijkstra_(edges, nb, des)
        # print('short_1', shrot_dis)
        # print('short_2', short_2, 'path', path)
        # Adjacent = [[0, 1, 12, Inf, Inf, Inf],
        #             [Inf, 0, 9, 3, Inf, Inf],
        #             [Inf, Inf, 0, Inf, 5, Inf],
        #             [Inf, Inf, 4, 0, 13, 15],
        #             [Inf, Inf, Inf, Inf, 0, 4],
        #             [Inf, Inf, Inf, Inf, Inf, 0]]
        # Src, Dst, N = 0, 5, 6
        # shrot_dis = self.dijstra(Adjacent, Src, Dst, N)
        return short_2

    def ifVaildNextAction(self, node, state, normal, k):
        D = state[0] * normal[0]
        des = state[3] * normal[3]
        # print('des', des)
        # h = int(state[2] * normal[2])
        num, child_list = self.graph.get_nb_num(node)
        # print('child_list', child_list)
        c_list = []
        q = []
        ucb_list = []
        sum = 0
        for i in range(num):
            # print('node', node, 'child', child_list[i], 'worst', self.graph.G[node][child_list[i]]['worst'] + self.graph.to_des_worst[child_list[i]], 'D', D)
            nb = child_list[i]
            key = str(nb) + ',' + str(k)  # + ',' + str(h)
            # print('key', key)
            # print('nb_delay', self.node_list[node].nb_delay)
            nb_delay = max(self.node_list[node].nb_delay[key])

            nb_to_des_worst = 0
            num, _ = self.graph.get_nb_num(nb)
            if num > 0 and nb < des - 1:
                # key2 = str(k) + ',' + str(h)
                # nb_to_des_worst = self.node_list[nb].worst_to_des[key2]
                if nx.has_path(self.graph.G, nb, des-1):
                    nb_to_des_worst = self.get_short_path(nb, des-1, k)
                else:
                    nb_to_des_worst = 10000
                #b_to_des_worst = self.get_short_path(nb, des - 1, k)
            elif num == 0 and nb < des - 1:
                nb_to_des_worst = self.max_reward
            elif nb > des - 1:
                nb_to_des_worst = self.max_reward
            # print('D', D)
            # print('nb', nb)
            # print('nb_d', nb_delay,'nb_to_des', nb_to_des_worst)
            # print('nb_d+nb_to_des', nb_delay + nb_to_des_worst, 'D', D, '-', nb_delay + nb_to_des_worst - D)
            if (nb_delay + nb_to_des_worst - D) <= 1e-15:
                sum += 1
        if sum == 0:
            return True
        else:
            return False

    def learning(self, deadline, dmax, flow):
        print('learning')
        fl = open("episode.txt", "w")
        d_max = dmax
        test_d = deadline

        print('flow', flow)

        # print('init_p')
        # for p in self.graph.node_list[0].agent.model.net.parameters():
        #     print('p', p)
        min_reward = 1000000
        for i in range(len(self.node_list) - 1):
            if flow > 0:
                self.node_list[i].agent.replay_memory.reset_memory_size()
                # self.node_list[i].agent.get_copy_memory()
                # Reset model
                # self.node_list[i].agent.model.restore()
                self.node_list[i].agent.model.last_t = 1
                self.node_list[i].agent.epsilon_greedy.num_actions = self.node_list[i].agent.action_dim

                self.node_list[i].agent.replay_memory.reset_memory_size()
                self.graph.node_list[i].num = 1
                self.graph.node_list[i].action_num = np.ones(self.graph.node_list[i].action_dim)
                self.graph.node_list[i].agent.model.load_parametersforf1()

        # num_states = 0
        reward_history = []
        # key = str(int(k)) + ',' + str(0)
        # print('source_to_des d_list', d_list)
        # print('len_d_list', len(d_list))
        flag_loss = False
        flag_ewc = False
        # print('min_worst', min_worst)
        # print('max_worst', max_worst)
        u = 0
        l = 1  # len(d_list)
        # print('min_worst', min_worst, 'max_worst', max_worst)

        while u < l:
            # for u in range(l):
            # u += 1
            # if k > 0 and u > 0:
            #     break
            # if k == 0:
            d = deadline
            # print('deadline', d)
            # else:
            #     d = max(d_list)
            # max_d = max(d_list)
            max_flow = self.flow_num
            max_h = self.graph.node_num
            normal = [d_max, max_flow, max_h, max_h]
            # print('d_list', d_list)
            # print('u', u)
            e = 0
            train_flag = True
            notchange_sum = 0
            episodes = self.episodes
            # if flow > 0:
            #     episodes = 1000
            while e <= episodes:
                # if notchange_sum >= 1000:
                #     break
                # print('f', flow, 'u', u, 'eposide', e)
                # print('f', k, 'u', u, 'eposide', e, file=fl)
                # print('u_len', len(d_list))
                # flow += 1.0
                # print('flow', flow)
                source = self.sources[flow]
                des = self.dess[flow]

                state = self.env.reset(source, des, d, flow + 1.0)
                # print('source', source, 'des', des)
                # print('bf_state', state)
                state = [a / b for a, b in zip(state, normal)]
                # print('af_state', state)
                reward_episode = 0
                e += 1
                node_id = source
                travel_nodes = []
                travel_nodes.append(node_id)
                h = 0
                self.graph.node_list[node_id].num += 1
                # print('node', node_id, 'num', self.graph.node_list[0].num)

                for t in range(100):
                    # Get q values
                    var_state = torch.tensor(state, dtype=torch.float)
                    var_state = var_state.to(self.device)
                    # print('state', state, 'temp_s', temp_s)
                    q_values = self.node_list[node_id].agent.model.calculate_q(1, var_state,
                                                                               flow)  ### self.model.get_q_values(state)[0]
                    # Determine action based on epsilon greedy
                    action, action_index = self.get_action(q_values, e, state, node_id, normal)
                    # print('state', state, 'action', action, 'node', node_id)
                    # print('q_values', q_values)
                    self.graph.node_list[action].num += 1
                    if action == -1:
                        break

                    nb_num, _ = self.graph.get_nb_num(action)

                    # print('node_num', self.graph.node_list[action].num, 'action_num', self.graph.node_list[node_id].action_num)

                    # Act the action and get result
                    next_state, reward, info = self.env.step(state, node_id, action, normal, training=True)

                    # print('learn_pass_weight_1', self.pass_weight)
                    # print('node', node_id, 'action', action, 'reward', reward, 'next_state', next_state)
                    self.graph.G[node_id][action]['num'] += 1
                    self.graph.G[node_id][action]['sum_d'] += reward
                    # avg_reward = self.graph.G[node_id][action]['sum_d'] / self.graph.G[node_id][action]['num']
                    # next_state = next_state/normal
                    next_state = [a / b for a, b in zip(next_state, normal)]

                    # print('next_state', next_state)

                    # print('ave_reward', avg_reward)
                    vaild_action = False
                    if action != des:
                        vaild_action = self.ifVaildNextAction(action, next_state, normal, flow)
                    # print('node', node_id, 'action', action, 'vaild', vaild_action)

                    # if vaild_action:
                    #     self.node_list[node_id].agent.replay_memory.add(temp_s, action, action_index,
                    #                                                     reward + 100,
                    #                                                     next_state, True)

                    temp_next_s = copy.deepcopy(next_state)

                    # self.update_all_q_values_node(node_id, action)
                    # var_temp_s = torch.tensor(temp_s, dtype=torch.float)
                    # var_temp_next_s = torch.tensor(temp_next_s, dtype=torch.float)
                    utility = reward
                    if (nb_num == 0 and action != des) or (vaild_action):
                        utility += 100
                    # print('nb_num', nb_num)
                    # print('utilty', utility)
                    sample = (state, action, action_index, utility, temp_next_s, info)
                    key = str(state) + ',' + str(action)
                    self.node_list[node_id].agent.replay_memory.add(key, sample)

                    # self.node_list[node_id].agent.train(state, q_values, action, reward, next_state, info, e) ### state, q_values, action, reward, end_life, num_states

                    node_id = action
                    travel_nodes.append(node_id)
                    state = next_state

                    # Add reward
                    reward_episode += reward

                    h += 1
                    if node_id == des or nb_num == 0:
                        break

                if e == self.episodes - 1:
                    flag_loss = True

                # print('len_d_list', len(d_list))
                # print('u', u)
                if u == l - 1 and e == self.episodes - 1:
                    flag_ewc = True

                _ = self.train(travel_nodes, e, flag_loss, flag_ewc, flow)

                # reward_history.append(reward_episode)
                # print('iter', u*self.episodes+e)
                # writer.add_scalar('rewards', reward_episode,  u*self.episodes+e)
                # logger.log_value('sum reward', reward_episode, u*self.episodes+e)

                iter = flow * (self.episodes) + u * self.episodes + e
                _, rewards = self.test(test_d, dmax, flow, e)

                # print('change_flag', change_flag, 'notchange', notchange_sum)

                for i in range(self.graph.node_num - 1):
                    self.graph.node_list[i].agent.model.save_parameters(flow)

            # for i in range(self.graph.node_num - 1):
            #     self.graph.node_list[i].agent.model.save_parameters(k)

            u += 1
            # opt_path = self.test(d, k, False)

            dd = deadline
            # if k == 1:
            #     dd = d_list[len(d_list)-1]
            path, _ = self.test(dd, dmax, flow, -1)
            # print('path', path)
            for i in range(len(path) - 1):
                node = path[i]
                next_node = path[i + 1]
                self.pass_weight[node][next_node] += 1
            # print('pass_weight', self.pass_weight)

            for i in range(len(self.graph.adj_matrix)):
                for j in range(len(self.graph.adj_matrix)):
                    if self.graph.adj_matrix[i][j] == 1:
                        # self.G[i][j]['off_worst'] = self.max_worst - worst
                        self.graph.G[i][j]['sum_d'] = 0
                        self.graph.G[i][j]['num'] = 0

        # plt.figure()
        # x = len(self.reward_history)
        # plt.plot(x, self.reward_history)
        # plt.show()

        return path

    def packetForward(self, node, current_slot, e_t, flow):
        #print('learn node', node, 'packetForward')
        # print('packets', self.graph.node_list[node].packets)
        # print('deadlines', self.graph.node_list[node].ddls)
        # print('slots', self.graph.node_list[node].slots)
        # print('current_slot', current_slot)
        # if node == 7:
        #     self.graph.node_list[node].printPacket()
        bl = True
        sum_delay = 0
        while bl:
            # print('packet_len', self.graph.node_list[node].getPacketlen())
            if self.graph.node_list[node].getPacketlen(self.dess) == 0:
                break
            packet = self.graph.node_list[node].getPacket(current_slot, self.dess)
            if packet == None:
                break
            # print('packet_id', packet.id)
            source = self.sources[packet.type]
            des = self.dess[packet.type]
            D = packet.reddl
            arrivalD = packet.arrivalD
            # print('packet', packet, 'D', D, 'arrivalD', arrivalD)
            slot = packet.slot
            # print('slot', slot)
            if slot > current_slot:
                break
            dmax = self.dmax_set[packet.type]
            # self.graph.node_list[node].popPacket()
            self.graph.node_list[node].setPacketFlag(packet)  ### mark forwarded

            normal = [dmax, self.flow_num, self.graph.node_num, self.graph.node_num]
            state = [D, packet.type + 1, source + 1, des + 1]
            #print('state1', state)
            state = [a / b for a, b in zip(state, normal)]
            #print('state2', state)

            nb_packets = self.graph.getNbPackets(node, packet.type, current_slot)
            if len(nb_packets) > 0:
                for i in range(len(nb_packets)):
                    state.append(nb_packets[i] / self.maxPackets)

            var_state = torch.tensor(state, dtype=torch.float)
            var_state = var_state.to(self.device)
            # if node == 7:
            #     print('nb_packets', nb_packets, 'state', state)
            # print('state', state, 'temp_s', temp_s)
            # print('node', node)
            # print('state', var_state)
            #print('get_values_type', packet.type)
            q_values = self.graph.node_list[node].agent.model.calculate_q(1, var_state, packet.type)  ### self.model.get_q_values(state)[0]
            # print('q_values', q_values)
            # Determine action based on epsilon greedy
            if packet.type == flow:
                iter = e_t + current_slot
                action, action_index = self.get_action(q_values, iter, state, node, normal)
            else:
                # print('get_max')
                # print('packet', packet, 'flow', flow)
                action, action_index = self.get_action_max(q_values, state, node, normal)

            # if action == self.dess[packet.type]:
            #     print('nb_packet', nb_packets)
            #     print('des', des)
            #     _, child_list = self.graph.get_nb_num(node)
            #     print('child_list', child_list)

            if action == -1:
                # print('finish')
                continue
            next_state, delay, info = self.env.step(state, node, action, normal, training=True)

            next_state = [a / b for a, b in zip(next_state, normal)]

            nb_packets = self.graph.getNbPackets(action, packet.type, current_slot)
            if len(nb_packets) > 0:
                for i in range(len(nb_packets)):
                    next_state.append(nb_packets[i] / self.maxPackets)
            # if node == 7:
            #     print('next_nb_packets', nb_packets, 'next_state', next_state, 'action', action)
            #
            # print('state', state, 'action', action, 'node', node)
            # print('qvalues', q_values)
            delay = int(delay)

            # print('delay', delay)

            # reward = delay + queue_delay
            sum_delay += delay

            # utility = reward
            nb_num, _ = self.graph.get_nb_num(action)

            if packet.type == flow:
                l = len(packet.path)
                if l > 1:
                    pre_node = packet.path[l - 2]
                    # print('packet_id', packet.id, 'packet_des', des)
                    # print('path', packet.path)
                    # print('node adds sample', pre_node)
                    # print('type', packet.type)
                    queue_delay = arrivalD - D
                    self.node_list[pre_node].addSample(packet, queue_delay, normal)

                    # print('nb_num', nb_num, 'des', des, 'action', action)

                if action < des and nb_num == 0:
                    utility = 100
                    sample = (state, action, action_index, utility, next_state, info)
                    key = str(state) + ',' + str(action)
                    self.node_list[node].agent.replay_memory.add(key, sample)
                    self.graph.node_list[node].popPacket(packet)
                elif action == des:
                    # print('action is the des')
                    sample = (state, action, action_index, delay, next_state, info)
                    key = str(state) + ',' + str(action)
                    self.node_list[node].agent.replay_memory.add(key, sample)
                    self.graph.node_list[node].popPacket(packet)
                    # if node == 7:
                    #     for i in range(len(self.graph.node_list[node].packets)):
                    #         print('id', self.graph.node_list[node].packets[i].id, 'state', self.graph.node_list[node].packets[i].state)

                    # packet.setState(state, action, action_index, delay, next_state, info)
                else:
                    # print('set')
                    packet.setState(state, action, action_index, delay, next_state, info)
                    # if node == 7:
                    #     for i in range(len(self.graph.node_list[node].packets)):
                    #         print('id', self.graph.node_list[node].packets[i].id, 'state', self.graph.node_list[node].packets[i].state)
            else:
                self.graph.node_list[node].popPacket(packet)
                # sample = (state, action, action_index, utility, next_state, info)
                # key = str(state) + ',' + str(action)
                # self.node_list[node].agent.replay_memory.add(key, sample)

            # print('after_forward', node, action)
            # print('next_node packets')
            # self.graph.node_list[action].printPacket()
            next_slot = current_slot + math.ceil(delay / self.slot_time)
            # print('packet', packet, 'type', packet.type)
            self.graph.node_list[action].addPacket(packet, D - delay, next_slot, self.dess)
            # print('after next_node packets')
            # self.graph.node_list[action].printPacket()
            # print('finish')
            # return 100000, -1, new_path

            if node == 0:
                if sum_delay > 10000:
                    break
            else:
                if sum_delay > 0:
                    break
            # print('delay', delay)

        self.graph.node_list[node].setddl(sum_delay, current_slot, self.dess)
        # print('after node forward')
        # self.graph.node_list[node].printPacket()

    def learnPacket(self, T, flow):
        test_Step = 5
        test_num = [0, 0, 0]
        if flow > 0:
            for i in range(self.graph.node_num - 1):
                self.node_list[i].agent.replay_memory.reset_memory_size()
                # self.node_list[i].agent.get_copy_memory()
                # Reset model
                # self.node_list[i].agent.model.restore()
                self.node_list[i].agent.model.last_t = 1
                self.node_list[i].agent.epsilon_greedy.num_actions = self.node_list[i].agent.action_dim

                self.node_list[i].agent.replay_memory.reset_memory_size()
                self.graph.node_list[i].num = 1
                self.graph.node_list[i].action_num = np.ones(self.graph.node_list[i].action_dim)
                #print('node', i)
                self.graph.node_list[i].agent.model.load_parametersforf1()
                if flow > 1:
                     self.graph.node_list[i].agent.model.load_parametersforf2()
                self.graph.node_list[i].clean()

        min_delay = 10000
        # d_max = max(d_list)
        # print('learn')
        id = 0
        for e in range(self.episodes):
            for t in range(1, T):
                for i in range(flow + 1):
                    if t % self.slots[i] == 0:
                        id += 1
                        packet = Packet(id, i, self.deadlines[i], t,
                                        self.sources[i])  # id, type, arrivalD, arrivalSlot, start_node
                        self.graph.node_list[self.sources[i]].addPacketFirst(packet)
                        # print('node', self.sources[i], 'add packets', id)

                nodes = []
                for i in range(self.N - 1):
                    nodes.append(i)
                    if self.graph.node_list[i].getPacketlen(self.dess) > 0:
                        # print('node', i, 'forward_packets', self.graph.node_list[i].packets)
                        self.packetForward(i, t, e*T, flow)
                        # print('fl_class', fl_class)
                        # print('node', i, 'af_packets', self.graph.node_list[i].packets)
                _ = self.train(nodes, t, flow)

                if t % test_Step == 0:
                    # print('test__')
                    avg_ete = self.testPacket(T, flow)
                    if flow == 0:
                        test_num[flow] += 1
                        l = test_num[flow]
                        writer1.add_scalar('rewards', avg_ete, l)
                        filename = 'reward1.txt'
                        with open(filename, 'a') as file_object:
                            file_object.write(str(avg_ete))
                            file_object.write('\n')
                    elif flow == 1:
                        test_num[flow] += 1
                        l = test_num[flow]
                        writer2.add_scalar('rewards', avg_ete, l)
                        self.reward_history2.append(avg_ete)
                        # f2.write(str(reward_episode))
                        filename = 'reward2.txt'
                        with open(filename, 'a') as file_object:
                            file_object.write(str(avg_ete))
                            file_object.write('\n')
                    else:
                        test_num[flow] += 1
                        filename = 'reward3.txt'
                        with open(filename, 'a') as file_object:
                            file_object.write(str(avg_ete))
                            file_object.write('\n')
                    # print('avg_ete', avg_ete)
                    if avg_ete > 0 and avg_ete <= min_delay:
                        min_delay = avg_ete
                        for i in range(self.graph.node_num - 1):
                            # print('node saves', i)
                            # print('state_dim', self.graph.node_list[i].agent.model.state_dim, 'agent_id', self.graph.node_list[i].agent.model.agent_id)
                            self.graph.node_list[i].agent.model.save_parameters(flow)

            # print('after_slot')
            bl = True
            t = T
            while bl:
                sum = 0
                nodes = []
                for i in range(self.N - 1):
                    nodes.append(i)
                    if self.graph.node_list[i].getPacketlen(self.dess) > 0:
                        sum += 1
                        # print('node', i, 'forward_packets', self.graph.node_list[i].packets)
                        self.packetForward(i, t, e*T, flow)
                        # print('node', i, 'af_packets', self.graph.node_list[i].packets)

                _ = self.train(nodes, t, flow)

                if t > 1 and t % test_Step == 0:
                    avg_ete = self.testPacket(T, flow)
                    # print('avg_ete', avg_ete)
                    if flow == 0:
                        test_num[flow] += 1
                        l = test_num[flow]
                        writer1.add_scalar('rewards', avg_ete, l)
                        filename = 'reward1.txt'
                        with open(filename, 'a') as file_object:
                            file_object.write(str(avg_ete))
                            file_object.write('\n')
                    elif flow == 1:
                        test_num[flow] += 1
                        l = test_num[flow]
                        writer2.add_scalar('rewards', avg_ete, l)
                        self.reward_history2.append(avg_ete)
                        # f2.write(str(reward_episode))
                        filename = 'reward2.txt'
                        with open(filename, 'a') as file_object:
                            file_object.write(str(avg_ete))
                            file_object.write('\n')
                    else:
                        test_num[flow] += 1
                        filename = 'reward3.txt'
                        with open(filename, 'a') as file_object:
                            file_object.write(str(avg_ete))
                            file_object.write('\n')

                    if avg_ete > 0 and avg_ete <= min_delay:
                        min_delay = avg_ete
                        for i in range(self.graph.node_num - 1):
                            self.graph.node_list[i].agent.model.save_parameters(flow)

                if sum == 0:
                    bl = False
                t += 1

        # print('spent_t', t)

    def testpacketForward(self, graph, node, current_slot, flow):
        #print('test', node)
        # print('node', node)
        # print('packets', self.graph.node_list[node].packets)
        # print('deadlines', self.graph.node_list[node].ddls)
        bl = True
        sum_delay = 0
        while bl:
            # print('len', graph.node_list[node].getPacketlen())
            if graph.node_list[node].getPacketlen(self.dess) == 0:
                break
            packet = graph.node_list[node].getPacket(current_slot, self.dess)
            if packet == None or node == self.dess[packet.type]:
                break
            source = self.sources[packet.type]
            des = self.dess[packet.type]
            D = packet.reddl
            # arrivalD = packet.arrivalD
            # path = graph.node_list[node].paths[0]
            slot = packet.slot
            if slot > current_slot:
                break
            dmax = self.dmax_set[packet.type]
            graph.node_list[node].popPacket(packet)

            normal = [dmax, self.flow_num, graph.node_num, graph.node_num]
            state = [D, packet.type + 1, source + 1, des + 1]
            #print('state1', state)
            state = [a / b for a, b in zip(state, normal)]
            #print('state2', state)

            nb_packets = graph.getNbPackets(node, packet.type, current_slot)
            if len(nb_packets) > 0:
                for i in range(len(nb_packets)):
                    state.append(nb_packets[i] / self.maxPackets)

            var_state = torch.tensor(state, dtype=torch.float)
            var_state = var_state.to(self.device)
            # print('state', state, 'temp_s', temp_s)
            # print('node', node)
            # print('state', var_state)
            #print('get_values_type', packet.type)
            q_values = graph.node_list[node].agent.model.calculate_q(1, var_state, packet.type)  ### self.model.get_q_values(state)[0]
            #q_values_1 = self.graph.node_list[node].agent.model.calculate_q(1, var_state, packet)  ### self.model.get_q_values(state)[0]

            #print('q_values_1', q_values_1, 'q_values', q_values)
            # Determine action based on epsilon greedy

            action, action_index = self.get_action_max(q_values, state, node, normal)

            if action == -1:
                # print('finish')
                continue
            next_state, delay, info = self.env.step(state, node, action, normal, training=False)

            # print('state', state, 'action', action, 'node', node)
            # print('qvalues', q_values)
            delay = int(delay)
            # print('delay', delay)
            # graph.node_list[node].setddl(delay, current_slot)

            sum_delay += delay

            # self.graph.node_list[node].setslot(slots)

            # print('after_forward', node, action)
            next_slot = current_slot + math.ceil(delay / self.slot_time)
            graph.node_list[action].addPacket(packet, D - delay, next_slot, self.dess)
            # print('finish')
            # return 100000, -1, new_path

            if node == 0:
                if sum_delay > 10000:
                    break
            else:
                if sum_delay > 0:
                    break

            # if sum_delay + 30 > self.slot_time:
            #     break
            # print('delay', delay)
        graph.node_list[node].setddl(sum_delay, current_slot, self.dess)
        # print('after test_forward')

    def testPacket(self, T, flow):
        # d_max = max(d_list)
        # print('testPacket')
        testGraph = copy.deepcopy(self.graph)
        for i in range(testGraph.node_num):
            testGraph.node_list[i].clean()
            # print('node_packets', testGraph.node_list[i].packets)
        id = 0
        for t in range(1, T):
            # print('t', t)
            for i in range(flow + 1):
                if t % self.slots[i] == 0:
                    id += 1
                    packet = Packet(id, i, self.deadlines[i], t, self.sources[i])
                    testGraph.node_list[self.sources[i]].addPacketFirst(packet)
                    # print('source_packets', testGraph.node_list[self.sources[i]].packets)
            nodes = []
            for i in range(self.N - 1):
                # print('node packet_len', testGraph.node_list[i].getPacketlen())
                if testGraph.node_list[i].getPacketlen(self.dess) > 0:
                    nodes.append(i)
                    # print('nodes', nodes)
                    # print('bf_packets', testGraph.node_list[i].packets)
                    self.testpacketForward(testGraph, i, t, flow)
                    # print('fl_class', fl_class)
                    # print('af_packets', testGraph.node_list[i].packets)

        # print('after_slot')
        bl = True
        t = T
        while bl:
            # print('t', t)
            sum = 0
            nodes = []
            for i in range(self.N - 1):
                len = testGraph.node_list[i].getPacketlen(self.dess)
                # print('node packet_len', i, len)
                if len > 0:
                    sum += 1
                    nodes.append(i)
                    # print('bf_packets', rand_graph.node_list[i].packets)
                    self.testpacketForward(testGraph, i, t, flow)
                    # print('af_packets', rand_graph.node_list[i].packets)
            # print('sum', sum)
            if sum == 0:
                bl = False
            t += 1

        ### calculate delay of packets
        avg_ete = testGraph.node_list[self.dess[flow]].get_ete(flow, self.deadlines[flow])
        return avg_ete

    def test(self, d, dmax, k, flag):
        source = self.sources[k]
        des = self.des[k]
        # print('test')
        if flag == -1:
            for i in range(self.graph.node_num - 1):
                if k == 0:
                    self.graph.node_list[i].agent.model.load_parametersforf1()
                elif k == 1:
                    self.graph.node_list[i].agent.model.load_parametersforf2()
                else:
                    self.graph.node_list[i].agent.model.load_parametersforf3()
            # print('test_d', d, 'flow', k)
        # else:
        #     print('episode_d', d, 'flow', k)
        d_max = dmax
        # for f in range(self.flow_num):
        #     key = str(int(f)) + ',' + str(0)

        # num_states = 0
        # print('flow', k)
        # key = str(int(k)) + ',' + str(0)
        # d_list = self.D_list[key]
        # print('source_to_des d_list', d_list)
        # max_d = max(d_list)
        max_flow = self.flow_num
        max_h = self.graph.node_num
        normal = [d_max, max_flow, max_h, max_h]
        # print('d_list', d_list)
        # print('u', u)

        state = self.env.reset(source, des, d, k + 1.0)
        # print('bf_state', state)
        state = [a / b for a, b in zip(state, normal)]
        # print('af_state', state)
        reward_episode = 0
        node_id = source
        travel_nodes = []
        travel_nodes.append(node_id)
        h = 0
        bl = 1
        penaty = 10000

        for t in range(100):
            # print('t', t)
            # Get q values
            # print('temp_s', temp_s)
            var_state = torch.tensor(state, dtype=torch.float)
            var_state = var_state.to(self.device)
            # print('state', state, 'temp_s', temp_s)
            q_values = self.node_list[node_id].agent.model.calculate_q(1,
                                                                       var_state,
                                                                       k)  ### self.model.get_q_values(state)[0]
            # Determine action based on epsilon greedy
            action, action_index = self.get_action_max(q_values, state, node_id, normal, Test=False)
            # print('state', state, 'action', action, 'node', node_id)
            # print('q_values', q_values)
            nb_num, _ = self.graph.get_nb_num(action)
            # print('nb_num', nb_num)
            if flag == -1:
                print('q_values', q_values)
                print('state', state, 'action', action, 'node', node_id)
            if action == -1 or (nb_num == 0 and action != des):
                bl = 0
                if k == 0:
                    penaty = 1000
                    l = len(self.reward_history1)
                    writer1.add_scalar('rewards', 1000, l)
                    self.reward_history1.append(1000)
                    # print(100, file=f1)
                    filename = 'reward1.txt'
                    with open(filename, 'a') as file_object:
                        file_object.write(str(1000))
                        file_object.write('\n')
                elif k == 1:
                    penaty = 1500
                    l = len(self.reward_history2)
                    writer2.add_scalar('rewards', 1500, l)
                    self.reward_history2.append(1500)
                    # print(500, file=f2)
                    filename = 'reward2.txt'
                    with open(filename, 'a') as file_object:
                        file_object.write(str(1500))
                        file_object.write('\n')
                else:
                    filename = 'reward3.txt'
                    with open(filename, 'a') as file_object:
                        file_object.write(str(1500))
                        file_object.write('\n')
                    # print(1000, file=f3)
                return travel_nodes, penaty

            # Act the action and get result
            next_state, _, info = self.env.step(state, node_id, action, normal, training=False)
            # print('next_state', next_state)
            # self.graph.G[node_id][action]['num'] += 1
            # self.graph.G[node_id][action]['sum_d'] += reward
            # avg_reward = self.graph.G[node_id][action]['sum_d'] / self.graph.G[node_id][action]['num']
            # next_state = next_state/normal
            next_state = [a / b for a, b in zip(next_state, normal)]

            # self.node_list[node_id].agent.train(state, q_values, action, reward, next_state, info, e) ### state, q_values, action, reward, end_life, num_states
            key = str(action) + ',' + str(int(k))  # + ',' + str(h + 1)
            real_d = self.graph.node_list[node_id].nb_delay[key]
            # print('key', key, 'real_d', real_d)
            real_pro = self.graph.node_list[node_id].nb_delay_pro[key]

            avg_reward = 0
            for i in range(len(real_d)):
                delay = real_d[i]
                avg_reward += delay * real_pro[i]

            # if flag == -1:
            #     #print('travel_nodes', travel_nodes)
            #     print('avg_reward', avg_reward)

            node_id = action
            travel_nodes.append(node_id)
            state = next_state

            # Add reward

            reward_episode += avg_reward

            h += 1
            if node_id == des:
                break
        if flag == -1:
            print('travel_nodes', travel_nodes)
            print('reward', reward_episode)
        # else:
        #     print('travel_nodes', travel_nodes)
        #     print('reward', reward_episode)
        # print('bl', bl)
        # reward_history.append(reward_episode)
        # print('iter', u*self.episodes+e)
        if bl == 1:
            if k == 0:
                l = len(self.reward_history1)
                writer1.add_scalar('rewards', reward_episode, l)
                self.reward_history1.append(reward_episode)
                # print(100, file=f1)
                # f1.write(str(reward_episode))
                filename = 'reward1.txt'
                with open(filename, 'a') as file_object:
                    file_object.write(str(reward_episode))
                    file_object.write('\n')
            elif k == 1:
                l = len(self.reward_history2)
                writer2.add_scalar('rewards', reward_episode, l)
                self.reward_history2.append(reward_episode)
                # f2.write(str(reward_episode))
                filename = 'reward2.txt'
                with open(filename, 'a') as file_object:
                    file_object.write(str(reward_episode))
                    file_object.write('\n')
            else:
                filename = 'reward3.txt'
                with open(filename, 'a') as file_object:
                    file_object.write(str(reward_episode))
                    file_object.write('\n')
        # logger.log_value('sum reward', reward_episode, u*self.episodes+e)

        # print('after_task1, parameters are')
        # for p in self.graph.node_list[0].agent.model.net.parameters():
        #     print('param', p)

        return travel_nodes, reward_episode

    def get_opt_path(self, d_set, dmax):
        for i in range(len(self.node_list) - 1):
            self.graph.node_list[i].agent.model.load_parameters()

        cost_S = []
        trip_S = []
        print('d_set', d_set)
        # print('get_opt_path')

        for k in range(self.flow_num):
            cost_set = []
            trip_set = []
            d_max = dmax[k]
            d = d_set[k]
            reward_history = []
            key = str(int(k)) + ',' + str(0)
            # max_d = max(d_list)
            max_flow = self.flow_num
            max_h = self.graph.node_num
            normal = [d_max, max_flow, max_h, max_h]
            # print('d_list', d_list)
            # print('d', d)
            # print('u', u)

            source = self.sources[k]
            des = self.des[k]

            state = self.env.reset(source, des, d, k + 1.0)
            # print('bf_state', state)
            state = [a / b for a, b in zip(state, normal)]
            # print('af_state', state)
            reward_episode = 0
            node_id = source
            travel_nodes = []
            travel_nodes.append(node_id)
            h = 0
            bl = 1

            for t in range(100):
                # Get q values
                var_state = torch.tensor(state, dtype=torch.float)
                var_state = var_state.to(self.device)
                # print('state', state, 'temp_s', temp_s)
                q_values = self.node_list[node_id].agent.model.calculate_q(1, var_state,
                                                                           k)  ### self.model.get_q_values(state)[0]
                # Determine action based on epsilon greedy
                print('q_values', q_values)
                action, action_index = self.get_action_max(q_values, state, node_id, normal, Test=True)
                print('state', state, 'action', action, 'node', node_id)
                if action == -1:
                    bl = 0
                    break

                # Act the action and get result
                next_state, _, info = self.env.step(state, node_id, action, normal, training=False)
                # self.graph.G[node_id][action]['num'] += 1
                # self.graph.G[node_id][action]['sum_d'] += reward
                # avg_reward = self.graph.G[node_id][action]['sum_d'] / self.graph.G[node_id][action]['num']
                # next_state = next_state/normal
                next_state = [a / b for a, b in zip(next_state, normal)]

                # self.node_list[node_id].agent.train(state, q_values, action, reward, next_state, info, e) ### state, q_values, action, reward, end_life, num_states
                key = str(action) + ',' + str(int(k))  # + ',' + str(h + 1)
                real_d = self.graph.node_list[node_id].nb_delay[key]
                # print('key', key, 'real_d', real_d)
                real_pro = self.graph.node_list[node_id].nb_delay_pro[key]

                avg_reward = 0
                # print('real_d', real_d)
                for i in range(len(real_d)):
                    # queuing_time = self.env.get_queueing_time(real_d[i], k)
                    delay = real_d[i]
                    # delay = real_d[i] + queuing_time
                    # print('delay', delay)
                    avg_reward += delay * real_pro[i]

                node_id = action
                travel_nodes.append(node_id)
                state = next_state

                # Add reward

                reward_episode += avg_reward

                h += 1
                if node_id == des:
                    break
            # print('reward', reward_episode)
            reward_history.append(reward_episode)
            # print('iter', u*self.episodes+e)
            if bl == 1:
                cost_set.append(reward_episode)
                trip_set.append(travel_nodes)
                print('cost', reward_episode)
                print('trip', travel_nodes)
            else:
                print('no path')
                cost_set.append(1000)
                trip_set.append(None)
            # print('d', d)
            # print('opt_path', travel_nodes)
            for i in range(len(travel_nodes) - 1):
                node = travel_nodes[i]
                next_node = travel_nodes[i + 1]
                # self.graph.pass_matrix[node][next_node] += 1
            # print('pass_weight', self.graph.pass_matrix)
            # logger.log_value('sum reward', reward_episode, u*self.episodes+e)

            # print('after_task1, parameters are')
            # for p in self.graph.node_list[0].agent.model.net.parameters():

            #     print('param', p)
            cost_S.append(cost_set)
            trip_S.append(trip_set)
        return cost_S, trip_S







