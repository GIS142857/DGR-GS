import json
import copy
import torch
import random
import numpy as np
from config import *
from agent import Agent
from packet import Packet
from tensorboardX import SummaryWriter
from GeneralTopology.TON24.TonMacLayer import TonMacLayer
from GeneralTopology.TON24.TonPhyLayer import TonPhyLayer

writer1 = SummaryWriter(log_dir='log999_11')
writer2 = SummaryWriter(log_dir='log999_22')
writer3 = SummaryWriter(log_dir='log999_33')


class Node(object):
    # set the parameters of a node, including its physical layer, mac layer, simulator,,,
    DEFAULT_MSG_NBITS = 1000 * 8
    def __init__(self, sim, id, src, dst, arrival_rates, pos, device):
        self.sim = sim
        self.id = id
        self.isSource = False
        if self.id in src:
            self.isSource = True
            self.packet_pri = src.index(self.id)
            self.arrival_rate = arrival_rates[FLOW_MAP[src.index(self.id)]]
        self.pos = pos
        self.phy = TonPhyLayer(self)
        self.mac = TonMacLayer(self)
        self.neighbors = ADJ_TABLE[self.id]
        self.nb_pri_queues = {key: {'flow1': 0, 'flow2': 0, 'flow3': 0} for key in self.neighbors}
        self.send_cnt = 0
        self.reces_for_me = []
        self.has_reces = []
        self.timeout = self.sim.timeout
        self.start_time = 0
        self.device = device
        self.time_slot = FRAME_SLOT[id]
        self.loss_history = []
        self.loss_num1 = 0
        self.loss_num2 = 0
        self.path_num = 0
        self.e2ed = []
        self.episode = 0
        self.sum_sources = len(SRC)
        self.sum_nodes = SUM_NODES
        self.state_dim = len(self.neighbors) + 1
        self.end_to_end_delay = {}

    @property
    def now(self):
        return self.sim.env.now

    def setAgent(self):
        # Set an agent for the node to learn the delay distribution
        self.action_dim = len(self.neighbors)
        self.agent = Agent(self.state_dim, self.action_dim, self.neighbors, self.id, self.device, TAU_N)

    def create_event(self):
        return self.sim.env.event()

    def delayed_exec(self, delay, func, *args, **kwargs):
        return self.sim.delayed_exec(delay, func, *args, **kwargs)

    ############################
    def set_layers(self, phy=None, mac=None):
        if phy is not None:
            self.phy = phy(self)
        if mac is not None:
            self.mac = mac(self)

    def get_nextnode(self, packet):
        # select the next node
        for nb in self.neighbors:
            if nb == packet.des_node_id:
                return nb
        # print('len(self.neighbors)', len(self.neighbors), self.neighbors)
        # print(packet.des, self.id, self.neighbors)
        # print(self.id, packet.des, self.dfs_find_all_paths(ADJ_TABLE, self.id, packet.des))
        next_node = random.choices(self.dfs_find_all_paths(ADJ_TABLE, self.id, packet.des_node_id))[0]
        # print('best:', self.get_best_action(packet))
        return next_node

    def dfs_find_all_paths(self, adj_table, start, target, path=None, all_paths=None):
        if path is None:
            path = []  # 当前路径
        if all_paths is None:
            all_paths = []  # 所有路径

        # 将当前节点添加到路径
        path.append(start)

        # 如果到达目标节点，保存当前路径
        if start == target:
            if path[1] not in all_paths:
                all_paths.append(path[1])
        else:
            # 遍历当前节点的所有邻居
            for neighbor in adj_table.get(start, []):
                if neighbor not in path:  # 防止循环
                    self.dfs_find_all_paths(adj_table, neighbor, target, path, all_paths)

        # 回溯，移除当前节点
        path.pop()

        return all_paths

    def get_best_action(self, packet):
        # calculate worst delay of each path
        # print('neighbor', neighbor)
        txtpath = './CDF_data/E2ED_' + str(self.id) + '.txt'
        with open(txtpath, 'r') as fl:
            js = fl.read()
            dis_set = json.loads(js)

        worstD_list = []
        aveD_list = []
        cand_neighbors = []
        # print('dis_set', dis_set)
        # print('keys', dis_set[str(packet.priority)].keys())
        path_set = []
        for key in dis_set[str(packet.priority)].keys():
            arry = key.split('_')
            arry = list(map(int, arry))
            path_set.append(arry)
        print('paths', path_set)
        for path in path_set:
            # path_list.append(path)
            state_flow = np.zeros(len(SRC))
            # print(state_flow, packet.priority)
            state_flow[packet.priority] = 1
            # print('state_flow', state_flow)
            state_node = np.zeros(SUM_NODES)
            neighbor = path[1]
            # print('i', i, 'sub_traverse_nodes', traverse_nodes[i:])
            # print('pri', packet.priority)
            for j in path:
                state_node[j] = 1
            print('state_node', state_node)
            state = np.concatenate((state_flow, state_node))
            print("state: ", neighbor)
            state = torch.tensor(state, dtype=torch.float32)
            qu = self.sim.nodes[neighbor].agent.model.forward(1, state)
            qu = qu.reshape(1, -1).squeeze().detach()
            # print('qu', qu)
            # print('worst_delay', qu[-1])
            worst_delay = qu[-1]
            l = int(len(qu) / 2)
            average_delay = qu.mean()
            print('aved', packet)
            if worst_delay <= packet.deadline:
                cand_neighbors.append(neighbor)
                aveD_list.append(average_delay)

        if len(aveD_list) > 0:
            idx = aveD_list.index(min(aveD_list))
            next_node = cand_neighbors[idx]
            return next_node
        # else:
        #     return None

    def run(self, episode):
        duration = 1
        deadline = DEADLINE[FLOW_MAP[self.id]]
        des_node_id = FLOW_DICT[self.id]
        flow_type = FLOW_MAP[self.id]
        priority = 1
        while self.now < self.start_time + duration:
            if self.id == 0:
                lamda = 1.0 / self.arrival_rate
                interval = random.expovariate(lamda)
            elif self.id == 1:
                rand = np.random.random()
                seed = 1
                if rand < 0.8:
                    seed = 2
                if seed == 2:
                    lamda = 1.0 / self.sim.arrival_rates[FLOW_MAP[self.id]]
                    interval = random.expovariate(lamda)
                    #print('interval', interval)
                else:
                    lamda = 1.0 / (self.sim.arrival_rates[FLOW_MAP[self.id]] / 2)
                    interval = random.expovariate(lamda)
            else:
                interval = self.sim.arrival_rates[FLOW_MAP[self.id]]

            yield self.sim.env.timeout(interval)
            self.send_cnt += 1
            packet_id = str(self.id) + '_' + str(self.send_cnt) + '_' + str(des_node_id) + '_' + str(episode)
            packet = Packet(self.id, des_node_id, packet_id, 'data', flow_type, 'training', priority, PACKET_LENGTH,
                            self.now)
            packet.arrival_time[self.id] = round(self.now, 2)
            packet.in_queue_time[self.id] = round(self.now, 2)
            next_node = self.get_nextnode(packet)
            self.mac.send_pdu(packet, next_node)

    def addsamples(self, packet, isUpdate):
        rewards = packet.e2ed_delay  # the cumulative rewards
        for i in range(len(packet.trans_path) - 1):
            pwd_node = packet.trans_path[i]
            next_node = packet.trans_path[i + 1]
            key = str(pwd_node) + '_' + str(next_node)
            state = packet.state_map[pwd_node]
            print('state', state)
            action = next_node
            done = 1
            action_idx = 0
            next_state = packet.state_map[next_node]
            print('next_state', next_state)
            rewards -= packet.all_one_hop_delay[key]
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            self.sim.nodes[pwd_node].agent.replay_memory.add(state, action, action_idx, rewards, next_state, done)

        if self.episode > 500 and isUpdate:
            temp = copy.deepcopy(packet.trans_path)
            temp.reverse()
            loss = self.train(temp)
            self.path_num += 1
            writer2.add_scalar('loss', loss, self.path_num)
            print('update_loss', loss)

    def train(self, trans_path):
        # train the qrdqn of each node
        sumloss = 0
        for node_id in trans_path[:-1]:  # nodes_list
            node = self.sim.nodes[node_id]
            replay_memory = self.sim.nodes[node].agent.replay_memory
            if replay_memory._num >= 1:  # batch_size:
                loss = self.optimize_adam(node)
                self.sim.nodes[node].agent.save_model('E2E')
                sumloss += loss
        sumloss = sumloss / (len(trans_path) - 1)
        return sumloss

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

    def calculate_loss(self, batch_size, states, actions, action_idxs, rewards, next_states, dones,
                       node):  ### batch_size, states, actions, rewards, next_states, dones
        # Calculate quantile values of current states and actions at taus.

        var_s = torch.tensor(states, dtype=torch.float)
        var_s = var_s.to(self.device)

        # print('node', node)
        #print('var_s', var_s)

        current_sa_quantiles = self.sim.nodes[node].agent.model.forward(batch_size, var_s)

        assert current_sa_quantiles.shape == (batch_size, self.sim.nodes[node].agent.tau_N, 1)

        with torch.no_grad():

            var_next_s = torch.tensor(next_states[0], dtype=torch.float)
            var_next_s = var_next_s.to(self.device)
            rewards = torch.tensor(rewards).to(self.device)

            next_node = int(actions[0])

            if dones[0] == 1:
                unit = [0.0]
                next_qu = []
                for m in range(self.sim.nodes[node].agent.tau_N):
                    next_qu.append(unit)
                next_qu = torch.tensor(next_qu).unsqueeze(0).to(self.device)
            else:
                next_qu = self.sim.nodes[next_node].agent.model.forward(1, var_next_s)

            target_sa_quantiles = next_qu
            for j in range(next_qu.shape[1]):
                target_sa_quantiles[0][j][0] = rewards[0] + (1.0 - dones[0]) * next_qu[0][j][0]

            for i in range(1, batch_size):
                var_next_s = torch.tensor(next_states[i], dtype=torch.float)
                var_next_s = var_next_s.to(self.device)
                next_node = int(actions[i])
                if dones[i] == 1:
                    unit = [0.0]
                    next_qu_ = []
                    for m in range(self.sim.nodes[node].agent.tau_N):
                        next_qu_.append(unit)
                    next_qu_ = torch.tensor(next_qu_).unsqueeze(0).to(self.device)

                else:
                    next_qu_ = self.sim.nodes[next_node].agent.model.forward(1, var_next_s)

                target_sa_quantiles_ = next_qu_
                for j in range(next_qu_.shape[1]):
                    target_sa_quantiles_[0][j][0] = rewards[i] + (1.0 - dones[i]) * next_qu_[0][j][0]

                target_sa_quantiles = torch.cat((target_sa_quantiles, target_sa_quantiles_), 0)

            target_qu = target_sa_quantiles
            target_qu = target_qu.transpose(1, 2)
            target_qu = target_qu.to(self.device)

            assert target_qu.shape == (batch_size, 1, self.sim.nodes[node].agent.tau_N)

        td_errors = target_qu - current_sa_quantiles
        assert td_errors.shape == (batch_size, self.sim.nodes[node].agent.tau_N, self.sim.nodes[node].agent.tau_N)
        tau_N = self.sim.nodes[node].agent.tau_N
        taus = torch.arange(0, tau_N + 1, device=self.device, dtype=torch.float32) / tau_N
        tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, tau_N)

        kappa = 1
        quantile_huber_loss = self.calculate_quantile_huber_loss(td_errors, tau_hats, kappa)

        return quantile_huber_loss, td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)

    def optimize_adam(self, node):  ### 优化QRDQN参数
        # print('node optimize', node)
        batch_size = self.sim.batch_size
        # Buffer for storing the loss-values of the most recent batches.
        sum_loss = 0
        sum_train = 0

        loss_sum = 0

        sample_keys = self.sim.nodes[node].agent.replay_memory.get_sample_keys()
        # print('sample_keys', sample_keys)
        for key in sample_keys:
            size = self.sim.batch_size
            epochs = int(self.sim.nodes[node].agent.replay_memory.memory[key]._n / batch_size)
            if epochs == 0:
                epochs = 1
            if epochs > 3:
                epochs = 3

            for _ in range(epochs):
                # print('memory_key', key, size)
                states, actions, action_idxs, rewards, next_states, dones = self.sim.nodes[
                    node].agent.replay_memory.sample_on_key(size, key)

                quantile_loss, errors = self.calculate_loss(
                    size, states, actions, action_idxs, rewards, next_states, dones, node)
                assert errors.shape == (size, 1)

                # print('errors', errors)

                loss = quantile_loss
                loss_sum += loss

                sum_loss += loss.data
                sum_train += 1
                self.sim.nodes[node].agent.model.lr = 0.0005

                print('loss', loss, 'node', node)

                # sum_loss += objective_loss
                # self.graph.node_list[node].agent.model.set_lr(learning_rate)

                opt = self.sim.nodes[node].agent.model.opt
                opt.zero_grad()
                loss.backward()
                opt.step()

        if node == 0:
            avg_loss = sum_loss / sum_train
            self.loss_history.append(avg_loss)
            self.loss_num1 += 1
            writer1.add_scalar('loss', avg_loss, self.loss_num1)
        elif node == 2:
            avg_loss = sum_loss / sum_train
            # self.loss_history.append(avg_loss)
            self.loss_num2 += 1
            writer3.add_scalar('loss', avg_loss, self.loss_num2)
        loss_sum /= sum_train
        return loss_sum

    def on_receive_pdu(self, packet):  # receive_pdu(pdu.src, pdu.payload)
        if packet.id in self.has_reces:
            return
        self.has_reces.append(packet.id)
        prev_node = packet.trans_path[-1]
        packet.trans_path.append(self.id)
        key1 = str(prev_node) + '_' + str(self.id)
        key2 = str(prev_node)
        # calculate the one hop delay from the last node to the current node
        one_hop_delay = self.now - packet.arrival_time[key2]
        packet.all_one_hop_delay[key1] = one_hop_delay
        packet.e2ed_delay += one_hop_delay
        # print(packet.des, self.id)
        if packet.des_node_id != self.id:
            next_node = self.get_nextnode(packet)
            self.mac.send_pdu(packet, next_node)
        else:
            self.reces_for_me.append(packet.id)
            self.e2ed.append(packet.e2ed_delay)
            print(packet.e2ed_delay)
            update = False
            if len(self.reces_for_me) % 10 == 0:
                update = True
            self.addsamples(packet, update)

            # Record the actual delay experienced by the packet
            for l in range(len(packet.trans_path) - 1):
                node_ = packet.trans_path[l]
                sub_path = packet.trans_path[l:]
                sum_delay = 0
                path = [str(x) for x in sub_path]
                path = "_".join(path)
                for j in range(len(sub_path) - 1):
                    key = str(sub_path[j]) + '_' + str(sub_path[j + 1])
                    sum_delay += packet.travase_delay[key]

                if str(packet.priority) not in self.sim.nodes[node_].end_to_end_delay.keys():
                    self.sim.nodes[node_].end_to_end_delay[str(packet.priority)] = {
                        path: {str(sum_delay): 1}}
                else:
                    if path not in self.sim.nodes[node_].end_to_end_delay[str(packet.priority)].keys():
                        # print('path not in')
                        self.sim.nodes[node_].end_to_end_delay[str(packet.priority)][path] = {
                            str(sum_delay): 1}
                        # print('bfnode2', node_, self.graph.node_list[node_].end_to_end_delay)
                    else:
                        if str(sum_delay) not in self.sim.nodes[node_].end_to_end_delay[str(packet.priority)][
                            path].keys():
                            self.sim.nodes[node_].end_to_end_delay[str(packet.priority)][path][
                                str(sum_delay)] = 1
                        else:
                            self.sim.nodes[node_].end_to_end_delay[str(packet.priority)][path][
                                str(sum_delay)] += 1