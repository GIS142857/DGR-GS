import json
import copy
import torch
import random
import numpy as np
from config import *
from agent import Agent
from packet import Packet
from tensorboardX import SummaryWriter
from tonMacLayer import TonMacLayer
from tonPhyLayer import TonPhyLayer


writer0 = SummaryWriter(log_dir='log999_0')
writer1 = SummaryWriter(log_dir='log999_1')
writer2 = SummaryWriter(log_dir='log999_2')
writer3 = SummaryWriter(log_dir='log999_3')


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
        self.recv_for_me = []
        self.has_recv = []
        self.timeout = self.sim.timeout
        self.start_time = 0
        self.device = device
        self.time_slot = FRAME_SLOT[id]
        self.loss_history = []
        self.loss_num1 = 0
        self.loss_num2 = 0
        self.path_num = 0
        self.e2ed = []
        self.sum_sources = len(SRC)
        self.sum_nodes = SUM_NODES
        self.state_dim = 4
        self.end_to_end_delay = {}

    @property
    def now(self):
        return self.sim.env.now

    def reset(self):
        self.phy.reset()
        self.mac.reset()
        self.nb_pri_queues = {key: {'flow1': 0, 'flow2': 0, 'flow3': 0} for key in self.neighbors}
        self.send_cnt = 0
        self.recv_for_me = []
        self.has_recv = []
        self.timeout = self.sim.timeout
        self.start_time = self.sim.env.now
        self.loss_history = []
        self.loss_num1 = 0
        self.loss_num2 = 0
        self.path_num = 0
        self.e2ed = []
        self.end_to_end_delay = {}

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
        if self.sim.episode > 500:
            print('best:', self.get_best_action(packet))

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
        if packet.des_node_id in self.neighbors:
            return packet.des_node_id
        state = [packet.des_node_id]
        for nb in self.neighbors:
            state.append(self.nb_pri_queues[nb][packet.flow_type])
        while len(state) < 4:
            state.append(0)
        state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        qu = self.agent.model.forward(1, state)
        qu = qu.reshape(1, -1).squeeze().detach()
        # print('qu', qu)
        worst_delay = qu[-1]
        print('worst_delay', worst_delay)
        l = int(len(qu) / 2)
        average_delay = qu.mean()
        return worst_delay


    def run(self):
        duration = self.sim.sim_time/2  # 发包时间
        deadline = DEADLINE[FLOW_MAP[self.id]]
        des_node_id = FLOW_DICT[self.id]
        flow_type = FLOW_MAP[self.id]
        while self.now < self.start_time + duration:
            if self.id == 0:
                priority = self.packet_pri
                interval = self.arrival_rate
            elif self.id == 1:
                priority = self.packet_pri
                interval = self.arrival_rate
            else:
                priority = self.packet_pri
                interval = self.arrival_rate

            yield self.sim.env.timeout(interval)
            self.send_cnt += 1
            packet_id = str(self.id) + '_' + str(self.send_cnt) + '_' + str(des_node_id) + '_' + str(self.sim.episode)
            packet = Packet(self.id, des_node_id, packet_id, 'data', flow_type, 'training', priority, PACKET_LENGTH,
                            self.now)
            packet.arrival_time[self.id] = round(self.now, 2)
            packet.in_queue_time[self.id] = round(self.now, 2)
            next_node = self.get_nextnode(packet)
            self.mac.send_pdu(packet, next_node)

    def add_samples(self, packet):
        for i in range(len(packet.trans_path) - 1):
            pwd_node = packet.trans_path[i]
            next_node = packet.trans_path[i + 1]
            state = packet.state_map[pwd_node]
            action = next_node
            done = 1
            # 判断 next_node 的下标
            action_idx = 0
            for nb in self.sim.nodes[pwd_node].neighbors:
                if nb == next_node:
                    break
                action_idx += 1
            next_state = copy.deepcopy(state)
            next_state[action_idx + 1] = next_state[action_idx + 1] + 1
            reward = packet.out_queue_time[next_node] - packet.out_queue_time[pwd_node]
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            self.sim.nodes[pwd_node].agent.replay_memory.add(state, action, action_idx, reward, next_state, done)

    def train(self):
        loss = 0
        replay_memory = self.agent.replay_memory
        if replay_memory._num >= self.sim.batch_size:  # batch_size:
            loss = self.optimize_adam()
            if self.sim.episode % 100 == 0:
                self.agent.save_model('E2E')
        return loss

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
        # Quantile huber loss.
        batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
        assert batch_quantile_huber_loss.shape == (batch_size, 1)
        if weights is not None:
            quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
        else:
            quantile_huber_loss = batch_quantile_huber_loss.mean()
        return quantile_huber_loss

    def calculate_loss(self, batch_size, states, actions, action_idxs, rewards, next_states, dones, node):
        # Calculate quantile values of current states and actions at taus.
        current_s_quantiles = node.agent.model.forward(batch_size, states)
        # print(current_s_quantiles, action_idxs)
        current_sa_quantiles = torch.zeros(batch_size, node.agent.tau_N, 1).to(self.device)
        for i in range(batch_size):
            current_sa_quantiles[i] = current_s_quantiles[i, :, int(action_idxs[i])].unsqueeze(-1)
        print(current_sa_quantiles.shape)
        # assert current_sa_quantiles.shape == (batch_size, node.agent.tau_N, node.agent.model.num_actions)

        # Initialize target_sa_quantiles as an empty tensor to concatenate in the loop
        target_sa_quantiles = torch.zeros(batch_size, node.agent.tau_N, 1).to(self.device)

        with torch.no_grad():
            for i in range(batch_size):
                var_next_s = next_states[i]
                next_node = int(actions[i])
                next_qu_ = node.agent.model.forward(1, var_next_s)
                # 计算每个动作分布的平均值
                action_means = next_qu_.mean(dim=-1)  # Shape: [64, 500]
                # 找到平均值最小的动作索引
                min_action_indices = action_means.argmin(dim=-1)  # Shape: [64]
                # 初始化用于存储最小动作分布的张量
                min_action_distributions = torch.zeros(1, 500, 1)  # Shape: [1, 500, 1]
                # 提取对应动作的分布
                min_action_distributions[:, :, 0] = next_qu_[:, :, min_action_indices[i]]
                target_sa_quantiles_ = min_action_distributions
                for j in range(next_qu_.shape[1]):
                    target_sa_quantiles_[0][j][0] = rewards[i]/1000 + (1.0 - dones[i]) * next_qu_[0][j][0]
                target_sa_quantiles = torch.cat((target_sa_quantiles, target_sa_quantiles_), 0)
            target_qu = target_sa_quantiles
            target_qu = target_qu.transpose(1, 2)
            target_qu = target_qu.to(self.device)

        # print(current_sa_quantiles, target_qu)
        print(target_qu.shape, current_sa_quantiles.shape)

        td_errors = target_qu - current_sa_quantiles
        tau_N = node.agent.tau_N
        taus = torch.arange(0, tau_N + 1, device=self.device, dtype=torch.float32) / tau_N
        tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, tau_N)
        kappa = 1
        quantile_huber_loss = self.calculate_quantile_huber_loss(td_errors, tau_hats, kappa)
        return quantile_huber_loss, td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)

    def optimize_adam(self):  ### 优化QRDQN参数
        batch_size = self.sim.batch_size
        # Buffer for storing the loss-values of the most recent batches.
        sum_loss = 0
        sum_train = 0
        loss_sum = 0
        sample_keys = self.agent.replay_memory.get_sample_keys(10)
        # print('sample_keys:', sample_keys[0], len(sample_keys))
        for key in sample_keys:
            states, actions, action_idxs, rewards, next_states, dones = self.agent.replay_memory.sample_on_key(batch_size, key)
            # print(states, actions, action_idxs, rewards, next_states, dones)
            quantile_loss, errors = self.calculate_loss(
                batch_size, states, actions, action_idxs, rewards, next_states, dones, self)
            assert errors.shape == (batch_size, 1)
            loss = quantile_loss
            loss_sum += loss
            sum_loss += loss.data
            sum_train += 1
            self.agent.model.lr = 0.001
            print('loss', loss, 'node', self.id)
            opt = self.agent.model.opt
            opt.zero_grad()
            loss.backward()
            opt.step()

        if self.id == 0:
            avg_loss = sum_loss / sum_train
            self.loss_history.append(avg_loss)
            self.loss_num1 += 1
            writer0.add_scalar('loss', avg_loss, self.loss_num1)
        elif self.id == 1:
            avg_loss = sum_loss / sum_train
            self.loss_num2 += 1
            writer1.add_scalar('loss', avg_loss, self.loss_num2)
        elif self.id == 2:
            avg_loss = sum_loss / sum_train
            self.loss_num2 += 1
            writer2.add_scalar('loss', avg_loss, self.loss_num2)
        loss_sum /= sum_train
        return loss_sum

    def on_receive_pdu(self, packet):  # receive_pdu(pdu.src, pdu.payload)
        if packet.id in self.has_recv:
            return
        self.has_recv.append(packet.id)
        prev_node = packet.trans_path[-1]
        packet.trans_path.append(self.id)
        # calculate the one hop delay from the last node to the current node
        one_hop_delay = self.now - packet.arrival_time[prev_node]
        packet.e2ed_delay += round(one_hop_delay, 2)
        if packet.des_node_id != self.id:
            next_node = self.get_nextnode(packet)
            self.mac.send_pdu(packet, next_node)
        else:
            packet.arrival_time[self.id] = round(self.now, 2)
            packet.in_queue_time[self.id] = round(self.now, 2)
            packet.out_queue_time[self.id] = round(self.now, 2)
            # print('arrival_time', packet.arrival_time)
            # print('in_queue_time', packet.in_queue_time)
            # print('out_queue_time', packet.out_queue_time)
            # print('e2ed_delay', packet.e2ed_delay)
            self.recv_for_me.append(packet.id)
            self.e2ed.append(round(packet.e2ed_delay, 2))
            self.add_samples(packet)