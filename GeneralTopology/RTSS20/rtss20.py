import simpy
import numpy as np
from utils import *
import random
from simpy.util import start_delayed
from mac_phy import DefaultPhyLayer, DefaultMacLayer, Packet

# 拓扑关系定义
sum_nodes = 11
src = [0]  # 设置源节点
des = [10]  # 设置目标节点
adj_M = {
    0:[1,4,7],
    1:[2],
    2:[3],
    3:[10],
    4:[5],
    5:[6],
    6:[10],
    7:[8],
    8:[9],
    9:[10],
    10:[],
}

position = {
    0: [-50, 0],
    1: [-30, 25],
    2: [0, 42],
    3: [30, 25],
    4: [-30, 0],
    5: [0, 0],
    6: [30, 0],
    7: [-30, -25],
    8: [0, -42],
    9: [30, -25],
    10: [50, 0],
}

# 网络传输相关参数设置：源节点发包频率
send_rates = {
    0:1500,
}
frame_slot = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 3,
    5: 5,
    6: 1,
    7: 4,
    8: 3,
    9: 2,
}

class PDU:
    def __init__(self, layer, nbits, **fields):
        self.layer = layer
        self.nbits = nbits
        for f in fields:
            setattr(self, f, fields[f])

class Simulator:
    def __init__(self, sum_nodes, sim_time, src, des, adj_M, position, send_rates, frame_slot, route_vector, seed):
        self.env = simpy.Environment()  # 设置环境
        self.nodes = []  # 存储环境中的所有节点
        self.sum_nodes = sum_nodes  # 节点总数
        self.start_time = self.env.now  # 模拟的开始时间
        self.sim_time = sim_time  # 设置模拟运行的截止时间。
        self.src = src  # 列表存储源节点。
        self.des = des  # 列表存储目的地节点。
        self.adj_M = adj_M  # 存储邻接矩阵表示节点间的连接。
        self.position = position  # 存储所有节点的位置信息(字典类型)
        self.send_rate = send_rates  # 源节点的发包速率
        self.frame_slot = frame_slot  # 帧槽的时隙分配
        self.each_slot = 290  # 每个 slot 的处理时间：0.29ms
        self.random = random.Random(seed)  # 初始化伪随机数生成器。
        self.e2ed_delay = {"0-1-2-3-10":[], "0-4-5-6-10":[], "0-7-8-9-10":[]}
        self.e2ed_delay_package = []
        self.route_vector = route_vector

        self.init()
        self.run()

    def run(self):
        self.env.run(until=self.sim_time)

    def init(self):
        for node_id in range(self.sum_nodes):
            self.nodes.append(Node(node_id, self, self.src, self.des, self.adj_M, self.position, self.send_rate))


    def get_avg_e2ed_delay(self):
        ############ test print ###########
        # print(self.e2ed_delay)
        print(len(self.nodes[len(self.nodes)-1].has_reces)) # 目标节点接受的数据包个数
        # for node in self.nodes:
            # print(node.id, node.mac.queue)
        ###################################
        data = []
        for i in self.e2ed_delay_package:
                data.append(round(i, 2))
        print(data)
        avg_e2ed_delay = round(np.mean(data), 2)
        return avg_e2ed_delay

    def delayed_exec(self, delay, func, *args, **kwargs): # 函数接收延迟时间和要执行的函数，使用环境设置延迟执行。
        func = ensure_generator(self.env, func, *args, **kwargs)
        start_delayed(self.env, func, delay=delay)

class Node:
    DEFAULT_MSG_NBITS = 1000 * 8
    def __init__(self, id, sim, src, des, adj_M, pos,  send_rates):
        self.id = id
        self.sim = sim
        self.isSrc = False
        self.isDes = False
        if self.id in src:
            self.isSrc = True
            # self.packet_pri = src.index(self.id)
            self.send_rate = send_rates[self.id]
        if self.id in des:
            self.isDes = True
        self.pos = pos[id]
        self.phy = DefaultPhyLayer(self)
        self.mac = DefaultMacLayer(self)
        self.neighbors = adj_M[id]
        self.neighbor_dis = {}
        self.neighbor_pdr = {}
        self.route_tb = {}
        self.sends = 0
        self.reces_for_me = []
        self.has_reces = []
        self.timeout = self.sim.env.timeout
        self.start_time = sim.env.now
        self.loss_history = []
        self.e2ed = []
        self.end_to_end_delay = {}

        # 启动节点服务进程
        if self.isSrc:
            self.route_tb = self.sim.route_vector
            self.sim.env.process(self.src_generate_packets())
        elif self.isDes:
            self.route_tb = {}
        else:
            self.route_tb = {self.neighbors[0]: 1}

    @property
    def now(self):  # 这是一个属性装饰器，返回当前的模拟环境时间。
        return self.sim.env.now

    def src_generate_packets(self):  # 源节点生成数据包
        duration = self.sim.sim_time  # 发包持续时间
        while self.sim.env.now < self.start_time + duration:
            interval = self.sim.send_rate[self.id]
            yield self.sim.env.timeout(interval)
            self.sends += 1
            packet_id = str(self.id) + '_' + str(self.sends)
            packet = Packet(self.sim, self.id, packet_id, 1)
            keys = list(self.sim.route_vector.keys())
            values = list(self.sim.route_vector.values())
            next_node_id = random.choices(keys, weights=values, k=1)[0]
            # next_node_id = max(self.sim.route_vector, key=self.sim.route_vector.get)
            next_node = self.sim.nodes[next_node_id]
            self.mac.send_pdu(packet, next_node)

    def des_receive_packet(self):

        pass

    def rx_tx_packets(self):
        pass


    ############################
    def create_event(self):
        return self.sim.env.event()

    ############################
    def create_process(self, func, *args, **kwargs):
        return ensure_generator(self.sim.env, func, *args, **kwargs)

    ############################
    def start_process(self, process):
        return self.sim.env.process(process)

    ############################
    def delayed_exec(self, delay, func, *args, **kwargs): # 延迟执行函数
        return self.sim.delayed_exec(delay, func, *args, **kwargs)

    def get_nextnode(self, packet): # 根据给定的数据包，选择下一跳节点进行传输。
        for node_id in self.neighbors:
            if node_id == packet.des:
                return self.sim.nodes[node_id]
        idx = max(self.route_tb, key=self.route_tb.get)
        next_node = self.sim.nodes[idx]
        return next_node

    def on_receive_pdu(self, packet):  # receive_pdu(pdu.src, pdu.payload)
        if packet.id in self.has_reces:
            return
        self.has_reces.append(packet.id)
        src = packet.trans_path[-1]
        packet.trans_path.append(self.id)
        key1 = str(src) + '_' + str(self.id)
        key2 = str(src)

        # calculate the one hop delay from the last node to the current node
        one_hop_delay = self.now - packet.arrival_time[key2]

        packet.travase_delay[key1] = one_hop_delay
        packet.end_to_end_delay += one_hop_delay
        if packet.des != self.id:
            next_node = self.get_nextnode(packet)
            self.mac.send_pdu(packet, next_node)
        else:
            self.reces_for_me.append(packet.id)
            self.sim.e2ed_delay_package.append(packet.end_to_end_delay)
            self.e2ed.append(packet.end_to_end_delay)
            self.sim.e2ed_delay[trans_path_reverse(packet.trans_path)].append(round(packet.end_to_end_delay, 2))





