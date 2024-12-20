import simpy
import random
import numpy as np
from pip._internal import network


class Node:  # 非 src/des 节点类
    def __init__(self, env, node_id):
        self.env = env
        self.node_id = node_id
        self.links = []

    def add_link(self, link):
        self.links.append(link)

class Link:  # 链路类
    def __init__(self, env, network, link_id, source_node, destination_node, service_type):
        self.env = env
        self.network = network
        self.link_id = link_id
        self.source_node = source_node
        self.destination_node = destination_node
        self.service_type = service_type
        self.queue = simpy.Store(env)

        # 启动链路服务进程
        env.process(self.serve())

    def serve(self):  # 链路服务进程，处理数据包的转发
        while True:
            packet = yield self.queue.get()  # 等待数据包到达
            # 根据服务类型定义服务能力（单位时间可以转发数据包的个数）
            if self.service_type == "Type I":
                service_time = 5
            elif self.service_type == "Type II":
                service_time = random.expovariate(1 / 5)
            else:
                service_time = 40 if random.random() < 1 / 8 else 0
            # 判断服务时间是不是 0
            if service_time != 0:
                yield self.env.timeout(1/service_time)  # 模拟服务时间
            else:
                yield self.env.timeout(1)
                self.network.links[self.link_id].receive_packet(packet)
                continue  # 服务时间为 0 的时候不执行数据包的转发
            if self.link_id in [3, 7, 11]:
                # 将数据包转发到目标节点
                packet.destination_node.receive_packet(packet)
            else:
                self.network.links[self.link_id + 1].receive_packet(packet)

    def receive_packet(self, packet):  # 将数据包加入到链路队列
        self.queue.put(packet)

class Source(Node):  # 源节点类
    def __init__(self, env, network, node ,load, routing_vector):
        super().__init__(env, node)
        self.network = network
        self.load = load
        self.routing_vector = routing_vector
        self.node = node

    def generate_packets(self):  # 源节点生成数据包进程
        while True:
            # 模拟源节点数据包生成时间
            yield self.env.timeout(1/self.load)
            # 创建数据包对象
            packet = Packet(self.env, self, self.env.now, self.routing_vector, self.network.destination)
            # 选择转发路径
            link_id = np.argmax(self.routing_vector)
            # 向临接的链路发送数据包
            self.node.links[link_id].receive_packet(packet)

class Destination(Node):  # 目标节点类
    def __init__(self, env, node_id):
        super().__init__(env, node_id)
        self.received_packets = 0

    def receive_packet(self, packet):
        self.received_packets += 1
        packet.queue.put(packet)

class Packet:  # 数据包类
    def __init__(self, env, source_node, arrival_time, routing_vector, destination_node):
        self.env = env
        self.source_node = source_node
        self.arrival_time = arrival_time
        self.routing_vector = routing_vector
        self.queue = simpy.Store(env)
        self.destination_node = destination_node

class Network:  # 网络模型类
    def __init__(self, env, load, routing_vector):
        self.env = env
        self.routing_vector = routing_vector  # 设置路由向量
        self.nodes = [Node(env, i) for i in range(11)]  # 创建节点

        # 创建链路
        self.links = [
            Link(env, self, 0, self.nodes[0], self.nodes[1], "Type I"),
            Link(env, self, 1, self.nodes[1], self.nodes[2], "Type I"),
            Link(env, self, 2, self.nodes[2], self.nodes[3], "Type I"),
            Link(env, self, 3, self.nodes[3], self.nodes[10], "Type I"),
            Link(env, self, 4, self.nodes[0], self.nodes[4], "Type II"),
            Link(env, self, 5, self.nodes[4], self.nodes[5], "Type II"),
            Link(env, self, 6, self.nodes[5], self.nodes[6], "Type II"),
            Link(env, self, 7, self.nodes[6], self.nodes[10], "Type II"),
            Link(env, self, 8, self.nodes[0], self.nodes[7], "Type III"),
            Link(env, self, 9, self.nodes[7], self.nodes[8], "Type III"),
            Link(env, self, 10, self.nodes[8], self.nodes[9], "Type III"),
            Link(env, self, 11, self.nodes[9], self.nodes[10], "Type III"),]

        # 连接节点和链路
        self.nodes[0].add_link(self.links[0])
        self.nodes[0].add_link(self.links[4])
        self.nodes[0].add_link(self.links[8])
        self.nodes[1].add_link(self.links[1])
        self.nodes[2].add_link(self.links[2])
        self.nodes[3].add_link(self.links[3])
        self.nodes[4].add_link(self.links[5])
        self.nodes[5].add_link(self.links[6])
        self.nodes[6].add_link(self.links[7])
        self.nodes[7].add_link(self.links[9])
        self.nodes[8].add_link(self.links[10])
        self.nodes[9].add_link(self.links[11])

        # 设置源节点和目标节点
        self.source = Source(env, self, self.nodes[0], load, routing_vector)
        self.destination = Destination(env, 10)

        # 连接源节点和目标节点
        self.links[3].destination_node = self.destination
        self.links[7].destination_node = self.destination
        self.links[11].destination_node = self.destination

        # 启动源节点的生成数据包进程
        env.process(self.source.generate_packets())


    def get_queue_lengths(self):  # 获取所有链路数据队列的长度
        queue_lengths = [len(link.queue.items) for link in self.links]
        return queue_lengths