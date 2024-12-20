from math import pi
import numpy as np
from scipy import integrate
from itertools import combinations
import math
from util.utils import distance
from net_trans_config import *


class PDU:
    def __init__(self, layer, nbits, **fields):
        self.layer = layer
        self.nbits = nbits
        for f in fields:
            setattr(self, f, fields[f])


class Packet:
    def __init__(self, sim, source, id, priority, nbits=103):
        # 构造函数接受多个参数：模拟器实例、数据包源节点、数据包ID、数据包优先级和数据包位数，默认设置为8*1000。
        # 初始化了数据包的源节点、ID、优先级和位数。
        self.source = source
        self.id = id
        self.priority = priority
        self.nbits = nbits
        self.start_time = sim.env.now  # 记录数据包的起始时间。
        # 初始化了几个用于存储数据包信息的字典：节点到达时间、单跳延迟、传输延迟、到达时间、下一个节点的队列等。
        self.node_arrival = dict()
        self.one_hop_delay = dict()
        self.travase_delay = {}
        self.arrival_time = {}
        self.queue_of_next_node = {}
        self.end_to_end_delay = 0
        self.trans_path = [source]  # 初始化数据包的传输路径为仅包含源节点的列表。
        self.des = 10  # 数据包的目标节点id


class DefaultMacLayer:
    LAYER_NAME = 'mac'
    HEADER_BITS = 24

    def __init__(self, node):
        self.node = node  # 当前节点
        self.queue = []  # 存储等待发送的数据包的队列
        self.ack_event = None  # 用于存储确认事件的变量
        self.total_tx_broadcast = 0
        self.total_tx_unicast = 0
        self.total_rx_broadcast = 0
        self.total_rx_unicast = 0
        self.total_retransmit = 0
        self.total_ack = 0  # 以上这些变量分别用于跟踪广播发送、单播发送、广播接收、单播接收、重传次数和确认次数
        self.has_reces = []  # 已接收的数据包ID列表，防止重复接收

    def addPacket(self, packet, pri):
        '''
        这个方法用于向队列中添加一个数据包，并按优先级排序。具体步骤如下：
            1.初始化队列长度。
            2.检查如果队列小于等于一个，则直接添加。
            3.否则，根据优先级插入到正确的位置。
        '''
        queue_len = 0
        if len(self.queue) <= 1:
            packet.payload.queue_of_next_node[str(self.node.id)] = queue_len
            self.queue.append(packet)
            return

        temp = [self.queue[0]]
        for i in range(1, len(self.queue)):
            if pri < self.queue[i].payload.priority:
                packet.payload.queue_of_next_node[str(self.node.id)] = queue_len
                temp.append(packet)
                for j in range(i, len(self.queue)):
                    temp.append(self.queue[j])
                self.queue = temp
                return
            else:
                queue_len += 1
                temp.append(self.queue[i])
        packet.payload.queue_of_next_node[str(self.node.id)] = queue_len
        self.queue.append(packet)

    def process_queue(self):
        '''
        通过模拟时序来处理队列中的包的发送：
            1.获取当前节点的发送时隙。
            2.计算下一个发送时隙的等待时间，并等待该时长。
            3.检查队列如果仍有数据包，则发送包。
        '''
        while len(self.queue) > 0:
            send_slot = self.node.sim.frame_slot[self.node.id]
            wait_time = (self.node.sim.env.now - self.node.sim.start_time) % (self.node.sim.each_slot * send_slot)
            interval = self.node.sim.each_slot * send_slot - wait_time
            yield self.node.sim.env.timeout(interval)
            if len(self.queue) == 0:
                return
            mac_pdu = self.queue[0]
            self.node.phy.send_pdu(mac_pdu)
            yield self.node.sim.env.timeout(self.node.sim.each_slot)

    def send_pdu(self, packet, next_node):
        '''
        用于发送一个数据包给下一个节点：
            1.生成新的 MAC PDU，包含源、目标、数据类型等信息。
            2.调用 addPacket 将PDU添加到队列中。
            3.如果是队列中的第一个包，则启动 process_queue 方法来处理队列。
        '''
        key = str(self.node.id)
        packet.arrival_time[key] = self.node.now
        mac_pdu = PDU(self.LAYER_NAME, packet.nbits + self.HEADER_BITS,
                      type='data',
                      src=self.node.id,
                      src_node=self.node,
                      dst=next_node.id,
                      dst_node=next_node,
                      payload=packet)
        self.addPacket(mac_pdu, packet.priority)
        if len(self.queue) == 1:
            self.node.sim.env.process(self.process_queue())

    def on_receive_pdu(self, pdu):
        '''
        处理接收到的PDU：
            1.检查是否是发送给当前节点的PDU。
            2.验证是否已经接收过该包，避免重复处理。
            3.接收新包，记录来源，并调用节点的接收处理函数。
            4.如果发送节点的队列不为空，则弹出队列中的第一个元素。
        '''
        if pdu.dst == self.node.id:
            if pdu.payload.id in self.has_reces:
                return
            self.has_reces.append(pdu.payload.id)
            self.node.on_receive_pdu(pdu.payload)
            last_node = pdu.src
            if len(self.node.sim.nodes[last_node].mac.queue) > 0:
                self.node.sim.nodes[last_node].mac.queue.pop(0)


# 计算路径损耗（Path Loss），参数d表示距离。其中使用了波长lambda和距离的三次方来计算路径损耗。
def getPL(d):
    alpha = 3
    lambda_ = (3 * (10 ** 8)) / (2.4 * (10 ** 9))
    PL = (lambda_ ** 2) / (4 * 4 * pi * pi * (d ** alpha))
    # print(PL)
    return PL


# print(getPL(1))
# 计算接收功率Pr。函数内部处理了不同情况下的距离计算，包括源节点和目的节点的不同组合。
def getPr(node1, node2):
    global Dis_sr
    global Dis
    global Dis_rd
    global D_sd
    global numberofsource
    global numberofdes

    Source_id = []
    Destinationid = []
    for s in range(numberofsource):  # %S id 100,101..
        Source_id.append(99 + s)
    for d in range(numberofsource):  # %D id 110,111..
        Destinationid.append(109 + d)
    d = numberofdes - 1
    Sourcenode = False
    Desnode = False
    Sourcenode = node1 in Source_id  # % % if the node1 is in the set of source_id, then return 1.
    Desnode = node2 in Destinationid
    if (Sourcenode == False) and (Desnode == False):
        # print("均是False时候:",node1,node1)
        d = Dis[node1, node2]
    elif (Sourcenode == True) and (Desnode == False):
        d = Dis_sr[s, node2]
    elif (Sourcenode == False) and (Desnode == True):
        #print('dis_rd',Dis_rd,'node1',node1,'d',d)
        d = Dis_rd[node1, d]
    elif (Sourcenode == True) and (Desnode == True):
        d = D_sd
    # print(d)
    Pe = 0.3  #0.151  # %#mW transmission power
    # %Pe = 151;%Journal paper
    # %Tpaquet = 2560*8; % the number of bits of a data packet-802.11(20160729)
    Tpaquet = 133 * 8  # % 802.15.4: 5B(SHR)+1B(PHR)+127B(PSDU) = 6+127=133B(SHR:synchronizaition header, PHR: PHY header,PSDU:PHY payload)
    NO = -192.0  # Journal paper
    B = 1000000  # Bandwidth 1MHz
    noise = (10 ** (NO / 10)) * B
    PL = 1
    if (d != 0):
        PL = getPL(d)
    else:
        PL = 1
    # print(PL)
    Pr = Pe * PL
    return Pr


# 根据接收功率Pr和干扰I计算包错误率（PER）。
def getPER_noise(Pr, I):
    Pe = 0.3  #0.151  ##mW transmission power
    # Tpaquet = 2560*8;% the number of bits of a data packet
    Tpaquet = 133 * 8
    # 802.15.4: 5B(SHR)+1B(PHR)+127B(PSDU) = 6+127=133B(SHR:synchronizaition header, PHR: PHY header,P
    N0 = -192
    # %N0 = -154;%Journal paper
    B = 1000000  # Bandwidth 1MHz
    noise = (10 ** (N0 / 10)) * B
    # SINR = Pr./(noise+I)
    SINR = Pr / (noise + I)
    # print("getPER_noise中SINR", SINR)

    ber = getBER(SINR)
    PER = (1 - ((1 - ber) ** Tpaquet))
    return PER


# 计算在特定干扰环境下，从节点node1传输到节点node2的成功概率。
# print(getPr(0,0))
def computePijtforDistribu(node1, node2, r, IterferIdset, R):
    global numberofsource
    m, n = IterferIdset.shape
    Iterfernum_r = 0
    Pe = 0.3  #0.151  # %  # mW transmission power
    N0 = -192
    # N0 = -154# % Journal paper
    B = 1000000  # # Bandwidth 1MHz
    noise = (10 ** (N0 / 10)) * B

    IterferIdset_r = []
    for i in range(n):
        if IterferIdset[r, i] >= 0 and IterferIdset[r, i] != node1:
            Iterfernum_r = Iterfernum_r + 1
            IterferIdset_r.append(IterferIdset[r, i])
    piju = np.zeros((Iterfernum_r, Iterfernum_r))
    if Iterfernum_r > 0:
        for k in range(Iterfernum_r):
            CombinIterferIdset = list(combinations(IterferIdset_r, k))
            CombinIterferIdset = np.array(CombinIterferIdset)
            m_c, n_c = CombinIterferIdset.shape
            # print("CombinIterferIdset",CombinIterferIdset)
            for i in range(m_c):
                Pl_a = 1  # active PL
                Pl_noa = 1  # not active PL
                Iter = 0
                PL = 1
                for j in range(n_c):
                    id = int(CombinIterferIdset[i, j])
                    if id == 99:  # sourcr S_1 id = 100
                        Pl_a = R[0, r] * Pl_a
                    elif id == 101:  # sourcr S_2 id=101
                        Pl_a = R[1, r] * Pl_a
                    else:
                        #print('id', id, 'numsource', numberofsource, 'r', r, 'R', R)
                        Pl_a = R[id + numberofsource, r] * Pl_a  # the first line is souce S_m

                    if id != node2:
                        Iter = Iter + getPr(id, node2)
                    else:
                        Iter = Iter + Pe
                # %when A and B are vectors returns the values that are not in the intersection of A and B.
                nonactiveset = np.setdiff1d(IterferIdset_r, CombinIterferIdset[i, :])
                if (len(nonactiveset) != 0):
                    num_noactivenode, w = nonactiveset.shape[0], 1
                    for q in range(num_noactivenode):
                        id = nonactiveset[q]
                        if id == 99:  # source_id
                            Pl_noa = (1 - R[0, r]) * Pl_noa
                        elif id == 100:
                            Pl_noa = (1 - R[1, r]) * Pl_noa
                        else:
                            # print('id',id)
                            # print('numberofsource',numberofsource)
                            # print('r',r)
                            id = int(id)
                            Pl_noa = (1 - R[id + numberofsource, r]) * Pl_noa
                PL = Pl_a * Pl_noa
                Pr = getPr(node1, node2)
                PER = getPER_noise(getPr(node1, node2), Iter)
                piju[k, i] = (1 - PER) * PL
    else:
        #print("node1",node1, "node2", node2,"piju", 1 - getPER_noise(getPr(node1, node2), 0))
        piju = 1 - getPER_noise(getPr(node1, node2), 0)
    return piju


# 根据信噪比（SINR）计算比特错误率（BER）。
def getBER(SINR):
    def f(x):
        return np.exp(-(x ** 2))

    v, err = integrate.quad(f, math.sqrt(SINR), float('inf'))
    v = v * 2 / math.sqrt(pi)
    #print('ber', 0.5*v)
    return 0.5 * v


# 计算包错误率，d为距离。
def getPER(d):
    if d != 0:
        Pe = 0.00088  #0.151  # mW transmission power 802.15.4 1wm=0dBm
        # Pe = 151;%Journal paper
        # Tpaquet = 2560*8;% the number of bits of a data packet-802.11(20160729)
        Tpaquet = 512 * 8  # 802.15.4: 5B(SHR)+1B(PHR)+127B(PSDU) = 6+127=133B(SHR:synchronizaition header, PHR: PHY header,PSDU:PHY payload)
        N0 = -192.0  # dBm / Hz
        # %N0 = -154;%Journal paper
        B = 6000000  # # Bandwidth 1MHz
        noise = (10 ** (N0 / 10)) * B
        #noise = 6.3096e-14
        #print('noise', noise)
        PL = getPL(d)
        Pr = Pe * PL
        #print('Pr', Pr, 'Pl', PL)
        # % I = 6.2691e-014;
        # % SINR = Pr. / (noise + I);
        SINR = Pr / noise
        #print("getPER中SINR", SINR)
        ber = getBER(SINR)
        #print('SINR', SINR, 'ber', ber)

        PER = (1 - ((1 - ber) ** Tpaquet))
        #print('per', PER)
        return PER
    else:
        PER = 0
        #print('per', PER)
        return PER


# print(1-getPER(20*1.4))
class DefaultPhyLayer:
    LAYER_NAME = 'phy'
    PHY_HEADER_LENGTH = 24

    def __init__(self, node):
        self.node = node
        self._current_rx_count = 0
        self._channel_busy_start = 0
        self.total_tx = 0
        self.total_rx = 0
        self.total_collision = 0
        self.total_error = 0
        self.total_bits_tx = 0
        self.total_bits_rx = 0
        self.total_channel_busy = 0
        self.total_channel_tx = 0

    def send_pdu(self, pdu):  # transmit a pdu
        tx_time = (pdu.nbits + PHY_HEADER_LENGTH) * BIT_TRANSMISSION_TIME  # pdu 的传输时间(单位：us)
        next_node = pdu.dst_node
        dist = distance(next_node.pos, self.node.pos)
        prop_time = dist * BIT_TRANSPORT_TIME + 1  # pdu 的传播时间(单位：us)
        self.node.delayed_exec(prop_time, next_node.phy.on_rx_start, pdu)
        self.node.delayed_exec(prop_time + tx_time, next_node.phy.on_rx_end, pdu)

    def on_tx_start(self, pdu):
        pass

    def on_tx_end(self, pdu):
        pass

    def on_rx_start(self, pdu):
        pass

    def on_rx_end(self, pdu):
        # receive a pdu
        prev_node = self.node.sim.nodes[pdu.src]
        dist = distance(prev_node.pos, self.node.pos)
        per = getPER(dist)
        # print(self.node.sim.random.random(), per)
        if self.node.sim.random.random() > per:
            # print('phy transmit successfully to', pdu.dst)
            self.node.mac.on_receive_pdu(pdu)
            self.total_rx += 1
            self.total_bits_rx += pdu.nbits
        else:
            # print("phy transmit failed to", pdu.dst)
            self.total_error += 1

    def on_collision(self, pdu):
        pass

    def cca(self):
        """Return True if the channel is clear"""
        return self._current_rx_count == 0
