import random
from math import pi
import numpy as np
from logger import Stat
from configure import *
from utils import distance
from scipy import integrate
from itertools import combinations
import math


def getPL(d): # 计算路径损耗（Path Loss），参数d表示距离。其中使用了波长lambda和距离的三次方来计算路径损耗。
    alpha = 3
    lambda_ = (3 * (10 ** 8)) / (2.4 * (10 ** 9))
    PL = (lambda_ ** 2) / (4 * 4 * pi * pi * (d ** alpha))
    # print(PL)
    return PL


# print(getPL(1))
def getPr(node1, node2): # 计算接收功率Pr。函数内部处理了不同情况下的距离计算，包括源节点和目的节点的不同组合。
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
    Pe = 0.3 #0.151  # %#mW transmission power
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

def getPER_noise(Pr, I): # 根据接收功率Pr和干扰I计算包错误率（PER）。
    Pe = 0.3 #0.151  ##mW transmission power
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

# print(getPr(0,0))
def computePijtforDistribu(node1, node2, r, IterferIdset, R): # 计算在特定干扰环境下，从节点node1传输到节点node2的成功概率。
    global numberofsource
    m, n = IterferIdset.shape
    Iterfernum_r = 0
    Pe = 0.3 #0.151  # %  # mW transmission power
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


def getBER(SINR): # 根据信噪比（SINR）计算比特错误率（BER）。
    def f(x):
        return np.exp(-(x ** 2))

    v, err = integrate.quad(f, math.sqrt(SINR), float('inf'))
    v = v * 2 / math.sqrt(pi)
    #print('ber', 0.5*v)
    return 0.5 * v


def getPER(d): # 计算包错误率，d为距离。
    #print('d', d)
    if d != 0:
        Pe = 0.00088 #0.151  # mW transmission power 802.15.4 1wm=0dBm
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
    def __init__(self, node, configuration, ber=0): # 初始化节点、配置和误差率等。
        self.node = node  # 此参数通常包含网络节点的对象或标识符，表示此物理层实例的归属节点。
        self.configuration = configuration  # 提供必要的配置信息，可能包括如传输功率、调制方案或频率设置等，用于物理层操作。
        self.ber = ber  # 用于模拟或计算数据传输的错误特性，影响通信的可靠性。
        self._current_rx_count = 0  # 内部计数器，用来跟踪正在进行的接收操作。在计算通道是否繁忙或确定接收重叠（可能导致碰撞）时非常有用。
        self._channel_busy_start = 0  # 记录通道开始使用进行传输的时间戳或计数器值。帮助确定通道繁忙的时间长度，用于评估利用率和效率。
        self.stat = Stat()  # 聚合各种通信统计数据，如总传输次数、接收次数、碰撞次数、错误累计等，这对于性能评估至关重要。
        self.stat.total_tx = 0  # 跟踪启动的总传输次数。
        self.stat.total_rx = 0  # 跟踪成功接收的总次数。
        self.stat.total_collision = 0  # 统计信号碰撞发生的次数。
        self.stat.total_error = 0  # 记录遇到的总错误数。
        self.stat.total_bits_tx = 0  # 计算总传输的比特数。
        self.stat.total_bits_rx = 0  # 记录总接收的比特数。
        self.stat.total_channel_busy = 0  # 量化通道繁忙的总时间/持续时间。。
        self.stat.total_channel_tx = 0  # 衡量实际数据传输花费的时间。
    # 发送数据包到邻居节点
    def send_pdu(self, pdu): # 发送数据单元函数，计算传输时间，调用发射开始和结束的处理。
        tx_time = (pdu.nbits + PHY_HEADER_LENGTH) * self.configuration["BIT_TRANSMISSION_TIME"]
        self.on_tx_start(pdu)
        # 调用发送结束函数
        self.node.delayed_exec(tx_time, self.on_tx_end, pdu)
        # print(self.node.neighbors)
        for node in self.node.neighbors:
            dist = distance(self.node.pos, node.pos)
            if dist <= self.configuration['tx_range']:
                prop_time = dist * BIT_TRASPORT_TIME + 1
                # 调用目的节点的物理层接受函数
                self.node.delayed_exec(
                    prop_time, node.phy.on_rx_start, pdu)
                self.node.delayed_exec(
                    prop_time + tx_time, node.phy.on_rx_end, pdu)

    # 开始发送数据包，切换到TX状态
    def on_tx_start(self, pdu): # 发送开始时的处理，记录开始时间。
        self.node.state_start_time = self.node.now

    # 发送数据包结束，切换回IDLE状态
    def on_tx_end(self, pdu): # 发送结束时的处理，更新状态。
        last_state_start_time = self.node.state_start_time
        self.node.state_start_time = self.node.now

    def on_rx_start(self, pdu): # 接收开始时的处理，增加当前接收计数，处理可能的冲突。
        self._current_rx_count += 1
        if self._current_rx_count > 1:
            self._collision = True
            self.on_collision(pdu)
        else:
            self._collision = False
            self.node.state_start_time = self.node.now

    def on_rx_end(self, pdu): # 接收结束时的处理，检查是否有冲突，调用相应的处理函数。
        self._current_rx_count -= 1
        if self._current_rx_count != 0:
            self._collision = True
        else:
            self._collision = False
            self._channel_busy_start = 0
            self.node.state_start_time = self.node.now
        if not self._collision:
            success_possible = 1
            # success_possible = 1 - getPER(distance(self.node.pos, self.node.sim.nodes[pdu.src].pos))
            if random.random() <= success_possible:
                self.node.sim.env.process(self.node.mac.on_receive_pdu(pdu))
            else:
                self.on_error(pdu)

        else:
            self.on_collision(pdu)

    def on_collision(self, pdu): # 处理数据包冲突情况。
        self.stat.total_collision += 1

    def on_error(self, pdu): # 处理数据包错误情况。
        self.stat.total_error += 1

    def cca(self): # 检查信道是否清晰（Channel Clear Assessment）。
        """Return True if the channel is clear"""
        return self._current_rx_count == 0

