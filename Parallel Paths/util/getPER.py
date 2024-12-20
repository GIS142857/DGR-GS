import random
from collections import deque
import simpy
from math import pi
import numpy as np
from scipy import integrate
from itertools import combinations
import math



# 计算路径损耗（Path Loss），参数d表示距离。其中使用了波长lambda和距离的三次方来计算路径损耗。
def getPL(d):
    alpha = 3
    lambda_ = (3 * (10 ** 8)) / (2.4 * (10 ** 9))
    PL = (lambda_ ** 2) / (4 * 4 * pi * pi * (d ** alpha))
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




# 根据信噪比（SINR）计算比特错误率（BER）。
def getBER(SINR):
    def f(x):
        return np.exp(-(x ** 2))
    v, err = integrate.quad(f, math.sqrt(SINR), float('inf'))
    v = v * 2 / math.sqrt(pi)
    return 0.5 * v


# 接收信号强度 = 射频发射功率 + 发射端天线增益 – 路径损耗 – 障碍物衰减 + 接收端天线增益

# 信噪比 SNR = 10lg（ PS / PN ），其中：
# SNR：信噪比，单位是dB。
# PS：信号的有效功率。
# PN：噪声的有效功率。

# 信干噪比 SINR = 10lg[ PS /( PI + PN ) ]，其中：
# SINR：信干噪比，单位是dB。
# PS：信号的有效功率。
# PI：干扰信号的有效功率。
# PN：噪声的有效功率。


# 数据速率 = 信道带宽 × 调制阶数 × 编码速率
# 信道带宽: 单位为 Hz。
# 调制阶数: 表示每个符号携带的信息位数，例如 BPSK 为 1，QPSK 为 2，16-QAM 为 4。
# 编码速率: 有效数据速率与编码速率之比，例如 1/2，3/4，5/6 等。

# 计算包错误率，d为距离。
def getPER(d):
    if d != 0:
        Pe = 0.00088  #0.151  # mW transmission power 802.15.4 1wm=0dBm
        Tpaquet = 512 * 8  # 802.15.4: 5B(SHR)+1B(PHR)+127B(PSDU) = 6+127=133B(SHR:synchronizaition header, PHR: PHY header,PSDU:PHY payload)
        N0 = -192.0  # dBm / Hz
        B = 6000000  # # Bandwidth 1MHz
        noise = (10 ** (N0 / 10)) * B
        PL = getPL(d)
        Pr = Pe * PL
        # % I = 6.2691e-014;
        # % SINR = Pr. / (noise + I);
        SINR = Pr / noise
        ber = getBER(SINR)
        PER = (1 - ((1 - ber) ** Tpaquet))
        return PER
    else:
        PER = 0
        return PER

print(getPER(31.6))