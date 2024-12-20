import os
from scipy.constants import c  # 从scipy的constants模块导入光速常量


ONE_HOP = 20  # 定义单跳的距离为20单位。
Z_BOUND = 400  # 定义仿真空间的Z轴边界为400单位。
X_BOUND = Y_BOUND = 600  # 同时定义仿真空间的X轴和Y轴边界为600单位。
BOUND = (X_BOUND, Y_BOUND, Z_BOUND)  # 将三个空间维度的边界封装为一个元组，方便后续引用。
UNIT = 1e6  # 定义一个时间单位为1,000,000（百万，可能表示微秒到秒的转换系数）。
RUN_TIME = 100000 * UNIT  # 定义仿真运行的总时间为100,000单位时间乘以前面定义的时间单位，即表示总时间。
# simpy
HELLO_INTERVAL = 0.25 * UNIT  # 定义发送"Hello"消息的时间间隔为0.25单位时间乘以时间单位。
KEY_NODE_PACKET = 2 * UNIT / 1e5  # 定义关键节点包的时间间隔，可能用于发送或处理特定功能的数据包。
VELOCITY_MAX = 44.7  # 定义最大速度为44.7单位（可能是米/秒）。
# 单位是us
POS_INTERVAL = 0.1 * UNIT  # 定义位置更新间隔为0.1单位时间乘以时间单位。
QGEO_TX_RANGE = 350  # 定义地理位置传输范围为350单位。
TX_RANGE = 1.57 * ONE_HOP  # 定义传输范围为1.57倍的单跳距离。

# statics
MAX_BUFFER_SIZE = 64  # 定义最大缓冲区大小为64。

MASK = -2147483648 / 2  # 定义一个掩码，用于某些位操作。

BIT_TRASPORT_TIME = 1 / 3e8 * UNIT  # 定义位传输时间，基于光速和时间单位。
PHY_HEADER_LENGTH = 128  # 定义物理层头部长度为128位。
PACKET_LOSS_RATE = 0.01  # 1% of packets are corrupted 定义数据包丢失率为1%。
### SIGNAL PARAMETERS
FREQUENCY = 2400000000  # 2.4 GHz  # 定义信号频率为2.4 GHz。


WAVELENGTH = c / FREQUENCY  # 计算信号的波长，为光速除以频率。
# mca layer
BROADCAST_ADDR = 0xFFFF  # 定义广播地址为16位全1，即65535。
MAC_QUEUE_MAX_LENGTH = 64  # 定义MAC层队列的最大长度为64。
# 定义MAC层的多个参数，包括队列最小值、最大值、重试次数、时隙持继时间、短帧间隔、长帧间隔、MAC头长度、ACK信息长度、最小和最大争用窗口等。
MAC_QUEUE_MIN = 0
MAC_QUEUE_MAX = 1
RETRIES_NUM = 7
SLOT_DURATION = 20  # 20 microseconds, 802.11g 2.4 GHz
SIFS_DURATION = 10  # 10 microseconds, 802.11g 2.4 GHz
DIFS_DURATION = SIFS_DURATION + (2 * SLOT_DURATION)
MAC_HEADER_LENGTH = 34 * 8  # 34 byte fixed fields of a mac packet
MAX_MAC_PAYLOAD_LENGTH = 2314 * 8
ACK_LENGTH = MAC_HEADER_LENGTH
CW_MIN = 16
CW_MAX = 1024

# MED3QN_MAC mac layer
TIME_SLOT = 10 / 1000 * UNIT  # 定义时间槽的长度，为10毫秒乘以时间单位。

# net
DEFAULT_NET_HEAD = 20 * 8  # 定义默认网络头部长度为20字节，每字节8位，共160位。

# mac_layer_type
DEFAULT_MAC = "802.11"
MED3QN_MAC = "MED3QN_mac"
