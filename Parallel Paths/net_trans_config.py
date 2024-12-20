from scipy.constants import c  # 从scipy的constants模块导入光速常量

# ==========================物理层和MAC层相关参数配置============================

ONE_HOP = 20  # 定义单跳的距离为20米。
Z_BOUND = 400  # 定义仿真空间的Z轴边界为400米。
X_BOUND = 600  # 同时定义仿真空间的X轴边界为600米。
Y_BOUND = 600  # 同时定义仿真空间的Y轴边界为600米。
UNIT = 1e6  # 用于转换时间单位 s -> us
# simpy
HELLO_INTERVAL = 0.25 * UNIT  # 定义发送"Hello"消息的时间间隔为0.25单位时间乘以时间单位。
KEY_NODE_PACKET = 2 * UNIT / 1e5  # 定义关键节点包的时间间隔，可能用于发送或处理特定功能的数据包。
VELOCITY_MAX = 44.7  # 定义最大速度为44.7单位（可能是米/秒）。
# 单位是us
TX_RANGE = 1.57 * ONE_HOP  # 定义传输范围为1.57倍的单跳距离。

################################ phy layer ################################
band_width = 11 * UNIT
BIT_TRANSPORT_TIME = 1 / 3e8 * UNIT  # 定义位传播时间，基于光速和时间单位。
BIT_TRANSMISSION_TIME = 1 / band_width * UNIT
PHY_HEADER_LENGTH = 24  # 定义物理层头部长度

########################### SIGNAL PARAMETERS #############################
FREQUENCY = 2400000000  # 2.4 GHz  # 定义信号频率为2.4 GHz。
WAVELENGTH = c / FREQUENCY  # 计算信号的波长，为光速除以频率。

################################ mac layer ################################
MAC_TYPE = 'TDMA'
SLOT_DURATION = 20  # 20 microseconds, 802.11g 2.4 GHz
SIFS_DURATION = 10  # 10 microseconds, 802.11g 2.4 GHz
DIFS_DURATION = SIFS_DURATION + (2 * SLOT_DURATION)
MAC_HEADER_LENGTH = 34 * 8  # 34 byte fixed fields of a mac packet
MAX_MAC_PAYLOAD_LENGTH = 2314 * 8

