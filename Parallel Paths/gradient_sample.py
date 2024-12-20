import simpy
import numpy as np
from sim_env import Simulator
import matplotlib.pyplot as plt
from net_trans_config import *
from collections import defaultdict
from itertools import chain
from scipy.stats import gamma, norm

# 算法参数设置
tau = 5 * UNIT  # 时隙(模拟时间)
delta = 0.02
eta = 1e-7
epsilon = 0.02

# 拓扑关系定义
sum_nodes = 11
src = [0]  # 设置源节点
des = [10]  # 设置目标节点
adj_M = {
    0: [1, 4, 7],
    1: [2],
    2: [3],
    3: [10],
    4: [5],
    5: [6],
    6: [10],
    7: [8],
    8: [9],
    9: [10],
    10: [],
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

# 网络传输相关参数设置
deadline = 160000  # 数据包的端到端时延期限
violate_Pr = 1e-5  # 设置违背概率
send_rates = {
    0: 1500,
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

# 用于存储时延模拟的数据，可视化
data = []


def sample_from_intersection(N):
    # 生成 N 维标准正态分布的随机点
    point = np.random.normal(size=N)

    # 归一化，使其落在单位球面上
    point /= np.linalg.norm(point)

    # 超平面的法向量
    normal = np.ones(N)
    normal /= np.linalg.norm(normal)

    # 计算点在法向量方向上的投影
    projection = np.dot(point, normal) * normal

    # 投影到超平面
    point_on_hyperplane = point - projection

    # 归一化，使其落在交线上
    point_on_hyperplane /= np.linalg.norm(point_on_hyperplane)

    return point


def gradient_sampling():
    # 初始化路由向量
    r = np.array([1 / 3, 1 / 3, 1 / 3])
    r = np.clip(r, epsilon, 1 - epsilon)
    print("initial routing vector:", r)
    print("-" * 50)

    # 迭代次数
    num_episodes = 100
    for i in range(num_episodes):
        print(f"Episode: {i + 1}")

        # 使用 r_i 模拟 tau 个时隙
        route_vector = {
            1: r[0],
            4: r[1],
            7: r[2],
        }
        sim0 = Simulator(sum_nodes, tau, src, des, adj_M, position, send_rates, frame_slot, route_vector, i)  # 创建网络模型
        # print(sim0.e2ed_delay)
        hat_D = sim0.get_avg_e2ed_delay()

        data.append(hat_D)

        # 采样扰动向量
        u = sample_from_intersection(3)

        r_plus_delta_u = r + delta * u
        r_minus_delta_u = r - delta * u
        print("Routing vector r+delta_u | r-delta_u:", r_plus_delta_u, r_minus_delta_u)
        # 使用 r_i + delta * u 模拟 tau 个时隙
        route_vector = {
            1: r_plus_delta_u[0],
            4: r_plus_delta_u[1],
            7: r_plus_delta_u[2],
        }
        sim1 = Simulator(sum_nodes, tau, src, des, adj_M, position, send_rates, frame_slot, route_vector, i)  # 创建网络模型
        hat_D_plus = sim1.get_avg_e2ed_delay()

        # 使用 r_i - delta * u 模拟 tau 个时隙
        route_vector = {
            1: r_minus_delta_u[0],
            4: r_minus_delta_u[1],
            7: r_minus_delta_u[2],
        }
        sim2 = Simulator(sum_nodes, tau, src, des, adj_M, position, send_rates, frame_slot, route_vector, i)  # 创建网络模型
        hat_D_minus = sim2.get_avg_e2ed_delay()

        # 计算梯度
        hat_nabla_D = 3 * (hat_D_plus - hat_D_minus) * u / (2 * delta)

        # 更新路由向量
        r = np.clip(r - eta * hat_nabla_D, epsilon, 1 - epsilon)
        r = r / np.sum(r)

        # 计算 r_j，首先将两次模拟所得的时延数据合并，合并之后按照路径绘制时延分布
        r_j = []
        e2ed_delay_all = {key: list(chain(sim1.e2ed_delay.get(key, []), sim2.e2ed_delay.get(key, [])))
                          for key in set(chain(sim1.e2ed_delay.keys(), sim2.e2ed_delay.keys()))}
        print(e2ed_delay_all)

        for label, values in e2ed_delay_all.items():
            k, loc, scale = gamma.fit(values)
            x_range = np.linspace(min(values), max(values), 100)
            y_cdf = gamma.cdf(x_range, a=k, loc=loc, scale=scale)  # 将 k 传递给 a 参数
            Pr_path = gamma.cdf(deadline, a=k, loc=loc, scale=scale)
            r_j.append(1 if 1 - Pr_path <= violate_Pr else 0)
        print("r_j:", r_j)

        # r 与 r_j 做按位乘积
        r = (np.array(r) * np.array(r_j)).tolist()
        r = r / np.sum(r)  # 归一化

        # 打印信息
        print("hat_D_plus | hat_D_minus: ", hat_D_plus, hat_D_minus)
        print("-" * 50)

        # 重置数据队列


def main():
    """
    主函数，运行模拟
    :return: None
    """
    gradient_sampling()


if __name__ == "__main__":
    main()
