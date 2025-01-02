import simpy
import numpy as np
from models import Network
import matplotlib.pyplot as plt

# 算法参数设置
tau = 50  # 时隙
delta = 0.02
eta = 0.05
epsilon = 0.02

# 用于存储时延模拟的数据，可视化
data = []


def gradient_sampling(env0, env1, env2, load):
    # 初始化路由向量
    r = np.array([1/3, 1/3, 1/3])
    r = np.clip(r, epsilon, 1-epsilon)
    print("initial routing vector:", r)
    print("-" * 50)

    # 迭代次数
    num_episodes = 50
    for i in range(num_episodes):
        print(f"Episode: {i + 1}")

        # 使用 r_i + delta * u 模拟 tau 个时隙
        network0 = Network(env0, load, r)  # 创建网络模型
        env0.run(until=env0.now + tau)  # 模拟 tau 个时隙
        hat_D = sum(network0.get_queue_lengths())

        print(hat_D, network0.destination.received_packets)
        data.append(hat_D)

        # 采样扰动向量
        # 1. 生成 K-1 维的随机向量
        v = np.random.randn(2)
        # 2. 将 v 扩展为 K 维，并保证所有元素之和为 0
        u = np.concatenate([v, [-np.sum(v)]])
        # 3. 将 u 投影到单位球面上
        u /= np.linalg.norm(u)

        r_plus_delta_u = (r + delta * u) / np.sum(r + delta * u)  # 归一化
        r_minus_delta_u = (r - delta * u) / np.sum(r - delta * u)  # 归一化
        print("Routing vector r+delta_u | r-delta_u:", r_plus_delta_u, r_minus_delta_u)
        # 使用 r_i + delta * u 模拟 tau 个时隙
        network1 = Network(env1, load, r_plus_delta_u)  # 创建网络模型
        env1.run(until=env1.now + tau)  # 模拟 tau 个时隙
        hat_D_plus = sum(network1.get_queue_lengths())

        # 使用 r_i - delta * u 模拟 tau 个时隙
        network2 = Network(env2, load, r_minus_delta_u)  # 创建网络模型
        env2.run(until=env2.now + tau)
        hat_D_minus = sum(network2.get_queue_lengths())

        print("Des received_packets:",network1.destination.received_packets, network2.destination.received_packets)

        # 计算梯度
        hat_nabla_D = 3 * (hat_D_plus - hat_D_minus) * u / (2 * delta)
        print(hat_nabla_D)
        # 更新路由向量
        r = np.clip(r - eta * hat_nabla_D, epsilon, 1-epsilon)
        r = r / np.sum(r)

        # 打印信息
        print("hat_D_plus | hat_D_minus: ",hat_D_plus, hat_D_minus)
        print("Link Queue Lengths: ", network1.get_queue_lengths(), network2.get_queue_lengths())
        print("-"*50)

        # 重置数据队列


def main():
    """
    主函数，运行模拟
    :return: None
    """

    # 创建 simpy 环境
    env0 = simpy.Environment()
    env1 = simpy.Environment()
    env2 = simpy.Environment()

    # 定义三种负载情况
    loads = [4]

    for load in loads:
        # 运行梯度采样算法
        print(f"Load: {load}")
        gradient_sampling(env0, env1, env2, load)
        print(data)
        for i in range(450):
            data.append(0)

        # 创建 x 轴数据，表示数据点的索引
        x = range(len(data))

        # 绘制折线图
        plt.plot(x, data)

        # 添加标题和标签
        plt.title("Low Load")
        plt.xlabel("Time")
        plt.ylabel("Average Delay")
        plt.show()


if __name__ == "__main__":
    main()
