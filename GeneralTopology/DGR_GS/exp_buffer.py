from collections import defaultdict, deque


class RouteExpBuffer:
    def __init__(self, max_experience_size):
        """
        初始化路由经验缓冲区。

        :param max_experience_size: 每条路径的最大经验数
        """
        self.max_experience_size = max_experience_size
        self.buffer = defaultdict(lambda: deque(maxlen=self.max_experience_size))

    def add_experience(self, path, experience):
        """
        向指定路径的经验列表中添加新的经验。

        :param path: 路径标识（例如 '0-3-5-7-9'）
        :param experience: 新的经验值
        """
        # 将经验添加到路径的经验队列中，队列会自动丢弃最旧的元素（如果超出了最大值）
        self.buffer[path].appendleft(experience)

    def get_experiences(self, path):
        """
        获取指定路径的所有经验值。

        :param path: 路径标识
        :return: 指定路径的经验列表
        """
        return list(self.buffer[path])

    def get_all_experiences(self):
        """
        获取所有路径的经验值。

        :return: 所有路径及其对应经验的字典
        """
        return {path: list(experiences) for path, experiences in self.buffer.items()}

    def clear(self):
        """
        清空所有路径的经验值。
        """
        self.buffer.clear()

# # 示例使用
# if __name__ == "__main__":
#     # 初始化一个最大经验数为 5 的路由经验缓冲区
#     max_size = 5
#     buffer = RouteExpBuffer(max_experience_size=max_size)
#
#     # 向 '0-3-5-7-9' 路径添加经验
#     buffer.add_experience('0-3-5-7-9', 5954.83)
#     buffer.add_experience('0-3-5-7-9', 4594.83)
#     buffer.add_experience('0-3-5-7-9', 5554.83)
#     buffer.add_experience('0-3-5-7-9', 6000.00)
#     buffer.add_experience('0-3-5-7-9', 6100.00)
#
#     # 超过最大值后，添加新的经验，最旧的会被丢弃
#     buffer.add_experience('0-3-5-7-9', 6200.00)  # 5954.83 会被丢弃
#     buffer.add_experience('0-3-5-7-9', 6300.00)
#
#     # 打印路径 '0-3-5-7-9' 的经验
#     print(buffer.get_experiences('0-3-5-7-9'))  # [6200.00, 6100.00, 6000.00, 5554.83, 4594.83]
#
#     # 获取所有路径的经验
#     print(buffer.get_all_experiences())
