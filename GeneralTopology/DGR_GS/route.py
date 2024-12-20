from GeneralTopology.util.utils import *
from config import *


def clip_and_normal_pr(prob_dict, epsilon):
    # 计算裁剪前的总和
    total_sum = sum(prob_dict.values())

    # 裁剪和归一化
    clipped_normalized_probs = {}
    for k, v in prob_dict.items():
        # 裁剪值
        clipped_value = np.clip(v / total_sum, epsilon, 1 - epsilon)
        clipped_normalized_probs[k] = clipped_value

    # 计算裁剪后值的总和
    clipped_sum = sum(clipped_normalized_probs.values())

    # 重新归一化
    for k in clipped_normalized_probs:
        clipped_normalized_probs[k] /= clipped_sum

    return clipped_normalized_probs


def find_all_paths(adj_tb, start_node, end_node):
    """
    find all paths between start_node and end_node using DFS
    """

    def dfs(current_node, path):
        if current_node == end_node:
            all_paths.append(path)
            return
        for neighbor in adj_tb[current_node]:
            dfs(neighbor, path + [neighbor])

    all_paths = []
    dfs(start_node, [start_node])
    return all_paths


def format_paths_and_init_pr(paths):
    """
    Format paths and calculate probabilities
    """
    total_paths = len(paths)
    paths_dict = {}
    for path in paths:
        path_str = '-'.join(map(str, path))
        paths_dict[path_str] = 1 / total_paths
    return paths_dict


def combine_path_u(paths, point):
    vector_u = {}
    i = 0
    for path in paths:
        path_str = '-'.join(map(str, path))
        vector_u[path_str] = point[i]
        i += 1
    return vector_u


# print(format_paths_and_init_pr(find_all_paths(ADJ_TABLE, 0, 9)))


class RouteTb:
    def __init__(self, node):
        self.node = node
        self.route_vector = {}  # {9:{3:0.5, 4:0.5}}
        self.vector_u = {}
        self.all_paths_cdf = {}  # {'0-3-5-7-9':[k, loc, scale]}  gamma distribution parameters(k, loc, scale)
        self.init_route_vector()
        self.sample_vector_u()

    def init_route_vector(self):
        if self.node in DES:
            pass
        if self.node.node_id in [0, 1, 2]:
            # print(self.node.node_id, FLOW_DICT[self.node.node_id])
            route = format_paths_and_init_pr(find_all_paths(ADJ_TABLE, self.node.node_id, FLOW_DICT[self.node.node_id]))
            self.route_vector = {FLOW_DICT[self.node.node_id]: clip_and_normal_pr(route, epsilon)}
        else:
            route_i_to_DES_0 = format_paths_and_init_pr(find_all_paths(ADJ_TABLE, self.node.node_id, DES[0]))
            route_i_to_DES_1 = format_paths_and_init_pr(find_all_paths(ADJ_TABLE, self.node.node_id, DES[1]))
            route_i_to_DES_2 = format_paths_and_init_pr(find_all_paths(ADJ_TABLE, self.node.node_id, DES[2]))
            self.route_vector = {
                DES[0]: clip_and_normal_pr(route_i_to_DES_0, epsilon),
                DES[1]: clip_and_normal_pr(route_i_to_DES_1, epsilon),
                DES[2]: clip_and_normal_pr(route_i_to_DES_2, epsilon)}

    def sample_vector_u(self):
        if self.node.node_id in [10, 11, 14, 15, 16, 17]:
            pass
        if self.node.node_id in [0, 1, 2]:
            all_paths = find_all_paths(ADJ_TABLE, self.node.node_id, FLOW_DICT[self.node.node_id])
            self.vector_u = {FLOW_DICT[self.node.node_id]: combine_path_u(all_paths, sample_u(len(all_paths)))}
        else:
            all_paths_i_to_DES_0 = find_all_paths(ADJ_TABLE, self.node.node_id, DES[0])
            all_paths_i_to_DES_1 = find_all_paths(ADJ_TABLE, self.node.node_id, DES[1])
            all_paths_i_to_DES_2 = find_all_paths(ADJ_TABLE, self.node.node_id, DES[2])
            self.vector_u = {
                DES[0]: combine_path_u(all_paths_i_to_DES_0, sample_u(len(all_paths_i_to_DES_0))),
                DES[1]: combine_path_u(all_paths_i_to_DES_1, sample_u(len(all_paths_i_to_DES_1))),
                DES[2]: combine_path_u(all_paths_i_to_DES_2, sample_u(len(all_paths_i_to_DES_2)))}
