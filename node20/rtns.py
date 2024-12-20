import numpy as np
import networkx as nx
import random
from Random_graph import randomG
import copy
from collections import defaultdict
import queue
from heapq import *

class RTNS:
    def __init__(self, g, D, f, source, des):
        self.graph = g
        self.deadline = D
        self.f = f
        self.num_node = self.graph.node_num
        self.q_table = np.zeros((self.num_node, self.num_node))
        self.episodes = 2000
        self.alpha = 0.4
        self.gamma = 0.99
        self.source = source
        self.des = des
        self.pass_weight = np.ones((g.node_num, g.node_num))

    def get_maxq_nb(self, u, child_list):
        max_nb = -1
        max_q = -10000
        # print('get_maxq_nb')
        # print('child_list', child_list)
        for i in range(len(child_list)):
            #print('v_q', self.q_table[u][child_list[i]])
            if self.q_table[u][child_list[i]] > max_q:
                max_q = self.q_table[u][child_list[i]]
                max_nb = child_list[i]
        return max_nb

    def get_nb(self, u, child_list, d, h):
        # print('get_best_nb', u)
        # print('child_list', child_list)
        max_nb = -1
        max_q = -10000
        for i in range(len(child_list)):
            nb = child_list[i]
            nb_num, _ = self.graph.get_nb_num(nb)
            if nb_num == 0 and nb < self.des:
                continue
            dmin = 0
            if nb < self.des:
                # print('node', u)
                # print('key', key)
                # print('worst_to_des', self.graph.node_list[nb].worst_to_des[key])
                dmin = self.get_short_path(nb, self.des)
            elif nb > self.des:
                dmin = 10000
            key1 = str(nb) + ',' + str(self.f)
            #print('dmin', dmin, 'nb_worst', max(self.graph.node_list[u].nb_delay[key1]), 'd', d)
            if (dmin + max(self.graph.node_list[u].nb_delay[key1])) <= d:
                #print('nb', nb, 'q', self.q_table[u][child_list[i]])
                if self.q_table[u][child_list[i]] > max_q:
                    max_q = self.q_table[u][child_list[i]]
                    max_nb = child_list[i]
        return max_nb


    def random_pick(self, some_list, probabilities):
        x = random.uniform(0, 1)
        # print('x',x)
        cumulative_probability = 0.0
        # print('some_list',some_list)
        # print('probabilities',probabilities)
        if len(some_list) == 1 and probabilities[0] != 0:
            return some_list[0]

        l = len(some_list)
        for item, item_probability in zip(some_list, probabilities):
            cumulative_probability += item_probability
            # print('item_probability',item_probability)
            if x < cumulative_probability:
                return item
        # print('uu', u)
        return some_list[l-1]

    def get_reward(self, u, v, k, training):
        # print('u',u,'v',v,'G',table.graph.G)
        # print('some_list',table.graph.G[u][v]['real_d'])
        # print('probabilities__',table.graph.G[u][v]['real_prob'])
        if v == -1:
            return 5000

        key = str(v) + ',' + str(k)
        real_d = self.graph.node_list[u].nb_delay[key]
        real_pro = self.graph.node_list[u].nb_delay_pro[key]
        if training:
            return self.random_pick(real_d, real_pro)
        else:
            return max(real_d)

    def get_reward_process(self, v, theta):
        #print('v', v, 'des', self.des)
        if v == self.des:
            #print('deadline', self.deadline, 'theta', theta)
            return self.deadline - theta
        else:
            return 0

    def get_next_maxq(self, v):
        #print('get_next_maxq')
        num, child_list = self.graph.get_nb_num(v)
        #print('nb', v, 'num', num)
        max_q = -10000
        for i in range(num):
            if self.q_table[v][child_list[i]] > max_q:
                max_q = self.q_table[v][child_list[i]]
        return max_q


    def update_q_table(self, u, v, R):
        #print('update_q_table')
        max_q = 0
        if v < self.num_node-1:
            max_q = self.get_next_maxq(v)
        #print('q_uv', self.q_table[u][v], 'R', R, 'max_q', max_q)
        self.q_table[u][v] = self.q_table[u][v] + self.alpha * (R + self.gamma * max_q - self.q_table[u][v])
        #print('q_uvv', self.q_table[u][v])


    def get_avg_d(self, u, v, adj_weight):
        f = self.f
        if adj_weight[u][v] > 1:
            #print('adj_weightt', adj_weight)
            self.graph.get_delayBasedNC(u, v, f, adj_weight)
        key = str(v) + ',' + str(f)  # + ',' + str(h+1)
        real_d = self.graph.node_list[u].nb_delay[key]
        real_pro = self.graph.node_list[u].nb_delay_pro[key]
        avg_d = 0
        for i in range(len(real_d)):
            delay = real_d[i]
            avg_d += delay * real_pro[i]
        return avg_d

    def get_max_d(self, u, v, h, f):
        key = str(v) + ',' + str(f) + ',' + str(h + 1)
        real_d = self.graph.node_list[u].nb_delay[key]
        return max(real_d)

    def dijkstra_raw(self, edges, from_node, to_node):
        # print('from', from_node)
        # print('to', to_node)
        g = defaultdict(list)
        for l, r, c in edges:
            g[l].append((c, r))
        q, seen = [(0, from_node, ())], set()
        # print('q', q)
        while q:
            (cost, v1, path) = heappop(q)
            # print('qq', q)
            # print('v1', v1)
            # print('seen', seen)
            if v1 not in seen:
                seen.add(v1)
                path = (v1, path)
                if v1 == to_node:
                    # print('cost', cost)
                    # print('path', path)
                    return cost, path
                for c, v2 in g.get(v1, ()):
                    if v2 not in seen:
                        heappush(q, (cost + c, v2, path))
        return 10000, None

    def dijkstra_(self, edges, from_node, to_node):
        # print('from_node',from_node)
        # print('to', to_node)
        len_shortest_path = -1
        ret_path = []
        # exist = nx.has_path(self.graph, from_node, to_node)
        # print('exist', exist)
        length, path_queue = self.dijkstra_raw(edges, from_node, to_node)
        if path_queue == None:
            return length, None
        if len(path_queue) > 0:
            len_shortest_path = length  ## 1. Get the length firstly;
            ## 2. Decompose the path_queue, to get the passing nodes in the shortest path.
            left = path_queue[0]
            ret_path.append(left)  ## 2.1 Record the destination node firstly;
            right = path_queue[1]
            while len(right) > 0:
                left = right[0]
                ret_path.append(left)  ## 2.2 Record other nodes, till the source-node.
                right = right[1]
            ret_path.reverse()  ## 3. Reverse the list finally, to make it be normal sequence.
        return len_shortest_path, ret_path

    def pop_list(self, next_list):
        l = len(next_list)
        node = next_list[l - 1]
        temp = []
        for i in range(l - 1):
            temp.append(next_list[i])
        return node, temp

    def push(self, child, next_list):
        for i in range(len(next_list)):
            if child == next_list[i]:
                return next_list
        next_list.append(child)
        return next_list

    def get_short_path(self, nb, des):
        # print('get_short_path', nb, h)
        M = 100000
        graph = self.graph
        adj_weight = np.ones((graph.node_num, graph.node_num)) * M
        edges = []
        next_list = []
        pre_node = nb
        next_list.append(pre_node)
        while len(next_list) > 0:
            # print('node', node)
            node, next_list = self.pop_list(next_list)
            # print('node', node)
            if node == des:
                continue
            num, child_list = graph.get_nb_num(node)
            # print('childs', child_list)
            for i in range(num):
                child = child_list[i]
                # print('child', child)
                key = str(child) + ',' + str(self.f)
                # print('key', key)
                nb_delay = max(graph.node_list[node].nb_delay[key])
                adj_weight[node][child] = nb_delay
                next_list = self.push(child, next_list)
            # h += 1
        # print('pass_weight', self.graph.pass_matrix)
        # print('adj_weight', adj_weight)
        for i in range(len(adj_weight)):
            for j in range(len(adj_weight[0])):
                if i != j and adj_weight[i][j] != M:
                    edges.append((i, j, adj_weight[i][j]))

        # shrot_dis = self.dijstra(adj_weight, nb, self.graph.node_num - 1, self.graph.node_num)
        short_2, path = self.dijkstra_(edges, nb, des)
        # print('short_1', shrot_dis)
        # print('short_2', short_2, 'path', path)
        # Adjacent = [[0, 1, 12, Inf, Inf, Inf],
        #             [Inf, 0, 9, 3, Inf, Inf],
        #             [Inf, Inf, 0, Inf, 5, Inf],
        #             [Inf, Inf, 4, 0, 13, 15],
        #             [Inf, Inf, Inf, Inf, 0, 4],
        #             [Inf, Inf, Inf, Inf, Inf, 0]]
        # Src, Dst, N = 0, 5, 6
        # shrot_dis = self.dijstra(Adjacent, Src, Dst, N)
        return short_2


    def get_best_path(self, adj_weight):
        epsilon = 0.1
        max_iter = 50
        for e in range(self.episodes):
            #print('e', e)
            u = self.source
            theta = 0
            D_u = self.deadline
            h = 0
            for t in range(max_iter):
                #print('u', u)
                num, child_list = self.graph.get_nb_num(u)
                #print('child_list', child_list, 'num', num)
                if num == 0:
                    break
                p = np.zeros(num)
                v_max = self.get_maxq_nb(u, child_list)
                #print('v_max', v_max)
                for i in range(num):
                    nb = child_list[i]
                    nb_num, _ = self.graph.get_nb_num(nb)
                    if nb_num == 0 and nb < self.des:
                        continue
                    dmin = 0
                    if nb < self.des:
                        dmin = self.get_short_path(nb, self.des)
                    elif nb > self.des:
                        dmin = 10000
                    key1 = str(nb) + ',' + str(self.f)
                    if (dmin + max(self.graph.node_list[u].nb_delay[key1])) > D_u:
                        p[i] = 0
                    elif nb == v_max:
                        p[i] = 1 - epsilon
                    else:
                        p[i] = epsilon / (num-1)
                #print('p', p)
                if max(p) == 0:
                    break
                v = self.random_pick(child_list, p)
                #print('v', v)
                h += 1
                theta_uv = self.get_reward(u, v, self.f, training=True)
                #print('theta_uv', theta_uv)
                theta += theta_uv
                D_u -= theta_uv
                R = self.get_reward_process(v, theta)
                #print('R', R)
                self.update_q_table(u, v, R)
                u = v
                if v >= self.des:
                    break

            #self.test(adj_weight)

        #### best_path
        # path = []
        # u = self.source
        # path.append(u)
        # theta = 0
        # D_u = self.deadline
        # h = 0
        # for t in range(max_iter):
        #     #print('u', u)
        #     num, child_list = self.graph.get_nb_num(u)
        #     if num == 0:
        #         break
        #     p = np.zeros(num)
        #     v = self.get_nb(u, child_list, D_u, h)
        #     path.append(v)
        #     #print('v', v)
        #     h += 1
        #     theta_uv = self.get_reward(u, v, self.f, training=False)
        #     #print('theta_uv', theta_uv)
        #     theta += theta_uv
        #     D_u -= theta_uv
        #     u = v
        #     if v >= self.des:
        #         break
        # cost = 0
        # ave_sum = 0
        # worst_sum = 0
        # for i in range(len(path) - 1):
        #     # print('node', trip[i], 'n_node', trip[i+1], 'avg_d', table.graph.G[trip[i]][trip[i+1]]['ave_d'])
        #     # ave_sum += table.graph.G[trip[i]][trip[i+1]]['ave_d']
        #     ave_sum += self.get_avg_d(path[i], path[i + 1], adj_weight)
        # for j in range(len(path) - 1):
        #     adj_weight[path[j]][path[j + 1]] += 1
        #
        #print('save_qTable')
        addr = 'q_table' + '_' + str(self.deadline)
        np.save(addr, self.q_table)
        #print('After_save_qTable')
        #
        # return ave_sum, path, adj_weight


    def test(self, adj_weight):
        u = self.source
        theta = 0
        D_u = self.deadline
        h = 0
        path = []
        path.append(u)
        max_iter = 500
        for t in range(max_iter):
            # print('u', u)
            num, child_list = self.graph.get_nb_num(u)
            if num == 0:
                break
            p = np.zeros(num)
            v = self.get_nb(u, child_list, D_u, h)
            path.append(v)
            # print('v', v)
            h += 1
            theta_uv = self.get_reward(u, v, self.f, training=False)
            # print('theta_uv', theta_uv)
            theta += theta_uv
            D_u -= theta_uv
            R = self.get_reward_process(v, theta)
            # print('R', R)
            self.update_q_table(u, v, R)
            u = v
            if v >= self.des:
                break
        ave_sum = 0
        for i in range(len(path) - 1):
            # print('node', trip[i], 'n_node', trip[i+1], 'avg_d', table.graph.G[trip[i]][trip[i+1]]['ave_d'])
            # ave_sum += table.graph.G[trip[i]][trip[i+1]]['ave_d']
            ave_sum += self.get_avg_d(path[i], path[i + 1], adj_weight)

        # if self.f == 0:
        #     filename = 'rtns_1.txt'
        #     with open(filename, 'a') as file_object:
        #         file_object.write(str(ave_sum))
        #         file_object.write('\n')
        # elif self.f == 1:
        #     filename = 'rtns_2.txt'
        #     with open(filename, 'a') as file_object:
        #         file_object.write(str(ave_sum))
        #         file_object.write('\n')
        # elif self.f == 2:
        #     filename = 'rtns_3.txt'
        #     with open(filename, 'a') as file_object:
        #         file_object.write(str(ave_sum))
        #         file_object.write('\n')
        # else:
        #     filename = 'rtns_4.txt'
        #     with open(filename, 'a') as file_object:
        #         file_object.write(str(ave_sum))
        #         file_object.write('\n')

# if __name__ == "__main__":
#     N = 10
#     p = 0.7
#     delay_num = 3
#     rand_graph = randomG(N, p)
#     M, g = rand_graph.generate_graph()
#     rand_graph.add_delay(delay_num)
#     # print('g1',g1.G,'g2',g2.G,'rand_graph',rand_graph.G)
#     #print('M', M)
#     for i in range(len(M)):
#         for j in range(len(M)):
#             if M[i][j] == 1:
#                 print(i, j, g[i][j])
#                 sum = 0
#                 for k in range(delay_num):
#                     sum += rand_graph.G[i][j]['real_d'][k] * rand_graph.G[i][j]['real_prob'][k]
#                 rand_graph.G[i][j]['ave_d'] = sum
#                 #print('edge',i,j,'ave',rand_graph.G[i][j]['ave_d'])
#                 # print('g1', g1.G[i][j],'prob',g1.G[i][j]['real_prob'])
#                 # print('g2', g2.G[i][j])
#         # print(graph[node])
#     ave_path = nx.dijkstra_path(rand_graph.G, source=0, target=N - 1, weight='worst')
#     ave_path_dis = nx.dijkstra_path_length(rand_graph.G, source=0, target=N - 1, weight='worst')
#     print('min_ave_path', ave_path,'dis',ave_path_dis)
#     max_ave_path = nx.dijkstra_path(rand_graph.G, source=0, target=N - 1, weight='off_worst')
#     max_ave_path_dis = nx.dijkstra_path_length(rand_graph.G, source=0, target=N - 1, weight='off_worst')
#     print('max_ave_path', max_ave_path, 'dis', max_ave_path_dis)
#
#     D = 500
#     rtns = RTNS(rand_graph, D)
#     path, cost = rtns.get_best_path()
#     print('q_table', rtns.q_table)
#     print('path', path, 'cost', cost)


