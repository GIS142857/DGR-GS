# coding:utf-8

import copy
import random
from queue import Queue, LifoQueue
# import Queue as qu
import numpy as np
import networkx as nx
from node import NODE
from collections import defaultdict
import math
from heapq import *


MIN_PRO = 0.03


class randomG:
    def __init__(self, N, p, flow_num, R, per_num, device, episodes, sources, dess, packetLen):
        self.node_num = N
        self.flow_num = int(flow_num)
        self.pdr_min = 0.7
        self.per_min = 0.5
        self.per_max = 0.8
        self.per_num = per_num
        self.ave_channel_d = 0
        self.var_channel_d = 0
        self.packet_len = 8000
        self.R = R
        self.packetLen = packetLen
        self.MaxRetrainsmission = 5
        self.sources = sources
        self.dess = dess
        self.node_list = []
        self.adj_matrix = np.zeros((N, N))
        self.weight_adj = []
        # self.node_feature = np.zeros((N,2))
        self.G = nx.DiGraph()
        self.to_des_worst = np.zeros(N)
        self.to_des_worst_path = np.zeros(N)
        self.N = N
        self.p = p
        self.beita = 0.6  # 0.8
        self.rho = 0.7
        self.gamma = 20
        self.max_worst = 300
        self.max_single = self.beita * np.math.sqrt(self.N) * (1 + self.rho) * self.gamma
        self.device = device
        self.loss_history = []
        self.episodes = episodes

    def sucess(self, adj_M):
        G = nx.DiGraph()
        for i in range(len(adj_M)):
            for j in range(len(adj_M)):
                if adj_M[i][j] == 1:
                    G.add_edge(i, j)
        # print('G0',G.has_node(0))
        # print('G9', G.has_node(0))
        if not G.has_node(0) or not G.has_node(len(adj_M) - 1):
            return False
        if nx.has_path(G, 0, len(adj_M) - 1):
            return True
        else:
            return False

    def generate_graph(self, packets, sources):
        # print('generate_graph')
        done = True
        # M = np.zeros((self.N, self.N))
        while done:
            adj_M = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    rand = random.random()
                    # print('node', i, 'next_node', j, 'rand', rand)
                    if rand < self.p and (j - i) <= self.beita * np.math.sqrt(self.N):
                        adj_M[i][j] = 1
                    else:
                        adj_M[i][j] = 0
            # print('sucess',self.sucess(adj_M))
            # print('adj',adj_M)
            if self.sucess(adj_M):
                done = False
                # G = nx.DiGraph()
                for i in range(len(adj_M)):
                    for j in range(len(adj_M)):
                        if adj_M[i][j] == 1:
                            self.G.add_edge(i, j)
                self.adj_matrix = adj_M

                for i in range(self.node_num):
                    num, child_list = self.get_nb_num(i)
                    node = NODE(i, num, child_list, self.device, self.episodes, packets, sources)  ### id, action_dim, action_list
                    self.node_list.append(node)
                # for k in range(self.flow_num):
                #     key = str(k)
                #     self.node_list[self.sources[k]].arrival_rate[key] = self.R[k]
                #     self.node_list[self.sources[k]].maxWaitTime[key] = 0
                return adj_M, self.G

    def load_graph(self, packets, sources):
        for i in range(self.node_num):
            #num, child_list = self.get_nb_num(i)
            #print('num', num)
            node = NODE(i, 0, [], self.device, self.episodes, packets, sources)  ### id, action_dim, action_list
            self.node_list.append(node)
        f = open("graph.txt", "r")
        #print('MM', self.adj_matrix)
        for i in range(self.N):
            line = f.readline()
            #print(line)
            s = line.split(' ')
            #print('s', s)
            #print('i', i, 'N', self.N)
            for j in range(len(s)):
                #print('s_j', s[j])
                if s[j] == '\n':
                    break
                nb = int(s[j])
                #print('nb', nb)
                self.adj_matrix[i][nb] = 1
        #print('M', self.adj_matrix)
        for i in range(len(self.adj_matrix)):
            for j in range(len(self.adj_matrix)):
                if self.adj_matrix[i][j] == 1:
                    self.G.add_edge(i, j)

        line = f.readline()
        for i in range(self.N*5):
            line = f.readline()
            if i % 5 == 0:
                continue
            if i % 5 == 1:
                s = line.split('-') # 2,0,1:[16, 12, 21];2,1,1:[19, 15, 72]
                key_s = s[1].split(';')
                # print('s', s)
                # print('key_s', key_s)
                for j in range(len(key_s)-1):
                    v_s = key_s[j].split(':')
                    #print('v_s', v_s)
                    node = int(i/5)
                    #print('node', node)
                    l = len(v_s[1])
                    # print('v_s_1', v_s[1], 'len_v_s_1', l)
                    # print('v_s_1-l', v_s[1][1:l-1])
                    s_delay = v_s[1][1:l-1].split(', ')
                    #print('s_delay', s_delay)
                    delay_set = []
                    for d in s_delay:
                        delay_set.append(float(d))
                    # print('d_set', delay_set)
                    # print('v_s', v_s)
                    self.node_list[node].nb_delay[v_s[0]] = delay_set

            if i % 5 == 2:
                s = line.split('-') # 2,0,1:[16, 12, 21];2,1,1:[19, 15, 72]
                key_s = s[1].split(';')
                # print('s', s)
                # print('key_s', key_s)
                for j in range(len(key_s)-1):
                    v_s = key_s[j].split(':')
                    node = int(i/5)
                    l = len(v_s[1])
                    # print('v_s_1', v_s[1], 'len_v_s_1', l)
                    # print('v_s_1-l', v_s[1][1:l - 1])
                    s_pro = v_s[1][1:l - 1].split(', ')
                    # print('s_delay', s_delay)
                    pro_set = []
                    for p in s_pro:
                        pro_set.append(float(p))
                    self.node_list[node].nb_delay_pro[v_s[0]] = pro_set

            if i % 5 == 3:
                s = line.split('-') # 2,0,1:[16, 12, 21];2,1,1:[19, 15, 72]
                key_s = s[1].split(';')
                # print('s', s)
                # print('key_s', key_s)
                # #v_s = key_s.split(',')
                # print('key_s[j]', key_s[0])
                for j in range(len(key_s)-1):
                    node = int(i/5)
                    delay = float(key_s[j])
                    # print('list(key_s[j])', list(key_s[j]))
                    # print('delay', delay)
                    self.node_list[node].all_worst_to_des.append(delay)

            if i % 5 == 4:
                s = line.split('-') # 2,0,1:[16, 12, 21];2,1,1:[19, 15, 72]
                key_s = s[1].split(';')
                # print('s', s)
                #print('key_s', key_s)
                if len(key_s) > 1:
                    node = int(i / 5)
                    self.node_list[node].worst_to_des = float(key_s[0])
        #print('adj_M', self.adj_matrix)
        #print('node_num', self.node_num)
        for i in range(self.node_num-1):
            num, child_list = self.get_nb_num(i)
            print('node', i, 'num', num)
            #node = NODE(i, num, child_list, self.device, self.episodes)  ### id, action_dim, action_list
            self.node_list[i].setChilds(num, child_list)

        for i in range(self.N):
             print('node', i)
             print('nb_delay', self.node_list[i].nb_delay)
             print('nb_delay_pro', self.node_list[i].nb_delay_pro)
             print('all_worst', self.node_list[i].all_worst_to_des)
             print('min_worst', self.node_list[i].worst_to_des)

    def exist_delay(self, delay, d_list):
        for i in range(len(d_list)):
            if delay == d_list[i]:
                return True
        return False

    def dijkstra_raw(self, edges, from_node, to_node):
        # print('from', from_node)
        # print('to', to_node)
        g = defaultdict(list)
        for l, r, c in edges:
            g[l].append((c, r))
        q, seen = [(0, from_node, ())], set()
        #print('q', q)
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

    def dijstra(self, adj, src, dst, n):
        #print('dijstra', 'src', src, 'des', dst)
        Inf = 100000
        dist = [Inf] * n
        dist[src] = 0
        book = [0] * n  # 记录已经确定的顶点
        # 每次找到起点到该点的最短途径
        u = src
        for _ in range(n - 1):  # 找n-1次
            # print('u', u)
            if u == self.node_num - 1 or u == None:
                break
            book[u] = 1  # 已经确定
            # 更新距离并记录最小距离的结点
            next_u, minVal = None, float('inf')
            for v in range(u + 1, n):  # w
                w = adj[u][v]
                #print('u', u, 'v', v, 'w', w)
                if w == Inf:  # 结点u和v之间没有边
                    continue
                #print('book_v', book[v], 'dis1',  dist[u] + w, 'dis2', dist[v])
                if not book[v] and dist[u] + w < dist[v]:  # 判断结点是否已经确定了，
                    dist[v] = dist[u] + w
                    if dist[v] < minVal:
                        next_u, minVal = v, dist[v]
            # 开始下一轮遍历
            u = next_u
        #print('short path', dist[dst])
        return dist[dst]

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

    def get_short_path(self, flow):
        #print('get_short_path')
        M = 100000
        adj_weight = np.ones((self.node_num, self.node_num)) * M
        edges = []
        next_list = []
        pre_node = self.sources[flow]
        des = self.dess[flow]
        #print('pre_node', pre_node, 'des', des)
        next_list.append(pre_node)
        while len(next_list) > 0:
            # print('node', node)
            node, next_list = self.pop_list(next_list)
            #print('node', node)
            if node == des:
                continue
            num, child_list = self.get_nb_num(node)
            #print('childs', child_list)
            for i in range(num):
                child = child_list[i]
                # print('child', child)
                key = str(child) + ',' + str(flow)
                # print('key', key)
                # print('real_d', self.node_list[node].nb_delay[key])
                worst_delay = max(self.node_list[node].nb_delay[key])
                #print('worst_delay', worst_delay)
                adj_weight[node][child] = worst_delay
                next_list = self.push(child, next_list)
            # h += 1
        # print('pass_weight', self.graph.pass_matrix)
        # print('adj_weight', adj_weight)
        for i in range(len(adj_weight)):
            for j in range(len(adj_weight[0])):
                if i != j and adj_weight[i][j] != M:
                    edges.append((i, j, adj_weight[i][j]))

        #shrot_dis = self.dijstra(adj_weight, nb, self.graph.node_num - 1, self.graph.node_num)
        short_2, path = self.dijkstra_(edges, pre_node, des)
        #print('short_1', shrot_dis)
        #print('short_2', short_2, 'path', path)
        # Adjacent = [[0, 1, 12, Inf, Inf, Inf],
        #             [Inf, 0, 9, 3, Inf, Inf],
        #             [Inf, Inf, 0, Inf, 5, Inf],
        #             [Inf, Inf, 4, 0, 13, 15],
        #             [Inf, Inf, Inf, Inf, 0, 4],
        #             [Inf, Inf, Inf, Inf, Inf, 0]]
        # Src, Dst, N = 0, 5, 6
        # shrot_dis = self.dijstra(Adjacent, Src, Dst, N)
        return short_2

    def get_link_delay(self, u, v, f, h):
        # print('K', K)
        per = random.uniform(0.5, 1)
        # per = 1
        delay_list = []
        min_b = int((0.8 * self.gamma * abs(u - v) * (1 + f) * h * 0.3) / per)
        max_b = int((1.2 * self.gamma * abs(u - v) * (1 + f) * h * 0.3) / per)
        num = 0
        while True:
            delay = int(random.uniform(min_b, max_b))
            if not self.exist_delay(delay, delay_list):
                delay_list.append(delay)
                num += 1
            if num > self.per_num - 2:
                break
        # for i in range(self.per_num-1):
        #     delay_list.append(int(random.uniform(min_b, max_b)))
        delay_list.append(max_b)
        # print('delay_list', delay_list)
        prob_list = self.generate_fix_rand(1, self.per_num)
        return max_b, delay_list, prob_list

    def not_in_pers(self, r_per, per):
        for p in per:
            if p == r_per:
                return False
        return True

    def generate_parameters(self, old_per):
        #print('in generate_parameters')
        # pdr = random.randrange(int(self.pdr_min*10), 1*10)/10
        #print('old_per', old_per)
        while True:
            per = random.randrange(int(self.per_min * 10), int((self.per_max) * 10)) / 10.0
            sum = 0
            for i in range(len(old_per)):
                if per == old_per[i]:
                    sum += 1
            if sum == 0:
                break
        #print('per', per)
        # p_min = random.randrange(int(self.per_min*10), int((self.per_min + 0.3)*10))/10
        # p_max = random.randrange(int(self.per_max * 10), int((self.per_max + 0.3)*10)) / 10
        # for i in range(self.per_num):
        #     while True:bas
        #         r_per = random.randrange(int(self.per_min * 10), int((self.per_max) * 10)) / 10.0
        #         if self.not_in_pers(r_per, per):
        #             per.append(r_per)
        #             break
        return per

    def add_parameters(self):  ### PDR, PER
        ### link parameter-per
        for i in range(len(self.adj_matrix)):
            old_per = [0.3]
            for j in range(len(self.adj_matrix)):
                if self.adj_matrix[i][j] == 1:
                    print('generate_parameters')
                    per = self.generate_parameters(old_per)
                    #print('node', i, j, 'per', per)
                    # self.G[i][j]['off_worst'] = self.max_worst - worst
                    self.G[i][j]['per'] = per
                    #print('node', i, 'n_node', j, 'per', per)
                    self.G[i][j]['sum_d'] = 0
                    self.G[i][j]['num'] = 0
                    old_per.append(per)
        ### node parameter-pdr
        for i in range(self.N):
            pdr = random.randrange(int(self.pdr_min * 10), int((self.pdr_min + 0.2) * 10)) / 10
            self.node_list[i].set_pdr(pdr)

    def get_nb_num(self, node):
        num = 0
        nb_list = []
        for i in range(self.node_num):
            # print('node',node,'i',i)
            # print(self.adj_matrix[node][i])
            if self.adj_matrix[node][i] == 1:
                num += 1
                nb_list.append(i)
        return num, nb_list

    def getNbPackets(self, node, flow, current_slot):
        packet_set = []
        num, child_list = self.get_nb_num(node)
        for i in range(num):
            packets = self.node_list[child_list[i]].getPacketsLen(flow, current_slot, self.dess)
            packet_set.append(packets)
        return packet_set

    def choose_child(self, child_list, node, visit_M):
        for i in range(len(child_list)):
            if visit_M[node][child_list[i]] == 0:
                return child_list[i]
        return -1

    def isNotexist(self, worst_to_des, lists, k):
        if k == 0:
            for i in range(len(lists)):
                if abs(worst_to_des - lists[i]) <= 10:
                    return False
        else:
            for i in range(len(lists)):
                if abs(worst_to_des - lists[i]) <= 100:
                    return False
        return True

    def opt_worst_to_des(self, node, worst_to_des, k, h, path):
        h = int(h)
        key = str(k) + ',' + str(h)
        # worst_to_des = worst_to_des * 100000
        if key in self.node_list[node].worst_to_des.keys():  # key1 in adict
            old_worst = self.node_list[node].worst_to_des[key]
            if worst_to_des < old_worst:
                self.node_list[node].worst_to_des[key] = worst_to_des
                self.node_list[node].min_path[key] = path
        else:
            self.node_list[node].worst_to_des[key] = worst_to_des
            self.node_list[node].min_path[key] = path
            self.node_list[node].all_worst_to_des[key] = []

        if self.isNotexist(worst_to_des, self.node_list[node].all_worst_to_des[key], k):
            self.node_list[node].all_worst_to_des[key].append(worst_to_des)
        # print('key', key)
        # print('worst_to_des_list', self.node_list[node].worst_to_des)

    def generate_fix_rand(self, sum, num):
        list1 = []
        for i in range(0, num - 1):
            a = random.random()
            list1.append(a)
        list1.sort()
        list1.append(sum)

        list2 = []
        for i in range(len(list1)):
            if i == 0:
                b = list1[i]
            else:
                b = list1[i] - list1[i - 1]
            list2.append(b)
        # print(list2)
        return list2

    def get_all_delay(self, node, path):
        arrival_rate = []
        pdr_dot = 1
        id = 0
        for i in range(len(path) - 1):
            if path[i] == node:
                break
            # print('i', i, 'node', path[i])
            pdr_dot *= (1 - self.node_list[int(path[i])].pdr)
            id += 1
        # print('id', id, 'node', node)
        self.node_list[int(node)].add_hop(id)

        for k in range(self.flow_num):
            arrival_rate.append(self.R[k] * pdr_dot)

        # print('path', path)
        last_node = int(path[id - 1])

        key = str(node) + ',' + str(0) + ',' + str(id)
        if key in self.node_list[last_node].nb_delay.keys():
            return

        # print('last_node', last_node, 'node', path[id])
        per = self.G[last_node][int(path[id])]['per']
        # print('per', per)
        # print('arrival_rate', arrival_rate)

        for k in range(self.flow_num):
            real_delay = []
            for p in per:
                e_x = self.packet_len / (self.trans[k] * (1 - p))
                # print('pk_len', self.packet_len, 'trans', self.trans[k])
                e_s_1 = e_x + self.ave_channel_d
                e_x_2 = (self.packet_len * self.packet_len * (1 + p)) / (
                        self.trans[k] * self.trans[k] * (1 - p) * (1 - p))
                e_s_2 = e_x_2 + self.var_channel_d
                # print('e_x', e_x, 'e_s_1', e_s_1, 'e_x_2', e_x_2, 'e_s_2', e_s_2)
                E_1 = 0
                E_2 = 0
                E_3 = 0
                for j in range(k + 1):
                    E_1 += arrival_rate[j] * e_s_2
                    if j < k:
                        E_2 += arrival_rate[j] * e_s_1
                    E_3 += arrival_rate[j] * e_s_1
                # print('E_1', E_1, 'E_2', E_2, 'E_3', E_3)
                if (1 - E_2) == 0:
                    E_2 = 0.999
                if (1 - E_3) == 0:
                    E_3 = 0.999
                # if E_3 > 1:
                #     print('k', k, 'per', p, 'e_s_1', e_s_1, 'arrival_rate', arrival_rate, 'E_3', E_3)
                e_w = E_1 / (2 * (1 - E_2) * (1 - E_3))
                real_delay.append(round(e_w * 100000))
            prob_list = self.generate_fix_rand(1, self.per_num)
            node = int(node)
            key = str(node) + ',' + str(k) + ',' + str(id)
            if key not in self.node_list[last_node].nb_delay.keys():
                self.node_list[last_node].nb_delay[key] = real_delay
                self.node_list[last_node].nb_delay_pro[key] = prob_list

    def get_maxRe(self, node, next_node, flow):
        print('node', node, 'next_node', next_node)
        Cmin = 1000 * self.packetLen / self.node_list[next_node].bandwidth
        Cmin = int(Cmin)
        per = self.G[node][next_node]['per']
        min_pro = MIN_PRO
        maxRe = math.log(min_pro/per, 1-per)
        Cmax = Cmin * int(maxRe)
        #print('maxRe', maxRe)
        return Cmax

    def get_worstDelay(self, node, next_node, flow):
        #print('get_worstDelay')
        # print('node', node,'next_node', next_node, 'flow', flow)
        W = 0
        bl = True
        while bl:
            packets = 0
            for k in range(flow):
                key = str(k)
                if key in self.node_list[next_node].arrival_rate.keys():
                    T = self.node_list[next_node].arrival_rate[key]
                    #print('key', key, 'T', T, 'W', W)
                    if W < T:
                        packets += 1
                    else:
                        packets += math.ceil(W / T)
            #     print('packets', packets)
            # print('packets', packets)
            packets += 1
            if packets > 3:
                packets = 3
            Cmax = self.get_maxRe(node, next_node, flow)
            new_W = packets * Cmax
            if new_W == W:
                #print('W', W, 'Cmax', Cmax)
                return W, Cmax
            W = new_W
            #print('W', W, 'Cmax', Cmax)

    def get_delayBasedNC(self, node, next_node, flow, pass_weight): ### h is the hop of next_node
        #print('node', node, 'next_node', next_node, 'flow', flow)
        #print('trans', self.trans)
        if flow == 0:
            worst_d = self.get_maxRe(node, next_node, flow)
            per = self.G[node][next_node]['per']
            Cmin = int((1000 * self.packetLen / (self.node_list[next_node].bandwidth))/per)
            Cmin = int(Cmin)
            #print('worst_d', worst_d, 'cmin', Cmin)
            min_pro = MIN_PRO
            maxRe = int(math.log(min_pro / per, 1 - per))
            real_d = []
            real_pro = []
            for l in range(1, maxRe + 1):
                d = Cmin * l + np.random.randint(5, 15)
                pro = math.pow((1 - per), l - 1) * per
                real_d.append(d)
                real_pro.append(pro)
            key = str(next_node) + str(',') + str(flow)  # nb + k
            self.node_list[node].nb_delay[key] = real_d
            #print('node', node, 'real_d', real_d, self.node_list[node].nb_delay[key])
            self.node_list[node].nb_delay_pro[key] = real_pro
            return worst_d + 10
        else:
            if pass_weight[node][next_node] > 1:
                #print('1111')
                worst_d, Cmax = self.get_worstDelay(node, next_node, flow)
                packets = worst_d / Cmax
                real_d = []
                real_pro = []
                Cmin = int((1000 * self.packetLen / (self.node_list[next_node].bandwidth))/per)
                Cmin = int(Cmin)
                per = self.G[node][next_node]['per']
                min_pro = MIN_PRO
                maxRe = int(math.log(min_pro / per, 1 - per))
                #print('maxRe', maxRe)
                for l in range(1, maxRe + 1):
                    d = Cmin * l * packets + 20
                    #print('d', d)
                    pro = math.pow((1 - per), l - 1) * per
                    real_d.append(d)
                    real_pro.append(pro)
                key = str(next_node) + str(',') + str(flow)  # nb + k
                self.node_list[node].nb_delay[key] = real_d
                self.node_list[node].nb_delay_pro[key] = real_pro
            else:
                #print('flow', flow)
                #worst_d = self.get_maxRe(node, next_node, flow)
                key = str(next_node) + str(',') + str(0)  # nb + k
                if key in self.node_list[node].nb_delay.keys():
                    key1 = str(next_node) + str(',') + str(flow)
                    # print('key', key, 'key1', key1)
                    self.node_list[node].nb_delay[key1] = self.node_list[node].nb_delay[key]
                    self.node_list[node].nb_delay_pro[key1] = self.node_list[node].nb_delay_pro[key]
                    worst_d = max(self.node_list[node].nb_delay[key1])
                else:
                    worst_d = self.get_maxRe(node, next_node, flow)
                    per = self.G[node][next_node]['per']
                    Cmin = int((1000 * self.packetLen / (self.node_list[next_node].bandwidth)) / per)
                    Cmin = int(Cmin)
                    # print('worst_d', worst_d, 'cmin', Cmin)
                    min_pro = MIN_PRO
                    maxRe = int(math.log(min_pro / per, 1 - per))
                    real_d = []
                    real_pro = []
                    for l in range(1, maxRe + 1):
                        d = Cmin * l + np.random.randint(5, 15)
                        pro = math.pow((1 - per), l - 1) * per
                        real_d.append(d)
                        real_pro.append(pro)
                    key1 = str(next_node) + str(',') + str(flow)  # nb + k
                    self.node_list[node].nb_delay[key1] = real_d
                    # print('node', node, 'real_d', real_d, self.node_list[node].nb_delay[key])
                    self.node_list[node].nb_delay_pro[key1] = real_pro
            return worst_d + 10

    def update_arrivalInterval(self, path, flow):
        for i in range(len(path)-1):
            key = str(flow)
            Cmin = 1000 * self.packetLen / self.trans[flow]
            node = path[i]
            next_node = path[i+1]
            if next_node == -1:
                return
            #per = self.G[node][next_node]['per']
            Cmax = self.get_maxRe(node, next_node, flow)
            key1 = str(next_node) + str(',') + str(flow)  # nb + k
            W = self.node_list[node].nb_worst_delay[key1]
            # print('W', W, 'cmin', Cmin)
            # print('node', node, 'arrival', self.node_list[node].arrival_rate[key])
            T = self.node_list[node].arrival_rate[key]
            if W > T:
                self.node_list[next_node].arrival_rate[key] = Cmin
            else:
                self.node_list[next_node].arrival_rate[key] = T + Cmin - W
        #     print('key', key)
        #     print('node', next_node, 'arrival_rate', self.node_list[next_node].arrival_rate[key])
        # print('finish_update')


    def get_worst_delay(self, arrival_rate, e_s_1, exp_d):
        p = pow(10,-6)
        #print('p', p)
        worst_d = []
        for k in range(self.flow_num):
            A = 0
            for j in range(k + 1):
                A = arrival_rate[j] * e_s_1[j]
            d = -1 * (exp_d[k] / A) * math.log(p / A)
            worst_d.append(int(d))
        return worst_d


    def get_delay(self, node, path):
        exp_d = []
        arrival_rate = []
        pdr_dot = 1
        id = 0
        for i in range(len(path) - 1):
            if path[i] == node:
                break
            # print('i', i, 'node', path[i])
            pdr_dot *= (1 - self.node_list[int(path[i])].pdr)
            id += 1

        for k in range(self.flow_num):
            arrival_rate.append(self.R[k] * pdr_dot)

        # print('path', path)
        last_node = int(path[id - 1])
        # print('last_node', last_node, 'node', path[id])
        per = self.G[last_node][int(path[id])]['per']
        #print('per', per)
        # print('arrival_rate', arrival_rate)
        E_S = []

        for k in range(self.flow_num):
            e_x = self.packet_len / (self.trans[k] * (1 - per))
            # print('pk_len', self.packet_len, 'trans', self.trans[k])
            e_s_1 = e_x + self.ave_channel_d
            E_S.append(e_s_1)
            e_x_2 = (self.packet_len * self.packet_len * (1 + per)) / (
                        self.trans[k] * self.trans[k] * (1 - per) * (1 - per))
            e_s_2 = e_x_2 + self.var_channel_d
            # print('e_x', e_x, 'e_s_1', e_s_1, 'e_x_2', e_x_2, 'e_s_2', e_s_2)
            E_1 = 0
            E_2 = 0
            E_3 = 0
            for j in range(k + 1):
                E_1 += arrival_rate[j] * e_s_2
                if j < k:
                    E_2 += arrival_rate[j] * e_s_1
                E_3 += arrival_rate[j] * e_s_1
                # print('E_33', E_3)
            #     print('E_3', E_3)
            # print('k', k, 'arrival_rate', arrival_rate, 'e_s_1', e_s_1)
            # print('E_1', E_1, 'E_2', E_2, 'E_3', E_3)
            if (1 - E_2) == 0:
                E_2 = 0.999
            if (1 - E_3) == 0:
                E_3 = 0.999
            # print('E_2', E_2, '1-e_2', 1 - E_2, 'E_3', E_3, '1-e_3', 1 - E_3)
            # if (1 - E_2) == 0 or (1 - E_3) == 0:
            #     print('000')
            deta = 2 * (1.0 - E_2) * (1.0 - E_3)
            # print('deta', deta)
            e_w = E_1 / deta
            # print('e_w', e_w)
            exp_d.append(round(e_w * 100000))
        # print('last_node', last_node, 'node', path[id])
        # print('exp_d', exp_d)

        worst_d = self.get_worst_delay(arrival_rate, E_S, exp_d)
        return worst_d

    def exist_hop(self, node, parent_hop):
        for h in self.node_list[node].hop:
            if h == parent_hop:
                return True
        return False

    def get_parent_node_hop(self, node, parent_hop):
        # print('node', node, 'parent_hop', parent_hop)
        parents = []
        num = 0
        for i in range(self.node_num):
            if self.adj_matrix[i][node] == 1 and self.exist_hop(i, parent_hop):
                parents.append(i)
                num += 1
        return num, parents

    def calculate_worst(self, q, node_h, flow, pass_weight):
        #print('calculate_worst')
        # queue = qu.LifoQueue()
        queue = LifoQueue()
        queue.queue = copy.deepcopy(q.queue)
        path_len = queue.qsize()
        #delay = np.zeros((path_len - 1, self.flow_num))
        path = np.zeros(path_len)
        for i in range(path_len):
            node = queue.get()
            path[path_len - 1 - i] = node
            self.node_list[int(node)].add_hop(path_len - 1 - i)
        #print('path', path)
        # for i in range(1, path_len): ### 计算每条链路的时延, 从前往后
        #     pre_node = path[i-1]
        #     node = path[i]
        #     for j in range(self.flow_num):
        #         key = str(int(node)) + ',' + str(j) + ',' + str(i)
        #         if key not in self.node_list[int(pre_node)].nb_delay.keys():
        #             max_d, real_d, real_pro = self.get_link_delay(pre_node, node, j, i)
        #             self.node_list[int(pre_node)].nb_delay[key] = real_d
        #             self.node_list[int(pre_node)].nb_delay_pro[key] = real_pro
        #         else:
        #             max_d = max(self.node_list[int(pre_node)].nb_delay[key])
        #         delay[i-1][j] = max_d
        #         # print('key', key)
        #         # print('pre_node', pre_node, 'node', node, 'real_d', real_d, 'real_pro', real_pro)
        #     #print('the link delay', delay[i - 1])

        for i in range(1, path_len):  ### 计算每条链路的时延, 从前往后
            pre_node = int(path[i - 1])
            node = int(path[i])
            #print('pre_node', pre_node, 'node', node)
            #worst_d = self.get_delay(node, path)
            worst_d = self.get_delayBasedNC(pre_node, node, flow, pass_weight)
            key = str(node) + str(',') + str(flow)   # nb + k
            # if key in self.node_list[pre_node].nb_delay.keys():
            #     old_d = max(self.node_list[pre_node].nb_delay[key])
            #     # print('old_worst', old_d)
            #     # print('new_worst', worst_d)
            self.node_list[pre_node].nb_worst_delay[key] = worst_d
        #print('finish_path')

            # self.get_all_delay(node, path)
            # for j in range(len(worst_d)):
            #     delay[i - 1][j] = worst_d[j]
            # print('the link worst delay', delay[i - 1])

        # if path[path_len - 1] < self.node_num - 1:
        #     print('path--', path)
        #     return

        # for k in range(self.flow_num):
        #     for i in range(len(path) - 1):
        #         worst_to_des = 0
        #         for j in range(i, path_len - 1):
        #             worst_to_des += delay[j][k]
        #         n = int(path[i])
        #         # print('node', n, 'worst_to_des', worst_to_des*100000)
        #         self.opt_worst_to_des(n, worst_to_des, k, node_h[int(path[i])], path)

    def reset_visit(self, child_list, visit_M):
        l = len(visit_M)
        for c in range(len(child_list)):
            pre_node = child_list[c]
            # print('update_for_pre_node', pre_node)
            for i in range(l):
                visit_M[pre_node][i] = 0

    def calculate_min_worst(self, flow, pass_weight):
        #print('pass_weight', pass_weight)
        # q = qu.LifoQueue()
        q = LifoQueue()
        visit_M = np.zeros((self.N, self.N))
        node_h = np.zeros(self.N)
        source = self.node_list[self.sources[flow]].id
        node = source
        q.put(node)
        bl = True
        re_path = 0
        pre_node = -1
        des = self.node_list[self.dess[flow]].id
        #print('des', des)
        while bl:
            # print('queue', q.queue)
            # print('node', node)
            # print('pre_node', pre_node)
            num, child_list = self.get_nb_num(node)
            #print('node', node, 'num', num, 'child_list', child_list)
            if num > 0:
                next_node = self.choose_child(child_list, node, visit_M)
                #print('next_node', next_node)
                if node == source and next_node == -1:
                    bl = False
                    break
                if next_node >= 0:  ### 有未被访问的邻居
                    if re_path == 1:
                        q.put(node)
                        re_path = 0
                    node_h[next_node] = node_h[node] + 1
                    visit_M[node][next_node] = 1
                    q.put(next_node)
                    node = next_node
                    if next_node == des:
                        self.calculate_worst(q, node_h, flow, pass_weight)
                        temp = q.get()
                        # pre_node = node
                        node = q.get()
                        re_path = 1
                else:  ### 邻居节点均被访问
                    # print('update_for_pre_node', pre_node)
                    num, child_list = self.get_nb_num(node)
                    self.reset_visit(child_list, visit_M)
                    # pre_node = node
                    node = q.get()
                    re_path = 1
            else:  ### no next_node
                # pre_node = node
                # print('no next_childs')
                self.calculate_worst(q, node_h, flow, pass_weight)
                temp = q.get()
                node = q.get()
                re_path = 1

        # for i in range(self.node_num):
        #     self.node_list[i].D_list.sort()  ###升序






