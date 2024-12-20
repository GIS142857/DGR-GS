import math
import random
import numpy as np
from collections import defaultdict
import queue
import copy

class Table:
    def __init__(self, graph):
        self.inf = 1000000
        self.min_d = self.inf
        self.graph = graph
        self.dmax = 20
        self.state_count_dictionary = defaultdict(dict)
        self.state_count_dictionary[self.graph.node_num - 1]['worst'] = [0]
        self.state_count_dictionary[self.graph.node_num - 1]['policy'] = []
        self.state_count_dictionary[self.graph.node_num - 1]['expect'] = [0]
        self.state_count_dictionary[self.graph.node_num - 1]['flow'] = [0]
        self.state_count_dictionary[self.graph.node_num - 1]['hop'] = [0]
        for i in range(self.graph.node_num - 1):
            self.state_count_dictionary[i]['worst'] = []
            self.state_count_dictionary[i]['policy'] = []
            self.state_count_dictionary[i]['expect'] = []
            self.state_count_dictionary[i]['flow'] = []
            self.state_count_dictionary[i]['hop'] = []
        self.Q = queue.Queue(maxsize=10000)
        for i in range(self.graph.node_num):
            if self.graph.G.has_node(self.graph.node_num - 1 - i):
                self.Q.put(self.graph.node_num - 1 - i)
        # print('Q',self.Q.queue)
        # self.q = np.ones((self.graph.node_num,self.dmax,self.graph.node_num)) * self.inf
        self.q = defaultdict(dict)  # key: node + D + child + k + child_hop

    def not_in_queue(self, node, q_list):
        #print('q_list', q_list)
        for i in range(len(q_list)):
            #print('node', q_list[i])
            if node == q_list[i]:
                return False
        return True

    def exist(self, node_hop, key, hop):
        if key in node_hop.keys():
            for i in range(len(node_hop[key])):
                if node_hop[key][i] == hop:
                    return True
            node_hop[key].append(hop)
            return True
        return False

    def get_hop(self, node_hops):
        hop = node_hops[0]
        hop_set = []
        for i in range(1, len(node_hops)):
            hop_set.append(node_hops[i])
        return hop, hop_set

    def generate_tables(self):
        for f in range(self.graph.flow_num):
            #print('flow', f)
            hops = self.graph.node_list[self.graph.node_num - 1].hop
            flag = 0
            for j in range(len(hops)):
                #print('self_Q', self.Q.queue)
                Qu = queue.Queue(maxsize=10000)
                Qu.queue = copy.deepcopy(self.Q.queue)
                node = Qu.get()
                parent_hop = hops[j] - 1
                #print('parent_hop', parent_hop)
                node_visit = defaultdict(dict)
                node_hop = defaultdict(dict)
                key = str(self.graph.node_num - 1)
                node_visit[key] = 1
                node_hop[key] = list()
                node_hop[key].append(hops[j])
                #print('node', node, 'parent_num', nb_num, 'parent_list', parent_list)
                # print('node_visit', node_visit)
                # print('node_hop', node_hop)
                while not Qu.empty() or flag == 0:
                    flag = 1
                    nb_num, parent_list = self.graph.get_parent_node_hop(node, parent_hop)
                    if nb_num == 0:
                        break

                    for i in range(nb_num):
                        #print('i', i)
                        # print('111')
                        parent = parent_list[i]
                        key = str(parent)
                        node_visit[key] = 1
                        bl = self.exist(node_hop, key, parent_hop)
                        if not bl:
                            node_hop[key] = []
                            node_hop[key].append(parent_hop)
                        bl = self.relax(parent_list[i], node, f, parent_hop + 1)
                        if bl:
                            # print('222')
                            # print('parent_list', parent_list[i])
                            #print('queue', Qu.queue)
                            if self.not_in_queue(parent_list[i], Qu.queue):
                                Qu.put(parent_list[i])
                    #             print('Q_len', len(self.Q.queue))
                    #             print('333')
                    # print('b_len', len(Qu.queue), 'queue', Qu.queue)
                    # print('node', node, 'parent_num', nb_num, 'parent_list', parent_list)
                    # print('node_visit', node_visit)
                    # print('node_hop', node_hop)
                    while True:
                        node = Qu.get()
                        if node_visit[str(node)] == 1:
                            #print('seleted_node', node)
                            parent_hop, hop_set = self.get_hop(node_hop[str(node)])
                            parent_hop -= 1
                            node_hop[str(node)] = hop_set
                            if len(node_hop[str(node)]) == 0:
                                node_visit[str(node)] = 0
                            break

                #     print('a_len', len(Qu.queue))
                #     print('change table node', node)
                #     print('parent_hop', parent_hop)
                # print('after_generate_graph')


    def get_bound_expected(self, d, node, k, h):
        # print('get_bound_node', node, 'hop', h, 'k', k)
        # print('node_v',node,'d',d,'worst_list',self.state_count_dictionary[node]['worst'])
        min_delay = self.inf
        for i in range(len(self.state_count_dictionary[node]['worst'])):
            if node == self.graph.node_num - 1:
                if int(d * 100) >= int(100 * self.state_count_dictionary[node]['worst'][i]) and \
                        self.state_count_dictionary[node]['expect'][i] < min_delay:
                    # print('d',d,'worst_i',self.state_count_dictionary[node]['worst'][i])
                    min_delay = self.state_count_dictionary[node]['expect'][i]

            else:
                # print('d', d, 'worst', self.state_count_dictionary[node]['worst'], 'expect',
                #       self.state_count_dictionary[node]['expect'][i], 'min_delay', min_delay, 'flow', self.state_count_dictionary[node]['flow'][i], 'k',k,'hop', self.state_count_dictionary[node]['hop'][i]-1, 'h', h)
                #h -= 1
                node_flow = self.state_count_dictionary[node]['flow'][i]
                node_hop = self.state_count_dictionary[node]['hop'][i]-1
                # print('node_flow', node_flow, 'k',k)
                # print('node_hop', node_hop, 'h', h)
                if int(d * 100) >= int(100 * self.state_count_dictionary[node]['worst'][i]) and self.state_count_dictionary[node]['expect'][i] < min_delay and (node_flow == k) and (node_hop == h):
                    #print('111111')
                    min_delay = self.state_count_dictionary[node]['expect'][i]
        return min_delay

    def get_expected(self, d, real_d_list, real_prob_list, node_v, k, h):
        sum = 0
        for i in range(len(real_d_list)):
            #print('d', d, 'real_d', real_d_list[i])
            e = self.get_bound_expected(d - real_d_list[i], node_v, k, h)
            #print('eeee',e)
            if e == self.inf:
                return sum, False

            sum += real_prob_list[i] * (real_d_list[i] + e)
        #print('sum', sum)
        return sum, True

    def get_worst_hop(self, node_v, h):
        #print()
        l = len(self.state_count_dictionary[node_v]['worst'])
        worst_set = []
        for i in range(l):
            if self.state_count_dictionary[node_v]['hop'][i] == h+1:
                worst_set.append(self.state_count_dictionary[node_v]['worst'][i])
        return worst_set


    def relax(self, node_u, node_v, k, h):  # h is the hop of node_v
        #print('relax', node_u, node_v, 'h', h)
        key = str(node_v) + str(',') + str(k) + ',' + str(h)
        #print('key', key)
        real_d_list = self.graph.node_list[node_u].nb_delay[key] ### nb + k + h
        real_prob_list = self.graph.node_list[node_u].nb_delay_pro[key]
        #print('real_d', real_d_list, 'real_prob', real_prob_list)
        change_num = 0
        #print('child_to_worst', self.state_count_dictionary[node_v]['worst'])
        worst_set = [0]
        if node_v < self.graph.node_num -1:
            worst_set = self.get_worst_hop(node_v, h)
        #print('worst_set', worst_set)
        for d0 in worst_set:
            for x0 in real_d_list:
                #print('d0', d0, 'x0', x0)
                d = d0 + x0
                #print('dd',d)
                e, bll = self.get_expected(d, real_d_list, real_prob_list, node_v, k, h)
                #print('bl', bll)
                if bll:
                    #print('dd', d, 'ee', e)
                    bl = self.merge(d, e, node_u, node_v, k, h)
                    #print('merge_bl', bl)
                    if bl:
                        change_num += 1
        if change_num > 0:
            return True
        else:
            return False

    def check_tuple(self, d, node, f, h):
        # print('node',node,'worst',self.state_count_dictionary[node]['worst'],'d',d)
        for i in range(len(self.state_count_dictionary[node]['worst'])):
            if self.state_count_dictionary[node]['worst'][i] == d and self.state_count_dictionary[node]['flow'][i] == f and self.state_count_dictionary[node]['hop'][i] == h:
                pi = self.state_count_dictionary[node]['policy'][i]
                e = self.state_count_dictionary[node]['expect'][i]
                flow = self.state_count_dictionary[node]['flow'][i]
                hop = self.state_count_dictionary[node]['hop'][i]
                return True, [d, pi, e, flow, hop], i
        return False, [], -1

    def check_index(self, index, list_index):
        for i in range(len(list_index)):
            if index == list_index[i]:
                return True
        return False

    def remove_tuple(self, list_index, node_u):
        temp_w_list = []
        temp_p_list = []
        temp_e_list = []
        temp_f_list = []
        temp_h_list = []
        k = 0
        # print('remove')
        # print('worst_list',self.state_count_dictionary[node_u]['worst'])
        for i in range(len(self.state_count_dictionary[node_u]['worst'])):
            if self.check_index(i, list_index):
                continue
            else:
                temp_w_list.append(self.state_count_dictionary[node_u]['worst'][i])
                temp_p_list.append(self.state_count_dictionary[node_u]['policy'][i])
                temp_e_list.append(self.state_count_dictionary[node_u]['expect'][i])
                temp_f_list.append(self.state_count_dictionary[node_u]['flow'][i])
                temp_h_list.append(self.state_count_dictionary[node_u]['hop'][i])
        self.state_count_dictionary[node_u]['worst'] = temp_w_list
        self.state_count_dictionary[node_u]['policy'] = temp_p_list
        self.state_count_dictionary[node_u]['expect'] = temp_e_list
        self.state_count_dictionary[node_u]['flow'] = temp_f_list
        self.state_count_dictionary[node_u]['hop'] = temp_h_list
        # print('after remove',self.state_count_dictionary[node_u]['worst'],self.state_count_dictionary[node_u]['policy'],self.state_count_dictionary[node_u]['expect'])

    def insert(self, d, node_v, e, node_u, f, h):
        # print('insert')
        # print('d', d, 'policy', node_v, 'e', e, 'f', f, 'h', h)
        #l = len(self.state_count_dictionary[node_u]['worst'])
        #if l == 0:
        self.state_count_dictionary[node_u]['worst'].append(d)
        self.state_count_dictionary[node_u]['policy'].append(node_v)
        self.state_count_dictionary[node_u]['expect'].append(e)
        self.state_count_dictionary[node_u]['flow'].append(f)
        self.state_count_dictionary[node_u]['hop'].append(h)
            #return l
        # else:
        #     k = 0
        #     for i in range(l):
        #         k += 1
        #         if e < self.state_count_dictionary[node_u]['expect'][i]:
        #             self.state_count_dictionary[node_u]['worst'].append(
        #                 self.state_count_dictionary[node_u]['worst'][l - 1])
        #             self.state_count_dictionary[node_u]['policy'].append(
        #                 self.state_count_dictionary[node_u]['policy'][l - 1])
        #             self.state_count_dictionary[node_u]['expect'].append(
        #                 self.state_count_dictionary[node_u]['expect'][l - 1])
        #             self.state_count_dictionary[node_u]['flow'].append(
        #                 self.state_count_dictionary[node_u]['flow'][l - 1])
        #             self.state_count_dictionary[node_u]['hop'].append(
        #                 self.state_count_dictionary[node_u]['hop'][l - 1])
        #             j = l - 1
        #             while j >= i:
        #                 self.state_count_dictionary[node_u]['worst'][j + 1] = \
        #                     self.state_count_dictionary[node_u]['worst'][j]
        #                 self.state_count_dictionary[node_u]['policy'][j + 1] = \
        #                     self.state_count_dictionary[node_u]['policy'][j]
        #                 self.state_count_dictionary[node_u]['expect'][j + 1] = \
        #                     self.state_count_dictionary[node_u]['expect'][j]
        #                 self.state_count_dictionary[node_u]['flow'][j + 1] = \
        #                     self.state_count_dictionary[node_u]['flow'][j]
        #                 self.state_count_dictionary[node_u]['hop'][j + 1] = \
        #                     self.state_count_dictionary[node_u]['hop'][j]
        #                 j -= 1
        #             self.state_count_dictionary[node_u]['worst'][i] = d
        #             self.state_count_dictionary[node_u]['policy'][i] = node_v
        #             self.state_count_dictionary[node_u]['expect'][i] = e
        #             self.state_count_dictionary[node_u]['flow'][i] = f
        #             self.state_count_dictionary[node_u]['hop'][i] = h
        #             break
        #     if k == l:
        #         self.state_count_dictionary[node_u]['worst'].append(d)
        #         self.state_count_dictionary[node_u]['policy'].append(node_v)
        #         self.state_count_dictionary[node_u]['expect'].append(e)
        #         self.state_count_dictionary[node_u]['flow'].append(f)
        #         self.state_count_dictionary[node_u]['hop'].append(h)
        #     # print('node inster', node_u, 'policy', self.state_count_dictionary[node_u]['worst'], self.state_count_dictionary[node_u]['policy'],self.state_count_dictionary[node_u]['expect'])
        #     return

    def merge(self, d, e, node_u, node_v, f, h): ### h is the hop of node_u
        #print('merge_node', node_u)
        bl, tuple, index = self.check_tuple(d, node_u, f, h)
        #print('check_result', 'tuple', tuple, 'index', index, 'bl', bl)
        change = False
        #print('policy1', self.state_count_dictionary[node_u])
        if bl:
            # print('e', e, 'old_e', tuple[2])
            if e < tuple[2]:
                self.state_count_dictionary[node_u]['worst'][index] = d  ## replace
                self.state_count_dictionary[node_u]['policy'][index] = node_v
                self.state_count_dictionary[node_u]['expect'][index] = e
                change = True
        else:
            self.insert(d, node_v, e, node_u, f, h)
            # self.state_count_dictionary[node_u]['worst'].append(d)
            # self.state_count_dictionary[node_u]['policy'].append(node_v)
            # self.state_count_dictionary[node_u]['expect'].append(e)
            change = True
        # print('node',node_u,'worst_list',self.state_count_dictionary[node_u]['worst'],'policy',self.state_count_dictionary[node_u]['policy'],'expect',self.state_count_dictionary[node_u]['expect'])
        ### remove all 3-tules from table[u] with d'>d and e'>=e
        #print('policy2', self.state_count_dictionary[node_u])
        remove_index = []
        for i in range(len(self.state_count_dictionary[node_u]['worst'])-1):
            #print('d', d, 'worst', self.state_count_dictionary[node_u]['worst'][i])
            if self.state_count_dictionary[node_u]['worst'][i] > d and self.state_count_dictionary[node_u]['expect'][i] >= e and self.state_count_dictionary[node_u]['flow'][i] == f and self.state_count_dictionary[node_u]['hop'][i] == h:
                remove_index.append(i)
            if self.state_count_dictionary[node_u]['worst'][i] <= d and self.state_count_dictionary[node_u]['expect'][i] <= e and self.state_count_dictionary[node_u]['flow'][i] == f and self.state_count_dictionary[node_u]['hop'][i] == h:
                l = len(self.state_count_dictionary[node_u]['worst'])
                remove_index.append(l-1)
        if remove_index:
            self.remove_tuple(remove_index, node_u)
        #print('policy3', self.state_count_dictionary[node_u])
        return change

    def get_policy(self, D, node, flow, hop):
        opt_child = -1
        min_delay = 100000
        #print('get_policy_node', node, 'D', D, 'flow', flow, 'h', hop)
        for i in range(len(self.state_count_dictionary[node]['worst'])):
            if self.state_count_dictionary[node]['worst'][i] <= D and self.state_count_dictionary[node]['expect'][i] < min_delay and self.state_count_dictionary[node]['flow'][i] == flow and self.state_count_dictionary[node]['hop'][i] == hop:
                min_delay = self.state_count_dictionary[node]['expect'][i]
                opt_child = self.state_count_dictionary[node]['policy'][i]
        return opt_child, min_delay

    def get_avg_d(self, u, v, h, f):
        key = str(v) + ',' + str(f) + ',' + str(h+1)
        real_d = self.graph.node_list[u].nb_delay[key]
        real_pro = self.graph.node_list[u].nb_delay_pro[key]
        avg_d = 0
        for i in range(len(real_d)):
            avg_d += real_d[i] * real_pro[i]
        return avg_d

    def get_goal_result(self, D, flow):
        trip = []
        trip.append(0)
        id = 0
        h = 1
        opt_min_delay = 100000
        while id < self.graph.node_num - 1:
            #print('id', id, 'D', D)
            child, min_delay = self.get_policy(D, id, flow, h)
            #print('child', child, 'min_delay', min_delay)
            key = str(child) + ',' + str(flow) + ',' + str(h)
            if key not in self.graph.node_list[id].nb_delay.keys():
                #print('key', key)
                return None, 10000
            real_d = self.graph.node_list[id].nb_delay[key]
            # print('key', key)
            # print('real_d', real_d)
            # if len(real_d) == 0:
            #     return None, 10000
            reward = real_d[0]
            #print('reward', reward)
            D -= reward
            if h == 1:
                opt_min_delay = min_delay
            # print('policy', table.state_count_dictionary[id]['policy'])
            # print('id', id)
            # print('child', child)
            trip.append(child)
            # print('node', id, 'worst', self.state_count_dictionary[id]['worst'], 'policy',
            #       self.state_count_dictionary[id]['policy'], 'expect', self.state_count_dictionary[id]['expect'])
            id = child
            h += 1
        ave_sum = 0
        for i in range(len(trip) - 1):
            # print('node', trip[i], 'n_node', trip[i+1], 'avg_d', table.graph.G[trip[i]][trip[i+1]]['ave_d'])
            # ave_sum += table.graph.G[trip[i]][trip[i+1]]['ave_d']
            ave_sum += self.get_avg_d(trip[i], trip[i + 1], i, flow)
        return trip, ave_sum



