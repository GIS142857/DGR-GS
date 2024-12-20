#coding:utf-8

from Random_graph import randomG
from learnforq import learnAgent
import os
import torch
from rtss2020 import *
import copy
from getOptResult import Table
from rtns import RTNS

# 设备配置
#torch.cuda.set_device(1) # 这句用来设置pytorch在哪块GPU上运行
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('device', device)
print(torch.cuda.is_available())

### params
# episodes
# eplision
# EpsilonGreedy
# log

episodes = 20
if __name__ == '__main__':
    R = [16, 8, 3, 80, 60]
    Trans = [500000, 500000, 500000, 20 * 1024, 20 * 1024]
    N = 20
    p = 0.8
    per_num = 1
    flow_num = 3
    sources = np.zeros(flow_num, dtype=int)
    des = 18 * np.ones(flow_num, dtype=int)
    packetLen = 8000
    packets = -1 * np.ones(N)
    for i in range(flow_num):
        packets[sources[i]] = i

    rand_graph = randomG(N, p, flow_num, R, per_num, device, episodes, sources, des, packetLen)  ###N, p, flow_num, R, Trans
    #M, g = rand_graph.generate_graph(packets, sources)
    rand_graph.load_graph(packets, sources)
    #print('g', rand_graph.adj_matrix)
    #rand_graph.add_parameters()
    #rand_graph.load_graph(packets, sources)
    g2 = copy.deepcopy(rand_graph)
    g3 = copy.deepcopy(rand_graph)
    # for i in range(flow_num):
    #     rand_graph.calculate_min_worst(0)

    # print('source_to_des_delay', rand_graph.node_list[0].all_worst_to_des)
    # #print('len_d_list', len(rand_graph.node_list[0].all_worst_to_des[0]), len(rand_graph.node_list[0].all_worst_to_des[1]))
    # print('min_source_to_des_delay', rand_graph.node_list[0].worst_to_des)
    # print('min_path', rand_graph.node_list[0].min_path)

    # for k in range(flow_num):
    #     i = 0
    #     for d in rand_graph.node_list[sources[k]].all_worst_to_des:
    #         if k < 2:
    #             rand_graph.node_list[sources[k]].all_worst_to_des[i] += 300
    #         else:
    #             rand_graph.node_list[sources[k]].all_worst_to_des[i] += 600
    #         i += 1


    tau_N = 50

    # source1 = 0
    # source2 = 1
    # sources = []
    # sources.append(source1)
    # sources.append(source2)
    # dess = [N - 1, N - 1]
    arrives_at = [6, 30, 10] ### ms[3,6,10]
    d_set = [1000, 2500, 5000, 1000, 2500, 5000, 1000, 2500, 5000]  #[800, 1000, 1500]
    slots = [3, 2, 1, 3, 2, 1, 3, 2, 1]
    dmax_set = d_set * 2

    T = 100
    epoch = 100

    learn_agent = learnAgent(flow_num, rand_graph, device, episodes, tau_N, sources, des, slots, d_set, dmax_set)


    for k in range(flow_num):
        #learn_agent.graph.calculate_min_worst(k, learn_agent.pass_weight)
        # f = open("graph.txt", "w")
        # for i in range(N):
        #     s = ''
        #     for j in range(i, N):
        #         if rand_graph.adj_matrix[i][j] == 1:
        #             s += str(j) + ' '
        #     f.write(s + '\n')
        # for i in range(N):
        #     f.write('\nnode' + str(i))
        #     f.write('\nnode_nb_delay-')
        #     # f.write('\n')
        #     for key, v in learn_agent.graph.node_list[i].nb_delay.items():
        #         f.write(key + ':')
        #         f.write(str(v) + ';')
        #     f.write('\nnode_nb_delay_pro-')
        #     for key, v in learn_agent.graph.node_list[i].nb_delay_pro.items():
        #         f.write(key + ':')
        #         f.write(str(v) + ';')
        #     f.write('\nall_worst-')
        #     for v in learn_agent.graph.node_list[i].all_worst_to_des:
        #         # f.write(key + ':')
        #         f.write(str(v) + ';')
        #     f.write('\nmin_worst-')
        # 
        #     f.write(str(rand_graph.node_list[i].worst_to_des) + ';')
        # f.close()
        #print('get_min_dis')
        min_worst_to_des = learn_agent.graph.get_short_path(k)
        print('min_worst_to_des', min_worst_to_des)
        #deadline = min_worst_to_des + 50
        # d_set.append(deadline)
        # dmax = 2 * deadline
        # dmax_set.append(dmax)
        learn_agent.learnPacket(T, k)
        # opt_path = learn_agent.learning(d_set[k], dmax_set[k], k)
        # print('opt_path', opt_path)
        # learn_agent.graph.update_arrivalInterval(opt_path, k)


    print('d_set', d_set)

    #my_cost, my_trip = learn_agent.get_opt_path(d_set, dmax_set)
    # print('my_result', ' cost', my_cost, 'trip', my_trip)

    pass_weight = np.ones((N, N))
    for k in range(flow_num):
        g2.calculate_min_worst(k, pass_weight)
    # for i in range(N):
    #     print('node', i)
    #     print('\nnode_nb_delay-')
    #     # f.write('\n')
    #     for key, v in g2.node_list[i].nb_delay.items():
    #         print(key + ':')
    #         print(str(v) + ';')
    #     print('\nnode_nb_delay_pro-')
    #     for key, v in g2.node_list[i].nb_delay_pro.items():
    #         print(key + ':')
    #         print(str(v) + ';')
    #     print('\nall_worst-')
    #     for v in g2.node_list[i].all_worst_to_des:
    #         # f.write(key + ':')
    #         print(str(v) + ';')


    #
    # get_best_path_rtss(g2, d_set, flow_num, sources, des)
    # #
    # pass_weight = np.ones((N, N))
    # for k in range(flow_num):
    #     g3.calculate_min_worst(k, pass_weight)
    # # for i in range(N):
    #     print('node', i)
    #     print('\nnode_nb_delay-')
    #     # f.write('\n')
    #     for key, v in g3.node_list[i].nb_delay.items():
    #         print(key + ':')
    #         print(str(v) + ';')
    #     print('\nnode_nb_delay_pro-')
    #     for key, v in g3.node_list[i].nb_delay_pro.items():
    #         print(key + ':')
    #         print(str(v) + ';')
    #     print('\nall_worst-')
    #     for v in g3.node_list[i].all_worst_to_des:
    #         # f.write(key + ':')
    #         print(str(v) + ';')

    # adj_weight = np.ones((N, N))
    # f_graph = open("rtns_graph.txt", "a+")
    # for k in range(flow_num):
    #     rtns = RTNS(g3, d_set[k], k, sources[k], des[k])
    #     #print('adj_weight', adj_weight)
    #     rtns.get_best_path(adj_weight)

        # for i in range(len(rtns_trip)):
        #     print('node', rtns_trip[i], 'arrival_interval', rtns.graph.node_list[rtns_trip[i]].arrival_rate)
        #     print('node', rtns_trip[i], 'arrival_interval', g3.node_list[rtns_trip[i]].arrival_rate)
    #
    # table = Table(rand_graph)
    # table.generate_tables()
    #
    # for i in range(table.graph.node_num):
    #     print('node', i, 'worst', table.state_count_dictionary[i]['worst'], 'policy',
    #           table.state_count_dictionary[i]['policy'], 'expect', table.state_count_dictionary[i]['expect'], 'flow',
    #           table.state_count_dictionary[i]['flow'], 'hop', table.state_count_dictionary[i]['hop'])
    #
    # opt_cost = []
    # opt_trip = []
    # for i in range(len(d_set1)):
    #     trip, opt_delay = table.get_goal_result(d_set1[i], 0)
    #     opt_cost.append(opt_delay)
    #     opt_trip.append(trip)
    #
    # for i in range(len(d_set2)):
    #     trip, opt_delay = table.get_goal_result(d_set2[i], 1)
    #     opt_cost.append(opt_delay)
    #     opt_trip.append(trip)
    #
    # for i in range(len(d_set3)):
    #     trip, opt_delay = table.get_goal_result(d_set3[i], 2)
    #     opt_cost.append(opt_delay)
    #     opt_trip.append(trip)
    #
    #
    # filename = 'result.txt'
    # with open(filename, 'w') as file_object:
    #     file_object.write('rtss_result  ' + ' ' + str(rtss_cost))
    #     file_object.write('\n')
    #     file_object.write('rtss_trip  ' + ' ' + str(rtss_trip))
    #     file_object.write('\n')
    #     file_object.write('my_result  ' + ' ' + str(my_cost))
    #     file_object.write('\n')
    #     file_object.write('my_trip  ' + ' ' + str(my_trip))
    # #     file_object.write('\n')
    # #     file_object.write('opt_result  ' + ' ' + str(opt_cost))
    # #     file_object.write('\n')
    # #     file_object.write('opt_trip  ' + ' ' + str(opt_trip))
    # #     file_object.write('\n')
    # #
