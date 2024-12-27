import json
import os
import torch
from config import *
from simulator import Simulator
from node import Node

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build a Simulator
sim = Simulator(SIM_TIME, SRC, DST, ARRIVAL_RATE, SUM_NODES, ADJ_TABLE, FRAME_SLOT, SLOT_DURATION, SUM_NODES, device)
for id in range(SUM_NODES):
    sim.nodes.append(Node(sim, id, SRC, DST, ARRIVAL_RATE, NODE_POSITION[id], device))

sim.init_agent()

# start the simulation
episodes = 2000
# sim.episode = episodes
for e in range(episodes):
    print('episode', e)
    if e > 0:
        for n in sim.nodes:
            n.mac.queues = {'flow1': [], 'flow2': [], 'flow3': []}
    sim.run(e)

    if (e + 1) % 100 == 0:
        store_path = './CDF_data/E2ED'
        for n in sim.nodes:
            sub_path = store_path + '_' + str(n.id) + '.txt'
            with open(sub_path, 'w') as fl:
                fl.write(json.dumps(n.end_to_end_delay))

    # for i in range(3):
    #     print('src', sim.nodes[i], sim.nodes[i].sends)
    #     print('des', sim.nodes[17-i], 'receives', len(sim.nodes[17-i].reces_for_me))
    #     print('des', sim.nodes[17-i], sim.nodes[17-i].e2ed)
