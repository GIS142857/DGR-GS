import os
import json
import torch
import numpy as np
from config import *
from node import Node
from simulator import Simulator


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build a Simulator
sim = Simulator(SIM_TIME, SRC, DST, ARRIVAL_RATE, SUM_NODES, ADJ_TABLE, FRAME_SLOT, SLOT_DURATION, SUM_NODES, device, BATCH_SIZE)
for id in range(SUM_NODES):
    sim.nodes.append(Node(sim, id, SRC, DST, ARRIVAL_RATE, NODE_POSITION[id], device))

sim.init_agent()

# start the simulation
episodes = 5000
# sim.episode = episodes
for episode in range(episodes):
    print('\nepisode', episode)
    sim.run(episode)

    if (episode + 1) % 100 == 0:
        store_path = './CDF_data/E2ED'
        for node in sim.nodes:
            sub_path = store_path + '_' + str(node.id) + '.txt'
            with open(sub_path, 'w') as fl:
                fl.write(json.dumps(node.end_to_end_delay))
    if episode > 100:
        for node in sim.nodes:
            if node.id in DST:
                continue
            node.train()
    for key in FLOW_DICT.keys():
        src = sim.nodes[key]
        dst = sim.nodes[FLOW_DICT[key]]
        print('\nsrc', src.id, 'send_cnt:', src.send_cnt)
        print('des', dst.id, 'recv_cnt:', len(dst.recv_for_me))
        print('avg_e2ed_delay:', str(src.id)+ '-' +str(dst.id), round(np.mean(dst.e2ed)/1000, 2))