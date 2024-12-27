from GeneralTopology.DGR_GS.simulator import Simulator
from config import *

def main():
    sim = Simulator(SIM_TIME, SUM_NODES, SRC, DES, ADJ_TABLE, NODE_POSITION, FRAME_SLOT, SLOT_DURATION, ARRIVAL_RATE)
    for e in range(episodes):
        print('\nepisode:', e)
        sim.episode = e

        # print(vars(sim))

        sim.run()
        # print(sim.paths_delay)
        # print(sim.e2ed_delay_plus.keys())
        # print(sim.e2ed_delay_minus.keys())

        print("send_cnt: ", sim.nodes[0].send_cnt, sim.nodes[1].send_cnt, sim.nodes[2].send_cnt)
        print("recv_cnt: ", sim.nodes[15].recv_cnt, sim.nodes[16].recv_cnt, sim.nodes[17].recv_cnt)
        print("can't guarantee count: ", sim.can_not_dg)
        print("loss_history: ", sim.loss_cnt)
        sim.get_avg_queue_delay(sim.queue_delay)
        sim.get_avg_e2ed_delay(sim.e2ed_delay)
        sim.get_worst_case_e2ed_delay(sim.paths_delay)

        sim.update()

        if e % 10 == 0:
            for node in sim.nodes:
                print(node.route_tb.route_vector)

        sim.reset()


if __name__ == "__main__":
    main()
