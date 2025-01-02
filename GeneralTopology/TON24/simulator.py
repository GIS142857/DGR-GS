import simpy
import random
from simpy.util import start_delayed
from config import *
from utils import *


class Simulator:
    ############################
    def __init__(self, sim_time, src, dst, arrival_rates, sum_nodes, adj_T, frame_slot, slot_duration, num_nodes, device, batch_size, seed=0):
        self.env = simpy.Environment()
        self.nodes = []
        self.sim_time = sim_time
        self.arrival_rates = arrival_rates
        self.random = random.Random(seed)
        self.src = src
        self.dst = dst
        self.range = range
        self.timeout = self.env.timeout
        self.sum_nodes = sum_nodes
        self.adj_T = adj_T
        self.device = device
        self.batch_size = batch_size
        self.frame_slot = frame_slot
        self.slot_duration = slot_duration
        self.num_nodes = num_nodes
        self.start_time = self.env.now
        self.cumulative_reward = [[], [], []]
        self.episode = 0

    ############################
    def init(self):
        pass

    ############################
    @property
    def now(self):
        return self.env.now

    def delayed_exec(self, delay, func, *args, **kwargs):
        func = ensure_generator(self.env, func, *args, **kwargs)
        start_delayed(self.env, func, delay=delay)

    ############################
    def init_agent(self):
        for id in range(SUM_NODES):
            me = self.nodes[id]
            me.setAgent()

    ############################
    def run(self, episode):
        self.episode = episode
        for node in self.nodes:
            node.reset()

        for id in self.src:
            node = self.nodes[id]
            node.start_time = self.env.now
            self.env.process(node.run())
        self.env.run(until=self.env.now + self.sim_time)