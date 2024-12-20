import networkx as nx
from GeneralTopology.util.model_config import NODE_POSITION
from GeneralTopology.util.utils import distance

# 计算两点之间的距离并构建干扰图
def build_interference_graph(node_positions, interference_distance):
    G = nx.Graph()
    for i in node_positions:
        G.add_node(i)
    for i in node_positions:
        for j in range(i + 1, len(node_positions)):
            dist = distance(node_positions[i], node_positions[j])
            if dist <= interference_distance:
                G.add_edge(i, j)
    return G

# 分配TDMA时隙
def assign_tdma_slots(graph):
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
    return coloring

# 构建干扰图
interference_distance = 125
interference_graph = build_interference_graph(NODE_POSITION, interference_distance)

# 分配TDMA时隙
tdma_slots = assign_tdma_slots(interference_graph)

# 输出时隙分配结果
for node, slot in tdma_slots.items():
    print(f"Node {node}: Slot {slot}")
