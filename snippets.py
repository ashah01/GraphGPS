"""
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
g = nx.from_edgelist(data.edge_index.T.tolist())
f = plt.figure()
nx.draw_networkx(g, ax=f.add_subplot(111))
f.savefig("graph.png")

"""