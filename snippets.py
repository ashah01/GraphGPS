"""
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
g = nx.from_edgelist(edge_index.T.tolist())
f = plt.figure()
nx.draw_networkx(g, ax=f.add_subplot(111))
f.savefig("graph.png")

def exclude(Pk, P, index):
   return torch.cat([Pk[0][15][:index], Pk[0][15][index+1:]]) @ torch.cat([P[0][15][:index], P[0][15][index+1:]])

"""