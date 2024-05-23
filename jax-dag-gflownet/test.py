import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pgmpy.models import BayesianNetwork
from collections import Counter

adj_array = np.load(
    "/Users/puw/Workspace/Vscode_Python/BayesianNetwork/jax-dag-gflownet/output/posterior_3.npy"
)
tupled_arrays = [tuple(map(tuple, adj_array[i])) for i in range(adj_array.shape[0])]
counter = Counter(tupled_arrays)
most_common_array, count = counter.most_common(1)[0]
print(count)
adj_array = most_common_array
# G = nx.from_numpy_array(adj_array)

# with open(
#     "/Users/puw/Workspace/Vscode_Python/BayesianNetwork/jax-dag-gflownet/graph3.pkl",
#     "rb",
# ) as file:
#     graph = pickle.load(file)
# G = graph.to_directed()
# print(list(G.nodes())[0])
# pos = nx.layout.circular_layout(G)
# nx.draw(
#     G,
#     pos,
#     with_labels=True,
#     node_size=2000,
#     node_color="lightblue",
#     font_size=10,
#     font_weight="bold",
# )
# plt.title("Linear Gaussian Bayesian Network")
# plt.show()

node_names = [
    "TOT_INJ",
    "REST1_0",
    "ALCFLAG",
    "MEDCAUSE",
    "SPEED_LIMIT",
    # "CITY",
    "V1CMPDIR",
    "V1EVENT1",
    "V2CMPDIR",
    "V2EVENT1",
    "NUMVEHS",
    "ST_FUNC",
    # "V1DIRCDE",
    "ACCTYPE",
    # "V2DIRCDE",
    "SEVERITY",
    "RDSURF",
    "LOC_TYPE",
    # "RODWYCLS",
    "REST1_1",
    "ROUTE_MILEPOST",
    "LIGHT",
    "TIME",
    "WEEKDAY",
    "NO_PEDS",
]

model = BayesianNetwork()
model.add_nodes_from(node_names)
for i in range(len(adj_array)):
    for j in range(len(adj_array[i])):
        if adj_array[i][j] == 1:  # 如果i到j有边
            model.add_edge(node_names[i], node_names[j])

model.remove_nodes_from(["TOT_INJ", "ACCTYPE"])
G = nx.DiGraph()
G.add_edges_from(model.edges())
nx.draw(G, with_labels=True)
plt.show()
# =====inj=====


with open(
    "/Users/puw/Workspace/Vscode_Python/BayesianNetwork/model/SEV_model.pkl", "wb"
) as f:
    pickle.dump(model, f)
