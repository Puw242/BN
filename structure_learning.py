import numpy as np
import pandas as pd
import pgmpy
from pgmpy.estimators import PC
import pickle
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import HillClimbSearch, BicScore, BDeuScore
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import MmhcEstimator
# import bnlearn as bn
from pgmpy.estimators import K2Score
from pgmpy.metrics import structure_score
import sys

sys.path.insert(0, "BNSL")

# from bnsl.estimators import HC,GA


data1 = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_1.csv"
)
data2 = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_2.csv"
)
data3 = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_3.csv"
)

# data1 = pd.read_csv(
#     "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/ori/wa_2022_January.csv"
# )
# data2 = pd.read_csv(
#     "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/ori/wa_2022_February.csv"
# )
# data3 = pd.read_csv(
#     "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/ori/wa_2022_March.csv"
# )

data4 = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_4.csv"
)

data_train = pd.concat([data1, data2, data3], axis=0)
data_test = data4
data_train.drop(
    [
        "Unnamed: 0",
        "ROUTE_ID",
        "MILEPOST",
        "TOT_KILL",
        "ACCYR",
        "MONTH",
    ],
    inplace=True,
    axis=1,
)
data_all = pd.concat([data_train, data_test], axis=0)
for i in range(5, 13):
    t = pd.read_csv(
        "/Users/puw/Workspace/Vscode_Python/BayesianNetwork/data/digit/wa_2022_"
        + str(i)
        + ".csv"
    )
    data_all = pd.concat([data_all, t], axis=0)


userd_attr = [
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
    # "ACCTYPE",
    # "V2DIRCDE",
    # "SEVERITY",
    "RDSURF",
    "LOC_TYPE",
    # "RODWYCLS",
    # "REST1_1",
    "ROUTE_MILEPOST",
    "LIGHT",
    "TIME",
    "WEEKDAY",
    "NO_PEDS",
]
data_all = data_all.loc[:, userd_attr]
print(data_all)
# for i, v in data_train.apply(lambda x: (x == -1) | (x == 0)).mean().items():
#     if v > 0.5:
#         data_train.drop(i, axis=1, inplace=True)

# for i, v in data_train.apply(lambda x: (pd.isna(x)) | (pd.isnull(x))).mean().items():
#     if v > 0.8:
#         data_train.drop(i, axis=1, inplace=True)
#         print(i)


# print(data_train.columns)


# est = HillClimbSearch(data_train)
# best_model = est.estimate(
#     scoring_method=BDeuScore(data_train, equivalent_sample_size=3),
#     epsilon=1e-8,
#     max_indegree=5,
# )
# est = MmhcEstimator(data_train)
# ================================
def score_function(node, parents, data):
    """
    分数函数，根据当前节点和给定的父节点集合来评估数据的拟合程度
    """

    G = nx.DiGraph()
    G.add_nodes_from(userd_attr)
    G.add_edges_from([(parent, node) for parent in parents])
    model = BayesianNetwork(G)
    # nodes = list(parents)
    # nodes.append(node)
    score = structure_score(model, data=data, scoring_method="k2")
    return score


def k2_algorithm(data, nodes, max_parents):
    """
    data: 观测数据, 二维列表或者numpy数组。
    nodes: 变量列表。
    max_parents: 每个结点允许的最大父结点数量。
    """

    # 初始化网络结构
    network_structure = {node: [] for node in nodes}

    # 按照给定的节点顺序处理每个节点
    for node in nodes:
        print("doing" + node)
        # 父节点候选集初始化
        parents = set()
        best_score = float("-inf")

        # 考虑添加新的父节点，直到不能改善分数或已达到最大父节点数
        while len(parents) < max_parents:
            candidates = set(nodes) - set(parents) - {node}
            if not candidates:
                break

            # 寻找可以最大化分数的最好父节点
            best_candidate = None
            for candidate in candidates:
                score = score_function(node, parents | {candidate}, data)
                if score > best_score:
                    best_candidate = candidate
                    best_score = score

            # 如果找到更好的父节点，更新父节点集合与最佳分数
            if best_candidate is not None:
                parents.add(best_candidate)
            else:
                break

        # 结束后为节点更新父节点列表
        network_structure[node] = list(parents)

    return network_structure


# =================================
# hc = GA(data_all)
# hc.run()
# print("Structure learning Done!")
# dag = hc.result
# dag.show()

# =================================
# struct = k2_algorithm(data_all, nodes=userd_attr, max_parents=6)

# G = nx.DiGraph()
# for chid, p in struct.items():
#     G.add_nodes_from(userd_attr)
#     G.add_edges_from([(pr, chid) for pr in p])
# print(struct)
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True)
# plt.show()
# model = BayesianModel(G)
# with open(
#     "/Users/puw/Workspace/Vscode_Python/BayesianNetwork/model/model.pkl", "wb"
# ) as f:
#     pickle.dump(model, f)

# ==============================bn learn
# from_to_tuples = [(userd_attr[i], userd_attr[-1]) for i in range(len(userd_attr) - 1)]


# model_sl = bn.structure_learning.fit(
#     data_all,
#     methodtype="hc",
#     scoretype="k2",
#     white_list=userd_attr,
#     bw_list_method='ndoes',
#     n_jobs=-1,
# )
# bn.plot(model_sl,params_static={'font_size':6})
# print(bn.get_edge_properties(model_sl))

# =============================
# est = PC(data=data_all)
# best_model = est.estimate(
#     variant="parallel", max_cond_vars=2, significance_level=0.01, ci_test="g_sq"
# )

# G = nx.DiGraph(best_model.edges())
# pos = nx.spring_layout(G)
# nx.draw(
#     G,
#     pos,
#     with_labels=True,
#     node_size=200,
#     node_color="lightgreen",
#     edge_color="gray",
#     width=2,
#     font_size=12,
#     font_weight="bold",
# )

# plt.show()

# # est = PC(data=data)
# # estimated_model = est.estimate(variant="parallel", max_cond_vars=1)
# with open(
#     "/Users/puw/Workspace/Vscode_Python/BayesianNetwork/model/model.pkl", "wb"
# ) as f:
#     pickle.dump(best_model, f)


# ========================
# train_num = round(data.shape[0] * 0.8)
# np.random.seed(1)
# index = np.random.permutation(data.shape[0])[0:train_num]

# train_d = data.iloc[index, :]
# test_d = data.drop(index, axis=0)


# from pgmpy.models import BayesianModel

# with open(
#     "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/model.pkl", "rb"
# ) as f:
#     loaded_model = pickle.load(f)

# model_r = BayesianNetwork(loaded_model.edges())

# user_v = list(model_r.nodes())
# model_train = train_d[user_v]
# model_test = test_d[user_v]
# model_test = model_test.drop(["SEVERITY"], axis=1)

# model_r.fit(data=model_train, estimator=MaximumLikelihoodEstimator, n_jobs=10)
