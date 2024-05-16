import numpy as np
import pandas as pd
import json
import pickle
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, labels_name, title):
    np.set_printoptions(precision=2)
    fig, ax = plt.subplots()
    # print(cm)
    plt.imshow(cm, interpolation="nearest")  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="black" if cm[i, j] > thresh else "white",
            )

    # show confusion matrix
    # plt.savefig("./fig/" + title + ".png", format="png")
    plt.show()


target = "TOT_INJ"
with open(
    "/Users/puw/Workspace/Vscode_Python/BayesianNetwork/model/INJ_model.pkl", "rb"
) as f:
    loaded_model = pickle.load(f)

# data1 = pd.read_csv(
#     "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_1.csv"
# )
# data2 = pd.read_csv(
#     "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_2.csv"
# )
# data3 = pd.read_csv(
#     "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_3.csv"
# )

# data4 = pd.read_csv(
#     "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_11.csv"
# )
# data_train = pd.concat([data1, data2, data3], axis=0)
# data_test = data4
# data_all = pd.DataFrame()
# for i in range(1, 12):
#     t = pd.read_csv(
#         "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_"
#         + str(i)
#         + ".csv"
#     )
#     data_all = pd.concat([data_all, t], axis=0)


data_train = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/output/WA/split_data/train.csv"
)
data_test = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/output/WA/split_data/test.csv"
)

data_all = pd.concat([data_train, data_test], axis=0)


# print(loaded_model.edges())
model = BayesianNetwork(loaded_model.edges())


data_test = data_test.loc[:, list(model.nodes())]
data_train = data_train.loc[:, list(model.nodes())]

state_names = {}
for key, _ in data_all.items():
    state_names[key] = list(set(data_all[key]))


def calculate_accuracy(df_true, df_pred):
    total_matches = (df_true == df_pred).sum().sum()  # 比较所有元素并计数相等的结果
    total_elements = df_true.size  # 总元素数
    accuracy = total_matches / total_elements
    return accuracy


G = nx.DiGraph()
G.add_edges_from(loaded_model.edges())
nx.draw(G, with_labels=True)
plt.show()

res_acc = []

# G.fit_update(data=data_test.iloc[:5, :], n_jobs=-1)
# print(G.get_cpds())
# for i in range(0, len(data_train), 200):
#     if i == 0:
#         model.fit(
#             data=data_train.iloc[i : i + 200, :],
#             estimator=MaximumLikelihoodEstimator,
#             n_jobs=-1,
#             state_names=state_names,
#         )
#     else:
#         model.fit_update(
#             data=data_train.iloc[i : i + 200, :], n_jobs=-1, n_prev_samples=i
#         )
#     r = model.predict(data_t)
#     acc = calculate_accuracy(data_test.iloc[:100, :].loc[:, [target]], r)
#     print(acc)
#     res_acc.append(acc)
data_train = data_train.reset_index(drop=True)

model.fit(
    data_train,
    estimator=BayesianEstimator,
    n_jobs=-1,
    state_names=state_names,
)
print("------")

print(model.get_markov_blanket("TOT_INJ"))

r = model.predict(data_test.drop([target], axis=1))
gt = data_test.loc[:, [target]].values.tolist()
pre = r[target].tolist()
# labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# labels = [0, 1, 2, 3, 4]
labels = [0,1,2,3]
cm = confusion_matrix(gt, pre, labels=labels)

# print(pre)
print(calculate_accuracy(data_test.loc[:, [target]], r))

plot_confusion_matrix(cm, labels, "confusion_matrix")
# model.save("trained.bif", filetype="bif")
