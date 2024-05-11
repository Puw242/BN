import numpy as np
import pandas as pd
import json
import pickle
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# model = BayesianNetwork.load(
#     "/Users/puw/Workspace/Vscode_Python/trained.bif", filetype="bif"
# )

data1 = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_1.csv"
)
data2 = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_2.csv"
)
data3 = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_3.csv"
)

data4 = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_4.csv"
)


data_train = pd.concat([data1, data2, data3], axis=0)
data_test = data4

print(data_train['ACCTYPE'].value_counts()/data_train.shape[0])
print(data_test['ACCTYPE'].value_counts()/data_test.shape[0])