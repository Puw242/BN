import pandas as pd
import numpy as np
import json

input_file = "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/ori_wa.csv"
data = pd.read_csv(input_file)

print(data.head())
print(data.info())

map_MONTH = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

map_INJ = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "MORE THAN FOUR": 5,
}

map_SEVER = {
    "UNKNOWN": 0,
    "NO APPARENT INJURY": 1,
    "POSSIBLE INJURY": 2,
    "MINOR INJURY": 3,
    "SERIOUS INJURY": 4,
    "FATAL": 5,
}

data["MEDCAUSE"] = pd.factorize(data["MEDCAUSE"])[0]
data["COUNTY"] = pd.factorize(data["COUNTY"])[0]
data["CITY"] = pd.factorize(data["CITY"])[0]
data["MONTH"] = data["MONTH"].map(map_MONTH)
data["TOT_INJ"] = data["TOT_INJ"].map(map_INJ)
data["ROUTE_ID"] = pd.factorize(data["ROUTE_ID"])[0] + 1
data["ST_FUNC"] = pd.factorize(data["ST_FUNC"])[0]
data["RUR_URB"] = pd.factorize(data["RUR_URB"])[0]
data["V1CMPDIR"] = pd.factorize(data["V1CMPDIR"])[0]
data["V1EVENT1"] = pd.factorize(data["V1EVENT1"])[0]

data["V1DIRCDE"] = pd.factorize(data["V1DIRCDE"])[0]
data["ACCTYPE"] = pd.factorize(data["ACCTYPE"])[0]
data["V2CMPDIR"] = pd.factorize(data["V2CMPDIR"])[0]
data["V2EVENT1"] = pd.factorize(data["V2EVENT1"])[0]

data["V2DIRCDE"] = pd.factorize(data["V2DIRCDE"])[0]
data["SEVERITY"] = data["SEVERITY"].map(map_SEVER)
data["OBJECT1"] = pd.factorize(data["OBJECT1"])[0]
data["OBJECT2"] = pd.factorize(data["OBJECT2"])[0]

data["LOC_TYPE"] = pd.factorize(data["LOC_TYPE"])[0]
data["RDSURF"] = pd.factorize(data["RDSURF"])[0]
data["LIGHT"] = pd.factorize(data["LIGHT"])[0]

data["LOC_CHAR"] = pd.factorize(data["LOC_CHAR"])[0]
data["WKZONE"] = pd.factorize(data["WKZONE"])[0]
data["RODWYCLS"] = pd.factorize(data["RODWYCLS"])[0]

data["CITY_NAME"] = pd.factorize(data["CITY_NAME"])[0]
data["ACCESS_CONTROL"] = pd.factorize(data["ACCESS_CONTROL"])[0]
data["MEDIAN_BARRIER"] = pd.factorize(data["MEDIAN_BARRIER"])[0]

data["LEFT_SHOULDER_SURFACE"] = pd.factorize(data["LEFT_SHOULDER_SURFACE"])[0]
data["SPECIAL_USE_LANE_ROAD_SIDE"] = pd.factorize(data["SPECIAL_USE_LANE_ROAD_SIDE"])[0]
data["LANE_SURFACE_TYPE"] = pd.factorize(data["LANE_SURFACE_TYPE"])[0]

data["DIVIDED_INDICATOR"] = pd.factorize(data["DIVIDED_INDICATOR"])[0]
data["SPECIAL_USE_LANE_ROAD_SIDE"] = pd.factorize(data["SPECIAL_USE_LANE_ROAD_SIDE"])[0]

for i in range(3):
    num = str(i)
    data["SEATPOS_" + num] = pd.factorize(data["SEATPOS_" + num])[0]
    data["SEX_" + num] = pd.factorize(data["SEX_" + num])[0]
    data["REST1_" + num] = pd.factorize(data["REST1_" + num])[0]
    data["HELMET_" + num] = pd.factorize(data["HELMET_" + num])[0]
    num = str(i + 1)
    data["DRASSESS_" + num] = pd.factorize(data["DRASSESS_" + num])[0]
    data["CONTRIB1_" + num] = pd.factorize(data["CONTRIB1_" + num])[0]
    data["CONTRIB2_" + num] = pd.factorize(data["CONTRIB2_" + num])[0]
    data["VEH_USE_" + num] = pd.factorize(data["VEH_USE_" + num])[0]

    data["DRV_ACTN_" + num] = pd.factorize(data["DRV_ACTN_" + num])[0]
    data["VEHCOND1_" + num] = pd.factorize(data["VEHCOND1_" + num])[0]
    data["VEHCOND2_" + num] = pd.factorize(data["VEHCOND2_" + num])[0]
    data["VEHCOND3_" + num] = pd.factorize(data["VEHCOND3_" + num])[0]
    data["INTER_A_" + num] = pd.factorize(data["INTER_A_" + num])[0]
    data["CMAXLES_" + num] = pd.factorize(data["CMAXLES_" + num])[0]
    data["DRAIRBAG_" + num] = pd.factorize(data["DRAIRBAG_" + num])[0]
    data["DRV_EJCT_" + num] = pd.factorize(data["DRV_EJCT_" + num])[0]
means = data[["AGE_0", "AGE_1", "AGE_2"]].mean()
print(means)
data[["AGE_0", "AGE_1", "AGE_2"]] = data[["AGE_0", "AGE_1", "AGE_2"]].fillna(means)
data["ROUTE_MILEPOST"] = data["ROUTE_ID"] * 1000 + data["MILEPOST"]

print(data.head())

dfs = {month: group for month, group in data.groupby("MONTH")}
for key, value in dfs.items():
    value.to_csv(
        "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/wa_2022_"
        + str(key)
        + ".csv",
        index=False,
    )
# data.to_csv(
#     "/Users/puw/Workspace/Vscode_Python/output/WA/baseline_data/digit/"
# )
