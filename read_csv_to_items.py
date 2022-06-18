 # read a csv of data and return a set of items
import numpy as np
import pandas as pd

from preference_classes import Item


def get_data_items(filename, max_items=99999, standardize_features=False, normalize_features=False, drop_cols=[]):

    df = pd.read_csv(filename)

    # df["IsInterpretable_int"] = df["IsInterpretable"].apply(lambda x: 1 if x else 0)

    # drop unused cols
    # drop_cols = drop_cols + ["id", "PolicyType", "Policy", "IsInterpretable"]
    # print(df.head())
    if "UsesProtectedFeatures" in df.columns.to_list():
        drop_cols = ["Id", "PolicyName", "Policy", "NumFeatures", "UsesProtectedFeatures"]
    else:
        drop_cols = ["Approach", "TrainingFile", "NumDatapoints", "TreeDepth", "BranchingLimit", "TimeLimit", "ProbTypePred", "SolverStatus", "ObjVal", "MIPGap", "SolvingTime", "NumBranchingNodes"]
    df.drop(columns=drop_cols, inplace=True)

    # df["NumBranchingFeatures"] = -df["NumBranchingFeatures"]
    # df["NumProtectedBranchingFeatures"] = -df["NumProtectedBranchingFeatures"]

    items = []
    for i, row in df.iterrows():
        items.append(Item(row.values, i))

    if len(items) > max_items:
        items = items[:max_items]

    if standardize_features:
        for i_feat in range(len(items[0].features)):
            feature_list = [i.features[i_feat] for i in items]
            mean_feat = np.mean(feature_list)
            stddev_feat = np.std(feature_list)

            if stddev_feat > 0:
                for item in items:
                    item.features[i_feat] = (
                        item.features[i_feat] - mean_feat
                    ) / stddev_feat
            else:
                for item in items:
                    item.features[i_feat] = 0.0

    if normalize_features:
        for i_feat in range(len(items[0].features)):
        # for i_feat in range(2):
            feature_list = [i.features[i_feat] for i in items]
            max_feat = np.max(feature_list)
            min_feat = np.min(feature_list)

            for item in items:
                item.features[i_feat] = (item.features[i_feat] - min_feat) / (max_feat - min_feat)

    return items
