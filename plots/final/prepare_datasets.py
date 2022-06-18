import os
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# define files and directories
from get_bounds import get_mmu_ub, get_mmu_lb, get_mmr_lb, get_mmr_ub
from preference_classes import generate_items
from read_csv_to_items import get_data_items


def prepare_offline_mmr():
    """
    read the original results from the offline MMR experiments and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'offline_mmr'
    u0_type = 'box'

    out_df_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/offline_mmr_final.csv'.format(version_str)
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/offline_mmr_final.csv"

    # results file (from static elicitation experiments)
    output_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/static_experiment_20200116_144436.csv'.format(version_str)

    # updated results, with 40 features, 10 items, and gamma = 0.2
    output_file_2 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/static_experiment_20200118_134948.csv'.format(version_str)

    # ----------------------------------------------------------------------------------------------------------------------
    # read results

    # data file: output from experiments
    # df = pd.read_csv(output_file, skiprows=1, delimiter=';')
    # df_2 = pd.read_csv(output_file_2, skiprows=1, delimiter=';')

    # df = df.append(df_2)

    # v_2
    output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_122255.csv"
    output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_122259.csv"
    output_file_3 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_122300.csv"
    output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_122303.csv"
    output_file_5 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_122308.csv"
    output_file_6 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_122310.csv"
    output_file_7 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_122313.csv"

    # df = pd.read_csv(data_file, delimiter=';')
    df = pd.read_csv(output_file, skiprows=1, delimiter=';')
    df2 = pd.read_csv(output_file_2, skiprows=1, delimiter=';')
    df3 = pd.read_csv(output_file_3, skiprows=1, delimiter=';')
    df4 = pd.read_csv(output_file_4, skiprows=1, delimiter=';')
    df5 = pd.read_csv(output_file_5, skiprows=1, delimiter=';')
    df6 = pd.read_csv(output_file_6, skiprows=1, delimiter=';')
    df7 = pd.read_csv(output_file_7, skiprows=1, delimiter=';')
    df = df.append(df2)
    df = df.append(df3)
    df = df.append(df4)
    df = df.append(df5)
    df = df.append(df6)
    df = df.append(df7)
    # find all cols where we will need to calculate different ub/lb values
    expt_cols = ["problem_seed", "num_items", "num_features", "item_sphere_size"]

    # create an id for each of these
    df["expt_params"] = df[expt_cols].apply(lambda x: tuple(x), axis=1)

    param_sets = df["expt_params"].unique()
    # param_sets = df[expt_cols].drop_duplicates().copy()

    df["mmr_ub"] = None
    df["mmr_lb"] = None
    df["mmu_ub"] = None
    df["mmu_lb"] = None

    # get the ub/lb values for each case
    for params in param_sets:
        # unpack...
        problem_seed, num_items, num_features, item_sphere_size = params
        print(f"params: {params}")
        # generate items
        items = generate_items(
            int(num_features),
            int(num_items),
            item_sphere_size=item_sphere_size,
            seed=int(problem_seed),
        )

        # mmu_lb = get_mmu_lb(items, u0_type)
        # mmu_ub = get_mmu_ub(items, u0_type)
        #
        # # set the ub/lb cols
        # df.loc[df["expt_params"] == params, "mmu_ub"] = mmu_ub
        # df.loc[df["expt_params"] == params, "mmu_lb"] = mmu_lb

        mmr_lb = get_mmr_lb(items, u0_type)
        mmr_ub = get_mmr_ub(items, u0_type)

        # set the ub/lb cols
        df.loc[df["expt_params"] == params, "mmr_ub"] = mmr_ub
        df.loc[df["expt_params"] == params, "mmr_lb"] = mmr_lb

    df["mmr_objval_normalized_new"] = (df["mmr_objval"] - df["mmr_lb"]) / (df["mmr_ub"] - df["mmr_lb"])

    df.to_csv(out_df_file, sep=";")



def prepare_offline_mmu():
    """
    read the original results from the offline MMU experiments and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'offline_mmu'
    objtype = 'mmu'
    u0_type = 'box'

    out_df_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/offline_mmu_final.csv'.format(version_str)
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/offline_mmu_final.csv"

    # results file (from static elicitation experiments)
    output_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/static_experiment_20200114_111428.csv'.format(
        version_str)

    # updated results, with 40 features, 10 items, and gamma = 0.2
    output_file_2 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/static_experiment_20200118_135254.csv'.format(
        version_str)

    # ----------------------------------------------------------------------------------------------------------------------
    # read results

    # data file: output from experiments
    # df = pd.read_csv(output_file, skiprows=1, delimiter=';')
    # df_2 = pd.read_csv(output_file_2, skiprows=1, delimiter=';')

    # df = df.append(df_2)

    # v_2
    output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_222541.csv"
    output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_222615.csv"
    output_file_3 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_003420.csv"
    output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_003423.csv"
    output_file_5 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_003431.csv"
    output_file_6 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_003435.csv"
    output_file_7 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210701_122253.csv"

    # df = pd.read_csv(data_file, delimiter=';')
    df = pd.read_csv(output_file, skiprows=1, delimiter=';')
    df2 = pd.read_csv(output_file_2, skiprows=1, delimiter=';')
    df3 = pd.read_csv(output_file_3, skiprows=1, delimiter=';')
    df4 = pd.read_csv(output_file_4, skiprows=1, delimiter=';')
    df5 = pd.read_csv(output_file_5, skiprows=1, delimiter=';')
    df6 = pd.read_csv(output_file_6, skiprows=1, delimiter=';')
    df7 = pd.read_csv(output_file_7, skiprows=1, delimiter=';')
    df = df.append(df2)
    df = df.append(df3)
    df = df.append(df4)
    df = df.append(df5)
    df = df.append(df6)
    df = df.append(df7)
    # find all cols where we will need to calculate different ub/lb values
    expt_cols = ["problem_seed", "num_items", "num_features", "item_sphere_size"]

    # create an id for each of these
    df["expt_params"] = df[expt_cols].apply(lambda x: tuple(x), axis=1)

    param_sets = df["expt_params"].unique()
    # param_sets = df[expt_cols].drop_duplicates().copy()

    df["mmu_ub"] = None
    df["mmu_lb"] = None

    # get the ub/lb values for each case
    for params in param_sets:
        # unpack...
        problem_seed, num_items, num_features, item_sphere_size = params
        print(f"params: {params}")
        # generate items
        items = generate_items(
            int(num_features),
            int(num_items),
            item_sphere_size=item_sphere_size,
            seed=int(problem_seed),
        )

        mmu_lb = get_mmu_lb(items, u0_type)
        mmu_ub = get_mmu_ub(items, u0_type)

        # set the ub/lb cols
        df.loc[df["expt_params"] == params, "mmu_ub"] = mmu_ub
        df.loc[df["expt_params"] == params, "mmu_lb"] = mmu_lb

        # mmr_lb = get_mmr_lb(items, u0_type)
        # mmr_ub = get_mmr_ub(items, u0_type)
        #
        # # set the ub/lb cols
        # df.loc[df["expt_params"] == params, "mmr_ub"] = mmr_ub
        # df.loc[df["expt_params"] == params, "mmr_lb"] = mmr_lb

    df["mmu_objval_normalized_new"] = (df["mmu_objval"] - df["mmu_lb"]) / (df["mmu_ub"] - df["mmu_lb"])

    df.to_csv(out_df_file, sep=";")


def prepare_online_mmr():
    """
    read the original results from the online MMR experiments and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'online_mmr'
    objtype = 'mmr'
    u0_type = 'box'

    out_df_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/online_mmr_final.csv'.format(version_str)

    # results file (from static elicitation experiments)
    output_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/adaptive_experiment_20200121_093418.csv'.format(
        version_str)

    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_final1.csv"
    output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210621_225103.csv"
    output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210622_145549.csv"
    output_file_3 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210622_171734.csv"
    output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210622_174620.csv"
    output_file_5 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210622_191403.csv"
    output_file_6 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210622_211048.csv"
    output_file_7 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210622_215620.csv"


    df = pd.read_csv(output_file, skiprows=1, delimiter=';')
    df2 = pd.read_csv(output_file_2, skiprows=1, delimiter=';')
    df3 = pd.read_csv(output_file_3, skiprows=1, delimiter=';')
    df4 = pd.read_csv(output_file_4, skiprows=1, delimiter=';')
    df5 = pd.read_csv(output_file_5, skiprows=1, delimiter=';')
    df6 = pd.read_csv(output_file_6, skiprows=1, delimiter=';')
    df7 = pd.read_csv(output_file_7, skiprows=1, delimiter=';')
    df = df.append(df2)
    df = df.append(df3)
    df = df.append(df4)
    df = df.append(df5)
    df = df.append(df6)
    df = df.append(df7)

    # ----------------------------------------------------------------------------------------------------------------------
    # read results

    # data file: output from experiments
    # df = pd.read_csv(output_file, skiprows=1, delimiter=';')

    # find all cols where we will need to calculate different ub/lb values
    expt_cols = ["problem_seed", "num_items", "num_features", "item_sphere_size"]

    # create an id for each of these
    df["expt_params"] = df[expt_cols].apply(lambda x: tuple(x), axis=1)

    param_sets = df["expt_params"].unique()
    # param_sets = df[expt_cols].drop_duplicates().copy()

    df["mmr_ub"] = None
    df["mmr_lb"] = None

    # get the ub/lb values for each case
    for params in param_sets:
        # unpack...
        problem_seed, num_items, num_features, item_sphere_size = params
        print(f"params: {params}")
        # generate items
        items = generate_items(
            int(num_features),
            int(num_items),
            item_sphere_size=item_sphere_size,
            seed=int(problem_seed),
        )

        # mmu_lb = get_mmu_lb(items, u0_type)
        # mmu_ub = get_mmu_ub(items, u0_type)
        #
        # # set the ub/lb cols
        # df.loc[df["expt_params"] == params, "mmu_ub"] = mmu_ub
        # df.loc[df["expt_params"] == params, "mmu_lb"] = mmu_lb

        mmr_lb = get_mmr_lb(items, u0_type)
        mmr_ub = get_mmr_ub(items, u0_type)

        # set the ub/lb cols
        df.loc[df["expt_params"] == params, "mmr_ub"] = mmr_ub
        df.loc[df["expt_params"] == params, "mmr_lb"] = mmr_lb

    df["mmr_objval_normalized_new"] = (df["mmr_objval"] - df["mmr_lb"]) / (df["mmr_ub"] - df["mmr_lb"])
    df["true_regret_normalized_new"] = (df["true_regret"] - df["mmr_lb"]) / (df["mmr_ub"] - df["mmr_lb"])

    df.to_csv(out_df_file, sep=";")


def prepare_online_mmu():
    """
    read the original results from the online MMU experiments and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'online_mmu'
    u0_type = 'box'

    # out_df_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/online_mmu_final.csv'.format(version_str)
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmu_final1.csv"

    # output_file_1 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/adaptive_experiment_20200119_194514.csv'.format(
    #     version_str)
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210610_172523_2.csv"
    # output_file_2 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/adaptive_experiment_20200121_182207.csv'.format(
    #     version_str)
    output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210611_211604.csv"
    output_file_3 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210612_164640.csv"


    # v_2
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210614_120056.csv"
    output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210615_110720.csv"
    output_file_3 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210611_211604.csv"
    # output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210613_161123.csv"
    # output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210617_161048.csv"
    output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210620_223228.csv"
    output_file_5 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210620_160517.csv"
    output_file_6 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210620_235629.csv"
    output_file_7 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210621_152309.csv"
    output_file_8 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210621_223816.csv"

    # ----------------------------------------------------------------------------------------------------------------------
    # read results

    # data file: output from experiments
    df_1 = pd.read_csv(output_file_1, skiprows=1, delimiter=';')
    df_2 = pd.read_csv(output_file_2, skiprows=1, delimiter=';')
    df_3 = pd.read_csv(output_file_3, skiprows=1, delimiter=';')
    df_4 = pd.read_csv(output_file_4, skiprows=1, delimiter=';')
    df_5 = pd.read_csv(output_file_5, skiprows=1, delimiter=';')
    df_6 = pd.read_csv(output_file_6, skiprows=1, delimiter=';')
    df_7 = pd.read_csv(output_file_7, skiprows=1, delimiter=';')
    df_8 = pd.read_csv(output_file_8, skiprows=1, delimiter=';')

    df = df_1.append(df_2)
    df = df.append(df_3)
    df = df.append(df_4)
    df = df.append(df_5)
    df = df.append(df_6)
    df = df.append(df_7)
    df = df.append(df_8)


    # find all cols where we will need to calculate different ub/lb values
    expt_cols = ["problem_seed", "num_items", "num_features", "item_sphere_size"]

    # create an id for each of these
    df["expt_params"] = df[expt_cols].apply(lambda x: tuple(x), axis=1)

    param_sets = df["expt_params"].unique()
    # param_sets = df[expt_cols].drop_duplicates().copy()

    df["mmu_ub"] = None
    df["mmu_lb"] = None

    # get the ub/lb values for each case
    for params in param_sets:
        # unpack...
        problem_seed, num_items, num_features, item_sphere_size = params
        print(f"params: {params}")
        # generate items
        items = generate_items(
            int(num_features),
            int(num_items),
            item_sphere_size=item_sphere_size,
            seed=int(problem_seed),
        )

        mmu_lb = get_mmu_lb(items, u0_type)
        mmu_ub = get_mmu_ub(items, u0_type)

        # set the ub/lb cols
        df.loc[df["expt_params"] == params, "mmu_ub"] = mmu_ub
        df.loc[df["expt_params"] == params, "mmu_lb"] = mmu_lb

        # mmr_lb = get_mmr_lb(items, u0_type)
        # mmr_ub = get_mmr_ub(items, u0_type)
        #
        # # set the ub/lb cols
        # df.loc[df["expt_params"] == params, "mmr_ub"] = mmr_ub
        # df.loc[df["expt_params"] == params, "mmr_lb"] = mmr_lb

    df["mmu_objval_normalized_new"] = (df["mmu_objval"] - df["mmu_lb"]) / (df["mmu_ub"] - df["mmu_lb"])
    df["true_u_normalized_new"] = (df["true_u"] - df["mmu_lb"]) / (df["mmu_ub"] - df["mmu_lb"])

    df.to_csv(out_df_file, sep=";")



# ----------------------------------------------------------------------------------------------------------------------

# Results with Data

# ----------------------------------------------------------------------------------------------------------------------


def prepare_offline_data():
    """
    read the original results from the offline MMR & MMU experiments (with data) and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'offline_data'
    file_str = version_str + "_final.csv"
    u0_type = 'positive_normed'

    out_df_file = f'/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{version_str}/{file_str}'
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/offline_mmr_data_0722_1.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/offline_mmr_data_0911_1.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/offline_mmr_data_0924_1_025.csv"
    # out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/offline_mmr_data_0924_2_000.csv"

    # # results file with mmr
    # mmr_output_file = f'/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/offline_data_mmr/static_experiment_20200218_154035.csv'

    # # results with random & mmu
    # mmu_output_file = f'/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/offline_data_mmu/static_experiment_20200219_082827.csv'

    # # results with random
    # random_output_file = f'/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/offline_data_random/static_experiment_20200220_140301.csv'

    # # ----------------------------------------------------------------------------------------------------------------------
    # # read results

    # # data file: output from experiments
    # df_1 = pd.read_csv(mmr_output_file, skiprows=1, delimiter=';')
    # df_2 = pd.read_csv(mmu_output_file, skiprows=1, delimiter=';')
    # df_3 = pd.read_csv(random_output_file, skiprows=1, delimiter=';')

    # df = pd.concat([df_1, df_2, df_3])
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210721_221510.csv"
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_20210909_210401.csv"
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_0_20210923_150236.csv"
    # output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_0_20210923_175527.csv"
    df = pd.read_csv(output_file_1, skiprows=1, delimiter=';')

    # there should be only one # items here
    num_items = df["num_items"].unique()
    print(num_items)

    assert len(num_items) == 1
    num_items = num_items[0]

    df["mmr_ub"] = None
    df["mmr_lb"] = None
    df["mmu_ub"] = None
    df["mmu_lb"] = None

    data_csv = "/Users/duncan/research/ActivePreferenceLearning/data/PolicyFeatures_RealData_HMIS-small_new.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_preprocessed_2_Robust_new25.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_20210906_preprocessed_final_Robust_25.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_20210922_preprocessed_final_Robust_all25.csv"
    items = get_data_items(
        data_csv,
        max_items=num_items,
        drop_cols=["IsInterpretable_int"],
        normalize_features=True,
    )

    mmu_lb = get_mmu_lb(items, u0_type)
    mmu_ub = get_mmu_ub(items, u0_type)

    # set the ub/lb cols
    df.loc[:, "mmu_ub"] = mmu_ub
    df.loc[:, "mmu_lb"] = mmu_lb

    mmr_lb = get_mmr_lb(items, u0_type)
    mmr_ub = get_mmr_ub(items, u0_type)
    print(mmr_ub)

    # set the ub/lb cols
    df.loc[:, "mmr_ub"] = mmr_ub
    df.loc[:, "mmr_lb"] = mmr_lb

    # calculate normalized objval for each method
    df["mmu_objval_normalized_new"] = None
    df["mmr_objval_normalized_new"] = None

    # for random, mmu and mmr objval col is populated
    # df.loc[df["method"] == "random", "mmu_objval_normalized_new"] = (df.loc[df["method"] == "random", "mmu_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)
    # df.loc[df["method"] == "random", "mmr_objval_normalized_new"] = (df.loc[df["method"] == "random", "mmr_objval"].astype(float) - mmr_lb) / (mmr_ub - mmr_lb)

    df.loc[df["method"] == "random", "mmu_objval_normalized_new"] = df.loc[df["method"] == "random", "mmu_objval"].astype(float) 
    df.loc[df["method"] == "random", "mmr_objval_normalized_new"] = df.loc[df["method"] == "random", "mmr_objval"].astype(float)

    # for mmu_heuristic/mmr_heuristic, only col method_objval is populated
    # df.loc[df["method"] == "mmu_heuristic", "mmu_objval_normalized_new"] = (df.loc[df["method"] == "mmu_heuristic", "method_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)
    # df.loc[df["method"] == "mmr_heuristic", "mmr_objval_normalized_new"] = (df[df["method"] == "mmr_heuristic"]["method_objval"].astype(float) - mmr_lb) / (mmr_ub - mmr_lb)
    df.loc[df["method"] == "mmu_heuristic", "mmu_objval_normalized_new"] = df.loc[df["method"] == "mmu_heuristic", "method_objval"].astype(float) 
    df.loc[df["method"] == "mmr_heuristic", "mmr_objval_normalized_new"] = df[df["method"] == "mmr_heuristic"]["method_objval"].astype(float) 

    df.to_csv(out_df_file, sep=";")


def prepare_offline_data_mmu():
    """
    read the original results from the offline MMR & MMU experiments (with data) and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'offline_data_mmu'
    file_str = version_str + "_final.csv"
    u0_type = 'positive_normed'

    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/offline_mmu_data_1016_1_000.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/offline_mmu_data_1016_1_025.csv"

    # # ----------------------------------------------------------------------------------------------------------------------
    # # read results

    # # data file: output from experiments
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_0_20211015_093025.csv"
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/static_experiment_0_20211015_090652.csv"
    df = pd.read_csv(output_file_1, skiprows=1, delimiter=';')

    # there should be only one # items here
    num_items = df["num_items"].unique()

    assert len(num_items) == 1
    num_items = num_items[0]

    df["mmr_ub"] = None
    df["mmr_lb"] = None
    df["mmu_ub"] = None
    df["mmu_lb"] = None

    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_20210906_preprocessed_final_Robust_25.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_20210922_preprocessed_final_Robust_all25.csv"
    items = get_data_items(
        data_csv,
        max_items=num_items,
        drop_cols=["IsInterpretable_int"],
        normalize_features=True,
    )

    mmu_lb = get_mmu_lb(items, u0_type)
    mmu_ub = get_mmu_ub(items, u0_type)

    # set the ub/lb cols
    df.loc[:, "mmu_ub"] = mmu_ub
    df.loc[:, "mmu_lb"] = mmu_lb

    mmr_lb = get_mmr_lb(items, u0_type)
    mmr_ub = get_mmr_ub(items, u0_type)

    # set the ub/lb cols
    df.loc[:, "mmr_ub"] = mmr_ub
    df.loc[:, "mmr_lb"] = mmr_lb

    # calculate normalized objval for each method
    df["mmu_objval_normalized_new"] = None
    df["mmr_objval_normalized_new"] = None

    # for random, mmu and mmr objval col is populated
    df.loc[df["method"] == "random", "mmu_objval_normalized_new"] = (df.loc[df["method"] == "random", "mmu_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)
    df.loc[df["method"] == "random", "mmr_objval_normalized_new"] = (df.loc[df["method"] == "random", "mmr_objval"].astype(float) - mmr_lb) / (mmr_ub - mmr_lb)

    # for mmu_heuristic/mmr_heuristic, only col method_objval is populated
    df.loc[df["method"] == "mmu_heuristic", "mmu_objval_normalized_new"] = (df.loc[df["method"] == "mmu_heuristic", "method_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)
    df.loc[df["method"] == "mmr_heuristic", "mmr_objval_normalized_new"] = (df[df["method"] == "mmr_heuristic"]["method_objval"].astype(float) - mmr_lb) / (mmr_ub - mmr_lb)

    df.to_csv(out_df_file, sep=";")


def prepare_online_data_mmr():
    """
    read the original results from the online MMR & MMU experiments (with data) and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'online_data_mmr'
    file_str = version_str + "_final.csv"
    u0_type = 'positive_normed'

    out_df_file = f'/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{version_str}/{file_str}'
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0718_1.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0911_1.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0915_1.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0922_1_veryold.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0923_1_025.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0923_1_000.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0925_1_025.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0925_2_000.csv"

    # results file
    mmr_output_file = f'/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{version_str}/adaptive_experiment_20200217_103901.csv'
    # v_2
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210706_182556.csv"
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210709_001708.csv"
    # output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210706_202910.csv"
    # output_file_3 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210706_205210.csv"

    # v_3_normalized_0714_1_0715_3
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210713_152830.csv"
    output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210713_152853.csv"
    output_file_3 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210713_153417.csv"
    output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210713_153556.csv"
    output_file_5 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210713_215454.csv"
    output_file_6 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210713_215502.csv"

    # v_3.5_normalized_0714_1_0715_4
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210713_224917.csv"

    # # v_4_normalized_0715_1
    # output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210714_222034.csv"
    # output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210715_135745.csv"
    # # v_5_unnormalized_0715_2
    # output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210715_150954.csv"
    # output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210715_152726.csv"
    # v_6_unnormalized_25new_nonneg_0715_1
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210715_212035.csv"
    output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210715_212118.csv"
    # v_6_unnormalized_40new_nonneg_0716_2
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210715_222946.csv"
    # v_7_unnormalized_25new_nonneg(new)_0717_1
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210717_192237.csv"
    output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210717_192254.csv"
    # v_8_unnormalized_41new_nonneg(new)_0718_1
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210717_232958.csv"
    # v_8_unnormalized_25_0911_1
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210910_153508.csv"
    # v_8_unnormalized_41_0915_1
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210914_150041.csv"
    # v__normalized_veryold_incon_0922_1
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_0_20210922_162004.csv"
    # v_10_normalized_latest_2.5incon_0923_1_25items
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_0_20210923_150221.csv"
    # v_11_normalized_latest_0incon_0923_1_25items
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_0_20210923_180610.csv"
    output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_0_20210924_201208.csv"
    # v_12_normalized_latest_025incon_0925_1_25items
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_0_20210925_153039.csv"
    # v_13_normalized_latest_0incon_0925_1_25items
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_0_20210925_153322.csv"
    # ----------------------------------------------------------------------------------------------------------------------
    # read results

    # data file: output from experiments
    # df = pd.read_csv(mmr_output_file, skiprows=1, delimiter=';')
    df = pd.read_csv(output_file_1, skiprows=1, delimiter=';')
    # df_2 = pd.read_csv(output_file_2, skiprows=1, delimiter=';')
    # df_3 = pd.read_csv(output_file_3, skiprows=1, delimiter=';')
    # df_4 = pd.read_csv(output_file_4, skiprows=1, delimiter=';')
    # df_5 = pd.read_csv(output_file_5, skiprows=1, delimiter=';')
    # df_6 = pd.read_csv(output_file_6, skiprows=1, delimiter=';')

    # df = df.append(df_2)
    # df = df.append(df_3)
    # df = df.append(df_4)
    # df = df.append(df_5)
    # df = df.append(df_6)

    # there should be only one # items here
    num_items = df["num_items"].unique()

    assert len(num_items) == 1
    num_items = num_items[0]

    df["mmr_ub"] = None
    df["mmr_lb"] = None
    df["mmu_ub"] = None
    df["mmu_lb"] = None

    data_csv = "/Users/duncan/research/ActivePreferenceLearning/data/PolicyFeatures_RealData_HMIS-small_new.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/DEBUG_folder/PolicyFeatures_RealData_LAHSA_normalized.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_preprocessed_2_Robust_new25.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_preprocessed_2_Robust_new41.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_20210906_preprocessed_final_Robust_41.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/PolicyFeatures_RealData_LAHSA_normalized.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_20210922_preprocessed_final_Robust_all25.csv"
    items = get_data_items(
        data_csv,
        max_items=num_items,
        drop_cols=["IsInterpretable_int"],
        normalize_features=True,
    )

    mmu_lb = get_mmu_lb(items, u0_type)
    mmu_ub = get_mmu_ub(items, u0_type)

    # set the ub/lb cols
    df.loc[:, "mmu_ub"] = mmu_ub
    df.loc[:, "mmu_lb"] = mmu_lb

    mmr_lb = get_mmr_lb(items, u0_type)
    mmr_ub = get_mmr_ub(items, u0_type)
    print(mmr_ub)

    # set the ub/lb cols
    df.loc[:, "mmr_ub"] = mmr_ub
    df.loc[:, "mmr_lb"] = mmr_lb

    # calculate normalized objval for each method
    # df["mmu_objval_normalized_new"] = None
    df["mmr_objval_normalized_new"] = None

    # for random, mmu and mmr objval col is populated
    # df.loc[df["method"] == "random", "mmu_objval_normalized_new"] = (df.loc[df["method"] == "random", "mmu_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)
    df.loc[:, "mmr_objval_normalized_new"] = (df[ "mmr_objval"].astype(float) - mmr_lb) / (mmr_ub - mmr_lb)

    # for mmu_heuristic/mmr_heuristic, only col method_objval is populated
    # df.loc[df["method"] == "mmu_heuristic", "mmu_objval_normalized_new"] = (df.loc[df["method"] == "mmu_heuristic", "method_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)
    df.loc[:, "true_regret_normalized_new"] = (df[ "true_regret"].astype(float) - mmr_lb) / (mmr_ub - mmr_lb)

    df.to_csv(out_df_file, sep=";")


def prepare_online_data_mmu():
    """
    read the original results from the online MMR & MMU experiments (with data) and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'online_data_mmu'
    file_str = version_str + "_final.csv"
    u0_type = 'positive_normed'

    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmu_data_0911_1.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmu_data_1016_2_000.csv"

    # results file
    mmr_output_file = f'/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{version_str}/adaptive_experiment_20200217_103901.csv'
    # v_8_unnormalized_25_1016_1
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_0_20211015_085037.csv"
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_0_20211015_093101.csv"
    # ----------------------------------------------------------------------------------------------------------------------
    # read results

    # data file: output from experiments
    # df = pd.read_csv(mmr_output_file, skiprows=1, delimiter=';')
    df = pd.read_csv(output_file_1, skiprows=1, delimiter=';')

    # there should be only one # items here
    num_items = df["num_items"].unique()

    assert len(num_items) == 1
    num_items = num_items[0]

    df["mmr_ub"] = None
    df["mmr_lb"] = None
    df["mmu_ub"] = None
    df["mmu_lb"] = None

    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_20210906_preprocessed_final_Robust_41.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_20210922_preprocessed_final_Robust_all25.csv"
    items = get_data_items(
        data_csv,
        max_items=num_items,
        drop_cols=["IsInterpretable_int"],
        normalize_features=True,
    )

    mmu_lb = get_mmu_lb(items, u0_type)
    mmu_ub = get_mmu_ub(items, u0_type)
    print(mmu_lb, mmu_ub)
    mmu_lb = min(df["mmu_objval"].astype(float))
    mmu_ub = max(df["mmu_objval"].astype(float))
    print(mmu_lb, mmu_ub)

    # set the ub/lb cols
    df.loc[:, "mmu_ub"] = mmu_ub
    df.loc[:, "mmu_lb"] = mmu_lb

    mmr_lb = get_mmr_lb(items, u0_type)
    mmr_ub = get_mmr_ub(items, u0_type)

    # set the ub/lb cols
    df.loc[:, "mmr_ub"] = mmr_ub
    df.loc[:, "mmr_lb"] = mmr_lb

    # calculate normalized objval for each method
    df["mmu_objval_normalized_new"] = None
    # df["mmr_objval_normalized_new"] = None

    # for random, mmu and mmr objval col is populated
    # df.loc[df["method"] == "random", "mmu_objval_normalized_new"] = (df.loc[df["method"] == "random", "mmu_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)
    df.loc[:, "mmu_objval_normalized_new"] = (df[ "mmu_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)

    # for mmu_heuristic/mmr_heuristic, only col method_objval is populated
    # df.loc[df["method"] == "mmu_heuristic", "mmu_objval_normalized_new"] = (df.loc[df["method"] == "mmu_heuristic", "method_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)
    df.loc[:, "true_u_normalized_new"] = (df[ "true_u"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)

    df.to_csv(out_df_file, sep=";")


# ----------------------------------------------------------------------------------------------------------------------

# Results comparing mmu & mmr

# ----------------------------------------------------------------------------------------------------------------------


def prepare_offline_mmu_mmr_comparison():
    """
    read the original results from the offline MMR/MMU experiments and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'compare_mmr_mmu'
    u0_type = 'box'

    out_df_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/compare_mmr_mmu_final.csv'.format(version_str)
    # results file (from static elicitation experiments)
    output_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/static_experiment_20200123_101913.csv'.format(version_str)

    # ----------------------------------------------------------------------------------------------------------------------
    # read results

    # data file: output from experiments
    df = pd.read_csv(output_file, skiprows=1, delimiter=';')

    # find all cols where we will need to calculate different ub/lb values
    expt_cols = ["problem_seed", "num_items", "num_features", "item_sphere_size"]

    # create an id for each of these
    df["expt_params"] = df[expt_cols].apply(lambda x: tuple(x), axis=1)

    param_sets = df["expt_params"].unique()
    # param_sets = df[expt_cols].drop_duplicates().copy()

    df["mmr_ub"] = None
    df["mmr_lb"] = None
    df["mmu_ub"] = None
    df["mmu_lb"] = None

    # get the ub/lb values for each case
    for params in param_sets:
        # unpack...
        problem_seed, num_items, num_features, item_sphere_size = params
        print(f"params: {params}")
        # generate items
        items = generate_items(
            int(num_features),
            int(num_items),
            item_sphere_size=item_sphere_size,
            seed=int(problem_seed),
        )

        mmu_lb = get_mmu_lb(items, u0_type)
        mmu_ub = get_mmu_ub(items, u0_type)
        #
        # # set the ub/lb cols
        df.loc[df["expt_params"] == params, "mmu_ub"] = mmu_ub
        df.loc[df["expt_params"] == params, "mmu_lb"] = mmu_lb

        mmr_lb = get_mmr_lb(items, u0_type)
        mmr_ub = get_mmr_ub(items, u0_type)

        # set the ub/lb cols
        df.loc[df["expt_params"] == params, "mmr_ub"] = mmr_ub
        df.loc[df["expt_params"] == params, "mmr_lb"] = mmr_lb

    df["mmr_objval_normalized_new"] = (df["mmr_objval"] - df["mmr_lb"]) / (df["mmr_ub"] - df["mmr_lb"])
    df["mmu_objval_normalized_new"] = (df["mmu_objval"] - df["mmu_lb"]) / (df["mmu_ub"] - df["mmu_lb"])

    df.to_csv(out_df_file, sep=";")



# ----------------------------------------------------------------------------------------------------------------------

# Results for timing

# ----------------------------------------------------------------------------------------------------------------------


def prepare_offline_timing():
    """
    read the original results from the timing experiments and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'timing'
    u0_type = 'box'

    out_df_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/timing_final.csv'.format(version_str)
    # results file (from static elicitation experiments)
    output_file_1 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/timing_experiment_20200131_193502.csv'.format(version_str)
    output_file_2 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/timing_experiment_20200131_194025.csv'.format(version_str)
    output_file_3 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/timing_experiment_20200127_122033.csv'.format(version_str)
    output_file_4 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/timing_experiment_20200202_154516.csv'.format(version_str)
    output_file_5 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/timing_experiment_20200125_141005.csv'.format(version_str)
    output_file_6 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/timing_experiment_20200220_201741.csv'.format(version_str)

    # ----------------------------------------------------------------------------------------------------------------------
    # read results

    # data file: output from experiments
    df_1 = pd.read_csv(output_file_1, skiprows=1, delimiter=';')
    df_2 = pd.read_csv(output_file_2, skiprows=1, delimiter=';')
    df_3 = pd.read_csv(output_file_3, skiprows=1, delimiter=';')
    df_4 = pd.read_csv(output_file_4, skiprows=1, delimiter=';')
    df_5 = pd.read_csv(output_file_5, skiprows=1, delimiter=';')
    df_6 = pd.read_csv(output_file_6, skiprows=1, delimiter=';')

    df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6])
    # find all cols where we will need to calculate different ub/lb values
    expt_cols = ["problem_seed", "num_items", "num_features", "item_sphere_size"]

    # create an id for each of these
    df["expt_params"] = df[expt_cols].apply(lambda x: tuple(x), axis=1)

    param_sets = df["expt_params"].unique()
    # param_sets = df[expt_cols].drop_duplicates().copy()

    # df["mmr_ub"] = None
    # df["mmr_lb"] = None
    df["mmu_ub"] = None
    df["mmu_lb"] = None

    # get the ub/lb values for each case
    for params in param_sets:
        # unpack...
        problem_seed, num_items, num_features, item_sphere_size = params
        print(f"params: {params}")
        # generate items
        items = generate_items(
            int(num_features),
            int(num_items),
            item_sphere_size=item_sphere_size,
            seed=int(problem_seed),
        )

        mmu_lb = get_mmu_lb(items, u0_type)
        mmu_ub = get_mmu_ub(items, u0_type)
        #
        # # set the ub/lb cols
        df.loc[df["expt_params"] == params, "mmu_ub"] = mmu_ub
        df.loc[df["expt_params"] == params, "mmu_lb"] = mmu_lb

        # mmr_lb = get_mmr_lb(items, u0_type)
        # mmr_ub = get_mmr_ub(items, u0_type)
        #
        # # set the ub/lb cols
        # df.loc[df["expt_params"] == params, "mmr_ub"] = mmr_ub
        # df.loc[df["expt_params"] == params, "mmr_lb"] = mmr_lb

    # df["mmr_objval_normalized_new"] = (df["mmr_objval"] - df["mmr_lb"]) / (df["mmr_ub"] - df["mmr_lb"])
    df["mmu_objval_normalized_new"] = (df["mmu_objval"] - df["mmu_lb"]) / (df["mmu_ub"] - df["mmu_lb"])

    df.to_csv(out_df_file, sep=";")


def finalize_gamma_comparison():
    """
    read the original results from the online MMR & MMU experiments (with data) and correctly normalize the results.

    for bookkeeping only.
    """

    version_str = 'online_data_mmr'
    file_str = version_str + "_final.csv"
    u0_type = 'positive_normed'

    out_df_file = f'/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{version_str}/{file_str}'
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/robust_to_inconsistency/comparison_online_0923_normalized.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/robust_to_inconsistency/comparison_offline_0924_normalized.csv"
    out_df_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/robust_to_inconsistency/comparison_online_0925_normalized.csv"


    # results file
    mmr_output_file = f'/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{version_str}/adaptive_experiment_20200217_103901.csv'
    # v_2
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/robust_to_inconsistency/comparison_online_0923.csv"
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/robust_to_inconsistency/comparison_offline_0924.csv"
    output_file_1 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/robust_to_inconsistency/comparison_online_0925.csv"

    # ----------------------------------------------------------------------------------------------------------------------
    # read results

    # data file: output from experiments
    # df = pd.read_csv(mmr_output_file, skiprows=1, delimiter=';')
    df = pd.read_csv(output_file_1)

    num_items = 20

    df["mmr_ub"] = None
    df["mmr_lb"] = None
    df["mmu_ub"] = None
    df["mmu_lb"] = None

    data_csv = "/Users/duncan/research/ActivePreferenceLearning/data/PolicyFeatures_RealData_HMIS-small_new.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/DEBUG_folder/PolicyFeatures_RealData_LAHSA_normalized.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_preprocessed_2_Robust_new25.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_preprocessed_2_Robust_new41.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_20210906_preprocessed_final_Robust_41.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/PolicyFeatures_RealData_LAHSA_normalized.csv"
    data_csv = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/AdultHMIS_20210922_preprocessed_final_Robust_all25.csv"
    items = get_data_items(
        data_csv,
        max_items=num_items,
        drop_cols=["IsInterpretable_int"],
        normalize_features=True,
    )

    mmu_lb = get_mmu_lb(items, u0_type)
    mmu_ub = get_mmu_ub(items, u0_type)

    # set the ub/lb cols
    df.loc[:, "mmu_ub"] = mmu_ub
    df.loc[:, "mmu_lb"] = mmu_lb

    mmr_lb = get_mmr_lb(items, u0_type)
    mmr_ub = get_mmr_ub(items, u0_type)

    col = df.columns.to_list()
    col = col[1:16]

    # for random, mmu and mmr objval col is populated
    # df.loc[df["method"] == "random", "mmu_objval_normalized_new"] = (df.loc[df["method"] == "random", "mmu_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)
    print(df.loc[:, col])
    df.loc[:, col] = (df.loc[:, col].astype(float) - mmr_lb) / (mmr_ub - mmr_lb)

    # # for mmu_heuristic/mmr_heuristic, only col method_objval is populated
    # # df.loc[df["method"] == "mmu_heuristic", "mmu_objval_normalized_new"] = (df.loc[df["method"] == "mmu_heuristic", "method_objval"].astype(float) - mmu_lb) / (mmu_ub - mmu_lb)
    # df.loc[:, "true_regret_normalized_new"] = (df[ "true_regret"].astype(float) - mmr_lb) / (mmr_ub - mmr_lb)
    df = df.round(4)
    df.to_csv(out_df_file)


# prepare_online_mmu()
# prepare_online_mmr()
# prepare_offline_mmu()
# prepare_offline_mmr()
# prepare_online_data_mmr()
# prepare_offline_data()
prepare_online_data_mmu()
# prepare_offline_data_mmu()
# finalize_gamma_comparison()