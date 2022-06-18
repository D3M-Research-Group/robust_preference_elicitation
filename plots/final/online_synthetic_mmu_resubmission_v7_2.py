# plot results
# changes from v6: added probpoly results with error
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set fonts
plt.rcParams["font.family"] = "Times New Roman"

plt.rc('figure', titlesize=10)  # fontsize of the figure title
plt.rc('font', size=10)

# figure size
fig_height = 5
fig_width = 7

label_bbox_props = {
    'pad': 0,
    'alpha': 0.2,
    'color': 'gray',
    'linewidth': 0.2,
    'boxstyle': "round,pad=0.3",
}

# overall marker size
markersize = 5

# plot style for optimal method
alpha = 0.3
opt_style = {'color': 'r',
             'linestyle': '-',
             'marker': 'o',
             'markersize': markersize
             }
opt_style_fill = {'color': 'r',
                   'alpha': alpha,
                   'linewidth': 0,
                   }

acac_style = {'color': 'g',
             'linestyle': '-',
             'marker': 'x',
             'markersize': markersize
             }
acac_style_fill = {'color': 'g',
                   'alpha': alpha,
                   'linewidth': 0,
                   }

acmm_style = {'color': 'g',
             'linestyle': '-',
             'marker': 'v',
             'markersize': markersize
             }
acmm_style_fill = {'color': 'g',
                   'alpha': alpha,
                   'linewidth': 0,
                   }

randmm_style = {'color': 'b',
             'linestyle': '-',
             'marker': '+',
             'markersize': markersize
             }
randmm_style_fill = {'color': 'b',
                   'alpha': alpha,
                   'linewidth': 0,
                   }
ellipsoidal_style = {'color': 'purple',
             'linestyle': '-.',
             'marker': '^',
             'markersize': markersize
             }

poly_style = {'color': 'black',
             'linestyle': '-.',
             'marker': 'v',
             'markersize': markersize
             }

probpoly_style = {'color': 'black',
             'linestyle': '-.',
             'marker': 'x',
             'markersize': markersize
             }

xlim = [0.95, 10]
xticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# ----------------------------------------------------------------------------------------------------------------------
# define files and directories
version_str = 'online_synthetic_mmu_resubmission_v7'

# objtype must be 'mmu' or 'mmr
objtype = 'mmu'
assert objtype in ['mmr', 'mmu']

if objtype == 'mmr':
    caption_objtype = 'MMR'
if objtype == 'mmu':
    caption_objtype = 'MMU'


# directory where images will be saved
# img_dir = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/fig/{}/'.format(version_str)
img_dir = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/"

# results files
# output_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/resubmission/adaptive_synthetic_mmu_new_error/adaptive_experiment_20201202_125032.csv'

# correct_v1
output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210610_172523_2.csv"
output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210611_211604.csv"
output_file_3 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210612_164640.csv"

# v_2
output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210614_120056.csv"
output_file_2 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210615_110720.csv"
output_file_3 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210611_211604.csv"
# output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210613_161123.csv"
# output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210617_161048.csv"
output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210620_223228.csv"
output_file_5 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210620_160517.csv"
output_file_6 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210620_235629.csv"
output_file_7 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210621_152309.csv"
output_file_8 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210621_223816.csv"

# output_file_3 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210613_161123.csv"
# output_file_4 = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/adaptive_experiment_20210614_012942.csv"

# output_file_zero_error = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/resubmission/adaptive_synthetic_mmu_new_gamma/adaptive_experiment_20201115_215139.csv'

# output_file_poly_zero_error = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/resubmission/adaptive_synthetic_mmu_poly/adaptive_experiment_20201222_145814.csv'

# output_file_probpoly_zero_error = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/resubmission/adaptive_synthetic_mmu_probpoly/adaptive_experiment_20201222_150429.csv'

# output_file_probpoly_error = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/resubmission/adaptive_synthetic_mmu_probpoly_error/adaptive_experiment_20201229_190237.csv'


# ----------------------------------------------------------------------------------------------------------------------
# read results

# data file: output from experiments
df = pd.read_csv(output_file, skiprows=1, delimiter=';')
df2 = pd.read_csv(output_file_2, skiprows=1, delimiter=';')
df3 = pd.read_csv(output_file_3, skiprows=1, delimiter=';')
df4 = pd.read_csv(output_file_4, skiprows=1, delimiter=';')
df5 = pd.read_csv(output_file_5, skiprows=1, delimiter=';')
df6 = pd.read_csv(output_file_6, skiprows=1, delimiter=';')
df7 = pd.read_csv(output_file_7, skiprows=1, delimiter=';')
df8 = pd.read_csv(output_file_8, skiprows=1, delimiter=';')
df = df.append(df2)
df = df.append(df3)
df = df.append(df4)
df = df.append(df5)
df = df.append(df6)
df = df.append(df7)
df = df.append(df8)

# df_zero_error = pd.read_csv(output_file_zero_error, skiprows=1, delimiter=';')

# # change gamma -> sigma and drop all sigma > 0 for zero error
# df_zero_error.rename(columns={"gamma": "sigma"}, inplace=True)
# df_zero_error = df_zero_error.loc[df_zero_error["sigma"] == 0.0]

# df = df.append(df_zero_error)

# # poly results with no error
# df_zero_error_poly = pd.read_csv(output_file_poly_zero_error, skiprows=1, delimiter=';')

# # change gamma -> sigma
# df_zero_error_poly.rename(columns={"gamma": "sigma"}, inplace=True)

# df = df.append(df_zero_error_poly)

# df_zero_error_probpoly = pd.read_csv(output_file_probpoly_zero_error, skiprows=1, delimiter=';')
# df_zero_error_probpoly.rename(columns={"gamma": "sigma"}, inplace=True)

# df = df.append(df_zero_error_probpoly)


# # probpoly results with error
# df_probpoly_error = pd.read_csv(output_file_probpoly_error, skiprows=1, delimiter=';')

# # change gamma -> sigma
# df_probpoly_error.rename(columns={"gamma": "sigma"}, inplace=True)

# df = df.append(df_probpoly_error)

# -------------------------------------------------------------------------------------------------------------
#  plots of objval
# ----------------------------------------------------------------------------------------------------------------------

# first row: increase num. items
# second row: increase num. features
# third row: increase gamma

# note that there should be only one random sample
def plot_objvals(df, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=None, legend=False):

    # plot static heuristic
    elicitation_method = 'maximin'
    recommendation_method = 'maximin'
    fill_style = opt_style_fill

    if plot_col == 'true_rank':
        val_func = np.max
    else:
        val_func = np.min

    val_list = [val_func(df[(df['elicitation_method'] == elicitation_method)
                   & (df['recommendation_method'] == recommendation_method)
                  & (df['K'] == k)][plot_col].values) for k in k_list]
    ax.plot(k_list, val_list, label='{}+{}'.format(caption_objtype, caption_objtype), **opt_style)

    # # plot fill between min/max
    # val_list_min = [df[(df['elicitation_method'] == elicitation_method)
    #                & (df['recommendation_method'] == recommendation_method)
    #                & (df['K'] == k)][plot_col].min() for k in k_list]
    # val_list_max = [df[(df['elicitation_method'] == elicitation_method)
    #                & (df['recommendation_method'] == recommendation_method)
    #                & (df['K'] == k)][plot_col].max() for k in k_list]
    # ax.fill_between(k_list, val_list_min, y2=val_list_max, **fill_style)

    # plot AC + AC
    # elicitation_method = 'AC'
    elicitation_method = 'robust_AC'
    # recommendation_method = 'AC'
    recommendation_method = 'robust_AC'
    fill_style = acac_style_fill


    val_list = [val_func(df[(df['elicitation_method'] == elicitation_method)
                   & (df['recommendation_method'] == recommendation_method)
                   & (df['K'] == k)][plot_col]) for k in k_list]
    ax.plot(k_list, val_list, label='AC+AC', **acac_style)

    # # plot fill between min/max
    # val_list_min = [df[(df['elicitation_method'] == elicitation_method)
    #                & (df['recommendation_method'] == recommendation_method)
    #                & (df['K'] == k)][plot_col].min() for k in k_list]
    # val_list_max = [df[(df['elicitation_method'] == elicitation_method)
    #                & (df['recommendation_method'] == recommendation_method)
    #                & (df['K'] == k)][plot_col].max() for k in k_list]
    # ax.fill_between(k_list, val_list_min, y2=val_list_max, **fill_style)

    # plot AC + maximin
    elicitation_method = 'AC'
    recommendation_method = 'maximin'
    fill_style = acmm_style_fill

    val_list = [val_func(df[(df['elicitation_method'] == elicitation_method)
                   & (df['recommendation_method'] == recommendation_method)
                   & (df['K'] == k)][plot_col]) for k in k_list]
    ax.plot(k_list, val_list, label='AC+{}'.format(caption_objtype), **acmm_style)

    # # plot fill between min/max
    # val_list_min = [df[(df['elicitation_method'] == elicitation_method)
    #                    & (df['recommendation_method'] == recommendation_method)
    #                    & (df['K'] == k)][plot_col].min() for k in k_list]
    # val_list_max = [df[(df['elicitation_method'] == elicitation_method)
    #                    & (df['recommendation_method'] == recommendation_method)
    #                    & (df['K'] == k)][plot_col].max() for k in k_list]
    # ax.fill_between(k_list, val_list_min, y2=val_list_max, **fill_style)

    # plot random + maximin
    elicitation_method = 'random'
    recommendation_method = 'maximin'
    fill_style = randmm_style_fill

    val_list = [val_func(df[(df['elicitation_method'] == elicitation_method)
                   & (df['recommendation_method'] == recommendation_method)
                   & (df['K'] == k)][plot_col]) for k in k_list]
    ax.plot(k_list, val_list, label='RAND+{}'.format(caption_objtype), **randmm_style)

    # # plot fill between min/max
    # val_list_min = [df[(df['elicitation_method'] == elicitation_method)
    #                    & (df['recommendation_method'] == recommendation_method)
    #                    & (df['K'] == k)][plot_col].min() for k in k_list]
    # val_list_max = [df[(df['elicitation_method'] == elicitation_method)
    #                    & (df['recommendation_method'] == recommendation_method)
    #                    & (df['K'] == k)][plot_col].max() for k in k_list]
    # ax.fill_between(k_list, val_list_min, y2=val_list_max, **fill_style)

    # plot ellipsoidal + mmr/mmu
    elicitation_method = 'ellipsoidal'
    recommendation_method = 'mean'

    val_list = [val_func(df[(df['elicitation_method'] == elicitation_method)
                   & (df['recommendation_method'] == recommendation_method)
                   & (df['K'] == k)][plot_col]) for k in k_list]
    ax.plot(k_list, val_list, label='ellip+{}'.format(caption_objtype), **ellipsoidal_style)


    # plot polyhedral + AC
    elicitation_method = 'polyhedral'
    recommendation_method = 'AC'

    val_list = [val_func(df[(df['elicitation_method'] == elicitation_method)
                   & (df['recommendation_method'] == recommendation_method)
                   & (df['K'] == k)][plot_col]) for k in k_list]
    ax.plot(k_list, val_list, label='poly+{}'.format(caption_objtype), **poly_style)

    # plot prob-polyhedral + AC
    elicitation_method = 'probpoly'
    recommendation_method = 'AC'

    val_list = [val_func(df[(df['elicitation_method'] == elicitation_method)
                   & (df['recommendation_method'] == recommendation_method)
                   & (df['K'] == k)][plot_col]) for k in k_list]
    ax.plot(k_list, val_list, label='probpoly+{}'.format(caption_objtype), **probpoly_style)


    # ticks and gridlines
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.set_xticks(xticks)
    ax.grid(linestyle='--', linewidth='0.2', color='black', which='major')

    ax.tick_params(
        axis='y',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        left=True,
        right=False,
        labelleft=True,
    )

    # set axis limits
    # ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    # add a plot label
    label_x = 1.03
    label_y = 0.5
    ax.text(label_x, label_y, plot_label,
            horizontalalignment='left',
            verticalalignment='center',
            rotation=-90,
            backgroundcolor='gray',
            bbox=label_bbox_props,
            transform=ax.transAxes,
            )
    if legend:
        ax.legend()

# ---------------
# Plot objective value
# ---------------


# which column to plot data from
plot_col = '{}_objval_normalized'.format(objtype)
k_list = [k for k in list(sorted(df['K'].unique())) if k > 0]

# get all num features, num items, gamma
num_features_list = list(sorted(df['num_features'].unique()))
num_items_list = list(sorted(df['num_items'].unique()))
gamma_list = list(sorted(df['sigma'].unique()))

num_cols = max(len(num_features_list), len(num_items_list), len(gamma_list))
num_rows = 3

if objtype == 'mmu':
    ylim = [0, 0.55]
    yticks = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
if objtype == 'mmu':
    ylim = [0.45, 1.05]
    yticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

fig, axs = plt.subplots(
        num_rows, num_cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
)

letter_dict = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g',
    7: 'h',
    8: 'i',
}
img_ind = 0

# plot first row: num items
i_row = 0
for i_col in range(len(num_items_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[1]
    num_items = num_items_list[i_col]
    gamma = gamma_list[0]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    # show_xlabels = False
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = None
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks)

# second first row: num num_features
i_row = 1
for i_col in range(len(num_features_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[i_col]
    num_items = num_items_list[1]
    gamma = gamma_list[0]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    # show_xlabels = False
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = None
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks)

# second first row: num num_features
i_row = 2
for i_col in range(len(gamma_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[1]
    num_items = num_items_list[1]
    gamma = gamma_list[i_col]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    legend = False
    # if i_col == 2:
    #     legend = True
    # show_xlabels = True
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = None
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks, legend=legend)

    ax.set_xlabel("K")

# set a single ylabel
if objtype == 'mmu':
    fig.text(0.0, 0.5, "Worst-Case Utility of Recommended Item",
             ha='center',
             va='center',
             rotation='vertical')
if objtype == 'mmr':
    fig.text(0.0, 0.5, "Worst-Case Regret of Recommended Item",
             ha='center',
             va='center',
             rotation='vertical')

plt.tight_layout()

plt.savefig(os.path.join(img_dir, 'online_objval_{}.pdf'.format(version_str)), bbox_inches='tight')


# ---------------
# Plot true objective value
# ---------------

# which column to plot data from
if objtype == 'mmu':
    plot_col = 'true_u_normalized'.format(objtype)
if objtype == 'mmr':
    plot_col = 'true_regret_normalized'.format(objtype)
    # plot_col = 'true_regret'.format(objtype)

k_list = [k for k in list(sorted(df['K'].unique())) if k > 0]

# get all num features, num items, gamma
num_features_list = list(sorted(df['num_features'].unique()))
num_items_list = list(sorted(df['num_items'].unique()))
gamma_list = list(sorted(df['sigma'].unique()))

num_cols = max(len(num_features_list), len(num_items_list), len(gamma_list))
num_rows = 3

ylim = [0, 1.05]
yticks = [0., 0.2, 0.4, 0.6, 0.8, 1.]

fig, axs = plt.subplots(
        num_rows, num_cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
)

letter_dict = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g',
    7: 'h',
    8: 'i',
}
img_ind = 0

# plot first row: num items
i_row = 0
for i_col in range(len(num_items_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[1]
    num_items = num_items_list[i_col]
    gamma = gamma_list[0]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    # show_xlabels = False
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = None
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks)

# second first row: num num_features
i_row = 1
for i_col in range(len(num_features_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[i_col]
    num_items = num_items_list[1]
    gamma = gamma_list[0]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    # show_xlabels = False
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = None
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks)

# second first row: num num_features
i_row = 2
for i_col in range(len(gamma_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[1]
    num_items = num_items_list[1]
    gamma = gamma_list[i_col]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    legend = False
    # if i_col == 2:
    #     legend = True
    # show_xlabels = True
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = None
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks, legend=legend)

    ax.set_xlabel("K")

# set a single ylabel
if objtype == 'mmu':
    fig.text(0.0, 0.5, "Utility of Recommended Item (worst-case over all agents)",
             ha='center',
             va='center',
             rotation='vertical')
if objtype == 'mmr':
    fig.text(0.0, 0.5, "Regret of Recommended Item (worst-case over all agents)",
             ha='center',
             va='center',
             rotation='vertical')
plt.tight_layout()

plt.savefig(os.path.join(img_dir, 'online_true_objval_{}.pdf'.format(version_str)), bbox_inches='tight')


# ---------------
# Plot objective value
# ---------------

# which column to plot data from
if objtype == 'mmu':
    plot_col = 'mmu_objval_normalized'.format(objtype)
if objtype == 'mmr':
    plot_col = 'mmr_objval_normalized'.format(objtype)

k_list = [k for k in list(sorted(df['K'].unique())) if k > 0]

# get all num features, num items, gamma
num_features_list = list(sorted(df['num_features'].unique()))
num_items_list = list(sorted(df['num_items'].unique()))
gamma_list = list(sorted(df['sigma'].unique()))

num_cols = max(len(num_features_list), len(num_items_list), len(gamma_list))
num_rows = 3

ylim = [0, 1.05]
yticks = [0., 0.2, 0.4, 0.6, 0.8, 1.]

fig, axs = plt.subplots(
        num_rows, num_cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
)

letter_dict = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g',
    7: 'h',
    8: 'i',
}
img_ind = 0

# plot first row: num items
i_row = 0
for i_col in range(len(num_items_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[1]
    num_items = num_items_list[i_col]
    gamma = gamma_list[0]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    # show_xlabels = False
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = None
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks)

# second first row: num num_features
i_row = 1
for i_col in range(len(num_features_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[i_col]
    num_items = num_items_list[1]
    gamma = gamma_list[0]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    # show_xlabels = False
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = None
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks)

# second first row: num num_features
i_row = 2
for i_col in range(len(gamma_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[1]
    num_items = num_items_list[1]
    gamma = gamma_list[i_col]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    legend = False
    # if i_col == 2:
    #     legend = True
    # show_xlabels = True
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = None
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks, legend=legend)

    ax.set_xlabel("K")

# set a single ylabel
if objtype == 'mmu':
    fig.text(0.0, 0.5, "Worst-Case Utility of Recommended Item (worst-case over all agents)",
             ha='center',
             va='center',
             rotation='vertical')
if objtype == 'mmr':
    fig.text(0.0, 0.5, "Worst-Case Regret of Recommended Item (worst-case over all agents)",
             ha='center',
             va='center',
             rotation='vertical')
plt.tight_layout()

plt.savefig(os.path.join(img_dir, 'online_objval_{}.pdf'.format(version_str)), bbox_inches='tight')



# ---------------
# Plot rank
# ---------------

# which column to plot data from
plot_col = 'true_rank'
rank_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

k_list = [k for k in list(sorted(df['K'].unique())) if k > 0]

# get all num features, num items, gamma
num_features_list = list(sorted(df['num_features'].unique()))
num_items_list = list(sorted(df['num_items'].unique()))
gamma_list = list(sorted(df['sigma'].unique()))

num_cols = max(len(num_features_list), len(num_items_list), len(gamma_list))
num_rows = 3

fig, axs = plt.subplots(
        num_rows, num_cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=False,
)

letter_dict = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g',
    7: 'h',
    8: 'i',
}
img_ind = 0

# plot first row: num items
i_row = 0
for i_col in range(len(num_items_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[1]
    num_items = num_items_list[i_col]
    gamma = gamma_list[0]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    # show_xlabels = False
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = [i for i in rank_list if i <= num_items]
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks)
    ax.set_ylim([num_items + 0.5, 0.5])

# second first row: num num_features
i_row = 1
for i_col in range(len(num_features_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[i_col]
    num_items = num_items_list[1]
    gamma = gamma_list[0]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    # show_xlabels = False
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = [i for i in rank_list if i <= num_items]
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks)
    ax.set_ylim([num_items + 0.5, 0.5])

# second first row: num num_features
i_row = 2
for i_col in range(len(gamma_list)):
    ax = axs[i_row, i_col]

    num_features = num_features_list[1]
    num_items = num_items_list[1]
    gamma = gamma_list[i_col]
    df_tmp = df[
        (df['num_features'] == num_features)
        & (df['num_items'] == num_items)
        & (df['sigma'] == gamma)
    ]

    legend = False
    # if i_col == 2:
    #     legend = True
    # show_xlabels = True
    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    yticks = [i for i in rank_list if i <= num_items]
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=yticks, legend=legend)

    ax.set_ylim([num_items + 0.5, 0.5])
    ax.set_xlabel("K")

# set a single ylabel
fig.text(0.0, 0.5, "Rank of Recommended Item (Worst-case over all agents)",
         ha='center',
         va='center',
         rotation='vertical')

plt.tight_layout()

plt.savefig(os.path.join(img_dir, 'online_rank_{}.pdf'.format(version_str)), bbox_inches='tight')