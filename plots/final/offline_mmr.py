# plot results
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set fonts
from matplotlib.ticker import NullFormatter, LogLocator, FuncFormatter

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
    # 'linecolor': 'black',
    'boxstyle': "round,pad=0.3",
}

# overall marker size
markersize = 5

# plot style for optimal method
opt_style = {'color': 'r',
             'linestyle': '-',
             'marker': 'o',
             'markersize': markersize
             }

# plot style for random (median)
rand_style_med = {'color': 'b',
                  'linestyle': '--',
                  'marker': '+',
                  'markersize': markersize
                  }

# random (fill)
rand_style_fill = {'color': 'b',
                   'alpha': 0.3,
                   'linewidth': 0,
                   }

xlim = [0.95, 10]
xticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# the data for these plots are produced with the following function call:
# args are:
# Namespace(DEBUG=False, gamma=[0.0, 0.05, 0.1], max_K=10, num_features=[10, 5, 20], num_items=[40, 20, 60],
# num_random_samples=50, output_dir='/home/dmcelfre/RobustActivePreferenceLearning_output/static_experiments/', problem_seed=0)

# ----------------------------------------------------------------------------------------------------------------------
# define files and directories

version_str = 'offline_mmr_final'

# objtype must be 'mmu' or 'mmr
objtype = 'mmr'
assert objtype in ['mmr', 'mmu']

# directory where images will be saved
img_dir = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/fig/{}/'.format(version_str)
img_dir = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/"

# # results file (from static elicitation experiments)
# output_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/static_experiment_20200116_144436.csv'.format(version_str)
#
# # updated results, with 40 features, 10 items, and gamma = 0.2
# output_file_2 = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{}/static_experiment_20200118_134948.csv'.format(version_str)
#
#
# # ----------------------------------------------------------------------------------------------------------------------
# # read results
#
# # data file: output from experiments
# df = pd.read_csv(output_file, skiprows=1, delimiter=';')
# df_2 = pd.read_csv(output_file_2, skiprows=1, delimiter=';')
#
# df = df.append(df_2)

# data_file = f"/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{version_str}/offline_mmr_final.csv"
# df = pd.read_csv(data_file, delimiter=';')
output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/offline_mmr_final.csv"
df = pd.read_csv(output_file, delimiter=';')
# -------------------------------------------------------------------------------------------------------------
#  plots of objval
# ----------------------------------------------------------------------------------------------------------------------

# first row: increase num. items
# second row: increase num. features
# third row: increase gamma

# note that there should be only one random sample
def plot_objvals(df, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks):

    # plot static heuristic
    val_list = [df[(df['method'] == '{}_heuristic'.format(objtype)) & (df['K'] == k)][plot_col].values[0] for k in k_list]
    ax.plot(k_list, val_list, label='OPT-hstc', **opt_style)

    # plot median val of random
    val_list = [df[(df['method'] == 'random') & (df['K'] == k)][plot_col].median() for k in k_list]
    ax.plot(k_list, val_list, label='RAND', **rand_style_med)

    # plot fill between min/max of random
    val_list_min = [df[(df['method'] == 'random') & (df['K'] == k)][plot_col].min() for k in k_list]
    val_list_max = [df[(df['method'] == 'random') & (df['K'] == k)][plot_col].max() for k in k_list]
    ax.fill_between(k_list, val_list_min, y2=val_list_max, **rand_style_fill)

    # ticks and gridlines
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
    ax.set_ylim(ylim)
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

# which column to plot data from
plot_col = '{}_objval_normalized_new'.format(objtype)
# plot_col = '{}_normalized_objval'.format(objtype)
k_list = [k for k in list(sorted(df['K'].unique())) if k > 0]

# get all num features, num items, gamma
num_features_list = list(sorted(df['num_features'].unique()))
num_items_list = list(sorted(df['num_items'].unique()))
gamma_list = [0.0, 0.1, 0.2]
gamma_list = [0.0, 1.0, 2.0]
# gamma_list = list(sorted(df['gamma'].unique()))

num_cols = max(len(num_features_list), len(num_items_list), len(gamma_list))
num_rows = 3

if objtype == 'mmu':
    ylim = [0, 1.05]
    yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
if objtype == 'mmr':
    ylim = [-0.01, 1.05]
    yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
        & (df['gamma'] == gamma)
    ]

    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks)

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
        & (df['gamma'] == gamma)
    ]

    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks)

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
        & (df['gamma'] == gamma)
    ]

    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    plot_objvals(df_tmp, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks)

    ax.set_xlabel("K")

# set a single ylabel
if objtype == 'mmu':
    fig.text(0.0, 0.5, "Normalized Worst-Case Utility",
             ha='center',
             va='center',
             rotation='vertical')
if objtype == 'mmr':
    fig.text(0.0, 0.5, "Normalized Worst-Case Regret",
             ha='center',
             va='center',
             rotation='vertical')

plt.tight_layout()

plt.savefig(os.path.join(img_dir, 'objval_{}.pdf'.format(version_str)), bbox_inches='tight')

# -------------------------------------------------------------------------------------------------------------
#  plots of runtime
# ----------------------------------------------------------------------------------------------------------------------

# first row: increase num. items
# second row: increase num. features
# third row: increase gamma

# which column to plot data from
k_list = [k for k in list(sorted(df['K'].unique())) if k > 0]

# note that there should be only one random sample
def plot_runtimes(df, ax, k_list, plot_label, xlim, xticks):

    # plot static heuristic
    time_list = [df[(df['method'] == '{}_heuristic'.format(objtype)) & (df['K'] == k)]['solve_time'].values[0] for k in k_list]
    val_list = np.cumsum(time_list)
    ax.plot(k_list, val_list, label='OPT-hstc', **opt_style)

    # median of random
    time_list = [df[(df['method'] == 'random') & (df['K'] == k)]['{}_objective_eval_time'.format(objtype)].median() for k in k_list]
    val_list = np.cumsum(time_list)
    ax.plot(k_list, val_list, label='RAND (obj. eval)', **rand_style_med)

    # plot fill between min/max of random
    val_list_min = np.cumsum([df[(df['method'] == 'random') & (df['K'] == k)]['{}_objective_eval_time'.format(objtype)].min() for k in k_list])
    val_list_max = np.cumsum([df[(df['method'] == 'random') & (df['K'] == k)]['{}_objective_eval_time'.format(objtype)].max() for k in k_list])
    ax.fill_between(k_list, val_list_min, y2=val_list_max, **rand_style_fill)

    ax.set_yscale('log')

    # # ticks and gridlines
    # yticks = [10**i for i in [-1, 0, 1, 2, 3, 4]]
    # ytick_labels = [' ', '1', ' ', '10^2', ' ', '10^4']

    ax.set_xticks(xticks)
    # ax.set_yticks(yticks, ytick_labels)
    ax.grid(linestyle='-', linewidth='0.2', color='black', which='major')
    ax.grid(linestyle='-', linewidth='0.1', color='black', which='minor')

    # ml = MultipleLocator(5)
    # ax.yaxis.set_minor_locator(ml)

    locmin = LogLocator(base=10.0, subs=(0.25, 0.5, 0.75, 1.), numticks=15)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(NullFormatter())

    def time_format_func(value, tick_number):
        if value == 10 ** 0:
            return '$10^0$'
        if value == 10 ** 2:
            return '$10^2$'
        if value == 10 ** 4:
            return '$10^4$'
        else:
            return ''

    locmaj = LogLocator(base=10.0, subs=(1.,), numticks=15)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_major_formatter(FuncFormatter(time_format_func))

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


# get all num features, num items, gamma
num_features_list = list(sorted(df['num_features'].unique()))
num_items_list = list(sorted(df['num_items'].unique()))
gamma_list = [0.0, 0.1, 0.2]
gamma_list = [0.0, 1.0, 2.0]
# gamma_list = list(sorted(df['gamma'].unique()))

num_cols = max(len(num_features_list), len(num_items_list), len(gamma_list))
num_rows = 3

fig, axs = plt.subplots(num_rows, num_cols,
                        figsize=(fig_width, fig_height),
                        sharey=True,
                        sharex=True)

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
        & (df['gamma'] == gamma)
    ]

    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    plot_runtimes(df_tmp, ax, k_list, plot_label, xlim, xticks)

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
        & (df['gamma'] == gamma)
    ]

    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    plot_runtimes(df_tmp, ax, k_list, plot_label, xlim, xticks)

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
        & (df['gamma'] == gamma)
    ]

    plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
    img_ind += 1
    plot_runtimes(df_tmp, ax, k_list, plot_label, xlim, xticks)
    ax.set_xlabel("K")

axs[1, 0].set_ylabel("Runtime (s)")

plt.tight_layout()

plt.savefig(os.path.join(img_dir, 'runtime_{}.pdf'.format(version_str)), bbox_inches='tight')

