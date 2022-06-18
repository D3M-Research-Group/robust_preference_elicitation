# plot results
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc('figure', titlesize=8)  # fontsize of the figure title
plt.rc('font', size=8)

# ----------------------------------------------------------------------------------------------------------------------
# initial plots for new submission
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# define files and directories
#
# the output file is produced using the following command:
# python -m experiments.static_experiment --max-K 10 --num-random-samples 100 --gamma 0.0 0.1 0.5 0.9
# --num-items 10 20 30 40 --num-features 3 5 10 20 --output-dir <output dir> &

# directory where images will be saved
img_dir = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/fig/compare_to_random_v2/'

# results file (from static elicitation experiments)
output_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/gold/v2/static_experiment_20200104_161209.csv'

# ----------------------------------------------------------------------------------------------------------------------
# read results

# data file: output from experiments
df = pd.read_csv(output_file, skiprows=1, delimiter=';')

# -------------------------------------------------------------------------------------------------------------
#  plots of objval
# ----------------------------------------------------------------------------------------------------------------------

# first row: increase num. items
# second row: increase num. features
# third row: increase gamma

# which column to plot data from
plot_col = 'normalized_u'
k_list = [k for k in list(sorted(df['K'].unique())) if k > 0]

def plot_objvals(df, ax, plot_col, k_list):

    opt_style = {'color': 'g',
               'linestyle': '-',
               'marker': '+',
               }
    rand_style_med = {'color': 'r',
               'linestyle': '-',
               'marker': 'x',
               }
    rand_style_min = {'color': 'r',
               'linestyle': ':',
                      }
    rand_style_max = {'color': 'r',
               'linestyle': ':',
                      }

    # plot static heuristic
    val_list = [df[(df['method'] == 'static_heuristic') & (df['K'] == k)][plot_col].values[0] for k in k_list]
    ax.plot(k_list, val_list, label='OPT-hstc', **opt_style)

    # plot median, min, and max val of random
    val_list = [df[(df['method'] == 'random') & (df['K'] == k)][plot_col].median() for k in k_list]
    ax.plot(k_list, val_list, label='RAND', **rand_style_med)

    val_list = [df[(df['method'] == 'random') & (df['K'] == k)][plot_col].min() for k in k_list]
    ax.plot(k_list, val_list, **rand_style_min)

    val_list = [df[(df['method'] == 'random') & (df['K'] == k)][plot_col].max() for k in k_list]
    ax.plot(k_list, val_list, **rand_style_max)


# get all num features, num items, gamma
num_features_list = list(sorted(df['num_features'].unique()))
num_items_list = list(sorted(df['num_items'].unique()))
gamma_list = list(sorted(df['gamma'].unique()))

num_cols = max(len(num_features_list), len(num_items_list), len(gamma_list))
num_rows = 3

fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# plot first row: num items
i_row = 0
for i_col in range(len(num_items_list)):
    ax = axs[i_row, i_col]
    ax.set_ylim([0, 0.5])

    df_tmp = df[
        (df['num_features'] == num_features_list[0])
        & (df['num_items'] == num_items_list[i_col])
        & (df['gamma'] == gamma_list[0])
    ]

    plot_objvals(df_tmp, ax, plot_col, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[0], num_items_list[i_col], gamma_list[0]))

    if i_col == 0:
        ax.set_ylabel(plot_col)

# second first row: num num_features
i_row = 1
for i_col in range(len(num_features_list)):
    ax = axs[i_row, i_col]
    ax.set_ylim([0, 0.5])

    df_tmp = df[
        (df['num_features'] == num_features_list[i_col])
        & (df['num_items'] == num_items_list[0])
        & (df['gamma'] == gamma_list[0])
    ]

    plot_objvals(df_tmp, ax, plot_col, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[i_col], num_items_list[0], gamma_list[0]))

    if i_col == 0:
        ax.set_ylabel(plot_col)

# second first row: num num_features
i_row = 2
for i_col in range(len(gamma_list)):
    ax = axs[i_row, i_col]
    ax.set_ylim([0, 0.5])

    df_tmp = df[
        (df['num_features'] == num_features_list[0])
        & (df['num_items'] == num_items_list[0])
        & (df['gamma'] == gamma_list[i_col])
    ]

    plot_objvals(df_tmp, ax, plot_col, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[0], num_items_list[0], gamma_list[i_col]))

    ax.set_xlabel("K")

    if i_col == 0:
        ax.set_ylabel(plot_col)


axs[0, 0].legend()
plt.suptitle("static heuristic compared to 100 random query samples", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(img_dir, 'objval_v2.pdf'))


# ----------------------------------------------------------------------------------------------------------------------
#  plot prob. random wins
# ----------------------------------------------------------------------------------------------------------------------

# first row: increase num. items
# second row: increase num. features
# third row: increase gamma

# which column to plot data from
plot_col = 'prob_random_wins'

# get all num features, num items, gamma
num_features_list = list(sorted(df['num_features'].unique()))
num_items_list = list(sorted(df['num_items'].unique()))
gamma_list = list(sorted(df['gamma'].unique()))

k_list = [k for k in list(sorted(df['K'].unique())) if k > 0]
num_cols = max(len(num_features_list), len(num_items_list), len(gamma_list))
num_rows = 3

fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# plot first row: num items
i_row = 0
for i_col in range(len(num_items_list)):
    ax = axs[i_row, i_col]
    ax.set_ylim([0, 1])

    df_tmp = df[
        (df['num_features'] == num_features_list[0])
        & (df['num_items'] == num_items_list[i_col])
        & (df['gamma'] == gamma_list[0])
    ]

    val_list = [df_tmp[(df_tmp['method'] == 'agg_random') & (df_tmp['K'] == k)][plot_col].values[0] for k in k_list]
    ax.plot(k_list, val_list, marker='+')

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[0], num_items_list[i_col], gamma_list[0]))

    if i_col == 0:
        ax.set_ylabel(plot_col)

# second first row: num num_features
i_row = 1
for i_col in range(len(num_features_list)):
    ax = axs[i_row, i_col]
    ax.set_ylim([0, 1])

    df_tmp = df[
        (df['num_features'] == num_features_list[i_col])
        & (df['num_items'] == num_items_list[0])
        & (df['gamma'] == gamma_list[0])
    ]

    # plot static heuristic
    val_list = [df_tmp[(df_tmp['method'] == 'agg_random') & (df_tmp['K'] == k)][plot_col].values[0] for k in k_list]
    ax.plot(k_list, val_list, marker='+')

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[i_col], num_items_list[0], gamma_list[0]))

    if i_col == 0:
        ax.set_ylabel(plot_col)


# second first row: num num_features
i_row = 2
for i_col in range(len(gamma_list)):
    ax = axs[i_row, i_col]
    ax.set_ylim([0, 1])

    df_tmp = df[
        (df['num_features'] == num_features_list[0])
        & (df['num_items'] == num_items_list[0])
        & (df['gamma'] == gamma_list[i_col])
    ]

    # plot static heuristic
    val_list = [df_tmp[(df_tmp['method'] == 'agg_random') & (df_tmp['K'] == k)][plot_col].values[0] for k in k_list]
    ax.plot(k_list, val_list, marker='+')

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[0], num_items_list[0], gamma_list[i_col]))

    ax.set_xlabel("K")

    if i_col == 0:
        ax.set_ylabel(plot_col)

plt.suptitle("static heuristic compared to 100 random query samples", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(os.path.join(img_dir, 'prob_random_wins_v2.pdf'))

# -------------------------------------------------------------------------------------------------------------
#  plots of runtime
# ----------------------------------------------------------------------------------------------------------------------

# first row: increase num. items
# second row: increase num. features
# third row: increase gamma

# which column to plot data from
k_list = [k for k in list(sorted(df['K'].unique())) if k > 0]

def plot_runtimes(df, ax, k_list):

    opt_style = {'color': 'g',
               'linestyle': '-',
               'marker': '+',
               }
    rand_style_avg = {'color': 'r',
               'linestyle': '-',
               'marker': 'x',
               }

    # plot static heuristic
    time_list = [df[(df['method'] == 'static_heuristic') & (df['K'] == k)]['solve_time'].values[0] for k in k_list]
    val_list = np.cumsum(time_list)
    ax.plot(k_list, val_list, label='OPT-hstc', **opt_style)

    # plot median, min, and max val of random
    time_list = [df[(df['method'] == 'random') & (df['K'] == k)]['objective_eval_time'].mean() for k in k_list]
    val_list = np.cumsum(time_list)
    ax.plot(k_list, val_list, label='RAND (obj. eval)', **rand_style_avg)

    ax.set_yscale('log')


# get all num features, num items, gamma
num_features_list = list(sorted(df['num_features'].unique()))
num_items_list = list(sorted(df['num_items'].unique()))
gamma_list = list(sorted(df['gamma'].unique()))

num_cols = max(len(num_features_list), len(num_items_list), len(gamma_list))
num_rows = 3

fig, axs = plt.subplots(num_rows, num_cols,
                        figsize=(12, 8),
                        sharey=True)

# plot first row: num items
i_row = 0
for i_col in range(len(num_items_list)):
    ax = axs[i_row, i_col]

    df_tmp = df[
        (df['num_features'] == num_features_list[0])
        & (df['num_items'] == num_items_list[i_col])
        & (df['gamma'] == gamma_list[0])
    ]

    plot_runtimes(df_tmp, ax, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[0], num_items_list[i_col], gamma_list[0]))

    if i_col == 0:
        ax.set_ylabel('total runtime (sec)')

# second first row: num num_features
i_row = 1
for i_col in range(len(num_features_list)):
    ax = axs[i_row, i_col]

    df_tmp = df[
        (df['num_features'] == num_features_list[i_col])
        & (df['num_items'] == num_items_list[0])
        & (df['gamma'] == gamma_list[0])
    ]

    plot_runtimes(df_tmp, ax, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[i_col], num_items_list[0], gamma_list[0]))

    if i_col == 0:
        ax.set_ylabel('total runtime (sec)')

# second first row: num num_features
i_row = 2
for i_col in range(len(gamma_list)):
    ax = axs[i_row, i_col]

    df_tmp = df[
        (df['num_features'] == num_features_list[0])
        & (df['num_items'] == num_items_list[0])
        & (df['gamma'] == gamma_list[i_col])
    ]

    plot_runtimes(df_tmp, ax, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[0], num_items_list[0], gamma_list[i_col]))

    ax.set_xlabel("K")

    if i_col == 0:
        ax.set_ylabel('total runtime (sec)')


axs[0, 0].legend()
plt.suptitle("static heuristic compared to 100 random query samples", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(os.path.join(img_dir, 'runtime_v2.pdf'))