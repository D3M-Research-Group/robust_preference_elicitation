# plot results
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc('figure', titlesize=8)  # fontsize of the figure title
plt.rc('font', size=8)

# the data for these plots are produced with the following function call:
# python -m experiments.static_experiment --max-K 10 --num-random-samples 1 --gamma 0.0 0.1 0.5 0.9 --num-items 40 20 60
# --num-features 10 5 20 --output-dir /home/dmcelfre/RobustActivePreferenceLearning_output/static_experiments/ &

# ----------------------------------------------------------------------------------------------------------------------
# define files and directories

# directory where images will be saved
img_dir = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/fig/scaling_v2/'

# results file (from static elicitation experiments)
output_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/gold/v2/static_experiment_20200104_174045.csv'

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

# note that there should be only one random sample
def plot_objvals(df, ax, plot_col, k_list):

    opt_style = {'color': 'g',
               'linestyle': '-',
               'marker': '+',
               }
    rand_style = {'color': 'r',
               'linestyle': '-',
               'marker': 'x',
               }

    # plot static heuristic
    val_list = [df[(df['method'] == 'static_heuristic') & (df['K'] == k)][plot_col].values[0] for k in k_list]
    ax.plot(k_list, val_list, label='OPT-hstc', **opt_style)

    # plot median, min, and max val of random
    val_list = [df[(df['method'] == 'random') & (df['K'] == k)][plot_col].values[0] for k in k_list]
    ax.plot(k_list, val_list, label='RAND', **rand_style)


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
        (df['num_features'] == num_features_list[1])
        & (df['num_items'] == num_items_list[i_col])
        & (df['gamma'] == gamma_list[0])
    ]

    plot_objvals(df_tmp, ax, plot_col, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[1], num_items_list[i_col], gamma_list[0]))

    if i_col == 0:
        ax.set_ylabel(plot_col)

# second first row: num num_features
i_row = 1
for i_col in range(len(num_features_list)):
    ax = axs[i_row, i_col]
    ax.set_ylim([0, 0.5])

    df_tmp = df[
        (df['num_features'] == num_features_list[i_col])
        & (df['num_items'] == num_items_list[1])
        & (df['gamma'] == gamma_list[0])
    ]

    plot_objvals(df_tmp, ax, plot_col, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[i_col], num_items_list[1], gamma_list[0]))

    if i_col == 0:
        ax.set_ylabel(plot_col)

# second first row: num num_features
i_row = 2
for i_col in range(len(gamma_list)):
    ax = axs[i_row, i_col]
    ax.set_ylim([0, 0.5])

    df_tmp = df[
        (df['num_features'] == num_features_list[1])
        & (df['num_items'] == num_items_list[1])
        & (df['gamma'] == gamma_list[i_col])
    ]

    plot_objvals(df_tmp, ax, plot_col, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[1], num_items_list[1], gamma_list[i_col]))

    ax.set_xlabel("K")

    if i_col == 0:
        ax.set_ylabel(plot_col)

plt.suptitle("static heuristic compared to one random query sample", fontsize=12)
axs[0, 0].legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(img_dir, 'objval_v2.pdf'))

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
    time_list = [df[(df['method'] == 'random') & (df['K'] == k)]['objective_eval_time'].values[0] for k in k_list]
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
        (df['num_features'] == num_features_list[1])
        & (df['num_items'] == num_items_list[i_col])
        & (df['gamma'] == gamma_list[0])
    ]

    plot_runtimes(df_tmp, ax, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[1], num_items_list[i_col], gamma_list[0]))

    if i_col == 0:
        ax.set_ylabel('total runtime (sec)')

# second first row: num num_features
i_row = 1
for i_col in range(len(num_features_list)):
    ax = axs[i_row, i_col]

    df_tmp = df[
        (df['num_features'] == num_features_list[i_col])
        & (df['num_items'] == num_items_list[1])
        & (df['gamma'] == gamma_list[0])
    ]

    plot_runtimes(df_tmp, ax, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[i_col], num_items_list[1], gamma_list[0]))

    if i_col == 0:
        ax.set_ylabel('total runtime (sec)')

# second first row: num num_features
i_row = 2
for i_col in range(len(gamma_list)):
    ax = axs[i_row, i_col]

    df_tmp = df[
        (df['num_features'] == num_features_list[1])
        & (df['num_items'] == num_items_list[1])
        & (df['gamma'] == gamma_list[i_col])
    ]

    plot_runtimes(df_tmp, ax, k_list)

    ax.set_title("#feat={}, #items={}, gam={}".format(num_features_list[1], num_items_list[1], gamma_list[i_col]))

    ax.set_xlabel("K")

    if i_col == 0:
        ax.set_ylabel('total runtime (sec)')


plt.suptitle("static heuristic compared to one random query sample", fontsize=12)
axs[0, 0].legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.savefig(os.path.join(img_dir, 'runtime_v2.pdf'))