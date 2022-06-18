# plot results
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc('figure', titlesize=8)  # fontsize of the figure title
plt.rc('font', size=8)

# the data for these plots are produced with the following function call:
# args are:
# Namespace(DEBUG=False, evaluate_mmr_objective=True, gamma=[0.0, 0.05, 0.1], max_K=10, num_features=[3, 4, 10],
# num_items=[10, 15, 20], num_random_samples=50, output_dir=<output_dir>, problem_seed=0, use_maximin_heuristic=True,
# use_mmr_heuristic=True)
# ----------------------------------------------------------------------------------------------------------------------
# define files and directories

version_str = 'v5_mmr_timing'

# directory where images will be saved
img_dir = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/fig/{}/'.format(version_str)

# results file (from static elicitation experiments)
output_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/gold/{}/static_experiment_20200109_140224.csv'.format(version_str)

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
plot_col = 'mmr_objval'
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
    val_list = [df[(df['method'] == 'mmr_heuristic') & (df['K'] == k)][plot_col].values[0] for k in k_list]
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

ylim = [0.0, 40.0]

# plot first row: num items
i_row = 0
for i_col in range(len(num_items_list)):
    ax = axs[i_row, i_col]
    ax.set_ylim(ylim)

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
    ax.set_ylim(ylim)

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
    ax.set_ylim(ylim)

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

plt.suptitle("MMR heuristic compared to 50 random query sample", fontsize=12)
axs[0, 0].legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(img_dir, 'objval_{}.pdf'.format(version_str)))

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
    time_list = [df[(df['method'] == 'mmr_heuristic') & (df['K'] == k)]['solve_time'].values[0] for k in k_list]
    val_list = np.cumsum(time_list)
    ax.plot(k_list, val_list, label='OPT-hstc', **opt_style)

    # plot median, min, and max val of random
    time_list = [df[(df['method'] == 'random') & (df['K'] == k)]['mmr_objective_eval_time'].median() for k in k_list]
    val_list = np.cumsum(time_list)
    ax.plot(k_list, val_list, label='RAND (obj. eval)', **rand_style_med)

    # plot median, min, and max val of random
    time_list = [df[(df['method'] == 'random') & (df['K'] == k)]['mmr_objective_eval_time'].min() for k in k_list]
    val_list = np.cumsum(time_list)
    ax.plot(k_list, val_list, label='RAND (obj. eval)', **rand_style_min)

    # plot median, min, and max val of random
    time_list = [df[(df['method'] == 'random') & (df['K'] == k)]['mmr_objective_eval_time'].max() for k in k_list]
    val_list = np.cumsum(time_list)
    ax.plot(k_list, val_list, label='RAND (obj. eval)', **rand_style_max)
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


plt.suptitle("MMR heuristic compared to 50 random query samples", fontsize=12)
axs[0, 0].legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.savefig(os.path.join(img_dir, 'runtime_{}.pdf'.format(version_str)))