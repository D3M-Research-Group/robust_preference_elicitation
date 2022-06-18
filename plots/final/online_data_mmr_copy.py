# plot results
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set fonts
plt.rcParams["font.family"] = "Times New Roman"

plt.rc("figure", titlesize=10)  # fontsize of the figure title
plt.rc("font", size=10)

# figure size
fig_height = 2.5
fig_width = 3

label_bbox_props = {
    "pad": 0,
    "alpha": 0.2,
    "color": "gray",
    "linewidth": 0.2,
    "boxstyle": "round,pad=0.3",
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

version_str = "online_data_mmr_0718_1"
version_str = "online_data_mmr_0911_1"
version_str = "online_data_mmr_0915_1"
version_str = "online_data_mmr_0922_1_veryold"
version_str = "online_data_mmr_0923_1_025gamma"
version_str = "online_data_mmr_0923_1_0gamma"
version_str = "online_data_mmr_0925_1_025gamma"
# version_str = "online_data_mmr_0925_2_0gamma"
file_str = version_str + "_final.csv"

# objtype must be 'mmu' or 'mmr
objtype = "mmr"
assert objtype in ["mmr", "mmu"]

if objtype == "mmr":
    caption_objtype = "MMR"
if objtype == "mmu":
    caption_objtype = "MMU"

# directory where images will be saved
img_dir = f"/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/fig/{version_str}/"
img_dir = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/"

# results file (from static elicitation experiments)
output_file = f"/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/final/{version_str}/{file_str}"
output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0718_1.csv"
output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0911_1.csv"
output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0915_1.csv"
output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0922_1_veryold.csv"
output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0923_1_025.csv"
output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0923_1_000.csv"
output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0925_1_025.csv"
# output_file = "/Users/yingxiao.ye/Dropbox/Research/Prefercence Elicitation/review_1/code_0603/RobustActivePreferenceLearning_private/test_results/online_mmr_data_0925_2_000.csv"

# ----------------------------------------------------------------------------------------------------------------------
# read results

# data file: output from experiments
df = pd.read_csv(output_file, delimiter=";")
# df = pd.read_csv(output_file, skiprows=1, delimiter=";")

# # create the correct normalized mmr value...
# df["mmr_objval_normalized"] = df["mmr_objval"].apply(lambda x: (x + 1.0) / 2.0)
# df["true_regret_normalized"] = df["true_regret"].apply(lambda x: (x + 1.0) / 2.0)

# -------------------------------------------------------------------------------------------------------------
#  plots of objval
# ----------------------------------------------------------------------------------------------------------------------

# first row: increase num. items
# second row: increase num. features
# third row: increase gamma

# note that there should be only one random sample
def plot_objvals(
    df, ax, plot_col, k_list, plot_label, xlim, ylim, xticks, yticks=None, legend=False
):

    # plot static heuristic
    if objtype == "mmu":
        method_name = "maximin"
    if objtype == "mmr":
        method_name = "mmr"

    elicitation_method = method_name
    recommendation_method = method_name
    fill_style = opt_style_fill

    if objtype == "mmu":
        val_func = np.min
        if plot_col == "true_rank":
            val_func = np.max
    if objtype == "mmr":
        val_func = np.max

    val_list = [
        val_func(
            df[
                (df["elicitation_method"] == elicitation_method)
                & (df["recommendation_method"] == recommendation_method)
                & (df["K"] == k)
            ][plot_col].values
        )
        for k in k_list
    ]
    ax.plot(
        k_list,
        val_list,
        label="{}+{}".format(caption_objtype, caption_objtype),
        **opt_style,
    )

    # # plot fill between min/max
    # val_list_min = [df[(df['elicitation_method'] == elicitation_method)
    #                & (df['recommendation_method'] == recommendation_method)
    #                & (df['K'] == k)][plot_col].min() for k in k_list]
    # val_list_max = [df[(df['elicitation_method'] == elicitation_method)
    #                & (df['recommendation_method'] == recommendation_method)
    #                & (df['K'] == k)][plot_col].max() for k in k_list]
    # ax.fill_between(k_list, val_list_min, y2=val_list_max, **fill_style)

    # plot AC + AC
    elicitation_method = "AC"
    recommendation_method = "AC"
    recommendation_method = 'mmr_AC'
    fill_style = acac_style_fill

    val_list = [
        val_func(
            df[
                (df["elicitation_method"] == elicitation_method)
                & (df["recommendation_method"] == recommendation_method)
                & (df["K"] == k)
            ][plot_col]
        )
        for k in k_list
    ]
    print(val_list)
    ax.plot(k_list, val_list, label="AC+AC", **acac_style)

    # # plot fill between min/max
    # val_list_min = [df[(df['elicitation_method'] == elicitation_method)
    #                & (df['recommendation_method'] == recommendation_method)
    #                & (df['K'] == k)][plot_col].min() for k in k_list]
    # val_list_max = [df[(df['elicitation_method'] == elicitation_method)
    #                & (df['recommendation_method'] == recommendation_method)
    #                & (df['K'] == k)][plot_col].max() for k in k_list]
    # ax.fill_between(k_list, val_list_min, y2=val_list_max, **fill_style)

    # plot AC + maximin
    elicitation_method = "AC"
    recommendation_method = method_name
    fill_style = acmm_style_fill

    val_list = [
        val_func(
            df[
                (df["elicitation_method"] == elicitation_method)
                & (df["recommendation_method"] == recommendation_method)
                & (df["K"] == k)
            ][plot_col]
        )
        for k in k_list
    ]
    ax.plot(k_list, val_list, label="AC+{}".format(caption_objtype), **acmm_style)

    # # plot fill between min/max
    # val_list_min = [df[(df['elicitation_method'] == elicitation_method)
    #                    & (df['recommendation_method'] == recommendation_method)
    #                    & (df['K'] == k)][plot_col].min() for k in k_list]
    # val_list_max = [df[(df['elicitation_method'] == elicitation_method)
    #                    & (df['recommendation_method'] == recommendation_method)
    #                    & (df['K'] == k)][plot_col].max() for k in k_list]
    # ax.fill_between(k_list, val_list_min, y2=val_list_max, **fill_style)

    # plot random + maximin
    elicitation_method = "random"
    recommendation_method = method_name
    fill_style = randmm_style_fill

    val_list = [
        val_func(
            df[
                (df["elicitation_method"] == elicitation_method)
                & (df["recommendation_method"] == recommendation_method)
                & (df["K"] == k)
            ][plot_col]
        )
        for k in k_list
    ]
    ax.plot(k_list, val_list, label="RAND+{}".format(caption_objtype), **randmm_style)

    # # plot fill between min/max
    # val_list_min = [df[(df['elicitation_method'] == elicitation_method)
    #                    & (df['recommendation_method'] == recommendation_method)
    #                    & (df['K'] == k)][plot_col].min() for k in k_list]
    # val_list_max = [df[(df['elicitation_method'] == elicitation_method)
    #                    & (df['recommendation_method'] == recommendation_method)
    #                    & (df['K'] == k)][plot_col].max() for k in k_list]
    # ax.fill_between(k_list, val_list_min, y2=val_list_max, **fill_style)


    # plot ellipsoidal + mean
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
    print(val_list)
    ax.plot(k_list, val_list, label='probpoly+{}'.format(caption_objtype), **probpoly_style)

    # ticks and gridlines
    if yticks is not None:
        ax.set_yticks(yticks)


    ax.set_xticks(xticks)
    ax.grid(linestyle="--", linewidth="0.2", color="black", which="major")

    ax.tick_params(
        axis="y",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        left=True,
        right=False,
        labelleft=True,
    )

    # set axis limits
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    # add a plot label
    label_x = 1.03
    label_y = 0.5
    ax.text(
        label_x,
        label_y,
        plot_label,
        horizontalalignment="left",
        verticalalignment="center",
        rotation=-90,
        backgroundcolor="gray",
        bbox=label_bbox_props,
        transform=ax.transAxes,
    )
    if legend:
        ax.legend()


# ---------------
# Plot objective value
# ---------------


# which column to plot data from
plot_col = "{}_objval_normalized_new".format(objtype)
k_list = [k for k in list(sorted(df["K"].unique())) if k > 0]

num_cols = 1
num_rows = 1

if objtype == 'mmu':
    ylim = None
    yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
if objtype == 'mmr':
    ylim = None
    yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

fig, ax = plt.subplots(
    num_rows, num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True,
)

# single plot


# plot_label = " ({}) {}\u2212{}\u2212{}".format(letter_dict[img_ind], num_items, num_features, gamma)
# yticks = None
ylim = None
yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

plot_objvals(df, ax, plot_col, k_list, "", xlim, ylim, xticks, yticks=yticks)

ax.set_xlabel("K")

# set a single ylabel
if objtype == "mmu":
    fig.text(
        0.0,
        0.5,
        "Worst-Case Utility",  # of Recommended Item",
        ha="center",
        va="center",
        rotation="vertical",
    )
if objtype == "mmr":
    fig.text(
        0.0,
        0.5,
        "Worst-Case Regret",  # of Recommended Item",
        ha="center",
        va="center",
        rotation="vertical",
    )

plt.tight_layout()

plt.savefig(
    os.path.join(img_dir, "objval_{}.pdf".format(version_str)), bbox_inches="tight"
)


# ---------------
# Plot true objective value
# ---------------

# which column to plot data from
if objtype == "mmu":
    plot_col = "true_u_normalized_new".format(objtype)
if objtype == "mmr":
    plot_col = "true_regret_normalized_new".format(objtype)
    # plot_col = 'true_regret'.format(objtype)

k_list = [k for k in list(sorted(df["K"].unique())) if k > 0]

# get all num features, num items, gamma
num_features_list = list(sorted(df["num_features"].unique()))
num_items_list = list(sorted(df["num_items"].unique()))
gamma_list = list(sorted(df["gamma"].unique()))

num_cols = 1
num_rows = 1

ylim = None
yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

fig, ax = plt.subplots(
    num_rows, num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True,
)

# yticks = None
plot_objvals(df, ax, plot_col, k_list, "", xlim, ylim, xticks, yticks=yticks)

ax.set_xlabel("K")

# set a single ylabel
if objtype == "mmu":
    fig.text(
        0.0,
        0.5,
        "True Normalized Utility",  # of Recommended Item", # (worst-case over all agents)",
        ha="center",
        va="center",
        rotation="vertical",
    )
if objtype == "mmr":
    fig.text(
        0.0,
        0.5,
        "True Normalized Regret", #  (worst-case over all agents)",
        ha="center",
        va="center",
        rotation="vertical",
    )
plt.tight_layout()

plt.savefig(
    os.path.join(img_dir, "true_objval_{}.pdf".format(version_str)), bbox_inches="tight"
)


# ---------------
# Plot objective value
# ---------------

# which column to plot data from
if objtype == "mmu":
    plot_col = "mmu_objval_normalized_new".format(objtype)
if objtype == "mmr":
    plot_col = "mmr_objval_normalized_new".format(objtype)

k_list = [k for k in list(sorted(df["K"].unique())) if k > 0]

num_cols = 1
num_rows = 1

#
# yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

fig, ax = plt.subplots(
    num_rows, num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True,
)

ylim = [0, 2]
ylim = [0, 10]
ylim = None
yticks = None
yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
plot_objvals(df, ax, plot_col, k_list, "", xlim, ylim, xticks, yticks=yticks)

ax.set_xlabel("K")

# set a single ylabel
if objtype == "mmu":
    fig.text(
        0.0,
        0.5,
        "Worst-Case Utility", #  (worst-case over all agents)",
        ha="center",
        va="center",
        rotation="vertical",
    )
if objtype == "mmr":
    fig.text(
        0.0,
        0.5,
        "Normalized Worst-Case Regret",  # of Recommended Item", #  (worst-case over all agents)",
        ha="center",
        va="center",
        rotation="vertical",
    )
plt.tight_layout()

plt.savefig(
    os.path.join(img_dir, "objval_{}.pdf".format(version_str)), bbox_inches="tight"
)


# ---------------
# Plot rank
# ---------------

# which column to plot data from
plot_col = "true_rank"

k_list = [k for k in list(sorted(df["K"].unique())) if k > 0]

num_cols = 1
num_rows = 1

fig, ax = plt.subplots(
    num_rows, num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=False,
)

num_items = df["num_items"].values[0]
yticks = [1, 5, 10, 15]
yticks = [1, 5, 10, 15, 20, 25]
# yticks = [1, 10, 30, 40]
plot_objvals(df, ax, plot_col, k_list, "", xlim, ylim, xticks, yticks=yticks)
ax.set_ylim([15 + 0.5, 0.5])
ax.set_ylim([25 + 0.5, 0.5])

ax.set_xlabel("K")

# set a single ylabel
fig.text(
    0.0,
    0.5,
    "Worst-Case Rank", #  (worst-case over all agents)",
    ha="center",
    va="center",
    rotation="vertical",
)

plt.tight_layout()

plt.savefig(
    os.path.join(img_dir, "rank_{}.pdf".format(version_str)), bbox_inches="tight"
)
