# plot results
import os
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# initial plots for new submission
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# define files and directories

# directory where images will be saved
img_dir = '/Users/duncan/research/RobustActivePreferenceLearning_output/fig/new/'

# results file (from adaptive elicitation experiments)
output_file = '/Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/gold/adaptive_v1/adaptive_experiment_20200116_143058.csv'

# ----------------------------------------------------------------------------------------------------------------------
# read results

# data file: output from experiments
df = pd.read_csv(output_file, skiprows=1, delimiter=';')


# ----------------------------------------------------------------------------------------------------------------------
#  example plot
# ----------------------------------------------------------------------------------------------------------------------

fig, axs = plt.subplots(1, 3, figsize=(6, 3))

sns.boxplot(y='worstcase_normalized_u',
            hue='method',
            x='K',
            flierprops={'marker': '+'},
            data=df,
            ax=axs[0]
)
axs[0].set_title("worst-case rec. utility")

sns.boxplot(y='true_normalized_u',
            hue='method',
            x='K',
            flierprops={'marker': '+'},
            data=df,
            ax=axs[1]
)
axs[1].set_title("true rec. utility")

sns.boxplot(y='true_rank',
            hue='method',
            x='K',
            flierprops={'marker': '+'},
            data=df,
            ax=axs[2]
)
num_items = df['num_items'].unique()[0]
axs[2].set_ylim([num_items, 1])
axs[2].set_title("true rec. rank")

axs[0].set_ylabel('')
axs[1].set_ylabel('')
axs[2].set_ylabel('')

plt.tight_layout()

