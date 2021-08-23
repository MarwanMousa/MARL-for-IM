import numpy as np
import matplotlib.pyplot as plt

labels = ["Single agent", "IPPO with shared rewards", "MAPPO with shared rewards", "IPPO with independent rewards",
          "MAPPO with independent rewards"]
num_experiments = len(labels)

three_mean = [138.8, 92.4, 99.4, 57.1, 63.3]
three_std = [29.6, 49.7, 48.1, 52, 47]
three_stage0_mean = [5.3, 10, 12.1, 12.8, 17.7]
three_stage0_std = [29.4, 36.3, 34, 30.9, 27.1]
three_stage1_mean = [69.6, 37.9, 39.9, 17.8, 19.5]
three_stage1_std = [11.4, 15.2, 14.4, 19.8, 18.1]
three_stage2_mean = [63.5, 44.4, 47.3, 26.5, 26.2]
three_stage2_std = [9.5, 11.5, 11.1, 13.9, 13.7]

three_mean_dp = [130.6, 103.4, 100, 66.2, 72.1]
three_std_dp = [30.7, 55.3, 46.2, 54.7, 49.7]
three_stage0_mean_dp = [4.1, 2, 13.8, 7.9, 11.6]
three_stage0_std_dp = [28.4, 43.3, 30.9, 35.6, 30.5]
three_stage1_mean_dp = [63.5, 45.9, 40, 22.9, 22.5]
three_stage1_std_dp = [12.3, 15.8, 16.4, 16.6, 17.4]
three_stage2_mean_dp = [63, 55.6, 48.8, 35.4, 35.8]
three_stage2_std_dp = [9.8, 7.8, 11.9, 10.7, 12]

colors = ["b", "r", "g", "y", "k"]
## First set of plots
fig, axs = plt.subplots(2, 2, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs = axs.ravel()
for i in range(num_experiments):
    axs[0].errorbar(i, three_mean[i], three_std[i], label=labels[i], fmt='o', lw=3, color=colors[i])
    axs[0].errorbar(i+.25, three_mean_dp[i], three_std_dp[i], fmt='o', lw=3, color=colors[i])
    axs[0].set_title("Total reward")
    axs[0].set_ylabel("Profit")

    axs[1].errorbar(i, three_stage0_mean[i], three_stage0_std[i], label=labels[i], fmt='o', lw=3, color=colors[i])
    axs[1].errorbar(i+.25, three_stage0_mean_dp[i], three_stage0_std_dp[i], fmt='o', lw=3, color=colors[i])
    axs[1].set_title("Stage 0")
    axs[1].set_ylabel("Profit")

    axs[2].errorbar(i, three_stage1_mean[i], three_stage1_std[i], label=labels[i], fmt='o', lw=3, color=colors[i])
    axs[2].errorbar(i+.25, three_stage1_mean_dp[i], three_stage1_std_dp[i], fmt='o', lw=3, color=colors[i])
    axs[2].set_title("Stage 2")
    axs[2].set_ylabel("Profit")
    axs[2].legend()

    axs[3].errorbar(i, three_stage2_mean[i], three_stage2_std[i], label=labels[i], fmt='o', lw=3, color=colors[i])
    axs[3].errorbar(i+.25, three_stage2_mean_dp[i], three_stage2_std_dp[i], fmt='o', lw=3, color=colors[i])
    axs[3].set_title("Stage 2")
    axs[3].set_ylabel("Profit")

plt.show()