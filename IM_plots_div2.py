import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


# Define plot settings
rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 13})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["figure.dpi"] = 200


#%% Load Single Rl Data
S2_results = np.load("checkpoints/single_agent/div_2/results.npy", allow_pickle=True)
S2_time = np.load("checkpoints/single_agent/div_2/time.npy", allow_pickle=True)
S2_reward_mean = np.load("checkpoints/single_agent/div_2/reward_mean.npy", allow_pickle=True)
S2_reward_std = np.load("checkpoints/single_agent/div_2/reward_std.npy", allow_pickle=True)
S2_inventory_mean = np.load("checkpoints/single_agent/div_2/inventory_mean.npy", allow_pickle=True)
S2_inventory_std = np.load("checkpoints/single_agent/div_2/inventory_std.npy", allow_pickle=True)
S2_backlog_mean = np.load("checkpoints/single_agent/div_2/backlog_mean.npy", allow_pickle=True)
S2_backlog_std = np.load("checkpoints/single_agent/div_2/backlog_std.npy", allow_pickle=True)
S2_customer_backlog_mean = np.load("checkpoints/single_agent/div_2/customer_backlog_mean.npy", allow_pickle=True)
S2_customer_backlog_std = np.load("checkpoints/single_agent/div_2/customer_backlog_std.npy", allow_pickle=True)
S2_profit = np.load("checkpoints/single_agent/div_2/profit.npy", allow_pickle=True)

#%% Load MA Rl Data
MA2_results = np.load("checkpoints/multi_agent/div_2/results.npy", allow_pickle=True)
MA2_time = np.load("checkpoints/multi_agent/div_2/time.npy", allow_pickle=True)
MA2_reward_mean = np.load("checkpoints/multi_agent/div_2/reward_mean.npy", allow_pickle=True)
MA2_reward_std = np.load("checkpoints/multi_agent/div_2/reward_std.npy", allow_pickle=True)
MA2_inventory_mean = np.load("checkpoints/multi_agent/div_2/inventory_mean.npy", allow_pickle=True)
MA2_inventory_std = np.load("checkpoints/multi_agent/div_2/inventory_std.npy", allow_pickle=True)
MA2_backlog_mean = np.load("checkpoints/multi_agent/div_2/backlog_mean.npy", allow_pickle=True)
MA2_backlog_std = np.load("checkpoints/multi_agent/div_2/backlog_std.npy", allow_pickle=True)
MA2_customer_backlog_mean = np.load("checkpoints/multi_agent/div_2/customer_backlog_mean.npy", allow_pickle=True)
MA2_customer_backlog_std = np.load("checkpoints/multi_agent/div_2/customer_backlog_std.npy", allow_pickle=True)
MA2_profit = np.load("checkpoints/multi_agent/div_2/profit.npy", allow_pickle=True)

#%% Load CC Rl Data
CC2_results = np.load("checkpoints/cc_agent/div_2/results.npy", allow_pickle=True)
CC2_time = np.load("checkpoints/cc_agent/div_2/time.npy", allow_pickle=True)
CC2_reward_mean = np.load("checkpoints/cc_agent/div_2/reward_mean.npy", allow_pickle=True)
CC2_reward_std = np.load("checkpoints/cc_agent/div_2/reward_std.npy", allow_pickle=True)
CC2_inventory_mean = np.load("checkpoints/cc_agent/div_2/inventory_mean.npy", allow_pickle=True)
CC2_inventory_std = np.load("checkpoints/cc_agent/div_2/inventory_std.npy", allow_pickle=True)
CC2_backlog_mean = np.load("checkpoints/cc_agent/div_2/backlog_mean.npy", allow_pickle=True)
CC2_backlog_std = np.load("checkpoints/cc_agent/div_2/backlog_std.npy", allow_pickle=True)
CC2_customer_backlog_mean = np.load("checkpoints/cc_agent/div_2/customer_backlog_mean.npy", allow_pickle=True)
CC2_customer_backlog_std = np.load("checkpoints/cc_agent/div_2/customer_backlog_std.npy", allow_pickle=True)
CC2_profit = np.load("checkpoints/cc_agent/div_2/profit.npy", allow_pickle=True)


#%% Load DFO Data
DFO2_reward_mean = np.load("checkpoints/single_agent/div_2/dfo_mean.npy", allow_pickle=True)
DFO2_reward_std = np.load("checkpoints/single_agent/div_2/dfo_std.npy", allow_pickle=True)

#%% Load Oracle Data
OR2_reward_mean = np.load("LP_results/div_2/Oracle/reward_mean.npy", allow_pickle=True)
OR2_reward_std = np.load("LP_results/div_2/Oracle/reward_std.npy", allow_pickle=True)
OR2_inventory_mean = np.load("LP_results/div_2/Oracle/inventory_mean.npy", allow_pickle=True)
OR2_inventory_std = np.load("LP_results/div_2/Oracle/inventory_std.npy", allow_pickle=True)
OR2_backlog_mean = np.load("LP_results/div_2/Oracle/backlog_mean.npy", allow_pickle=True)
OR2_backlog_std = np.load("LP_results/div_2/Oracle/backlog_std.npy", allow_pickle=True)
OR2_customer_backlog_mean = np.load("LP_results/div_2/Oracle/customer_backlog_mean.npy", allow_pickle=True)
OR2_customer_backlog_std = np.load("LP_results/div_2/Oracle/customer_backlog_std.npy", allow_pickle=True)
OR2_profit = np.load("LP_results/div_2/Oracle/profit.npy", allow_pickle=True)

#%% Load SHLP Data
SHLP2_time = np.load("LP_results/div_2/SHLP/time.npy", allow_pickle=True)
SHLP2_reward_mean = np.load("LP_results/div_2/SHLP/reward_mean.npy", allow_pickle=True)
SHLP2_reward_std = np.load("LP_results/div_2/SHLP/reward_std.npy", allow_pickle=True)
SHLP2_inventory_mean = np.load("LP_results/div_2/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP2_inventory_std = np.load("LP_results/div_2/SHLP/inventory_std.npy", allow_pickle=True)
SHLP2_backlog_mean = np.load("LP_results/div_2/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP2_backlog_std = np.load("LP_results/div_2/SHLP/backlog_std.npy", allow_pickle=True)
SHLP2_customer_backlog_mean = np.load("LP_results/div_2/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP2_customer_backlog_std = np.load("LP_results/div_2/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP2_profit = np.load("LP_results/div_2/SHLP/profit.npy", allow_pickle=True)

#%% Load DSHLP Data
DSHLP2_time = np.load("LP_results/div_2/DSHLP/time.npy", allow_pickle=True)
DSHLP2_reward_mean = np.load("LP_results/div_2/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP2_reward_std = np.load("LP_results/div_2/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP2_inventory_mean = np.load("LP_results/div_2/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP2_inventory_std = np.load("LP_results/div_2/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP2_backlog_mean = np.load("LP_results/div_2/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP2_backlog_std = np.load("LP_results/div_2/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP2_customer_backlog_mean = np.load("LP_results/div_2/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP2_customer_backlog_std = np.load("LP_results/div_2/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP2_profit = np.load("LP_results/div_2/DSHLP/profit.npy", allow_pickle=True)
DSHLP2_failed_tests = np.load("LP_results/div_2/DSHLP/failed_tests.npy", allow_pickle=True)

#%% Plots
colour_dict = {}
colour_dict['OR'] = 'k'
colour_dict['SHLP'] = 'tab:red'
colour_dict['DSHLP'] = 'tab:orange'
colour_dict['DecLP'] = 'yellow'
colour_dict['S'] = 'tab:blue'
colour_dict['MA'] = 'salmon'
colour_dict['MAS'] = 'tab:cyan'
colour_dict['CC'] = 'tab:green'
colour_dict['MAI'] = 'indigo'
colour_dict['MASI'] = 'tab:purple'
colour_dict['CCI'] = 'tab:olive'
#%% Div2 profit
OR2_profit_mean = np.mean(np.cumsum(OR2_profit, axis=1), axis=0)
OR2_profit_std = np.std(np.cumsum(OR2_profit, axis=1), axis=0)

SHLP2_profit_mean = np.mean(np.cumsum(SHLP2_profit, axis=1), axis=0)
SHLP2_profit_std = np.std(np.cumsum(SHLP2_profit, axis=1), axis=0)

DSHLP2_profit_mean = np.mean(np.cumsum(DSHLP2_profit, axis=1), axis=0)
DSHLP2_profit_std = np.std(np.cumsum(DSHLP2_profit, axis=1), axis=0)

S2_profit_mean = np.mean(np.cumsum(S2_profit, axis=1), axis=0)
S2_profit_std = np.std(np.cumsum(S2_profit, axis=1), axis=0)

MA2_profit_mean = np.mean(np.cumsum(MA2_profit, axis=1), axis=0)
MA2_profit_std = np.std(np.cumsum(MA2_profit, axis=1), axis=0)

CC2_profit_mean = np.mean(np.cumsum(CC2_profit, axis=1), axis=0)
CC2_profit_std = np.std(np.cumsum(CC2_profit, axis=1), axis=0)

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.plot(OR2_profit_mean, label='Oracle', lw=2, color=colour_dict['OR'])
#axs.fill_between(np.arange(0, 30), OR1_profit_mean-OR1_profit_std, OR1_profit_mean+OR1_profit_std, alpha=0.3)
axs.plot(SHLP2_profit_mean, label='SHILP', lw=2, color=colour_dict['SHLP'])
axs.plot(DSHLP2_profit_mean, label='DSHILP', lw=2, color=colour_dict['DSHLP'])
axs.plot(S2_profit_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(MA2_profit_mean, label='IPPO', lw=2, color=colour_dict['MA'])
#axs.plot(MA2_profit_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
#axs.plot(MA2I_profit_mean, label='IPPO Independent', lw=2, color=colour_dict['MAI'])
#axs.plot(MA2I_profit_mean, label='IPPO shared network Independent', lw=2, color=colour_dict['MASI'])
axs.plot(CC2_profit_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
#axs.plot(CC2I_profit_mean, label='MAPPO Independent', lw=2, color=colour_dict['CCI'])
axs.set_ylabel("Cumulative Profit")
axs.set_xlabel("Period")
axs.legend()
axs.set_xlim(0, 29)
plt.savefig('report_figures/div2_profit.png', dpi=200, bbox_inches='tight')
plt.show()


#%% Learning curves

p = 100
# Unpack values from each iteration
S2_rewards = np.hstack([i['hist_stats']['episode_reward'] for i in S2_results])
S2_Mean_rewards = np.array([np.mean(S2_rewards[i - p:i + 1]) if i >= p else np.mean(S2_rewards[:i + 1])
                         for i, _ in enumerate(S2_rewards)])
S2_Std_rewards = np.array([np.std(S2_rewards[i - p:i + 1]) if i >= p else np.std(S2_rewards[:i + 1])
                        for i, _ in enumerate(S2_rewards)])

MA2_rewards = np.hstack([i['episode_reward'] for i in MA2_results])
MA2_Mean_rewards = np.array([np.mean(MA2_rewards[i - p:i + 1]) if i >= p else np.mean(MA2_rewards[:i + 1])
                         for i, _ in enumerate(MA2_rewards)])
MA2_Std_rewards = np.array([np.std(MA2_rewards[i - p:i + 1]) if i >= p else np.std(MA2_rewards[:i + 1])
                        for i, _ in enumerate(MA2_rewards)])


CC2_rewards = np.hstack([i['hist_stats']['episode_reward'] for i in CC2_results])
CC2_Mean_rewards = np.array([np.mean(CC2_rewards[i - p:i + 1]) if i >= p else np.mean(CC2_rewards[:i + 1])
                         for i, _ in enumerate(CC2_rewards)])
CC2_Std_rewards = np.array([np.std(CC2_rewards[i - p:i + 1]) if i >= p else np.std(CC2_rewards[:i + 1])
                        for i, _ in enumerate(CC2_rewards)])

#%% MA Learning Curve plots
fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.fill_between(np.arange(len(MA2_Mean_rewards)),
                 MA2_Mean_rewards - MA2_Std_rewards,
                 MA2_Mean_rewards + MA2_Std_rewards,
                 alpha=0.1)
axs.plot(MA2_Mean_rewards, label='Mean IPPO rewards', lw=2)

axs.fill_between(np.arange(len(CC2_Mean_rewards)),
                 CC2_Mean_rewards - CC2_Std_rewards,
                 CC2_Mean_rewards + CC2_Std_rewards,
                 alpha=0.1)
axs.plot(CC2_Mean_rewards, label='Mean MAPPO rewards', lw=2)

# Plot DSHLP rewards
axs.fill_between(np.arange(len(CC2_Mean_rewards)),
                np.ones(len(CC2_Mean_rewards)) * (DSHLP2_reward_mean - DSHLP2_reward_std),
                np.ones(len(CC2_Mean_rewards)) * (DSHLP2_reward_mean + DSHLP2_reward_std),
                alpha=0.3)
axs.plot(np.arange(len(CC2_Mean_rewards)), np.ones(len(CC2_Mean_rewards)) * (DSHLP2_reward_mean), label='Mean DSHLP rewards', lw=2)


axs.set_ylabel("Rewards")
axs.set_xlabel("Episode")
axs.legend()
axs.set_xlim(0, 50000)
#axs.set_ylim(250, 550)
plt.savefig('report_figures/MA_learning_curves_div2.png', dpi=200, bbox_inches='tight')
plt.show()

#%% Printing Results
print('Div 2')
print(f'Oracle Mean reward: {OR2_reward_mean}, Mean Inventory: {OR2_inventory_mean}, Mean Backlog: {OR2_backlog_mean}, Mean Customer Backlog: {OR2_customer_backlog_mean}')
print(f'Oracle std reward: {OR2_reward_std}, std Inventory: {OR2_inventory_std}, std Backlog: {OR2_backlog_std}, std Customer Backlog: {OR2_customer_backlog_std} \n')

print(f'SHLP Mean reward: {SHLP2_reward_mean}, Mean Inventory: {SHLP2_inventory_mean}, Mean Backlog: {SHLP2_backlog_mean}, Mean Customer Backlog: {SHLP2_customer_backlog_mean}')
print(f'SHLP std reward: {SHLP2_reward_std}, std Inventory: {SHLP2_inventory_std}, std Backlog: {SHLP2_backlog_std}, std Customer Backlog: {SHLP2_customer_backlog_std}')
print(f'SHLP Mean relative to Oracle: {SHLP2_reward_mean/OR2_reward_mean}\n')

print(f'DSHLP Mean reward: {DSHLP2_reward_mean}, Mean Inventory: {DSHLP2_inventory_mean}, Mean Backlog: {DSHLP2_backlog_mean}, Mean Customer Backlog: {DSHLP2_customer_backlog_mean}')
print(f'DSHLP std reward: {DSHLP2_reward_std}, std Inventory: {DSHLP2_inventory_std}, std Backlog: {DSHLP2_backlog_std}, std Customer Backlog: {DSHLP2_customer_backlog_std}')
print(f'DSHLP Mean relative to Oracle: {DSHLP2_reward_mean/OR2_reward_mean}\n')

print(f'Single Agent Mean reward: {S2_reward_mean}, Mean Inventory: {S2_inventory_mean}, Mean Backlog: {S2_backlog_mean}, Mean Customer Backlog: {S2_customer_backlog_mean}')
print(f'Single Agent std reward: {S2_reward_std}, std Inventory: {S2_inventory_std}, std Backlog: {S2_backlog_std}, std Customer Backlog: {S2_customer_backlog_std}')
print(f'Single Agent Mean relative to Oracle: {S2_reward_mean/OR2_reward_mean}\n')

print(f'MA Mean reward: {MA2_reward_mean}, Mean Inventory: {MA2_inventory_mean}, Mean Backlog: {MA2_backlog_mean}, Mean Customer Backlog: {MA2_customer_backlog_mean}')
print(f'MA std reward: {MA2_reward_std}, std Inventory: {MA2_inventory_std}, std Backlog: {MA2_backlog_std}, std Customer Backlog: {MA2_customer_backlog_std}')
print(f'MA Mean relative to Oracle: {MA2_reward_mean/OR2_reward_mean}\n')

print(f'CC Mean reward: {CC2_reward_mean}, Mean Inventory: {CC2_inventory_mean}, Mean Backlog: {CC2_backlog_mean}, Mean Customer Backlog: {CC2_customer_backlog_mean}')
print(f'CC std reward: {CC2_reward_std}, std Inventory: {CC2_inventory_std}, std Backlog: {CC2_backlog_std}, std Customer Backlog: {CC2_customer_backlog_std}')
print(f'CC Mean relative to Oracle: {CC2_reward_mean/OR2_reward_mean}\n')

