import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle

# Define plot settings
rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 13})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["figure.dpi"] = 200

#%% For pickle load
def policy_mapping_fn(agent_id, episode, **kwargs):
    for i in range(num_stages):
        if agent_id.startswith(agent_ids[i]):
            return agent_ids[i]

def policy_mapping_single(agent_id, episode, **kwargs):
    return "stage"

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


#%% Load Single Rl Data
S4_results = np.load("checkpoints/single_agent/four_stage/results.npy", allow_pickle=True)
S4_time = np.load("checkpoints/single_agent/four_stage/time.npy", allow_pickle=True)
S4_reward_mean = np.load("checkpoints/single_agent/four_stage/reward_mean.npy", allow_pickle=True)
S4_reward_std = np.load("checkpoints/single_agent/four_stage/reward_std.npy", allow_pickle=True)
S4_inventory_mean = np.load("checkpoints/single_agent/four_stage/inventory_mean.npy", allow_pickle=True)
S4_inventory_std = np.load("checkpoints/single_agent/four_stage/inventory_std.npy", allow_pickle=True)
S4_backlog_mean = np.load("checkpoints/single_agent/four_stage/backlog_mean.npy", allow_pickle=True)
S4_backlog_std = np.load("checkpoints/single_agent/four_stage/backlog_std.npy", allow_pickle=True)
S4_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage/customer_backlog_mean.npy", allow_pickle=True)
S4_customer_backlog_std = np.load("checkpoints/single_agent/four_stage/customer_backlog_std.npy", allow_pickle=True)
S4_profit = np.load("checkpoints/single_agent/four_stage/profit.npy", allow_pickle=True)

S2_results = np.load("checkpoints/single_agent/two_stage/results.npy", allow_pickle=True)
S2_time = np.load("checkpoints/single_agent/two_stage/time.npy", allow_pickle=True)
S2_reward_mean = np.load("checkpoints/single_agent/two_stage/reward_mean.npy", allow_pickle=True)
S2_reward_std = np.load("checkpoints/single_agent/two_stage/reward_std.npy", allow_pickle=True)
S2_inventory_mean = np.load("checkpoints/single_agent/two_stage/inventory_mean.npy", allow_pickle=True)
S2_inventory_std = np.load("checkpoints/single_agent/two_stage/inventory_std.npy", allow_pickle=True)
S2_backlog_mean = np.load("checkpoints/single_agent/two_stage/backlog_mean.npy", allow_pickle=True)
S2_backlog_std = np.load("checkpoints/single_agent/two_stage/backlog_std.npy", allow_pickle=True)
S2_customer_backlog_mean = np.load("checkpoints/single_agent/two_stage/customer_backlog_mean.npy", allow_pickle=True)
S2_customer_backlog_std = np.load("checkpoints/single_agent/two_stage/customer_backlog_std.npy", allow_pickle=True)
S2_profit = np.load("checkpoints/single_agent/two_stage/profit.npy", allow_pickle=True)

S8_results = np.load("checkpoints/single_agent/eight_stage/results.npy", allow_pickle=True)
S8_time = np.load("checkpoints/single_agent/eight_stage/time.npy", allow_pickle=True)
S8_reward_mean = np.load("checkpoints/single_agent/eight_stage/reward_mean.npy", allow_pickle=True)
S8_reward_std = np.load("checkpoints/single_agent/eight_stage/reward_std.npy", allow_pickle=True)
S8_inventory_mean = np.load("checkpoints/single_agent/eight_stage/inventory_mean.npy", allow_pickle=True)
S8_inventory_std = np.load("checkpoints/single_agent/eight_stage/inventory_std.npy", allow_pickle=True)
S8_backlog_mean = np.load("checkpoints/single_agent/eight_stage/backlog_mean.npy", allow_pickle=True)
S8_backlog_std = np.load("checkpoints/single_agent/eight_stage/backlog_std.npy", allow_pickle=True)
S8_customer_backlog_mean = np.load("checkpoints/single_agent/eight_stage/customer_backlog_mean.npy", allow_pickle=True)
S8_customer_backlog_std = np.load("checkpoints/single_agent/eight_stage/customer_backlog_std.npy", allow_pickle=True)
S8_profit = np.load("checkpoints/single_agent/eight_stage/profit.npy", allow_pickle=True)

S4N2_results = np.load("checkpoints/single_agent/four_stage_noise_2/results.npy", allow_pickle=True)
S4N2_time = np.load("checkpoints/single_agent/four_stage_noise_2/time.npy", allow_pickle=True)
S4N2_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_2/reward_mean.npy", allow_pickle=True)
S4N2_reward_std = np.load("checkpoints/single_agent/four_stage_noise_2/reward_std.npy", allow_pickle=True)
S4N2_inventory_mean = np.load("checkpoints/single_agent/four_stage_noise_2/inventory_mean.npy", allow_pickle=True)
S4N2_inventory_std = np.load("checkpoints/single_agent/four_stage_noise_2/inventory_std.npy", allow_pickle=True)
S4N2_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_2/backlog_mean.npy", allow_pickle=True)
S4N2_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_2/backlog_std.npy", allow_pickle=True)
S4N2_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_2/customer_backlog_mean.npy", allow_pickle=True)
S4N2_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_2/customer_backlog_std.npy", allow_pickle=True)
S4N2_profit = np.load("checkpoints/single_agent/four_stage_noise_2/profit.npy", allow_pickle=True)

S4N5_results = np.load("checkpoints/single_agent/four_stage_noise_5/results.npy", allow_pickle=True)
S4N5_time = np.load("checkpoints/single_agent/four_stage_noise_5/time.npy", allow_pickle=True)
S4N5_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_5/reward_mean.npy", allow_pickle=True)
S4N5_reward_std = np.load("checkpoints/single_agent/four_stage_noise_5/reward_std.npy", allow_pickle=True)
S4N5_inventory_mean = np.load("checkpoints/single_agent/four_stage_noise_5/inventory_mean.npy", allow_pickle=True)
S4N5_inventory_std = np.load("checkpoints/single_agent/four_stage_noise_5/inventory_std.npy", allow_pickle=True)
S4N5_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_5/backlog_mean.npy", allow_pickle=True)
S4N5_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_5/backlog_std.npy", allow_pickle=True)
S4N5_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_5/customer_backlog_mean.npy", allow_pickle=True)
S4N5_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_5/customer_backlog_std.npy", allow_pickle=True)
S4N5_profit = np.load("checkpoints/single_agent/four_stage_noise_5/profit.npy", allow_pickle=True)

S4N10_results = np.load("checkpoints/single_agent/four_stage_noise_10/results.npy", allow_pickle=True)
S4N10_time = np.load("checkpoints/single_agent/four_stage_noise_10/time.npy", allow_pickle=True)
S4N10_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_10/reward_mean.npy", allow_pickle=True)
S4N10_reward_std = np.load("checkpoints/single_agent/four_stage_noise_10/reward_std.npy", allow_pickle=True)
S4N10_inventory_mean = np.load("checkpoints/single_agent/four_stage_noise_10/inventory_mean.npy", allow_pickle=True)
S4N10_inventory_std = np.load("checkpoints/single_agent/four_stage_noise_10/inventory_std.npy", allow_pickle=True)
S4N10_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_10/backlog_mean.npy", allow_pickle=True)
S4N10_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_10/backlog_std.npy", allow_pickle=True)
S4N10_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_10/customer_backlog_mean.npy", allow_pickle=True)
S4N10_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_10/customer_backlog_std.npy", allow_pickle=True)
S4N10_profit = np.load("checkpoints/single_agent/four_stage_noise_10/profit.npy", allow_pickle=True)

S4N20_results = np.load("checkpoints/single_agent/four_stage_noise_20/results.npy", allow_pickle=True)
S4N20_time = np.load("checkpoints/single_agent/four_stage_noise_20/time.npy", allow_pickle=True)
S4N20_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_20/reward_mean.npy", allow_pickle=True)
S4N20_reward_std = np.load("checkpoints/single_agent/four_stage_noise_20/reward_std.npy", allow_pickle=True)
S4N20_inventory_mean = np.load("checkpoints/single_agent/four_stage_noise_20/inventory_mean.npy", allow_pickle=True)
S4N20_inventory_std = np.load("checkpoints/single_agent/four_stage_noise_20/inventory_std.npy", allow_pickle=True)
S4N20_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_20/backlog_mean.npy", allow_pickle=True)
S4N20_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_20/backlog_std.npy", allow_pickle=True)
S4N20_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_20/customer_backlog_mean.npy", allow_pickle=True)
S4N20_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_20/customer_backlog_std.npy", allow_pickle=True)
S4N20_profit = np.load("checkpoints/single_agent/four_stage_noise_20/profit.npy", allow_pickle=True)

S4N30_results = np.load("checkpoints/single_agent/four_stage_noise_30/results.npy", allow_pickle=True)
S4N30_time = np.load("checkpoints/single_agent/four_stage_noise_30/time.npy", allow_pickle=True)
S4N30_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_30/reward_mean.npy", allow_pickle=True)
S4N30_reward_std = np.load("checkpoints/single_agent/four_stage_noise_30/reward_std.npy", allow_pickle=True)
S4N30_inventory_mean = np.load("checkpoints/single_agent/four_stage_noise_30/inventory_mean.npy", allow_pickle=True)
S4N30_inventory_std = np.load("checkpoints/single_agent/four_stage_noise_30/inventory_std.npy", allow_pickle=True)
S4N30_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_30/backlog_mean.npy", allow_pickle=True)
S4N30_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_30/backlog_std.npy", allow_pickle=True)
S4N30_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_30/customer_backlog_mean.npy", allow_pickle=True)
S4N30_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_30/customer_backlog_std.npy", allow_pickle=True)
S4N30_profit = np.load("checkpoints/single_agent/four_stage_noise_30/profit.npy", allow_pickle=True)

S4N40_results = np.load("checkpoints/single_agent/four_stage_noise_40/results.npy", allow_pickle=True)
S4N40_time = np.load("checkpoints/single_agent/four_stage_noise_40/time.npy", allow_pickle=True)
S4N40_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_40/reward_mean.npy", allow_pickle=True)
S4N40_reward_std = np.load("checkpoints/single_agent/four_stage_noise_40/reward_std.npy", allow_pickle=True)
S4N40_inventory_mean = np.load("checkpoints/single_agent/four_stage_noise_40/inventory_mean.npy", allow_pickle=True)
S4N40_inventory_std = np.load("checkpoints/single_agent/four_stage_noise_40/inventory_std.npy", allow_pickle=True)
S4N40_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_40/backlog_mean.npy", allow_pickle=True)
S4N40_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_40/backlog_std.npy", allow_pickle=True)
S4N40_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_40/customer_backlog_mean.npy", allow_pickle=True)
S4N40_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_40/customer_backlog_std.npy", allow_pickle=True)
S4N40_profit = np.load("checkpoints/single_agent/four_stage_noise_40/profit.npy", allow_pickle=True)

S4N50_results = np.load("checkpoints/single_agent/four_stage_noise_50/results.npy", allow_pickle=True)
S4N50_time = np.load("checkpoints/single_agent/four_stage_noise_50/time.npy", allow_pickle=True)
S4N50_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_50/reward_mean.npy", allow_pickle=True)
S4N50_reward_std = np.load("checkpoints/single_agent/four_stage_noise_50/reward_std.npy", allow_pickle=True)
S4N50_inventory_mean = np.load("checkpoints/single_agent/four_stage_noise_50/inventory_mean.npy", allow_pickle=True)
S4N50_inventory_std = np.load("checkpoints/single_agent/four_stage_noise_50/inventory_std.npy", allow_pickle=True)
S4N50_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_50/backlog_mean.npy", allow_pickle=True)
S4N50_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_50/backlog_std.npy", allow_pickle=True)
S4N50_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_noise_50/customer_backlog_mean.npy", allow_pickle=True)
S4N50_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_noise_50/customer_backlog_std.npy", allow_pickle=True)
S4N50_profit = np.load("checkpoints/single_agent/four_stage_noise_50/profit.npy", allow_pickle=True)

S4D10_results = np.load("checkpoints/single_agent/four_stage_delay_10/results.npy", allow_pickle=True)
S4D10_time = np.load("checkpoints/single_agent/four_stage_delay_10/time.npy", allow_pickle=True)
S4D10_reward_mean = np.load("checkpoints/single_agent/four_stage_delay_10/reward_mean.npy", allow_pickle=True)
S4D10_reward_std = np.load("checkpoints/single_agent/four_stage_delay_10/reward_std.npy", allow_pickle=True)
S4D10_inventory_mean = np.load("checkpoints/single_agent/four_stage_delay_10/inventory_mean.npy", allow_pickle=True)
S4D10_inventory_std = np.load("checkpoints/single_agent/four_stage_delay_10/inventory_std.npy", allow_pickle=True)
S4D10_backlog_mean = np.load("checkpoints/single_agent/four_stage_delay_10/backlog_mean.npy", allow_pickle=True)
S4D10_backlog_std = np.load("checkpoints/single_agent/four_stage_delay_10/backlog_std.npy", allow_pickle=True)
S4D10_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_delay_10/customer_backlog_mean.npy", allow_pickle=True)
S4D10_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_delay_10/customer_backlog_std.npy", allow_pickle=True)
S4D10_profit = np.load("checkpoints/single_agent/four_stage_delay_10/profit.npy", allow_pickle=True)

S4D20_results = np.load("checkpoints/single_agent/four_stage_delay_20/results.npy", allow_pickle=True)
S4D20_time = np.load("checkpoints/single_agent/four_stage_delay_20/time.npy", allow_pickle=True)
S4D20_reward_mean = np.load("checkpoints/single_agent/four_stage_delay_20/reward_mean.npy", allow_pickle=True)
S4D20_reward_std = np.load("checkpoints/single_agent/four_stage_delay_20/reward_std.npy", allow_pickle=True)
S4D20_inventory_mean = np.load("checkpoints/single_agent/four_stage_delay_20/inventory_mean.npy", allow_pickle=True)
S4D20_inventory_std = np.load("checkpoints/single_agent/four_stage_delay_20/inventory_std.npy", allow_pickle=True)
S4D20_backlog_mean = np.load("checkpoints/single_agent/four_stage_delay_20/backlog_mean.npy", allow_pickle=True)
S4D20_backlog_std = np.load("checkpoints/single_agent/four_stage_delay_20/backlog_std.npy", allow_pickle=True)
S4D20_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_delay_20/customer_backlog_mean.npy", allow_pickle=True)
S4D20_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_delay_20/customer_backlog_std.npy", allow_pickle=True)
S4D20_profit = np.load("checkpoints/single_agent/four_stage_delay_20/profit.npy", allow_pickle=True)

S4D30_results = np.load("checkpoints/single_agent/four_stage_delay_30/results.npy", allow_pickle=True)
S4D30_time = np.load("checkpoints/single_agent/four_stage_delay_30/time.npy", allow_pickle=True)
S4D30_reward_mean = np.load("checkpoints/single_agent/four_stage_delay_30/reward_mean.npy", allow_pickle=True)
S4D30_reward_std = np.load("checkpoints/single_agent/four_stage_delay_30/reward_std.npy", allow_pickle=True)
S4D30_inventory_mean = np.load("checkpoints/single_agent/four_stage_delay_30/inventory_mean.npy", allow_pickle=True)
S4D30_inventory_std = np.load("checkpoints/single_agent/four_stage_delay_30/inventory_std.npy", allow_pickle=True)
S4D30_backlog_mean = np.load("checkpoints/single_agent/four_stage_delay_30/backlog_mean.npy", allow_pickle=True)
S4D30_backlog_std = np.load("checkpoints/single_agent/four_stage_delay_30/backlog_std.npy", allow_pickle=True)
S4D30_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_delay_30/customer_backlog_mean.npy", allow_pickle=True)
S4D30_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_delay_30/customer_backlog_std.npy", allow_pickle=True)
S4D30_profit = np.load("checkpoints/single_agent/four_stage_delay_30/profit.npy", allow_pickle=True)

S4D40_results = np.load("checkpoints/single_agent/four_stage_delay_40/results.npy", allow_pickle=True)
S4D40_time = np.load("checkpoints/single_agent/four_stage_delay_40/time.npy", allow_pickle=True)
S4D40_reward_mean = np.load("checkpoints/single_agent/four_stage_delay_40/reward_mean.npy", allow_pickle=True)
S4D40_reward_std = np.load("checkpoints/single_agent/four_stage_delay_40/reward_std.npy", allow_pickle=True)
S4D40_inventory_mean = np.load("checkpoints/single_agent/four_stage_delay_40/inventory_mean.npy", allow_pickle=True)
S4D40_inventory_std = np.load("checkpoints/single_agent/four_stage_delay_40/inventory_std.npy", allow_pickle=True)
S4D40_backlog_mean = np.load("checkpoints/single_agent/four_stage_delay_40/backlog_mean.npy", allow_pickle=True)
S4D40_backlog_std = np.load("checkpoints/single_agent/four_stage_delay_40/backlog_std.npy", allow_pickle=True)
S4D40_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_delay_40/customer_backlog_mean.npy", allow_pickle=True)
S4D40_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_delay_40/customer_backlog_std.npy", allow_pickle=True)
S4D40_profit = np.load("checkpoints/single_agent/four_stage_delay_40/profit.npy", allow_pickle=True)

S4D50_results = np.load("checkpoints/single_agent/four_stage_delay_50/results.npy", allow_pickle=True)
S4D50_time = np.load("checkpoints/single_agent/four_stage_delay_50/time.npy", allow_pickle=True)
S4D50_reward_mean = np.load("checkpoints/single_agent/four_stage_delay_50/reward_mean.npy", allow_pickle=True)
S4D50_reward_std = np.load("checkpoints/single_agent/four_stage_delay_50/reward_std.npy", allow_pickle=True)
S4D50_inventory_mean = np.load("checkpoints/single_agent/four_stage_delay_50/inventory_mean.npy", allow_pickle=True)
S4D50_inventory_std = np.load("checkpoints/single_agent/four_stage_delay_50/inventory_std.npy", allow_pickle=True)
S4D50_backlog_mean = np.load("checkpoints/single_agent/four_stage_delay_50/backlog_mean.npy", allow_pickle=True)
S4D50_backlog_std = np.load("checkpoints/single_agent/four_stage_delay_50/backlog_std.npy", allow_pickle=True)
S4D50_customer_backlog_mean = np.load("checkpoints/single_agent/four_stage_delay_50/customer_backlog_mean.npy", allow_pickle=True)
S4D50_customer_backlog_std = np.load("checkpoints/single_agent/four_stage_delay_50/customer_backlog_std.npy", allow_pickle=True)
S4D50_profit = np.load("checkpoints/single_agent/four_stage_delay_50/profit.npy", allow_pickle=True)

#%% Load Noise trained Single RL Data
NS4N10_results = np.load("checkpoints/single_agent_noisy_demand/four_stage_10/results.npy", allow_pickle=True)
NS4N10_time = np.load("checkpoints/single_agent_noisy_demand/four_stage_10/time.npy", allow_pickle=True)
NS4N10_reward_mean = np.load("checkpoints/single_agent_noisy_demand/four_stage_10/reward_mean.npy", allow_pickle=True)
NS4N10_reward_std = np.load("checkpoints/single_agent_noisy_demand/four_stage_10/reward_std.npy", allow_pickle=True)
NS4N10_profit = np.load("checkpoints/single_agent_noisy_demand/four_stage_10/profit.npy", allow_pickle=True)

NS4N20_results = np.load("checkpoints/single_agent_noisy_demand/four_stage_20/results.npy", allow_pickle=True)
NS4N20_time = np.load("checkpoints/single_agent_noisy_demand/four_stage_20/time.npy", allow_pickle=True)
NS4N20_reward_mean = np.load("checkpoints/single_agent_noisy_demand/four_stage_20/reward_mean.npy", allow_pickle=True)
NS4N20_reward_std = np.load("checkpoints/single_agent_noisy_demand/four_stage_20/reward_std.npy", allow_pickle=True)
NS4N20_profit = np.load("checkpoints/single_agent_noisy_demand/four_stage_20/profit.npy", allow_pickle=True)

NS4N30_results = np.load("checkpoints/single_agent_noisy_demand/four_stage_30/results.npy", allow_pickle=True)
NS4N30_time = np.load("checkpoints/single_agent_noisy_demand/four_stage_30/time.npy", allow_pickle=True)
NS4N30_reward_mean = np.load("checkpoints/single_agent_noisy_demand/four_stage_30/reward_mean.npy", allow_pickle=True)
NS4N30_reward_std = np.load("checkpoints/single_agent_noisy_demand/four_stage_30/reward_std.npy", allow_pickle=True)
NS4N30_profit = np.load("checkpoints/single_agent_noisy_demand/four_stage_30/profit.npy", allow_pickle=True)

NS4N40_results = np.load("checkpoints/single_agent_noisy_demand/four_stage_40/results.npy", allow_pickle=True)
NS4N40_time = np.load("checkpoints/single_agent_noisy_demand/four_stage_40/time.npy", allow_pickle=True)
NS4N40_reward_mean = np.load("checkpoints/single_agent_noisy_demand/four_stage_40/reward_mean.npy", allow_pickle=True)
NS4N40_reward_std = np.load("checkpoints/single_agent_noisy_demand/four_stage_40/reward_std.npy", allow_pickle=True)
NS4N40_profit = np.load("checkpoints/single_agent_noisy_demand/four_stage_40/profit.npy", allow_pickle=True)

NS4N50_results = np.load("checkpoints/single_agent_noisy_demand/four_stage_50/results.npy", allow_pickle=True)
NS4N50_time = np.load("checkpoints/single_agent_noisy_demand/four_stage_50/time.npy", allow_pickle=True)
NS4N50_reward_mean = np.load("checkpoints/single_agent_noisy_demand/four_stage_50/reward_mean.npy", allow_pickle=True)
NS4N50_reward_std = np.load("checkpoints/single_agent_noisy_demand/four_stage_50/reward_std.npy", allow_pickle=True)
NS4N50_profit = np.load("checkpoints/single_agent_noisy_demand/four_stage_50/profit.npy", allow_pickle=True)

NS4D10_results = np.load("checkpoints/single_agent_noisy_delay/four_stage_10/results.npy", allow_pickle=True)
NS4D10_time = np.load("checkpoints/single_agent_noisy_delay/four_stage_10/time.npy", allow_pickle=True)
NS4D10_reward_mean = np.load("checkpoints/single_agent_noisy_delay/four_stage_10/reward_mean.npy", allow_pickle=True)
NS4D10_reward_std = np.load("checkpoints/single_agent_noisy_delay/four_stage_10/reward_std.npy", allow_pickle=True)
NS4D10_profit = np.load("checkpoints/single_agent_noisy_delay/four_stage_10/profit.npy", allow_pickle=True)

NS4D20_results = np.load("checkpoints/single_agent_noisy_delay/four_stage_20/results.npy", allow_pickle=True)
NS4D20_time = np.load("checkpoints/single_agent_noisy_delay/four_stage_20/time.npy", allow_pickle=True)
NS4D20_reward_mean = np.load("checkpoints/single_agent_noisy_delay/four_stage_20/reward_mean.npy", allow_pickle=True)
NS4D20_reward_std = np.load("checkpoints/single_agent_noisy_delay/four_stage_20/reward_std.npy", allow_pickle=True)
NS4D20_profit = np.load("checkpoints/single_agent_noisy_delay/four_stage_20/profit.npy", allow_pickle=True)

NS4D30_results = np.load("checkpoints/single_agent_noisy_delay/four_stage_30/results.npy", allow_pickle=True)
NS4D30_time = np.load("checkpoints/single_agent_noisy_delay/four_stage_30/time.npy", allow_pickle=True)
NS4D30_reward_mean = np.load("checkpoints/single_agent_noisy_delay/four_stage_30/reward_mean.npy", allow_pickle=True)
NS4D30_reward_std = np.load("checkpoints/single_agent_noisy_delay/four_stage_30/reward_std.npy", allow_pickle=True)
NS4D30_profit = np.load("checkpoints/single_agent_noisy_delay/four_stage_30/profit.npy", allow_pickle=True)

NS4D40_results = np.load("checkpoints/single_agent_noisy_delay/four_stage_40/results.npy", allow_pickle=True)
NS4D40_time = np.load("checkpoints/single_agent_noisy_delay/four_stage_40/time.npy", allow_pickle=True)
NS4D40_reward_mean = np.load("checkpoints/single_agent_noisy_delay/four_stage_40/reward_mean.npy", allow_pickle=True)
NS4D40_reward_std = np.load("checkpoints/single_agent_noisy_delay/four_stage_40/reward_std.npy", allow_pickle=True)
NS4D40_profit = np.load("checkpoints/single_agent_noisy_delay/four_stage_40/profit.npy", allow_pickle=True)

NS4D50_results = np.load("checkpoints/single_agent_noisy_delay/four_stage_50/results.npy", allow_pickle=True)
NS4D50_time = np.load("checkpoints/single_agent_noisy_delay/four_stage_50/time.npy", allow_pickle=True)
NS4D50_reward_mean = np.load("checkpoints/single_agent_noisy_delay/four_stage_50/reward_mean.npy", allow_pickle=True)
NS4D50_reward_std = np.load("checkpoints/single_agent_noisy_delay/four_stage_50/reward_std.npy", allow_pickle=True)
NS4D50_profit = np.load("checkpoints/single_agent_noisy_delay/four_stage_50/profit.npy", allow_pickle=True)
#%% Load MA Rl Data
MA4_results = np.load("checkpoints/multi_agent/four_stage/results.npy", allow_pickle=True)
MA4_time = np.load("checkpoints/multi_agent/four_stage/time.npy", allow_pickle=True)
MA4_reward_mean = np.load("checkpoints/multi_agent/four_stage/reward_mean.npy", allow_pickle=True)
MA4_reward_std = np.load("checkpoints/multi_agent/four_stage/reward_std.npy", allow_pickle=True)
MA4_inventory_mean = np.load("checkpoints/multi_agent/four_stage/inventory_mean.npy", allow_pickle=True)
MA4_inventory_std = np.load("checkpoints/multi_agent/four_stage/inventory_std.npy", allow_pickle=True)
MA4_backlog_mean = np.load("checkpoints/multi_agent/four_stage/backlog_mean.npy", allow_pickle=True)
MA4_backlog_std = np.load("checkpoints/multi_agent/four_stage/backlog_std.npy", allow_pickle=True)
MA4_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage/customer_backlog_mean.npy", allow_pickle=True)
MA4_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage/customer_backlog_std.npy", allow_pickle=True)
MA4_profit = np.load("checkpoints/multi_agent/four_stage/profit.npy", allow_pickle=True)

MA4I_results = np.load("checkpoints/multi_agent/four_stage_independent/results.npy", allow_pickle=True)
MA4I_time = np.load("checkpoints/multi_agent/four_stage_independent/time.npy", allow_pickle=True)
MA4I_reward_mean = np.load("checkpoints/multi_agent/four_stage_independent/reward_mean.npy", allow_pickle=True)
MA4I_reward_std = np.load("checkpoints/multi_agent/four_stage_independent/reward_std.npy", allow_pickle=True)
MA4i_inventory_mean = np.load("checkpoints/multi_agent/four_stage_independent/inventory_mean.npy", allow_pickle=True)
MA4I_inventory_std = np.load("checkpoints/multi_agent/four_stage_independent/inventory_std.npy", allow_pickle=True)
MA4I_backlog_mean = np.load("checkpoints/multi_agent/four_stage_independent/backlog_mean.npy", allow_pickle=True)
MA4I_backlog_std = np.load("checkpoints/multi_agent/four_stage_independent/backlog_std.npy", allow_pickle=True)
MA4I_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_independent/customer_backlog_mean.npy", allow_pickle=True)
MA4I_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_independent/customer_backlog_std.npy", allow_pickle=True)
MA4I_profit = np.load("checkpoints/multi_agent/four_stage_independent/profit.npy", allow_pickle=True)

MA2_results = np.load("checkpoints/multi_agent/two_stage/results.npy", allow_pickle=True)
MA2_time = np.load("checkpoints/multi_agent/two_stage/time.npy", allow_pickle=True)
MA2_reward_mean = np.load("checkpoints/multi_agent/two_stage/reward_mean.npy", allow_pickle=True)
MA2_reward_std = np.load("checkpoints/multi_agent/two_stage/reward_std.npy", allow_pickle=True)
MA2_inventory_mean = np.load("checkpoints/multi_agent/two_stage/inventory_mean.npy", allow_pickle=True)
MA2_inventory_std = np.load("checkpoints/multi_agent/two_stage/inventory_std.npy", allow_pickle=True)
MA2_backlog_mean = np.load("checkpoints/multi_agent/two_stage/backlog_mean.npy", allow_pickle=True)
MA2_backlog_std = np.load("checkpoints/multi_agent/two_stage/backlog_std.npy", allow_pickle=True)
MA2_customer_backlog_mean = np.load("checkpoints/multi_agent/two_stage/customer_backlog_mean.npy", allow_pickle=True)
MA2_customer_backlog_std = np.load("checkpoints/multi_agent/two_stage/customer_backlog_std.npy", allow_pickle=True)
MA2_profit = np.load("checkpoints/multi_agent/two_stage/profit.npy", allow_pickle=True)

MA8_results = np.load("checkpoints/multi_agent/eight_stage/results.npy", allow_pickle=True)
MA8_time = np.load("checkpoints/multi_agent/eight_stage/time.npy", allow_pickle=True)
MA8_reward_mean = np.load("checkpoints/multi_agent/eight_stage/reward_mean.npy", allow_pickle=True)
MA8_reward_std = np.load("checkpoints/multi_agent/eight_stage/reward_std.npy", allow_pickle=True)
MA8_inventory_mean = np.load("checkpoints/multi_agent/eight_stage/inventory_mean.npy", allow_pickle=True)
MA8_inventory_std = np.load("checkpoints/multi_agent/eight_stage/inventory_std.npy", allow_pickle=True)
MA8_backlog_mean = np.load("checkpoints/multi_agent/eight_stage/backlog_mean.npy", allow_pickle=True)
MA8_backlog_std = np.load("checkpoints/multi_agent/eight_stage/backlog_std.npy", allow_pickle=True)
MA8_customer_backlog_mean = np.load("checkpoints/multi_agent/eight_stage/customer_backlog_mean.npy", allow_pickle=True)
MA8_customer_backlog_std = np.load("checkpoints/multi_agent/eight_stage/customer_backlog_std.npy", allow_pickle=True)
MA8_profit = np.load("checkpoints/multi_agent/eight_stage/profit.npy", allow_pickle=True)


MA4N10_time = np.load("checkpoints/multi_agent/four_stage_noise_10/time.npy", allow_pickle=True)
MA4N10_reward_mean = np.load("checkpoints/multi_agent/four_stage_noise_10/reward_mean.npy", allow_pickle=True)
MA4N10_reward_std = np.load("checkpoints/multi_agent/four_stage_noise_10/reward_std.npy", allow_pickle=True)
MA4N10_inventory_mean = np.load("checkpoints/multi_agent/four_stage_noise_10/inventory_mean.npy", allow_pickle=True)
MA4N10_inventory_std = np.load("checkpoints/multi_agent/four_stage_noise_10/inventory_std.npy", allow_pickle=True)
MA4N10_backlog_mean = np.load("checkpoints/multi_agent/four_stage_noise_10/backlog_mean.npy", allow_pickle=True)
MA4N10_backlog_std = np.load("checkpoints/multi_agent/four_stage_noise_10/backlog_std.npy", allow_pickle=True)
MA4N10_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_noise_10/customer_backlog_mean.npy", allow_pickle=True)
MA4N10_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_noise_10/customer_backlog_std.npy", allow_pickle=True)
MA4N10_profit = np.load("checkpoints/multi_agent/four_stage_noise_10/profit.npy", allow_pickle=True)

MA4N20_time = np.load("checkpoints/multi_agent/four_stage_noise_20/time.npy", allow_pickle=True)
MA4N20_reward_mean = np.load("checkpoints/multi_agent/four_stage_noise_20/reward_mean.npy", allow_pickle=True)
MA4N20_reward_std = np.load("checkpoints/multi_agent/four_stage_noise_20/reward_std.npy", allow_pickle=True)
MA4N20_inventory_mean = np.load("checkpoints/multi_agent/four_stage_noise_20/inventory_mean.npy", allow_pickle=True)
MA4N20_inventory_std = np.load("checkpoints/multi_agent/four_stage_noise_20/inventory_std.npy", allow_pickle=True)
MA4N20_backlog_mean = np.load("checkpoints/multi_agent/four_stage_noise_20/backlog_mean.npy", allow_pickle=True)
MA4N20_backlog_std = np.load("checkpoints/multi_agent/four_stage_noise_20/backlog_std.npy", allow_pickle=True)
MA4N20_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_noise_20/customer_backlog_mean.npy", allow_pickle=True)
MA4N20_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_noise_20/customer_backlog_std.npy", allow_pickle=True)
MA4N20_profit = np.load("checkpoints/multi_agent/four_stage_noise_20/profit.npy", allow_pickle=True)

MA4N30_time = np.load("checkpoints/multi_agent/four_stage_noise_30/time.npy", allow_pickle=True)
MA4N30_reward_mean = np.load("checkpoints/multi_agent/four_stage_noise_30/reward_mean.npy", allow_pickle=True)
MA4N30_reward_std = np.load("checkpoints/multi_agent/four_stage_noise_30/reward_std.npy", allow_pickle=True)
MA4N30_inventory_mean = np.load("checkpoints/multi_agent/four_stage_noise_30/inventory_mean.npy", allow_pickle=True)
MA4N30_inventory_std = np.load("checkpoints/multi_agent/four_stage_noise_30/inventory_std.npy", allow_pickle=True)
MA4N30_backlog_mean = np.load("checkpoints/multi_agent/four_stage_noise_30/backlog_mean.npy", allow_pickle=True)
MA4N30_backlog_std = np.load("checkpoints/multi_agent/four_stage_noise_30/backlog_std.npy", allow_pickle=True)
MA4N30_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_noise_30/customer_backlog_mean.npy", allow_pickle=True)
MA4N30_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_noise_30/customer_backlog_std.npy", allow_pickle=True)
MA4N30_profit = np.load("checkpoints/multi_agent/four_stage_noise_30/profit.npy", allow_pickle=True)

MA4N50_time = np.load("checkpoints/multi_agent/four_stage_noise_50/time.npy", allow_pickle=True)
MA4N50_reward_mean = np.load("checkpoints/multi_agent/four_stage_noise_50/reward_mean.npy", allow_pickle=True)
MA4N50_reward_std = np.load("checkpoints/multi_agent/four_stage_noise_50/reward_std.npy", allow_pickle=True)
MA4N50_inventory_mean = np.load("checkpoints/multi_agent/four_stage_noise_50/inventory_mean.npy", allow_pickle=True)
MA4N50_inventory_std = np.load("checkpoints/multi_agent/four_stage_noise_50/inventory_std.npy", allow_pickle=True)
MA4N50_backlog_mean = np.load("checkpoints/multi_agent/four_stage_noise_50/backlog_mean.npy", allow_pickle=True)
MA4N50_backlog_std = np.load("checkpoints/multi_agent/four_stage_noise_50/backlog_std.npy", allow_pickle=True)
MA4N50_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_noise_50/customer_backlog_mean.npy", allow_pickle=True)
MA4N50_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_noise_50/customer_backlog_std.npy", allow_pickle=True)
MA4N50_profit = np.load("checkpoints/multi_agent/four_stage_noise_50/profit.npy", allow_pickle=True)

MA4D10_time = np.load("checkpoints/multi_agent/four_stage_delay_10/time.npy", allow_pickle=True)
MA4D10_reward_mean = np.load("checkpoints/multi_agent/four_stage_delay_10/reward_mean.npy", allow_pickle=True)
MA4D10_reward_std = np.load("checkpoints/multi_agent/four_stage_delay_10/reward_std.npy", allow_pickle=True)
MA4D10_inventory_mean = np.load("checkpoints/multi_agent/four_stage_delay_10/inventory_mean.npy", allow_pickle=True)
MA4D10_inventory_std = np.load("checkpoints/multi_agent/four_stage_delay_10/inventory_std.npy", allow_pickle=True)
MA4D10_backlog_mean = np.load("checkpoints/multi_agent/four_stage_delay_10/backlog_mean.npy", allow_pickle=True)
MA4D10_backlog_std = np.load("checkpoints/multi_agent/four_stage_delay_10/backlog_std.npy", allow_pickle=True)
MA4D10_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_delay_10/customer_backlog_mean.npy", allow_pickle=True)
MA4D10_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_delay_10/customer_backlog_std.npy", allow_pickle=True)
MA4D10_profit = np.load("checkpoints/multi_agent/four_stage_delay_10/profit.npy", allow_pickle=True)

MA4D20_time = np.load("checkpoints/multi_agent/four_stage_delay_20/time.npy", allow_pickle=True)
MA4D20_reward_mean = np.load("checkpoints/multi_agent/four_stage_delay_20/reward_mean.npy", allow_pickle=True)
MA4D20_reward_std = np.load("checkpoints/multi_agent/four_stage_delay_20/reward_std.npy", allow_pickle=True)
MA4D20_inventory_mean = np.load("checkpoints/multi_agent/four_stage_delay_20/inventory_mean.npy", allow_pickle=True)
MA4D20_inventory_std = np.load("checkpoints/multi_agent/four_stage_delay_20/inventory_std.npy", allow_pickle=True)
MA4D20_backlog_mean = np.load("checkpoints/multi_agent/four_stage_delay_20/backlog_mean.npy", allow_pickle=True)
MA4D20_backlog_std = np.load("checkpoints/multi_agent/four_stage_delay_20/backlog_std.npy", allow_pickle=True)
MA4D20_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_delay_20/customer_backlog_mean.npy", allow_pickle=True)
MA4D20_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_delay_20/customer_backlog_std.npy", allow_pickle=True)
MA4D20_profit = np.load("checkpoints/multi_agent/four_stage_delay_20/profit.npy", allow_pickle=True)

MA4D30_time = np.load("checkpoints/multi_agent/four_stage_delay_30/time.npy", allow_pickle=True)
MA4D30_reward_mean = np.load("checkpoints/multi_agent/four_stage_delay_30/reward_mean.npy", allow_pickle=True)
MA4D30_reward_std = np.load("checkpoints/multi_agent/four_stage_delay_30/reward_std.npy", allow_pickle=True)
MA4D30_inventory_mean = np.load("checkpoints/multi_agent/four_stage_delay_30/inventory_mean.npy", allow_pickle=True)
MA4D30_inventory_std = np.load("checkpoints/multi_agent/four_stage_delay_30/inventory_std.npy", allow_pickle=True)
MA4D30_backlog_mean = np.load("checkpoints/multi_agent/four_stage_delay_30/backlog_mean.npy", allow_pickle=True)
MA4D30_backlog_std = np.load("checkpoints/multi_agent/four_stage_delay_30/backlog_std.npy", allow_pickle=True)
MA4D30_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_delay_30/customer_backlog_mean.npy", allow_pickle=True)
MA4D30_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_delay_30/customer_backlog_std.npy", allow_pickle=True)
MA4D30_profit = np.load("checkpoints/multi_agent/four_stage_delay_30/profit.npy", allow_pickle=True)

MA4N40_time = np.load("checkpoints/multi_agent/four_stage_noise_40/time.npy", allow_pickle=True)
MA4N40_reward_mean = np.load("checkpoints/multi_agent/four_stage_noise_40/reward_mean.npy", allow_pickle=True)
MA4N40_reward_std = np.load("checkpoints/multi_agent/four_stage_noise_40/reward_std.npy", allow_pickle=True)
MA4N40_inventory_mean = np.load("checkpoints/multi_agent/four_stage_noise_40/inventory_mean.npy", allow_pickle=True)
MA4N40_inventory_std = np.load("checkpoints/multi_agent/four_stage_noise_40/inventory_std.npy", allow_pickle=True)
MA4N40_backlog_mean = np.load("checkpoints/multi_agent/four_stage_noise_40/backlog_mean.npy", allow_pickle=True)
MA4N40_backlog_std = np.load("checkpoints/multi_agent/four_stage_noise_40/backlog_std.npy", allow_pickle=True)
MA4N40_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_noise_40/customer_backlog_mean.npy", allow_pickle=True)
MA4N40_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_noise_40/customer_backlog_std.npy", allow_pickle=True)
MA4N40_profit = np.load("checkpoints/multi_agent/four_stage_noise_40/profit.npy", allow_pickle=True)

MA4D40_time = np.load("checkpoints/multi_agent/four_stage_delay_40/time.npy", allow_pickle=True)
MA4D40_reward_mean = np.load("checkpoints/multi_agent/four_stage_delay_40/reward_mean.npy", allow_pickle=True)
MA4D40_reward_std = np.load("checkpoints/multi_agent/four_stage_delay_40/reward_std.npy", allow_pickle=True)
MA4D40_inventory_mean = np.load("checkpoints/multi_agent/four_stage_delay_40/inventory_mean.npy", allow_pickle=True)
MA4D40_inventory_std = np.load("checkpoints/multi_agent/four_stage_delay_40/inventory_std.npy", allow_pickle=True)
MA4D40_backlog_mean = np.load("checkpoints/multi_agent/four_stage_delay_40/backlog_mean.npy", allow_pickle=True)
MA4D40_backlog_std = np.load("checkpoints/multi_agent/four_stage_delay_40/backlog_std.npy", allow_pickle=True)
MA4D40_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_delay_40/customer_backlog_mean.npy", allow_pickle=True)
MA4D40_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_delay_40/customer_backlog_std.npy", allow_pickle=True)
MA4D40_profit = np.load("checkpoints/multi_agent/four_stage_delay_40/profit.npy", allow_pickle=True)

MA4D50_time = np.load("checkpoints/multi_agent/four_stage_delay_50/time.npy", allow_pickle=True)
MA4D50_reward_mean = np.load("checkpoints/multi_agent/four_stage_delay_50/reward_mean.npy", allow_pickle=True)
MA4D50_reward_std = np.load("checkpoints/multi_agent/four_stage_delay_50/reward_std.npy", allow_pickle=True)
MA4D50_inventory_mean = np.load("checkpoints/multi_agent/four_stage_delay_50/inventory_mean.npy", allow_pickle=True)
MA4D50_inventory_std = np.load("checkpoints/multi_agent/four_stage_delay_50/inventory_std.npy", allow_pickle=True)
MA4D50_backlog_mean = np.load("checkpoints/multi_agent/four_stage_delay_50/backlog_mean.npy", allow_pickle=True)
MA4D50_backlog_std = np.load("checkpoints/multi_agent/four_stage_delay_50/backlog_std.npy", allow_pickle=True)
MA4D50_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_delay_50/customer_backlog_mean.npy", allow_pickle=True)
MA4D50_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_delay_50/customer_backlog_std.npy", allow_pickle=True)
MA4D50_profit = np.load("checkpoints/multi_agent/four_stage_delay_50/profit.npy", allow_pickle=True)

#%% Load noise trained MA Rl Data
NMA4N10_time = np.load("checkpoints/multi_agent_noisy_demand/four_stage_10/time.npy", allow_pickle=True)
NMA4N10_reward_mean = np.load("checkpoints/multi_agent_noisy_demand/four_stage_10/reward_mean.npy", allow_pickle=True)
NMA4N10_reward_std = np.load("checkpoints/multi_agent_noisy_demand/four_stage_10/reward_std.npy", allow_pickle=True)
NMA4N10_profit = np.load("checkpoints/multi_agent_noisy_demand/four_stage_10/profit.npy", allow_pickle=True)

NMA4N20_time = np.load("checkpoints/multi_agent_noisy_demand/four_stage_20/time.npy", allow_pickle=True)
NMA4N20_reward_mean = np.load("checkpoints/multi_agent_noisy_demand/four_stage_20/reward_mean.npy", allow_pickle=True)
NMA4N20_reward_std = np.load("checkpoints/multi_agent_noisy_demand/four_stage_20/reward_std.npy", allow_pickle=True)
NMA4N20_profit = np.load("checkpoints/multi_agent_noisy_demand/four_stage_20/profit.npy", allow_pickle=True)

NMA4N30_time = np.load("checkpoints/multi_agent_noisy_demand/four_stage_30/time.npy", allow_pickle=True)
NMA4N30_reward_mean = np.load("checkpoints/multi_agent_noisy_demand/four_stage_30/reward_mean.npy", allow_pickle=True)
NMA4N30_reward_std = np.load("checkpoints/multi_agent_noisy_demand/four_stage_30/reward_std.npy", allow_pickle=True)
NMA4N30_profit = np.load("checkpoints/multi_agent_noisy_demand/four_stage_30/profit.npy", allow_pickle=True)

NMA4N40_time = np.load("checkpoints/multi_agent_noisy_demand/four_stage_40/time.npy", allow_pickle=True)
NMA4N40_reward_mean = np.load("checkpoints/multi_agent_noisy_demand/four_stage_40/reward_mean.npy", allow_pickle=True)
NMA4N40_reward_std = np.load("checkpoints/multi_agent_noisy_demand/four_stage_40/reward_std.npy", allow_pickle=True)
NMA4N40_profit = np.load("checkpoints/multi_agent_noisy_demand/four_stage_40/profit.npy", allow_pickle=True)

NMA4N50_time = np.load("checkpoints/multi_agent_noisy_demand/four_stage_50/time.npy", allow_pickle=True)
NMA4N50_reward_mean = np.load("checkpoints/multi_agent_noisy_demand/four_stage_50/reward_mean.npy", allow_pickle=True)
NMA4N50_reward_std = np.load("checkpoints/multi_agent_noisy_demand/four_stage_50/reward_std.npy", allow_pickle=True)
NMA4N50_profit = np.load("checkpoints/multi_agent_noisy_demand/four_stage_50/profit.npy", allow_pickle=True)

NMA4D10_time = np.load("checkpoints/multi_agent_noisy_delay/four_stage_10/time.npy", allow_pickle=True)
NMA4D10_reward_mean = np.load("checkpoints/multi_agent_noisy_delay/four_stage_10/reward_mean.npy", allow_pickle=True)
NMA4D10_reward_std = np.load("checkpoints/multi_agent_noisy_delay/four_stage_10/reward_std.npy", allow_pickle=True)
NMA4D10_profit = np.load("checkpoints/multi_agent_noisy_delay/four_stage_10/profit.npy", allow_pickle=True)

NMA4D20_time = np.load("checkpoints/multi_agent_noisy_delay/four_stage_20/time.npy", allow_pickle=True)
NMA4D20_reward_mean = np.load("checkpoints/multi_agent_noisy_delay/four_stage_20/reward_mean.npy", allow_pickle=True)
NMA4D20_reward_std = np.load("checkpoints/multi_agent_noisy_delay/four_stage_20/reward_std.npy", allow_pickle=True)
NMA4D20_profit = np.load("checkpoints/multi_agent_noisy_delay/four_stage_20/profit.npy", allow_pickle=True)

NMA4D30_time = np.load("checkpoints/multi_agent_noisy_delay/four_stage_30/time.npy", allow_pickle=True)
NMA4D30_reward_mean = np.load("checkpoints/multi_agent_noisy_delay/four_stage_30/reward_mean.npy", allow_pickle=True)
NMA4D30_reward_std = np.load("checkpoints/multi_agent_noisy_delay/four_stage_30/reward_std.npy", allow_pickle=True)
NMA4D30_profit = np.load("checkpoints/multi_agent_noisy_delay/four_stage_30/profit.npy", allow_pickle=True)

NMA4D40_time = np.load("checkpoints/multi_agent_noisy_delay/four_stage_40/time.npy", allow_pickle=True)
NMA4D40_reward_mean = np.load("checkpoints/multi_agent_noisy_delay/four_stage_40/reward_mean.npy", allow_pickle=True)
NMA4D40_reward_std = np.load("checkpoints/multi_agent_noisy_delay/four_stage_40/reward_std.npy", allow_pickle=True)
NMA4D40_profit = np.load("checkpoints/multi_agent_noisy_delay/four_stage_40/profit.npy", allow_pickle=True)

NMA4D50_time = np.load("checkpoints/multi_agent_noisy_delay/four_stage_50/time.npy", allow_pickle=True)
NMA4D50_reward_mean = np.load("checkpoints/multi_agent_noisy_delay/four_stage_50/reward_mean.npy", allow_pickle=True)
NMA4D50_reward_std = np.load("checkpoints/multi_agent_noisy_delay/four_stage_50/reward_std.npy", allow_pickle=True)
NMA4D50_profit = np.load("checkpoints/multi_agent_noisy_delay/four_stage_50/profit.npy", allow_pickle=True)

#%% Load MA Shared Rl Data
MAS4_results = np.load("checkpoints/multi_agent/four_stage_share/results.npy", allow_pickle=True)
MAS4_time = np.load("checkpoints/multi_agent/four_stage_share/time.npy", allow_pickle=True)
MAS4_reward_mean = np.load("checkpoints/multi_agent/four_stage_share/reward_mean.npy", allow_pickle=True)
MAS4_reward_std = np.load("checkpoints/multi_agent/four_stage_share/reward_std.npy", allow_pickle=True)
MAS4_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share/inventory_mean.npy", allow_pickle=True)
MAS4_inventory_std = np.load("checkpoints/multi_agent/four_stage_share/inventory_std.npy", allow_pickle=True)
MAS4_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share/backlog_mean.npy", allow_pickle=True)
MAS4_backlog_std = np.load("checkpoints/multi_agent/four_stage_share/backlog_std.npy", allow_pickle=True)
MAS4_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share/customer_backlog_mean.npy", allow_pickle=True)
MAS4_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share/customer_backlog_std.npy", allow_pickle=True)
MAS4_profit = np.load("checkpoints/multi_agent/four_stage_share/profit.npy", allow_pickle=True)

MAS4I_results = np.load("checkpoints/multi_agent/four_stage_share_independent/results.npy", allow_pickle=True)
MAS4I_time = np.load("checkpoints/multi_agent/four_stage_share_independent/time.npy", allow_pickle=True)
MAS4I_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_independent/reward_mean.npy", allow_pickle=True)
MAS4I_reward_std = np.load("checkpoints/multi_agent/four_stage_share_independent/reward_std.npy", allow_pickle=True)
MAS4i_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_independent/inventory_mean.npy", allow_pickle=True)
MAS4I_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_independent/inventory_std.npy", allow_pickle=True)
MAS4I_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_independent/backlog_mean.npy", allow_pickle=True)
MAS4I_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_independent/backlog_std.npy", allow_pickle=True)
MAS4I_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_independent/customer_backlog_mean.npy", allow_pickle=True)
MAS4I_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_independent/customer_backlog_std.npy", allow_pickle=True)
MAS4I_profit = np.load("checkpoints/multi_agent/four_stage_share_independent/profit.npy", allow_pickle=True)

MAS2_results = np.load("checkpoints/multi_agent/two_stage_share/results.npy", allow_pickle=True)
MAS2_time = np.load("checkpoints/multi_agent/two_stage_share/time.npy", allow_pickle=True)
MAS2_reward_mean = np.load("checkpoints/multi_agent/two_stage_share/reward_mean.npy", allow_pickle=True)
MAS2_reward_std = np.load("checkpoints/multi_agent/two_stage_share/reward_std.npy", allow_pickle=True)
MAS2_inventory_mean = np.load("checkpoints/multi_agent/two_stage_share/inventory_mean.npy", allow_pickle=True)
MAS2_inventory_std = np.load("checkpoints/multi_agent/two_stage_share/inventory_std.npy", allow_pickle=True)
MAS2_backlog_mean = np.load("checkpoints/multi_agent/two_stage_share/backlog_mean.npy", allow_pickle=True)
MAS2_backlog_std = np.load("checkpoints/multi_agent/two_stage_share/backlog_std.npy", allow_pickle=True)
MAS2_customer_backlog_mean = np.load("checkpoints/multi_agent/two_stage_share/customer_backlog_mean.npy", allow_pickle=True)
MAS2_customer_backlog_std = np.load("checkpoints/multi_agent/two_stage_share/customer_backlog_std.npy", allow_pickle=True)
MAS2_profit = np.load("checkpoints/multi_agent/two_stage_share/profit.npy", allow_pickle=True)

MAS8_results = np.load("checkpoints/multi_agent/eight_stage_share/results.npy", allow_pickle=True)
MAS8_time = np.load("checkpoints/multi_agent/eight_stage_share/time.npy", allow_pickle=True)
MAS8_reward_mean = np.load("checkpoints/multi_agent/eight_stage_share/reward_mean.npy", allow_pickle=True)
MAS8_reward_std = np.load("checkpoints/multi_agent/eight_stage_share/reward_std.npy", allow_pickle=True)
MAS8_inventory_mean = np.load("checkpoints/multi_agent/eight_stage_share/inventory_mean.npy", allow_pickle=True)
MAS8_inventory_std = np.load("checkpoints/multi_agent/eight_stage_share/inventory_std.npy", allow_pickle=True)
MAS8_backlog_mean = np.load("checkpoints/multi_agent/eight_stage_share/backlog_mean.npy", allow_pickle=True)
MAS8_backlog_std = np.load("checkpoints/multi_agent/eight_stage_share/backlog_std.npy", allow_pickle=True)
MAS8_customer_backlog_mean = np.load("checkpoints/multi_agent/eight_stage_share/customer_backlog_mean.npy", allow_pickle=True)
MAS8_customer_backlog_std = np.load("checkpoints/multi_agent/eight_stage_share/customer_backlog_std.npy", allow_pickle=True)
MAS8_profit = np.load("checkpoints/multi_agent/eight_stage_share/profit.npy", allow_pickle=True)

MAS4N10_time = np.load("checkpoints/multi_agent/four_stage_share_noise_10/time.npy", allow_pickle=True)
MAS4N10_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_10/reward_mean.npy", allow_pickle=True)
MAS4N10_reward_std = np.load("checkpoints/multi_agent/four_stage_share_noise_10/reward_std.npy", allow_pickle=True)
MAS4N10_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_10/inventory_mean.npy", allow_pickle=True)
MAS4N10_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_noise_10/inventory_std.npy", allow_pickle=True)
MAS4N10_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_10/backlog_mean.npy", allow_pickle=True)
MAS4N10_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_10/backlog_std.npy", allow_pickle=True)
MAS4N10_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_10/customer_backlog_mean.npy", allow_pickle=True)
MAS4N10_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_10/customer_backlog_std.npy", allow_pickle=True)
MAS4N10_profit = np.load("checkpoints/multi_agent/four_stage_share_noise_10/profit.npy", allow_pickle=True)

MAS4N20_time = np.load("checkpoints/multi_agent/four_stage_share_noise_20/time.npy", allow_pickle=True)
MAS4N20_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_20/reward_mean.npy", allow_pickle=True)
MAS4N20_reward_std = np.load("checkpoints/multi_agent/four_stage_share_noise_20/reward_std.npy", allow_pickle=True)
MAS4N20_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_20/inventory_mean.npy", allow_pickle=True)
MAS4N20_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_noise_20/inventory_std.npy", allow_pickle=True)
MAS4N20_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_20/backlog_mean.npy", allow_pickle=True)
MAS4N20_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_20/backlog_std.npy", allow_pickle=True)
MAS4N20_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_20/customer_backlog_mean.npy", allow_pickle=True)
MAS4N20_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_20/customer_backlog_std.npy", allow_pickle=True)
MAS4N20_profit = np.load("checkpoints/multi_agent/four_stage_share_noise_20/profit.npy", allow_pickle=True)

MAS4N30_time = np.load("checkpoints/multi_agent/four_stage_share_noise_30/time.npy", allow_pickle=True)
MAS4N30_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_30/reward_mean.npy", allow_pickle=True)
MAS4N30_reward_std = np.load("checkpoints/multi_agent/four_stage_share_noise_30/reward_std.npy", allow_pickle=True)
MAS4N30_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_30/inventory_mean.npy", allow_pickle=True)
MAS4N30_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_noise_30/inventory_std.npy", allow_pickle=True)
MAS4N30_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_30/backlog_mean.npy", allow_pickle=True)
MAS4N30_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_30/backlog_std.npy", allow_pickle=True)
MAS4N30_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_30/customer_backlog_mean.npy", allow_pickle=True)
MAS4N30_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_30/customer_backlog_std.npy", allow_pickle=True)
MAS4N30_profit = np.load("checkpoints/multi_agent/four_stage_share_noise_30/profit.npy", allow_pickle=True)

MAS4N40_time = np.load("checkpoints/multi_agent/four_stage_share_noise_40/time.npy", allow_pickle=True)
MAS4N40_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_40/reward_mean.npy", allow_pickle=True)
MAS4N40_reward_std = np.load("checkpoints/multi_agent/four_stage_share_noise_40/reward_std.npy", allow_pickle=True)
MAS4N40_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_40/inventory_mean.npy", allow_pickle=True)
MAS4N40_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_noise_40/inventory_std.npy", allow_pickle=True)
MAS4N40_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_40/backlog_mean.npy", allow_pickle=True)
MAS4N40_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_40/backlog_std.npy", allow_pickle=True)
MAS4N40_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_40/customer_backlog_mean.npy", allow_pickle=True)
MAS4N40_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_40/customer_backlog_std.npy", allow_pickle=True)
MAS4N40_profit = np.load("checkpoints/multi_agent/four_stage_share_noise_40/profit.npy", allow_pickle=True)

MAS4N50_time = np.load("checkpoints/multi_agent/four_stage_share_noise_50/time.npy", allow_pickle=True)
MAS4N50_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_50/reward_mean.npy", allow_pickle=True)
MAS4N50_reward_std = np.load("checkpoints/multi_agent/four_stage_share_noise_50/reward_std.npy", allow_pickle=True)
MAS4N50_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_50/inventory_mean.npy", allow_pickle=True)
MAS4N50_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_noise_50/inventory_std.npy", allow_pickle=True)
MAS4N50_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_50/backlog_mean.npy", allow_pickle=True)
MAS4N50_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_50/backlog_std.npy", allow_pickle=True)
MAS4N50_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_50/customer_backlog_mean.npy", allow_pickle=True)
MAS4N50_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_50/customer_backlog_std.npy", allow_pickle=True)
MAS4N50_profit = np.load("checkpoints/multi_agent/four_stage_share_noise_50/profit.npy", allow_pickle=True)

MAS4D10_time = np.load("checkpoints/multi_agent/four_stage_share_delay_10/time.npy", allow_pickle=True)
MAS4D10_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_10/reward_mean.npy", allow_pickle=True)
MAS4D10_reward_std = np.load("checkpoints/multi_agent/four_stage_share_delay_10/reward_std.npy", allow_pickle=True)
MAS4D10_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_10/inventory_mean.npy", allow_pickle=True)
MAS4D10_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_delay_10/inventory_std.npy", allow_pickle=True)
MAS4D10_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_10/backlog_mean.npy", allow_pickle=True)
MAS4D10_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_delay_10/backlog_std.npy", allow_pickle=True)
MAS4D10_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_10/customer_backlog_mean.npy", allow_pickle=True)
MAS4D10_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_delay_10/customer_backlog_std.npy", allow_pickle=True)
MAS4D10_profit = np.load("checkpoints/multi_agent/four_stage_share_delay_10/profit.npy", allow_pickle=True)

MAS4D20_time = np.load("checkpoints/multi_agent/four_stage_share_delay_20/time.npy", allow_pickle=True)
MAS4D20_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_20/reward_mean.npy", allow_pickle=True)
MAS4D20_reward_std = np.load("checkpoints/multi_agent/four_stage_share_delay_20/reward_std.npy", allow_pickle=True)
MAS4D20_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_20/inventory_mean.npy", allow_pickle=True)
MAS4D20_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_delay_20/inventory_std.npy", allow_pickle=True)
MAS4D20_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_20/backlog_mean.npy", allow_pickle=True)
MAS4D20_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_delay_20/backlog_std.npy", allow_pickle=True)
MAS4D20_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_20/customer_backlog_mean.npy", allow_pickle=True)
MAS4D20_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_delay_20/customer_backlog_std.npy", allow_pickle=True)
MAS4D20_profit = np.load("checkpoints/multi_agent/four_stage_share_delay_20/profit.npy", allow_pickle=True)

MAS4D30_time = np.load("checkpoints/multi_agent/four_stage_share_delay_30/time.npy", allow_pickle=True)
MAS4D30_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_30/reward_mean.npy", allow_pickle=True)
MAS4D30_reward_std = np.load("checkpoints/multi_agent/four_stage_share_delay_30/reward_std.npy", allow_pickle=True)
MAS4D30_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_30/inventory_mean.npy", allow_pickle=True)
MAS4D30_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_delay_30/inventory_std.npy", allow_pickle=True)
MAS4D30_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_30/backlog_mean.npy", allow_pickle=True)
MAS4D30_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_delay_30/backlog_std.npy", allow_pickle=True)
MAS4D30_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_30/customer_backlog_mean.npy", allow_pickle=True)
MAS4D30_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_delay_30/customer_backlog_std.npy", allow_pickle=True)
MAS4D30_profit = np.load("checkpoints/multi_agent/four_stage_share_delay_30/profit.npy", allow_pickle=True)

MAS4D40_time = np.load("checkpoints/multi_agent/four_stage_share_delay_40/time.npy", allow_pickle=True)
MAS4D40_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_40/reward_mean.npy", allow_pickle=True)
MAS4D40_reward_std = np.load("checkpoints/multi_agent/four_stage_share_delay_40/reward_std.npy", allow_pickle=True)
MAS4D40_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_40/inventory_mean.npy", allow_pickle=True)
MAS4D40_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_delay_40/inventory_std.npy", allow_pickle=True)
MAS4D40_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_40/backlog_mean.npy", allow_pickle=True)
MAS4D40_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_delay_40/backlog_std.npy", allow_pickle=True)
MAS4D40_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_40/customer_backlog_mean.npy", allow_pickle=True)
MAS4D40_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_delay_40/customer_backlog_std.npy", allow_pickle=True)
MAS4D40_profit = np.load("checkpoints/multi_agent/four_stage_share_delay_40/profit.npy", allow_pickle=True)

MAS4D50_time = np.load("checkpoints/multi_agent/four_stage_share_delay_50/time.npy", allow_pickle=True)
MAS4D50_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_50/reward_mean.npy", allow_pickle=True)
MAS4D50_reward_std = np.load("checkpoints/multi_agent/four_stage_share_delay_50/reward_std.npy", allow_pickle=True)
MAS4D50_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_50/inventory_mean.npy", allow_pickle=True)
MAS4D50_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_delay_50/inventory_std.npy", allow_pickle=True)
MAS4D50_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_50/backlog_mean.npy", allow_pickle=True)
MAS4D50_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_delay_50/backlog_std.npy", allow_pickle=True)
MAS4D50_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_delay_50/customer_backlog_mean.npy", allow_pickle=True)
MAS4D50_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_delay_50/customer_backlog_std.npy", allow_pickle=True)
MAS4D50_profit = np.load("checkpoints/multi_agent/four_stage_share_delay_50/profit.npy", allow_pickle=True)

MAS4N10_time = np.load("checkpoints/multi_agent/four_stage_share_noise_10/time.npy", allow_pickle=True)
MAS4N10_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_10/reward_mean.npy", allow_pickle=True)
MAS4N10_reward_std = np.load("checkpoints/multi_agent/four_stage_share_noise_10/reward_std.npy", allow_pickle=True)
MAS4N10_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_10/inventory_mean.npy", allow_pickle=True)
MAS4N10_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_noise_10/inventory_std.npy", allow_pickle=True)
MAS4N10_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_10/backlog_mean.npy", allow_pickle=True)
MAS4N10_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_10/backlog_std.npy", allow_pickle=True)
MAS4N10_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_10/customer_backlog_mean.npy", allow_pickle=True)
MAS4N10_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_10/customer_backlog_std.npy", allow_pickle=True)
MAS4N10_profit = np.load("checkpoints/multi_agent/four_stage_share_noise_10/profit.npy", allow_pickle=True)

MAS4N20_time = np.load("checkpoints/multi_agent/four_stage_share_noise_20/time.npy", allow_pickle=True)
MAS4N20_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_20/reward_mean.npy", allow_pickle=True)
MAS4N20_reward_std = np.load("checkpoints/multi_agent/four_stage_share_noise_20/reward_std.npy", allow_pickle=True)
MAS4N20_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_20/inventory_mean.npy", allow_pickle=True)
MAS4N20_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_noise_20/inventory_std.npy", allow_pickle=True)
MAS4N20_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_20/backlog_mean.npy", allow_pickle=True)
MAS4N20_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_20/backlog_std.npy", allow_pickle=True)
MAS4N20_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_20/customer_backlog_mean.npy", allow_pickle=True)
MAS4N20_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_20/customer_backlog_std.npy", allow_pickle=True)
MAS4N20_profit = np.load("checkpoints/multi_agent/four_stage_share_noise_20/profit.npy", allow_pickle=True)

MAS4N30_time = np.load("checkpoints/multi_agent/four_stage_share_noise_30/time.npy", allow_pickle=True)
MAS4N30_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_30/reward_mean.npy", allow_pickle=True)
MAS4N30_reward_std = np.load("checkpoints/multi_agent/four_stage_share_noise_30/reward_std.npy", allow_pickle=True)
MAS4N30_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_30/inventory_mean.npy", allow_pickle=True)
MAS4N30_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_noise_30/inventory_std.npy", allow_pickle=True)
MAS4N30_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_30/backlog_mean.npy", allow_pickle=True)
MAS4N30_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_30/backlog_std.npy", allow_pickle=True)
MAS4N30_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_30/customer_backlog_mean.npy", allow_pickle=True)
MAS4N30_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_30/customer_backlog_std.npy", allow_pickle=True)
MAS4N30_profit = np.load("checkpoints/multi_agent/four_stage_share_noise_30/profit.npy", allow_pickle=True)

MAS4N50_time = np.load("checkpoints/multi_agent/four_stage_share_noise_50/time.npy", allow_pickle=True)
MAS4N50_reward_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_50/reward_mean.npy", allow_pickle=True)
MAS4N50_reward_std = np.load("checkpoints/multi_agent/four_stage_share_noise_50/reward_std.npy", allow_pickle=True)
MAS4N50_inventory_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_50/inventory_mean.npy", allow_pickle=True)
MAS4N50_inventory_std = np.load("checkpoints/multi_agent/four_stage_share_noise_50/inventory_std.npy", allow_pickle=True)
MAS4N50_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_50/backlog_mean.npy", allow_pickle=True)
MAS4N50_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_50/backlog_std.npy", allow_pickle=True)
MAS4N50_customer_backlog_mean = np.load("checkpoints/multi_agent/four_stage_share_noise_50/customer_backlog_mean.npy", allow_pickle=True)
MAS4N50_customer_backlog_std = np.load("checkpoints/multi_agent/four_stage_share_noise_50/customer_backlog_std.npy", allow_pickle=True)
MAS4N50_profit = np.load("checkpoints/multi_agent/four_stage_share_noise_50/profit.npy", allow_pickle=True)

#%% Load noise trained MA Shared Rl Data
NMAS4N10_time = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_10/time.npy", allow_pickle=True)
NMAS4N10_reward_mean = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_10/reward_mean.npy", allow_pickle=True)
NMAS4N10_reward_std = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_10/reward_std.npy", allow_pickle=True)
NMAS4N10_profit = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_10/profit.npy", allow_pickle=True)

NMAS4N20_time = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_20/time.npy", allow_pickle=True)
NMAS4N20_reward_mean = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_20/reward_mean.npy", allow_pickle=True)
NMAS4N20_reward_std = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_20/reward_std.npy", allow_pickle=True)
NMAS4N20_profit = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_20/profit.npy", allow_pickle=True)

NMAS4N30_time = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_30/time.npy", allow_pickle=True)
NMAS4N30_reward_mean = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_30/reward_mean.npy", allow_pickle=True)
NMAS4N30_reward_std = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_30/reward_std.npy", allow_pickle=True)
NMAS4N30_profit = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_30/profit.npy", allow_pickle=True)

NMAS4N40_time = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_40/time.npy", allow_pickle=True)
NMAS4N40_reward_mean = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_40/reward_mean.npy", allow_pickle=True)
NMAS4N40_reward_std = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_40/reward_std.npy", allow_pickle=True)
NMAS4N40_profit = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_40/profit.npy", allow_pickle=True)

NMAS4N50_time = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_50/time.npy", allow_pickle=True)
NMAS4N50_reward_mean = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_50/reward_mean.npy", allow_pickle=True)
NMAS4N50_reward_std = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_50/reward_std.npy", allow_pickle=True)
NMAS4N50_profit = np.load("checkpoints/multi_agent_share_noisy_demand/four_stage_50/profit.npy", allow_pickle=True)

NMAS4D10_time = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_10/time.npy", allow_pickle=True)
NMAS4D10_reward_mean = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_10/reward_mean.npy", allow_pickle=True)
NMAS4D10_reward_std = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_10/reward_std.npy", allow_pickle=True)
NMAS4D10_profit = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_10/profit.npy", allow_pickle=True)

NMAS4D20_time = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_20/time.npy", allow_pickle=True)
NMAS4D20_reward_mean = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_20/reward_mean.npy", allow_pickle=True)
NMAS4D20_reward_std = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_20/reward_std.npy", allow_pickle=True)
NMAS4D20_profit = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_20/profit.npy", allow_pickle=True)

NMAS4D30_time = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_30/time.npy", allow_pickle=True)
NMAS4D30_reward_mean = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_30/reward_mean.npy", allow_pickle=True)
NMAS4D30_reward_std = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_30/reward_std.npy", allow_pickle=True)
NMAS4D30_profit = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_30/profit.npy", allow_pickle=True)

NMAS4D40_time = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_40/time.npy", allow_pickle=True)
NMAS4D40_reward_mean = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_40/reward_mean.npy", allow_pickle=True)
NMAS4D40_reward_std = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_40/reward_std.npy", allow_pickle=True)
NMAS4D40_profit = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_40/profit.npy", allow_pickle=True)

NMAS4D50_time = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_50/time.npy", allow_pickle=True)
NMAS4D50_reward_mean = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_50/reward_mean.npy", allow_pickle=True)
NMAS4D50_reward_std = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_50/reward_std.npy", allow_pickle=True)
NMAS4D50_profit = np.load("checkpoints/multi_agent_share_noisy_delay/four_stage_50/profit.npy", allow_pickle=True)
#%% Load CC Rl Data
CC4_results = np.load("checkpoints/cc_agent/four_stage/results.npy", allow_pickle=True)
CC4_time = np.load("checkpoints/cc_agent/four_stage/time.npy", allow_pickle=True)
CC4_reward_mean = np.load("checkpoints/cc_agent/four_stage/reward_mean.npy", allow_pickle=True)
CC4_reward_std = np.load("checkpoints/cc_agent/four_stage/reward_std.npy", allow_pickle=True)
CC4_inventory_mean = np.load("checkpoints/cc_agent/four_stage/inventory_mean.npy", allow_pickle=True)
CC4_inventory_std = np.load("checkpoints/cc_agent/four_stage/inventory_std.npy", allow_pickle=True)
CC4_backlog_mean = np.load("checkpoints/cc_agent/four_stage/backlog_mean.npy", allow_pickle=True)
CC4_backlog_std = np.load("checkpoints/cc_agent/four_stage/backlog_std.npy", allow_pickle=True)
CC4_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage/customer_backlog_mean.npy", allow_pickle=True)
CC4_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage/customer_backlog_std.npy", allow_pickle=True)
CC4_profit = np.load("checkpoints/cc_agent/four_stage/profit.npy", allow_pickle=True)

CC4I_results = np.load("checkpoints/cc_agent/four_stage_independent/results.npy", allow_pickle=True)
CC4I_time = np.load("checkpoints/cc_agent/four_stage_independent/time.npy", allow_pickle=True)
CC4I_reward_mean = np.load("checkpoints/cc_agent/four_stage_independent/reward_mean.npy", allow_pickle=True)
CC4I_reward_std = np.load("checkpoints/cc_agent/four_stage_independent/reward_std.npy", allow_pickle=True)
CC4i_inventory_mean = np.load("checkpoints/cc_agent/four_stage_independent/inventory_mean.npy", allow_pickle=True)
CC4I_inventory_std = np.load("checkpoints/cc_agent/four_stage_independent/inventory_std.npy", allow_pickle=True)
CC4I_backlog_mean = np.load("checkpoints/cc_agent/four_stage_independent/backlog_mean.npy", allow_pickle=True)
CC4I_backlog_std = np.load("checkpoints/cc_agent/four_stage_independent/backlog_std.npy", allow_pickle=True)
CC4I_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_independent/customer_backlog_mean.npy", allow_pickle=True)
CC4I_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_independent/customer_backlog_std.npy", allow_pickle=True)
CC4I_profit = np.load("checkpoints/cc_agent/four_stage_independent/profit.npy", allow_pickle=True)

CC2_results = np.load("checkpoints/cc_agent/two_stage/results.npy", allow_pickle=True)
CC2_time = np.load("checkpoints/cc_agent/two_stage/time.npy", allow_pickle=True)
CC2_reward_mean = np.load("checkpoints/cc_agent/two_stage/reward_mean.npy", allow_pickle=True)
CC2_reward_std = np.load("checkpoints/cc_agent/two_stage/reward_std.npy", allow_pickle=True)
CC2_inventory_mean = np.load("checkpoints/cc_agent/two_stage/inventory_mean.npy", allow_pickle=True)
CC2_inventory_std = np.load("checkpoints/cc_agent/two_stage/inventory_std.npy", allow_pickle=True)
CC2_backlog_mean = np.load("checkpoints/cc_agent/two_stage/backlog_mean.npy", allow_pickle=True)
CC2_backlog_std = np.load("checkpoints/cc_agent/two_stage/backlog_std.npy", allow_pickle=True)
CC2_customer_backlog_mean = np.load("checkpoints/cc_agent/two_stage/customer_backlog_mean.npy", allow_pickle=True)
CC2_customer_backlog_std = np.load("checkpoints/cc_agent/two_stage/customer_backlog_std.npy", allow_pickle=True)
CC2_profit = np.load("checkpoints/cc_agent/two_stage/profit.npy", allow_pickle=True)

CC8_results = np.load("checkpoints/cc_agent/eight_stage/results.npy", allow_pickle=True)
CC8_time = np.load("checkpoints/cc_agent/eight_stage/time.npy", allow_pickle=True)
CC8_reward_mean = np.load("checkpoints/cc_agent/eight_stage/reward_mean.npy", allow_pickle=True)
CC8_reward_std = np.load("checkpoints/cc_agent/eight_stage/reward_std.npy", allow_pickle=True)
CC8_inventory_mean = np.load("checkpoints/cc_agent/eight_stage/inventory_mean.npy", allow_pickle=True)
CC8_inventory_std = np.load("checkpoints/cc_agent/eight_stage/inventory_std.npy", allow_pickle=True)
CC8_backlog_mean = np.load("checkpoints/cc_agent/eight_stage/backlog_mean.npy", allow_pickle=True)
CC8_backlog_std = np.load("checkpoints/cc_agent/eight_stage/backlog_std.npy", allow_pickle=True)
CC8_customer_backlog_mean = np.load("checkpoints/cc_agent/eight_stage/customer_backlog_mean.npy", allow_pickle=True)
CC8_customer_backlog_std = np.load("checkpoints/cc_agent/eight_stage/customer_backlog_std.npy", allow_pickle=True)
CC8_profit = np.load("checkpoints/cc_agent/eight_stage/profit.npy", allow_pickle=True)

CC4N2_results = np.load("checkpoints/cc_agent/four_stage_noise_2/results.npy", allow_pickle=True)
CC4N2_time = np.load("checkpoints/cc_agent/four_stage_noise_2/time.npy", allow_pickle=True)
CC4N2_reward_mean = np.load("checkpoints/cc_agent/four_stage_noise_2/reward_mean.npy", allow_pickle=True)
CC4N2_reward_std = np.load("checkpoints/cc_agent/four_stage_noise_2/reward_std.npy", allow_pickle=True)
CC4N2_inventory_mean = np.load("checkpoints/cc_agent/four_stage_noise_2/inventory_mean.npy", allow_pickle=True)
CC4N2_inventory_std = np.load("checkpoints/cc_agent/four_stage_noise_2/inventory_std.npy", allow_pickle=True)
CC4N2_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_2/backlog_mean.npy", allow_pickle=True)
CC4N2_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_2/backlog_std.npy", allow_pickle=True)
CC4N2_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_2/customer_backlog_mean.npy", allow_pickle=True)
CC4N2_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_2/customer_backlog_std.npy", allow_pickle=True)
CC4N2_profit = np.load("checkpoints/cc_agent/four_stage_noise_2/profit.npy", allow_pickle=True)

CC4N5_results = np.load("checkpoints/cc_agent/four_stage_noise_5/results.npy", allow_pickle=True)
CC4N5_time = np.load("checkpoints/cc_agent/four_stage_noise_5/time.npy", allow_pickle=True)
CC4N5_reward_mean = np.load("checkpoints/cc_agent/four_stage_noise_5/reward_mean.npy", allow_pickle=True)
CC4N5_reward_std = np.load("checkpoints/cc_agent/four_stage_noise_5/reward_std.npy", allow_pickle=True)
CC4N5_inventory_mean = np.load("checkpoints/cc_agent/four_stage_noise_5/inventory_mean.npy", allow_pickle=True)
CC4N5_inventory_std = np.load("checkpoints/cc_agent/four_stage_noise_5/inventory_std.npy", allow_pickle=True)
CC4N5_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_5/backlog_mean.npy", allow_pickle=True)
CC4N5_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_5/backlog_std.npy", allow_pickle=True)
CC4N5_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_5/customer_backlog_mean.npy", allow_pickle=True)
CC4N5_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_5/customer_backlog_std.npy", allow_pickle=True)
CC4N5_profit = np.load("checkpoints/cc_agent/four_stage_noise_5/profit.npy", allow_pickle=True)

CC4N10_results = np.load("checkpoints/cc_agent/four_stage_noise_10/results.npy", allow_pickle=True)
CC4N10_time = np.load("checkpoints/cc_agent/four_stage_noise_10/time.npy", allow_pickle=True)
CC4N10_reward_mean = np.load("checkpoints/cc_agent/four_stage_noise_10/reward_mean.npy", allow_pickle=True)
CC4N10_reward_std = np.load("checkpoints/cc_agent/four_stage_noise_10/reward_std.npy", allow_pickle=True)
CC4N10_inventory_mean = np.load("checkpoints/cc_agent/four_stage_noise_10/inventory_mean.npy", allow_pickle=True)
CC4N10_inventory_std = np.load("checkpoints/cc_agent/four_stage_noise_10/inventory_std.npy", allow_pickle=True)
CC4N10_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_10/backlog_mean.npy", allow_pickle=True)
CC4N10_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_10/backlog_std.npy", allow_pickle=True)
CC4N10_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_10/customer_backlog_mean.npy", allow_pickle=True)
CC4N10_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_10/customer_backlog_std.npy", allow_pickle=True)
CC4N10_profit = np.load("checkpoints/cc_agent/four_stage_noise_10/profit.npy", allow_pickle=True)

CC4N20_results = np.load("checkpoints/cc_agent/four_stage_noise_20/results.npy", allow_pickle=True)
CC4N20_time = np.load("checkpoints/cc_agent/four_stage_noise_20/time.npy", allow_pickle=True)
CC4N20_reward_mean = np.load("checkpoints/cc_agent/four_stage_noise_20/reward_mean.npy", allow_pickle=True)
CC4N20_reward_std = np.load("checkpoints/cc_agent/four_stage_noise_20/reward_std.npy", allow_pickle=True)
CC4N20_inventory_mean = np.load("checkpoints/cc_agent/four_stage_noise_20/inventory_mean.npy", allow_pickle=True)
CC4N20_inventory_std = np.load("checkpoints/cc_agent/four_stage_noise_20/inventory_std.npy", allow_pickle=True)
CC4N20_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_20/backlog_mean.npy", allow_pickle=True)
CC4N20_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_20/backlog_std.npy", allow_pickle=True)
CC4N20_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_20/customer_backlog_mean.npy", allow_pickle=True)
CC4N20_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_20/customer_backlog_std.npy", allow_pickle=True)
CC4N20_profit = np.load("checkpoints/cc_agent/four_stage_noise_20/profit.npy", allow_pickle=True)

CC4N30_results = np.load("checkpoints/cc_agent/four_stage_noise_30/results.npy", allow_pickle=True)
CC4N30_time = np.load("checkpoints/cc_agent/four_stage_noise_30/time.npy", allow_pickle=True)
CC4N30_reward_mean = np.load("checkpoints/cc_agent/four_stage_noise_30/reward_mean.npy", allow_pickle=True)
CC4N30_reward_std = np.load("checkpoints/cc_agent/four_stage_noise_30/reward_std.npy", allow_pickle=True)
CC4N30_inventory_mean = np.load("checkpoints/cc_agent/four_stage_noise_30/inventory_mean.npy", allow_pickle=True)
CC4N30_inventory_std = np.load("checkpoints/cc_agent/four_stage_noise_30/inventory_std.npy", allow_pickle=True)
CC4N30_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_30/backlog_mean.npy", allow_pickle=True)
CC4N30_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_30/backlog_std.npy", allow_pickle=True)
CC4N30_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_30/customer_backlog_mean.npy", allow_pickle=True)
CC4N30_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_30/customer_backlog_std.npy", allow_pickle=True)
CC4N30_profit = np.load("checkpoints/cc_agent/four_stage_noise_30/profit.npy", allow_pickle=True)

CC4N50_results = np.load("checkpoints/cc_agent/four_stage_noise_50/results.npy", allow_pickle=True)
CC4N50_time = np.load("checkpoints/cc_agent/four_stage_noise_50/time.npy", allow_pickle=True)
CC4N50_reward_mean = np.load("checkpoints/cc_agent/four_stage_noise_50/reward_mean.npy", allow_pickle=True)
CC4N50_reward_std = np.load("checkpoints/cc_agent/four_stage_noise_50/reward_std.npy", allow_pickle=True)
CC4N50_inventory_mean = np.load("checkpoints/cc_agent/four_stage_noise_50/inventory_mean.npy", allow_pickle=True)
CC4N50_inventory_std = np.load("checkpoints/cc_agent/four_stage_noise_50/inventory_std.npy", allow_pickle=True)
CC4N50_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_50/backlog_mean.npy", allow_pickle=True)
CC4N50_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_50/backlog_std.npy", allow_pickle=True)
CC4N50_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_50/customer_backlog_mean.npy", allow_pickle=True)
CC4N50_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_50/customer_backlog_std.npy", allow_pickle=True)
CC4N50_profit = np.load("checkpoints/cc_agent/four_stage_noise_50/profit.npy", allow_pickle=True)

CC4D10_results = np.load("checkpoints/cc_agent/four_stage_delay_10/results.npy", allow_pickle=True)
CC4D10_time = np.load("checkpoints/cc_agent/four_stage_delay_10/time.npy", allow_pickle=True)
CC4D10_reward_mean = np.load("checkpoints/cc_agent/four_stage_delay_10/reward_mean.npy", allow_pickle=True)
CC4D10_reward_std = np.load("checkpoints/cc_agent/four_stage_delay_10/reward_std.npy", allow_pickle=True)
CC4D10_inventory_mean = np.load("checkpoints/cc_agent/four_stage_delay_10/inventory_mean.npy", allow_pickle=True)
CC4D10_inventory_std = np.load("checkpoints/cc_agent/four_stage_delay_10/inventory_std.npy", allow_pickle=True)
CC4D10_backlog_mean = np.load("checkpoints/cc_agent/four_stage_delay_10/backlog_mean.npy", allow_pickle=True)
CC4D10_backlog_std = np.load("checkpoints/cc_agent/four_stage_delay_10/backlog_std.npy", allow_pickle=True)
CC4D10_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_delay_10/customer_backlog_mean.npy", allow_pickle=True)
CC4D10_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_delay_10/customer_backlog_std.npy", allow_pickle=True)
CC4D10_profit = np.load("checkpoints/cc_agent/four_stage_delay_10/profit.npy", allow_pickle=True)

CC4D20_results = np.load("checkpoints/cc_agent/four_stage_delay_20/results.npy", allow_pickle=True)
CC4D20_time = np.load("checkpoints/cc_agent/four_stage_delay_20/time.npy", allow_pickle=True)
CC4D20_reward_mean = np.load("checkpoints/cc_agent/four_stage_delay_20/reward_mean.npy", allow_pickle=True)
CC4D20_reward_std = np.load("checkpoints/cc_agent/four_stage_delay_20/reward_std.npy", allow_pickle=True)
CC4D20_inventory_mean = np.load("checkpoints/cc_agent/four_stage_delay_20/inventory_mean.npy", allow_pickle=True)
CC4D20_inventory_std = np.load("checkpoints/cc_agent/four_stage_delay_20/inventory_std.npy", allow_pickle=True)
CC4D20_backlog_mean = np.load("checkpoints/cc_agent/four_stage_delay_20/backlog_mean.npy", allow_pickle=True)
CC4D20_backlog_std = np.load("checkpoints/cc_agent/four_stage_delay_20/backlog_std.npy", allow_pickle=True)
CC4D20_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_delay_20/customer_backlog_mean.npy", allow_pickle=True)
CC4D20_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_delay_20/customer_backlog_std.npy", allow_pickle=True)
CC4D20_profit = np.load("checkpoints/cc_agent/four_stage_delay_20/profit.npy", allow_pickle=True)

CC4D30_results = np.load("checkpoints/cc_agent/four_stage_delay_30/results.npy", allow_pickle=True)
CC4D30_time = np.load("checkpoints/cc_agent/four_stage_delay_30/time.npy", allow_pickle=True)
CC4D30_reward_mean = np.load("checkpoints/cc_agent/four_stage_delay_30/reward_mean.npy", allow_pickle=True)
CC4D30_reward_std = np.load("checkpoints/cc_agent/four_stage_delay_30/reward_std.npy", allow_pickle=True)
CC4D30_inventory_mean = np.load("checkpoints/cc_agent/four_stage_delay_30/inventory_mean.npy", allow_pickle=True)
CC4D30_inventory_std = np.load("checkpoints/cc_agent/four_stage_delay_30/inventory_std.npy", allow_pickle=True)
CC4D30_backlog_mean = np.load("checkpoints/cc_agent/four_stage_delay_30/backlog_mean.npy", allow_pickle=True)
CC4D30_backlog_std = np.load("checkpoints/cc_agent/four_stage_delay_30/backlog_std.npy", allow_pickle=True)
CC4D30_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_delay_30/customer_backlog_mean.npy", allow_pickle=True)
CC4D30_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_delay_30/customer_backlog_std.npy", allow_pickle=True)
CC4D30_profit = np.load("checkpoints/cc_agent/four_stage_delay_30/profit.npy", allow_pickle=True)

CC4N40_results = np.load("checkpoints/cc_agent/four_stage_noise_40/results.npy", allow_pickle=True)
CC4N40_time = np.load("checkpoints/cc_agent/four_stage_noise_40/time.npy", allow_pickle=True)
CC4N40_reward_mean = np.load("checkpoints/cc_agent/four_stage_noise_40/reward_mean.npy", allow_pickle=True)
CC4N40_reward_std = np.load("checkpoints/cc_agent/four_stage_noise_40/reward_std.npy", allow_pickle=True)
CC4N40_inventory_mean = np.load("checkpoints/cc_agent/four_stage_noise_40/inventory_mean.npy", allow_pickle=True)
CC4N40_inventory_std = np.load("checkpoints/cc_agent/four_stage_noise_40/inventory_std.npy", allow_pickle=True)
CC4N40_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_40/backlog_mean.npy", allow_pickle=True)
CC4N40_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_40/backlog_std.npy", allow_pickle=True)
CC4N40_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_noise_40/customer_backlog_mean.npy", allow_pickle=True)
CC4N40_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_noise_40/customer_backlog_std.npy", allow_pickle=True)
CC4N40_profit = np.load("checkpoints/cc_agent/four_stage_noise_40/profit.npy", allow_pickle=True)

CC4D40_results = np.load("checkpoints/cc_agent/four_stage_delay_40/results.npy", allow_pickle=True)
CC4D40_time = np.load("checkpoints/cc_agent/four_stage_delay_40/time.npy", allow_pickle=True)
CC4D40_reward_mean = np.load("checkpoints/cc_agent/four_stage_delay_40/reward_mean.npy", allow_pickle=True)
CC4D40_reward_std = np.load("checkpoints/cc_agent/four_stage_delay_40/reward_std.npy", allow_pickle=True)
CC4D40_inventory_mean = np.load("checkpoints/cc_agent/four_stage_delay_40/inventory_mean.npy", allow_pickle=True)
CC4D40_inventory_std = np.load("checkpoints/cc_agent/four_stage_delay_40/inventory_std.npy", allow_pickle=True)
CC4D40_backlog_mean = np.load("checkpoints/cc_agent/four_stage_delay_40/backlog_mean.npy", allow_pickle=True)
CC4D40_backlog_std = np.load("checkpoints/cc_agent/four_stage_delay_40/backlog_std.npy", allow_pickle=True)
CC4D40_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_delay_40/customer_backlog_mean.npy", allow_pickle=True)
CC4D40_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_delay_40/customer_backlog_std.npy", allow_pickle=True)
CC4D40_profit = np.load("checkpoints/cc_agent/four_stage_delay_40/profit.npy", allow_pickle=True)

CC4D50_results = np.load("checkpoints/cc_agent/four_stage_delay_50/results.npy", allow_pickle=True)
CC4D50_time = np.load("checkpoints/cc_agent/four_stage_delay_50/time.npy", allow_pickle=True)
CC4D50_reward_mean = np.load("checkpoints/cc_agent/four_stage_delay_50/reward_mean.npy", allow_pickle=True)
CC4D50_reward_std = np.load("checkpoints/cc_agent/four_stage_delay_50/reward_std.npy", allow_pickle=True)
CC4D50_inventory_mean = np.load("checkpoints/cc_agent/four_stage_delay_50/inventory_mean.npy", allow_pickle=True)
CC4D50_inventory_std = np.load("checkpoints/cc_agent/four_stage_delay_50/inventory_std.npy", allow_pickle=True)
CC4D50_backlog_mean = np.load("checkpoints/cc_agent/four_stage_delay_50/backlog_mean.npy", allow_pickle=True)
CC4D50_backlog_std = np.load("checkpoints/cc_agent/four_stage_delay_50/backlog_std.npy", allow_pickle=True)
CC4D50_customer_backlog_mean = np.load("checkpoints/cc_agent/four_stage_delay_50/customer_backlog_mean.npy", allow_pickle=True)
CC4D50_customer_backlog_std = np.load("checkpoints/cc_agent/four_stage_delay_50/customer_backlog_std.npy", allow_pickle=True)
CC4D50_profit = np.load("checkpoints/cc_agent/four_stage_delay_50/profit.npy", allow_pickle=True)

#%% Load noise trained CC Rl Data
NCC4N10_results = np.load("checkpoints/cc_agent_noisy_demand/four_stage_10/results.npy", allow_pickle=True)
NCC4N10_time = np.load("checkpoints/cc_agent_noisy_demand/four_stage_10/time.npy", allow_pickle=True)
NCC4N10_reward_mean = np.load("checkpoints/cc_agent_noisy_demand/four_stage_10/reward_mean.npy", allow_pickle=True)
NCC4N10_reward_std = np.load("checkpoints/cc_agent_noisy_demand/four_stage_10/reward_std.npy", allow_pickle=True)
NCC4N10_profit = np.load("checkpoints/cc_agent_noisy_demand/four_stage_10/profit.npy", allow_pickle=True)

NCC4N20_results = np.load("checkpoints/cc_agent_noisy_demand/four_stage_20/results.npy", allow_pickle=True)
NCC4N20_time = np.load("checkpoints/cc_agent_noisy_demand/four_stage_20/time.npy", allow_pickle=True)
NCC4N20_reward_mean = np.load("checkpoints/cc_agent_noisy_demand/four_stage_20/reward_mean.npy", allow_pickle=True)
NCC4N20_reward_std = np.load("checkpoints/cc_agent_noisy_demand/four_stage_20/reward_std.npy", allow_pickle=True)
NCC4N20_profit = np.load("checkpoints/cc_agent_noisy_demand/four_stage_20/profit.npy", allow_pickle=True)

NCC4N30_results = np.load("checkpoints/cc_agent_noisy_demand/four_stage_30/results.npy", allow_pickle=True)
NCC4N30_time = np.load("checkpoints/cc_agent_noisy_demand/four_stage_30/time.npy", allow_pickle=True)
NCC4N30_reward_mean = np.load("checkpoints/cc_agent_noisy_demand/four_stage_30/reward_mean.npy", allow_pickle=True)
NCC4N30_reward_std = np.load("checkpoints/cc_agent_noisy_demand/four_stage_30/reward_std.npy", allow_pickle=True)
NCC4N30_profit = np.load("checkpoints/cc_agent_noisy_demand/four_stage_30/profit.npy", allow_pickle=True)

NCC4N40_results = np.load("checkpoints/cc_agent_noisy_demand/four_stage_40/results.npy", allow_pickle=True)
NCC4N40_time = np.load("checkpoints/cc_agent_noisy_demand/four_stage_40/time.npy", allow_pickle=True)
NCC4N40_reward_mean = np.load("checkpoints/cc_agent_noisy_demand/four_stage_40/reward_mean.npy", allow_pickle=True)
NCC4N40_reward_std = np.load("checkpoints/cc_agent_noisy_demand/four_stage_40/reward_std.npy", allow_pickle=True)
NCC4N40_profit = np.load("checkpoints/cc_agent_noisy_demand/four_stage_40/profit.npy", allow_pickle=True)

NCC4N50_results = np.load("checkpoints/cc_agent_noisy_demand/four_stage_50/results.npy", allow_pickle=True)
NCC4N50_time = np.load("checkpoints/cc_agent_noisy_demand/four_stage_50/time.npy", allow_pickle=True)
NCC4N50_reward_mean = np.load("checkpoints/cc_agent_noisy_demand/four_stage_50/reward_mean.npy", allow_pickle=True)
NCC4N50_reward_std = np.load("checkpoints/cc_agent_noisy_demand/four_stage_50/reward_std.npy", allow_pickle=True)
NCC4N50_profit = np.load("checkpoints/cc_agent_noisy_demand/four_stage_50/profit.npy", allow_pickle=True)

NCC4D10_results = np.load("checkpoints/cc_agent_noisy_delay/four_stage_10/results.npy", allow_pickle=True)
NCC4D10_time = np.load("checkpoints/cc_agent_noisy_delay/four_stage_10/time.npy", allow_pickle=True)
NCC4D10_reward_mean = np.load("checkpoints/cc_agent_noisy_delay/four_stage_10/reward_mean.npy", allow_pickle=True)
NCC4D10_reward_std = np.load("checkpoints/cc_agent_noisy_delay/four_stage_10/reward_std.npy", allow_pickle=True)
NCC4D10_profit = np.load("checkpoints/cc_agent_noisy_delay/four_stage_10/profit.npy", allow_pickle=True)

NCC4D20_results = np.load("checkpoints/cc_agent_noisy_delay/four_stage_20/results.npy", allow_pickle=True)
NCC4D20_time = np.load("checkpoints/cc_agent_noisy_delay/four_stage_20/time.npy", allow_pickle=True)
NCC4D20_reward_mean = np.load("checkpoints/cc_agent_noisy_delay/four_stage_20/reward_mean.npy", allow_pickle=True)
NCC4D20_reward_std = np.load("checkpoints/cc_agent_noisy_delay/four_stage_20/reward_std.npy", allow_pickle=True)
NCC4D20_profit = np.load("checkpoints/cc_agent_noisy_delay/four_stage_20/profit.npy", allow_pickle=True)

NCC4D30_results = np.load("checkpoints/cc_agent_noisy_delay/four_stage_30/results.npy", allow_pickle=True)
NCC4D30_time = np.load("checkpoints/cc_agent_noisy_delay/four_stage_30/time.npy", allow_pickle=True)
NCC4D30_reward_mean = np.load("checkpoints/cc_agent_noisy_delay/four_stage_30/reward_mean.npy", allow_pickle=True)
NCC4D30_reward_std = np.load("checkpoints/cc_agent_noisy_delay/four_stage_30/reward_std.npy", allow_pickle=True)
NCC4D30_profit = np.load("checkpoints/cc_agent_noisy_delay/four_stage_30/profit.npy", allow_pickle=True)

NCC4D40_results = np.load("checkpoints/cc_agent_noisy_delay/four_stage_40/results.npy", allow_pickle=True)
NCC4D40_time = np.load("checkpoints/cc_agent_noisy_delay/four_stage_40/time.npy", allow_pickle=True)
NCC4D40_reward_mean = np.load("checkpoints/cc_agent_noisy_delay/four_stage_40/reward_mean.npy", allow_pickle=True)
NCC4D40_reward_std = np.load("checkpoints/cc_agent_noisy_delay/four_stage_40/reward_std.npy", allow_pickle=True)
NCC4D40_profit = np.load("checkpoints/cc_agent_noisy_delay/four_stage_40/profit.npy", allow_pickle=True)

NCC4D50_results = np.load("checkpoints/cc_agent_noisy_delay/four_stage_50/results.npy", allow_pickle=True)
NCC4D50_time = np.load("checkpoints/cc_agent_noisy_delay/four_stage_50/time.npy", allow_pickle=True)
NCC4D50_reward_mean = np.load("checkpoints/cc_agent_noisy_delay/four_stage_50/reward_mean.npy", allow_pickle=True)
NCC4D50_reward_std = np.load("checkpoints/cc_agent_noisy_delay/four_stage_50/reward_std.npy", allow_pickle=True)
NCC4D50_profit = np.load("checkpoints/cc_agent_noisy_delay/four_stage_50/profit.npy", allow_pickle=True)
#%% Load DFO Data
DFO4_reward_mean = np.load("checkpoints/single_agent/four_stage/dfo_mean.npy", allow_pickle=True)
DFO4_reward_std = np.load("checkpoints/single_agent/four_stage/dfo_std.npy", allow_pickle=True)

DFO2_reward_mean = np.load("checkpoints/single_agent/two_stage/dfo_mean.npy", allow_pickle=True)
DFO2_reward_std = np.load("checkpoints/single_agent/two_stage/dfo_std.npy", allow_pickle=True)

DFO8_reward_mean = np.load("checkpoints/single_agent/eight_stage/dfo_mean.npy", allow_pickle=True)
DFO8_reward_std = np.load("checkpoints/single_agent/eight_stage/dfo_std.npy", allow_pickle=True)

DFO4N2_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_2/dfo_mean.npy", allow_pickle=True)
DFO4N2_reward_std = np.load("checkpoints/single_agent/four_stage_noise_2/dfo_std.npy", allow_pickle=True)

DFO4N5_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_5/dfo_mean.npy", allow_pickle=True)
DFO4N5_reward_std = np.load("checkpoints/single_agent/four_stage_noise_5/dfo_std.npy", allow_pickle=True)

DFO4N10_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_10/dfo_mean.npy", allow_pickle=True)
DFO4N10_reward_std = np.load("checkpoints/single_agent/four_stage_noise_10/dfo_std.npy", allow_pickle=True)

DFO4N20_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_20/dfo_mean.npy", allow_pickle=True)
DFO4N20_reward_std = np.load("checkpoints/single_agent/four_stage_noise_20/dfo_std.npy", allow_pickle=True)

DFO4N30_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_30/dfo_mean.npy", allow_pickle=True)
DFO4N30_reward_std = np.load("checkpoints/single_agent/four_stage_noise_30/dfo_std.npy", allow_pickle=True)

DFO4N40_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_40/dfo_mean.npy", allow_pickle=True)
DFO4N40_reward_std = np.load("checkpoints/single_agent/four_stage_noise_40/dfo_std.npy", allow_pickle=True)

DFO4N50_reward_mean = np.load("checkpoints/single_agent/four_stage_noise_50/dfo_mean.npy", allow_pickle=True)
DFO4N50_reward_std = np.load("checkpoints/single_agent/four_stage_noise_50/dfo_std.npy", allow_pickle=True)

DFO4D10_reward_mean = np.load("checkpoints/single_agent/four_stage_delay_10/dfo_mean.npy", allow_pickle=True)
DFO4D10_reward_std = np.load("checkpoints/single_agent/four_stage_delay_10/dfo_std.npy", allow_pickle=True)

DFO4D20_reward_mean = np.load("checkpoints/single_agent/four_stage_delay_20/dfo_mean.npy", allow_pickle=True)
DFO4D20_reward_std = np.load("checkpoints/single_agent/four_stage_delay_20/dfo_std.npy", allow_pickle=True)

DFO4D30_reward_mean = np.load("checkpoints/single_agent/four_stage_delay_30/dfo_mean.npy", allow_pickle=True)
DFO4D30_reward_std = np.load("checkpoints/single_agent/four_stage_delay_30/dfo_std.npy", allow_pickle=True)

DFO4D40_reward_mean = np.load("checkpoints/single_agent/four_stage_delay_40/dfo_mean.npy", allow_pickle=True)
DFO4D40_reward_std = np.load("checkpoints/single_agent/four_stage_delay_40/dfo_std.npy", allow_pickle=True)

DFO4D50_reward_mean = np.load("checkpoints/single_agent/four_stage_delay_50/dfo_mean.npy", allow_pickle=True)
DFO4D50_reward_std = np.load("checkpoints/single_agent/four_stage_delay_50/dfo_std.npy", allow_pickle=True)
#%% Load Oracle Data
OR4_reward_mean = np.load("LP_results/four_stage/Oracle/reward_mean.npy", allow_pickle=True)
OR4_reward_list = np.load("LP_results/four_stage/Oracle/reward_list.npy", allow_pickle=True)
OR4_reward_std = np.load("LP_results/four_stage/Oracle/reward_std.npy", allow_pickle=True)
OR4_inventory_mean = np.load("LP_results/four_stage/Oracle/inventory_mean.npy", allow_pickle=True)
OR4_inventory_std = np.load("LP_results/four_stage/Oracle/inventory_std.npy", allow_pickle=True)
OR4_backlog_mean = np.load("LP_results/four_stage/Oracle/backlog_mean.npy", allow_pickle=True)
OR4_backlog_std = np.load("LP_results/four_stage/Oracle/backlog_std.npy", allow_pickle=True)
OR4_customer_backlog_mean = np.load("LP_results/four_stage/Oracle/customer_backlog_mean.npy", allow_pickle=True)
OR4_customer_backlog_std = np.load("LP_results/four_stage/Oracle/customer_backlog_std.npy", allow_pickle=True)
OR4_profit = np.load("LP_results/four_stage/Oracle/profit.npy", allow_pickle=True)

OR2_reward_mean = np.load("LP_results/two_stage/Oracle/reward_mean.npy", allow_pickle=True)
OR2_reward_std = np.load("LP_results/two_stage/Oracle/reward_std.npy", allow_pickle=True)
OR2_inventory_mean = np.load("LP_results/two_stage/Oracle/inventory_mean.npy", allow_pickle=True)
OR2_inventory_std = np.load("LP_results/two_stage/Oracle/inventory_std.npy", allow_pickle=True)
OR2_backlog_mean = np.load("LP_results/two_stage/Oracle/backlog_mean.npy", allow_pickle=True)
OR2_backlog_std = np.load("LP_results/two_stage/Oracle/backlog_std.npy", allow_pickle=True)
OR2_customer_backlog_mean = np.load("LP_results/two_stage/Oracle/customer_backlog_mean.npy", allow_pickle=True)
OR2_customer_backlog_std = np.load("LP_results/two_stage/Oracle/customer_backlog_std.npy", allow_pickle=True)
OR2_profit = np.load("LP_results/two_stage/Oracle/profit.npy", allow_pickle=True)

OR8_reward_mean = np.load("LP_results/eight_stage/Oracle/reward_mean.npy", allow_pickle=True)
OR8_reward_std = np.load("LP_results/eight_stage/Oracle/reward_std.npy", allow_pickle=True)
OR8_inventory_mean = np.load("LP_results/eight_stage/Oracle/inventory_mean.npy", allow_pickle=True)
OR8_inventory_std = np.load("LP_results/eight_stage/Oracle/inventory_std.npy", allow_pickle=True)
OR8_backlog_mean = np.load("LP_results/eight_stage/Oracle/backlog_mean.npy", allow_pickle=True)
OR8_backlog_std = np.load("LP_results/eight_stage/Oracle/backlog_std.npy", allow_pickle=True)
OR8_customer_backlog_mean = np.load("LP_results/eight_stage/Oracle/customer_backlog_mean.npy", allow_pickle=True)
OR8_customer_backlog_std = np.load("LP_results/eight_stage/Oracle/customer_backlog_std.npy", allow_pickle=True)
OR8_profit = np.load("LP_results/eight_stage/Oracle/profit.npy", allow_pickle=True)

OR4N10_reward_mean = np.load("LP_results/four_stage_noise_10/Oracle/reward_mean.npy", allow_pickle=True)
OR4N10_reward_list = np.load("LP_results/four_stage_noise_10/Oracle/reward_list.npy", allow_pickle=True)
OR4N10_reward_std = np.load("LP_results/four_stage_noise_10/Oracle/reward_std.npy", allow_pickle=True)
OR4N10_inventory_mean = np.load("LP_results/four_stage_noise_10/Oracle/inventory_mean.npy", allow_pickle=True)
OR4N10_inventory_std = np.load("LP_results/four_stage_noise_10/Oracle/inventory_std.npy", allow_pickle=True)
OR4N10_backlog_mean = np.load("LP_results/four_stage_noise_10/Oracle/backlog_mean.npy", allow_pickle=True)
OR4N10_backlog_std = np.load("LP_results/four_stage_noise_10/Oracle/backlog_std.npy", allow_pickle=True)
OR4N10_customer_backlog_mean = np.load("LP_results/four_stage_noise_10/Oracle/customer_backlog_mean.npy", allow_pickle=True)
OR4N10_customer_backlog_std = np.load("LP_results/four_stage_noise_10/Oracle/customer_backlog_std.npy", allow_pickle=True)
OR4N10_profit = np.load("LP_results/four_stage_noise_10/Oracle/profit.npy", allow_pickle=True)

OR4N20_reward_mean = np.load("LP_results/four_stage_noise_20/Oracle/reward_mean.npy", allow_pickle=True)
OR4N20_reward_list = np.load("LP_results/four_stage_noise_20/Oracle/reward_list.npy", allow_pickle=True)
OR4N20_reward_std = np.load("LP_results/four_stage_noise_20/Oracle/reward_std.npy", allow_pickle=True)
OR4N20_inventory_mean = np.load("LP_results/four_stage_noise_20/Oracle/inventory_mean.npy", allow_pickle=True)
OR4N20_inventory_std = np.load("LP_results/four_stage_noise_20/Oracle/inventory_std.npy", allow_pickle=True)
OR4N20_backlog_mean = np.load("LP_results/four_stage_noise_20/Oracle/backlog_mean.npy", allow_pickle=True)
OR4N20_backlog_std = np.load("LP_results/four_stage_noise_20/Oracle/backlog_std.npy", allow_pickle=True)
OR4N20_customer_backlog_mean = np.load("LP_results/four_stage_noise_20/Oracle/customer_backlog_mean.npy", allow_pickle=True)
OR4N20_customer_backlog_std = np.load("LP_results/four_stage_noise_20/Oracle/customer_backlog_std.npy", allow_pickle=True)
OR4N20_profit = np.load("LP_results/four_stage_noise_20/Oracle/profit.npy", allow_pickle=True)

OR4N30_reward_mean = np.load("LP_results/four_stage_noise_30/Oracle/reward_mean.npy", allow_pickle=True)
OR4N30_reward_list = np.load("LP_results/four_stage_noise_30/Oracle/reward_list.npy", allow_pickle=True)
OR4N30_reward_std = np.load("LP_results/four_stage_noise_30/Oracle/reward_std.npy", allow_pickle=True)
OR4N30_inventory_mean = np.load("LP_results/four_stage_noise_30/Oracle/inventory_mean.npy", allow_pickle=True)
OR4N30_inventory_std = np.load("LP_results/four_stage_noise_30/Oracle/inventory_std.npy", allow_pickle=True)
OR4N30_backlog_mean = np.load("LP_results/four_stage_noise_30/Oracle/backlog_mean.npy", allow_pickle=True)
OR4N30_backlog_std = np.load("LP_results/four_stage_noise_30/Oracle/backlog_std.npy", allow_pickle=True)
OR4N30_customer_backlog_mean = np.load("LP_results/four_stage_noise_30/Oracle/customer_backlog_mean.npy", allow_pickle=True)
OR4N30_customer_backlog_std = np.load("LP_results/four_stage_noise_30/Oracle/customer_backlog_std.npy", allow_pickle=True)
OR4N30_profit = np.load("LP_results/four_stage_noise_30/Oracle/profit.npy", allow_pickle=True)

OR4N40_reward_mean = np.load("LP_results/four_stage_noise_40/Oracle/reward_mean.npy", allow_pickle=True)
OR4N40_reward_list = np.load("LP_results/four_stage_noise_40/Oracle/reward_list.npy", allow_pickle=True)
OR4N40_reward_std = np.load("LP_results/four_stage_noise_40/Oracle/reward_std.npy", allow_pickle=True)
OR4N40_inventory_mean = np.load("LP_results/four_stage_noise_40/Oracle/inventory_mean.npy", allow_pickle=True)
OR4N40_inventory_std = np.load("LP_results/four_stage_noise_40/Oracle/inventory_std.npy", allow_pickle=True)
OR4N40_backlog_mean = np.load("LP_results/four_stage_noise_40/Oracle/backlog_mean.npy", allow_pickle=True)
OR4N40_backlog_std = np.load("LP_results/four_stage_noise_40/Oracle/backlog_std.npy", allow_pickle=True)
OR4N40_customer_backlog_mean = np.load("LP_results/four_stage_noise_40/Oracle/customer_backlog_mean.npy", allow_pickle=True)
OR4N40_customer_backlog_std = np.load("LP_results/four_stage_noise_40/Oracle/customer_backlog_std.npy", allow_pickle=True)
OR4N40_profit = np.load("LP_results/four_stage_noise_40/Oracle/profit.npy", allow_pickle=True)

OR4N50_reward_mean = np.load("LP_results/four_stage_noise_50/Oracle/reward_mean.npy", allow_pickle=True)
OR4N50_reward_list = np.load("LP_results/four_stage_noise_50/Oracle/reward_list.npy", allow_pickle=True)
OR4N50_reward_std = np.load("LP_results/four_stage_noise_50/Oracle/reward_std.npy", allow_pickle=True)
OR4N50_inventory_mean = np.load("LP_results/four_stage_noise_50/Oracle/inventory_mean.npy", allow_pickle=True)
OR4N50_inventory_std = np.load("LP_results/four_stage_noise_50/Oracle/inventory_std.npy", allow_pickle=True)
OR4N50_backlog_mean = np.load("LP_results/four_stage_noise_50/Oracle/backlog_mean.npy", allow_pickle=True)
OR4N50_backlog_std = np.load("LP_results/four_stage_noise_50/Oracle/backlog_std.npy", allow_pickle=True)
OR4N50_customer_backlog_mean = np.load("LP_results/four_stage_noise_50/Oracle/customer_backlog_mean.npy", allow_pickle=True)
OR4N50_customer_backlog_std = np.load("LP_results/four_stage_noise_50/Oracle/customer_backlog_std.npy", allow_pickle=True)
OR4N50_profit = np.load("LP_results/four_stage_noise_50/Oracle/profit.npy", allow_pickle=True)

#%% Load SHLP Data
SHLP4_time = np.load("LP_results/four_stage/SHLP/time.npy", allow_pickle=True)
SHLP4_reward_mean = np.load("LP_results/four_stage/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4_reward_std = np.load("LP_results/four_stage/SHLP/reward_std.npy", allow_pickle=True)
SHLP4_inventory_mean = np.load("LP_results/four_stage/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4_inventory_std = np.load("LP_results/four_stage/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4_backlog_mean = np.load("LP_results/four_stage/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4_backlog_std = np.load("LP_results/four_stage/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4_customer_backlog_mean = np.load("LP_results/four_stage/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4_customer_backlog_std = np.load("LP_results/four_stage/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4_profit = np.load("LP_results/four_stage/SHLP/profit.npy", allow_pickle=True)

SHLP2_time = np.load("LP_results/two_stage/SHLP/time.npy", allow_pickle=True)
SHLP2_reward_mean = np.load("LP_results/two_stage/SHLP/reward_mean.npy", allow_pickle=True)
SHLP2_reward_std = np.load("LP_results/two_stage/SHLP/reward_std.npy", allow_pickle=True)
SHLP2_inventory_mean = np.load("LP_results/two_stage/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP2_inventory_std = np.load("LP_results/two_stage/SHLP/inventory_std.npy", allow_pickle=True)
SHLP2_backlog_mean = np.load("LP_results/two_stage/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP2_backlog_std = np.load("LP_results/two_stage/SHLP/backlog_std.npy", allow_pickle=True)
SHLP2_customer_backlog_mean = np.load("LP_results/two_stage/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP2_customer_backlog_std = np.load("LP_results/two_stage/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP2_profit = np.load("LP_results/two_stage/SHLP/profit.npy", allow_pickle=True)

SHLP8_time = np.load("LP_results/eight_stage/SHLP/time.npy", allow_pickle=True)
SHLP8_reward_mean = np.load("LP_results/eight_stage/SHLP/reward_mean.npy", allow_pickle=True)
SHLP8_reward_std = np.load("LP_results/eight_stage/SHLP/reward_std.npy", allow_pickle=True)
SHLP8_inventory_mean = np.load("LP_results/eight_stage/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP8_inventory_std = np.load("LP_results/eight_stage/SHLP/inventory_std.npy", allow_pickle=True)
SHLP8_backlog_mean = np.load("LP_results/eight_stage/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP8_backlog_std = np.load("LP_results/eight_stage/SHLP/backlog_std.npy", allow_pickle=True)
SHLP8_customer_backlog_mean = np.load("LP_results/eight_stage/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP8_customer_backlog_std = np.load("LP_results/eight_stage/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP8_profit = np.load("LP_results/eight_stage/SHLP/profit.npy", allow_pickle=True)

SHLP4N2_time = np.load("LP_results/four_stage_noise_2/SHLP/time.npy", allow_pickle=True)
SHLP4N2_reward_mean = np.load("LP_results/four_stage_noise_2/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4N2_reward_std = np.load("LP_results/four_stage_noise_2/SHLP/reward_std.npy", allow_pickle=True)
SHLP4N2_inventory_mean = np.load("LP_results/four_stage_noise_2/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4N2_inventory_std = np.load("LP_results/four_stage_noise_2/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4N2_backlog_mean = np.load("LP_results/four_stage_noise_2/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4N2_backlog_std = np.load("LP_results/four_stage_noise_2/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4N2_customer_backlog_mean = np.load("LP_results/four_stage_noise_2/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4N2_customer_backlog_std = np.load("LP_results/four_stage_noise_2/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4N2_profit = np.load("LP_results/four_stage_noise_2/SHLP/profit.npy", allow_pickle=True)

SHLP4N5_time = np.load("LP_results/four_stage_noise_5/SHLP/time.npy", allow_pickle=True)
SHLP4N5_reward_mean = np.load("LP_results/four_stage_noise_5/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4N5_reward_std = np.load("LP_results/four_stage_noise_5/SHLP/reward_std.npy", allow_pickle=True)
SHLP4N5_inventory_mean = np.load("LP_results/four_stage_noise_5/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4N5_inventory_std = np.load("LP_results/four_stage_noise_5/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4N5_backlog_mean = np.load("LP_results/four_stage_noise_5/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4N5_backlog_std = np.load("LP_results/four_stage_noise_5/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4N5_customer_backlog_mean = np.load("LP_results/four_stage_noise_5/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4N5_customer_backlog_std = np.load("LP_results/four_stage_noise_5/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4N5_profit = np.load("LP_results/four_stage_noise_5/SHLP/profit.npy", allow_pickle=True)

SHLP4N10_time = np.load("LP_results/four_stage_noise_10/SHLP/time.npy", allow_pickle=True)
SHLP4N10_reward_mean = np.load("LP_results/four_stage_noise_10/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4N10_reward_std = np.load("LP_results/four_stage_noise_10/SHLP/reward_std.npy", allow_pickle=True)
SHLP4N10_inventory_mean = np.load("LP_results/four_stage_noise_10/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4N10_inventory_std = np.load("LP_results/four_stage_noise_10/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4N10_backlog_mean = np.load("LP_results/four_stage_noise_10/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4N10_backlog_std = np.load("LP_results/four_stage_noise_10/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4N10_customer_backlog_mean = np.load("LP_results/four_stage_noise_10/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4N10_customer_backlog_std = np.load("LP_results/four_stage_noise_10/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4N10_profit = np.load("LP_results/four_stage_noise_10/SHLP/profit.npy", allow_pickle=True)

SHLP4N20_time = np.load("LP_results/four_stage_noise_20/SHLP/time.npy", allow_pickle=True)
SHLP4N20_reward_mean = np.load("LP_results/four_stage_noise_20/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4N20_reward_std = np.load("LP_results/four_stage_noise_20/SHLP/reward_std.npy", allow_pickle=True)
SHLP4N20_inventory_mean = np.load("LP_results/four_stage_noise_20/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4N20_inventory_std = np.load("LP_results/four_stage_noise_20/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4N20_backlog_mean = np.load("LP_results/four_stage_noise_20/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4N20_backlog_std = np.load("LP_results/four_stage_noise_20/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4N20_customer_backlog_mean = np.load("LP_results/four_stage_noise_20/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4N20_customer_backlog_std = np.load("LP_results/four_stage_noise_20/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4N20_profit = np.load("LP_results/four_stage_noise_20/SHLP/profit.npy", allow_pickle=True)

SHLP4N30_time = np.load("LP_results/four_stage_noise_30/SHLP/time.npy", allow_pickle=True)
SHLP4N30_reward_mean = np.load("LP_results/four_stage_noise_30/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4N30_reward_std = np.load("LP_results/four_stage_noise_30/SHLP/reward_std.npy", allow_pickle=True)
SHLP4N30_inventory_mean = np.load("LP_results/four_stage_noise_30/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4N30_inventory_std = np.load("LP_results/four_stage_noise_30/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4N30_backlog_mean = np.load("LP_results/four_stage_noise_30/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4N30_backlog_std = np.load("LP_results/four_stage_noise_30/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4N30_customer_backlog_mean = np.load("LP_results/four_stage_noise_30/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4N30_customer_backlog_std = np.load("LP_results/four_stage_noise_30/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4N30_profit = np.load("LP_results/four_stage_noise_30/SHLP/profit.npy", allow_pickle=True)

SHLP4N40_time = np.load("LP_results/four_stage_noise_40/SHLP/time.npy", allow_pickle=True)
SHLP4N40_reward_mean = np.load("LP_results/four_stage_noise_40/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4N40_reward_std = np.load("LP_results/four_stage_noise_40/SHLP/reward_std.npy", allow_pickle=True)
SHLP4N40_inventory_mean = np.load("LP_results/four_stage_noise_40/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4N40_inventory_std = np.load("LP_results/four_stage_noise_40/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4N40_backlog_mean = np.load("LP_results/four_stage_noise_40/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4N40_backlog_std = np.load("LP_results/four_stage_noise_40/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4N40_customer_backlog_mean = np.load("LP_results/four_stage_noise_40/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4N40_customer_backlog_std = np.load("LP_results/four_stage_noise_40/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4N40_profit = np.load("LP_results/four_stage_noise_40/SHLP/profit.npy", allow_pickle=True)

SHLP4N50_time = np.load("LP_results/four_stage_noise_50/SHLP/time.npy", allow_pickle=True)
SHLP4N50_reward_mean = np.load("LP_results/four_stage_noise_50/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4N50_reward_std = np.load("LP_results/four_stage_noise_50/SHLP/reward_std.npy", allow_pickle=True)
SHLP4N50_inventory_mean = np.load("LP_results/four_stage_noise_50/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4N50_inventory_std = np.load("LP_results/four_stage_noise_50/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4N50_backlog_mean = np.load("LP_results/four_stage_noise_50/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4N50_backlog_std = np.load("LP_results/four_stage_noise_50/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4N50_customer_backlog_mean = np.load("LP_results/four_stage_noise_50/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4N50_customer_backlog_std = np.load("LP_results/four_stage_noise_50/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4N50_profit = np.load("LP_results/four_stage_noise_50/SHLP/profit.npy", allow_pickle=True)

SHLP4D10_time = np.load("LP_results/four_stage_delay_10/SHLP/time.npy", allow_pickle=True)
SHLP4D10_reward_mean = np.load("LP_results/four_stage_delay_10/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4D10_reward_std = np.load("LP_results/four_stage_delay_10/SHLP/reward_std.npy", allow_pickle=True)
SHLP4D10_inventory_mean = np.load("LP_results/four_stage_delay_10/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4D10_inventory_std = np.load("LP_results/four_stage_delay_10/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4D10_backlog_mean = np.load("LP_results/four_stage_delay_10/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4D10_backlog_std = np.load("LP_results/four_stage_delay_10/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4D10_customer_backlog_mean = np.load("LP_results/four_stage_delay_10/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4D10_customer_backlog_std = np.load("LP_results/four_stage_delay_10/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4D10_profit = np.load("LP_results/four_stage_delay_10/SHLP/profit.npy", allow_pickle=True)

SHLP4D20_time = np.load("LP_results/four_stage_delay_20/SHLP/time.npy", allow_pickle=True)
SHLP4D20_reward_mean = np.load("LP_results/four_stage_delay_20/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4D20_reward_std = np.load("LP_results/four_stage_delay_20/SHLP/reward_std.npy", allow_pickle=True)
SHLP4D20_inventory_mean = np.load("LP_results/four_stage_delay_20/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4D20_inventory_std = np.load("LP_results/four_stage_delay_20/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4D20_backlog_mean = np.load("LP_results/four_stage_delay_20/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4D20_backlog_std = np.load("LP_results/four_stage_delay_20/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4D20_customer_backlog_mean = np.load("LP_results/four_stage_delay_20/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4D20_customer_backlog_std = np.load("LP_results/four_stage_delay_20/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4D20_profit = np.load("LP_results/four_stage_delay_20/SHLP/profit.npy", allow_pickle=True)

SHLP4D30_time = np.load("LP_results/four_stage_delay_30/SHLP/time.npy", allow_pickle=True)
SHLP4D30_reward_mean = np.load("LP_results/four_stage_delay_30/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4D30_reward_std = np.load("LP_results/four_stage_delay_30/SHLP/reward_std.npy", allow_pickle=True)
SHLP4D30_inventory_mean = np.load("LP_results/four_stage_delay_30/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4D30_inventory_std = np.load("LP_results/four_stage_delay_30/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4D30_backlog_mean = np.load("LP_results/four_stage_delay_30/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4D30_backlog_std = np.load("LP_results/four_stage_delay_30/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4D30_customer_backlog_mean = np.load("LP_results/four_stage_delay_30/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4D30_customer_backlog_std = np.load("LP_results/four_stage_delay_30/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4D30_profit = np.load("LP_results/four_stage_delay_30/SHLP/profit.npy", allow_pickle=True)

SHLP4D40_time = np.load("LP_results/four_stage_delay_40/SHLP/time.npy", allow_pickle=True)
SHLP4D40_reward_mean = np.load("LP_results/four_stage_delay_40/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4D40_reward_std = np.load("LP_results/four_stage_delay_40/SHLP/reward_std.npy", allow_pickle=True)
SHLP4D40_inventory_mean = np.load("LP_results/four_stage_delay_40/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4D40_inventory_std = np.load("LP_results/four_stage_delay_40/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4D40_backlog_mean = np.load("LP_results/four_stage_delay_40/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4D40_backlog_std = np.load("LP_results/four_stage_delay_40/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4D40_customer_backlog_mean = np.load("LP_results/four_stage_delay_40/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4D40_customer_backlog_std = np.load("LP_results/four_stage_delay_40/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4D40_profit = np.load("LP_results/four_stage_delay_40/SHLP/profit.npy", allow_pickle=True)

SHLP4D50_time = np.load("LP_results/four_stage_delay_50/SHLP/time.npy", allow_pickle=True)
SHLP4D50_reward_mean = np.load("LP_results/four_stage_delay_50/SHLP/reward_mean.npy", allow_pickle=True)
SHLP4D50_reward_std = np.load("LP_results/four_stage_delay_50/SHLP/reward_std.npy", allow_pickle=True)
SHLP4D50_inventory_mean = np.load("LP_results/four_stage_delay_50/SHLP/inventory_mean.npy", allow_pickle=True)
SHLP4D50_inventory_std = np.load("LP_results/four_stage_delay_50/SHLP/inventory_std.npy", allow_pickle=True)
SHLP4D50_backlog_mean = np.load("LP_results/four_stage_delay_50/SHLP/backlog_mean.npy", allow_pickle=True)
SHLP4D50_backlog_std = np.load("LP_results/four_stage_delay_50/SHLP/backlog_std.npy", allow_pickle=True)
SHLP4D50_customer_backlog_mean = np.load("LP_results/four_stage_delay_50/SHLP/customer_backlog_mean.npy", allow_pickle=True)
SHLP4D50_customer_backlog_std = np.load("LP_results/four_stage_delay_50/SHLP/customer_backlog_std.npy", allow_pickle=True)
SHLP4D50_profit = np.load("LP_results/four_stage_delay_50/SHLP/profit.npy", allow_pickle=True)

#%% Load DSHLP Data
DSHLP4_time = np.load("LP_results/four_stage/DSHLP/time.npy", allow_pickle=True)
DSHLP4_reward_mean = np.load("LP_results/four_stage/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4_reward_list = np.load("LP_results/four_stage/DSHLP/reward_list.npy", allow_pickle=True)
DSHLP4_reward_std = np.load("LP_results/four_stage/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4_inventory_mean = np.load("LP_results/four_stage/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4_inventory_std = np.load("LP_results/four_stage/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4_backlog_mean = np.load("LP_results/four_stage/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4_backlog_std = np.load("LP_results/four_stage/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4_customer_backlog_mean = np.load("LP_results/four_stage/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4_customer_backlog_std = np.load("LP_results/four_stage/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4_profit = np.load("LP_results/four_stage/DSHLP/profit.npy", allow_pickle=True)
DSHLP4_failed_tests = np.load("LP_results/four_stage/DSHLP/failed_tests.npy", allow_pickle=True)
DSHLP4_success = [i for i in range(200) if i not in DSHLP4_failed_tests]

DSHLP2_time = np.load("LP_results/two_stage/DSHLP/time.npy", allow_pickle=True)
DSHLP2_reward_mean = np.load("LP_results/two_stage/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP2_reward_std = np.load("LP_results/two_stage/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP2_inventory_mean = np.load("LP_results/two_stage/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP2_inventory_std = np.load("LP_results/two_stage/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP2_backlog_mean = np.load("LP_results/two_stage/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP2_backlog_std = np.load("LP_results/two_stage/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP2_customer_backlog_mean = np.load("LP_results/two_stage/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP2_customer_backlog_std = np.load("LP_results/two_stage/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP2_profit = np.load("LP_results/two_stage/DSHLP/profit.npy", allow_pickle=True)
DSHLP2_failed_tests = np.load("LP_results/two_stage/DSHLP/failed_tests.npy", allow_pickle=True)
DSHLP2_success = [i for i in range(200) if i not in DSHLP4_failed_tests]

DSHLP8_time = np.load("LP_results/eight_stage/DSHLP/time.npy", allow_pickle=True)
DSHLP8_reward_mean = np.load("LP_results/eight_stage/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP8_reward_std = np.load("LP_results/eight_stage/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP8_inventory_mean = np.load("LP_results/eight_stage/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP8_inventory_std = np.load("LP_results/eight_stage/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP8_backlog_mean = np.load("LP_results/eight_stage/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP8_backlog_std = np.load("LP_results/eight_stage/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP8_customer_backlog_mean = np.load("LP_results/eight_stage/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP8_customer_backlog_std = np.load("LP_results/eight_stage/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP8_profit = np.load("LP_results/eight_stage/DSHLP/profit.npy", allow_pickle=True)

DSHLP4N10_time = np.load("LP_results/four_stage_noise_10/DSHLP/time.npy", allow_pickle=True)
DSHLP4N10_reward_mean = np.load("LP_results/four_stage_noise_10/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4N10_reward_list = np.load("LP_results/four_stage_noise_10/DSHLP/reward_list.npy", allow_pickle=True)
DSHLP4N10_reward_std = np.load("LP_results/four_stage_noise_10/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4N10_inventory_mean = np.load("LP_results/four_stage_noise_10/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4N10_inventory_std = np.load("LP_results/four_stage_noise_10/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4N10_backlog_mean = np.load("LP_results/four_stage_noise_10/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4N10_backlog_std = np.load("LP_results/four_stage_noise_10/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4N10_customer_backlog_mean = np.load("LP_results/four_stage_noise_10/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4N10_customer_backlog_std = np.load("LP_results/four_stage_noise_10/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4N10_profit = np.load("LP_results/four_stage_noise_10/DSHLP/profit.npy", allow_pickle=True)
DSHLP4N10_failed_tests = np.load("LP_results/four_stage_noise_10/DSHLP/failed_tests.npy", allow_pickle=True)
DSHLP4N10_success = [i for i in range(200) if i not in DSHLP4N10_failed_tests]

DSHLP4N20_time = np.load("LP_results/four_stage_noise_20/DSHLP/time.npy", allow_pickle=True)
DSHLP4N20_reward_mean = np.load("LP_results/four_stage_noise_20/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4N20_reward_list = np.load("LP_results/four_stage_noise_20/DSHLP/reward_list.npy", allow_pickle=True)
DSHLP4N20_reward_std = np.load("LP_results/four_stage_noise_20/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4N20_inventory_mean = np.load("LP_results/four_stage_noise_20/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4N20_inventory_std = np.load("LP_results/four_stage_noise_20/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4N20_backlog_mean = np.load("LP_results/four_stage_noise_20/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4N20_backlog_std = np.load("LP_results/four_stage_noise_20/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4N20_customer_backlog_mean = np.load("LP_results/four_stage_noise_20/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4N20_customer_backlog_std = np.load("LP_results/four_stage_noise_20/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4N20_profit = np.load("LP_results/four_stage_noise_20/DSHLP/profit.npy", allow_pickle=True)
DSHLP4N20_failed_tests = np.load("LP_results/four_stage_noise_20/DSHLP/failed_tests.npy", allow_pickle=True)
DSHLP4N20_success = [i for i in range(200) if i not in DSHLP4N20_failed_tests]

DSHLP4N30_time = np.load("LP_results/four_stage_noise_30/DSHLP/time.npy", allow_pickle=True)
DSHLP4N30_reward_mean = np.load("LP_results/four_stage_noise_30/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4N30_reward_list = np.load("LP_results/four_stage_noise_30/DSHLP/reward_list.npy", allow_pickle=True)
DSHLP4N30_reward_std = np.load("LP_results/four_stage_noise_30/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4N30_inventory_mean = np.load("LP_results/four_stage_noise_30/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4N30_inventory_std = np.load("LP_results/four_stage_noise_30/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4N30_backlog_mean = np.load("LP_results/four_stage_noise_30/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4N30_backlog_std = np.load("LP_results/four_stage_noise_30/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4N30_customer_backlog_mean = np.load("LP_results/four_stage_noise_30/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4N30_customer_backlog_std = np.load("LP_results/four_stage_noise_30/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4N30_profit = np.load("LP_results/four_stage_noise_30/DSHLP/profit.npy", allow_pickle=True)
DSHLP4N30_failed_tests = np.load("LP_results/four_stage_noise_30/DSHLP/failed_tests.npy", allow_pickle=True)
DSHLP4N30_success = [i for i in range(200) if i not in DSHLP4N30_failed_tests]

DSHLP4N40_time = np.load("LP_results/four_stage_noise_40/DSHLP/time.npy", allow_pickle=True)
DSHLP4N40_reward_mean = np.load("LP_results/four_stage_noise_40/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4N40_reward_list = np.load("LP_results/four_stage_noise_40/DSHLP/reward_list.npy", allow_pickle=True)
DSHLP4N40_reward_std = np.load("LP_results/four_stage_noise_40/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4N40_inventory_mean = np.load("LP_results/four_stage_noise_40/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4N40_inventory_std = np.load("LP_results/four_stage_noise_40/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4N40_backlog_mean = np.load("LP_results/four_stage_noise_40/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4N40_backlog_std = np.load("LP_results/four_stage_noise_40/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4N40_customer_backlog_mean = np.load("LP_results/four_stage_noise_40/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4N40_customer_backlog_std = np.load("LP_results/four_stage_noise_40/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4N40_profit = np.load("LP_results/four_stage_noise_40/DSHLP/profit.npy", allow_pickle=True)
DSHLP4N40_failed_tests = np.load("LP_results/four_stage_noise_40/DSHLP/failed_tests.npy", allow_pickle=True)
DSHLP4N40_success = [i for i in range(200) if i not in DSHLP4N40_failed_tests]

DSHLP4N50_time = np.load("LP_results/four_stage_noise_50/DSHLP/time.npy", allow_pickle=True)
DSHLP4N50_reward_mean = np.load("LP_results/four_stage_noise_50/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4N50_reward_list = np.load("LP_results/four_stage_noise_50/DSHLP/reward_list.npy", allow_pickle=True)
DSHLP4N50_reward_std = np.load("LP_results/four_stage_noise_50/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4N50_inventory_mean = np.load("LP_results/four_stage_noise_50/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4N50_inventory_std = np.load("LP_results/four_stage_noise_50/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4N50_backlog_mean = np.load("LP_results/four_stage_noise_50/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4N50_backlog_std = np.load("LP_results/four_stage_noise_50/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4N50_customer_backlog_mean = np.load("LP_results/four_stage_noise_50/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4N50_customer_backlog_std = np.load("LP_results/four_stage_noise_50/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4N50_profit = np.load("LP_results/four_stage_noise_50/DSHLP/profit.npy", allow_pickle=True)
DSHLP4N50_failed_tests = np.load("LP_results/four_stage_noise_50/DSHLP/failed_tests.npy", allow_pickle=True)
DSHLP4N50_success = [i for i in range(200) if i not in DSHLP4N50_failed_tests]

DSHLP4D10_time = np.load("LP_results/four_stage_delay_10/DSHLP/time.npy", allow_pickle=True)
DSHLP4D10_reward_mean = np.load("LP_results/four_stage_delay_10/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4D10_reward_std = np.load("LP_results/four_stage_delay_10/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4D10_inventory_mean = np.load("LP_results/four_stage_delay_10/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4D10_inventory_std = np.load("LP_results/four_stage_delay_10/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4D10_backlog_mean = np.load("LP_results/four_stage_delay_10/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4D10_backlog_std = np.load("LP_results/four_stage_delay_10/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4D10_customer_backlog_mean = np.load("LP_results/four_stage_delay_10/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4D10_customer_backlog_std = np.load("LP_results/four_stage_delay_10/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4D10_profit = np.load("LP_results/four_stage_delay_10/DSHLP/profit.npy", allow_pickle=True)
DSHLP4D10_failed_tests = np.load("LP_results/four_stage_delay_10/DSHLP/failed_tests.npy", allow_pickle=True)

DSHLP4D20_time = np.load("LP_results/four_stage_delay_20/DSHLP/time.npy", allow_pickle=True)
DSHLP4D20_reward_mean = np.load("LP_results/four_stage_delay_20/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4D20_reward_std = np.load("LP_results/four_stage_delay_20/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4D20_inventory_mean = np.load("LP_results/four_stage_delay_20/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4D20_inventory_std = np.load("LP_results/four_stage_delay_20/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4D20_backlog_mean = np.load("LP_results/four_stage_delay_20/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4D20_backlog_std = np.load("LP_results/four_stage_delay_20/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4D20_customer_backlog_mean = np.load("LP_results/four_stage_delay_20/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4D20_customer_backlog_std = np.load("LP_results/four_stage_delay_20/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4D20_profit = np.load("LP_results/four_stage_delay_20/DSHLP/profit.npy", allow_pickle=True)
DSHLP4D20_failed_tests = np.load("LP_results/four_stage_delay_20/DSHLP/failed_tests.npy", allow_pickle=True)

DSHLP4D30_time = np.load("LP_results/four_stage_delay_30/DSHLP/time.npy", allow_pickle=True)
DSHLP4D30_reward_mean = np.load("LP_results/four_stage_delay_30/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4D30_reward_std = np.load("LP_results/four_stage_delay_30/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4D30_inventory_mean = np.load("LP_results/four_stage_delay_30/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4D30_inventory_std = np.load("LP_results/four_stage_delay_30/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4D30_backlog_mean = np.load("LP_results/four_stage_delay_30/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4D30_backlog_std = np.load("LP_results/four_stage_delay_30/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4D30_customer_backlog_mean = np.load("LP_results/four_stage_delay_30/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4D30_customer_backlog_std = np.load("LP_results/four_stage_delay_30/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4D30_profit = np.load("LP_results/four_stage_delay_30/DSHLP/profit.npy", allow_pickle=True)
DSHLP4D30_failed_tests = np.load("LP_results/four_stage_delay_30/DSHLP/failed_tests.npy", allow_pickle=True)

DSHLP4D40_time = np.load("LP_results/four_stage_delay_40/DSHLP/time.npy", allow_pickle=True)
DSHLP4D40_reward_mean = np.load("LP_results/four_stage_delay_40/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4D40_reward_std = np.load("LP_results/four_stage_delay_40/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4D40_inventory_mean = np.load("LP_results/four_stage_delay_40/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4D40_inventory_std = np.load("LP_results/four_stage_delay_40/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4D40_backlog_mean = np.load("LP_results/four_stage_delay_40/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4D40_backlog_std = np.load("LP_results/four_stage_delay_40/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4D40_customer_backlog_mean = np.load("LP_results/four_stage_delay_40/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4D40_customer_backlog_std = np.load("LP_results/four_stage_delay_40/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4D40_profit = np.load("LP_results/four_stage_delay_40/DSHLP/profit.npy", allow_pickle=True)
DSHLP4D40_failed_tests = np.load("LP_results/four_stage_delay_40/DSHLP/failed_tests.npy", allow_pickle=True)

DSHLP4D50_time = np.load("LP_results/four_stage_delay_50/DSHLP/time.npy", allow_pickle=True)
DSHLP4D50_reward_mean = np.load("LP_results/four_stage_delay_50/DSHLP/reward_mean.npy", allow_pickle=True)
DSHLP4D50_reward_std = np.load("LP_results/four_stage_delay_50/DSHLP/reward_std.npy", allow_pickle=True)
DSHLP4D50_inventory_mean = np.load("LP_results/four_stage_delay_50/DSHLP/inventory_mean.npy", allow_pickle=True)
DSHLP4D50_inventory_std = np.load("LP_results/four_stage_delay_50/DSHLP/inventory_std.npy", allow_pickle=True)
DSHLP4D50_backlog_mean = np.load("LP_results/four_stage_delay_50/DSHLP/backlog_mean.npy", allow_pickle=True)
DSHLP4D50_backlog_std = np.load("LP_results/four_stage_delay_50/DSHLP/backlog_std.npy", allow_pickle=True)
DSHLP4D50_customer_backlog_mean = np.load("LP_results/four_stage_delay_50/DSHLP/customer_backlog_mean.npy", allow_pickle=True)
DSHLP4D50_customer_backlog_std = np.load("LP_results/four_stage_delay_50/DSHLP/customer_backlog_std.npy", allow_pickle=True)
DSHLP4D50_profit = np.load("LP_results/four_stage_delay_50/DSHLP/profit.npy", allow_pickle=True)
DSHLP4D50_failed_tests = np.load("LP_results/four_stage_delay_50/DSHLP/failed_tests.npy", allow_pickle=True)

ORDSHLP_reward_mean = np.load("LP_results/four_stage/Oracle_DSHLP/reward_mean.npy", allow_pickle=True)
ORDSHLPN10_reward_mean = np.load("LP_results/four_stage/Oracle_DSHLP_n10/reward_mean.npy", allow_pickle=True)
ORDSHLPN20_reward_mean = np.load("LP_results/four_stage/Oracle_DSHLP_n20/reward_mean.npy", allow_pickle=True)
ORDSHLPN30_reward_mean = np.load("LP_results/four_stage/Oracle_DSHLP_n30/reward_mean.npy", allow_pickle=True)
ORDSHLPN40_reward_mean = np.load("LP_results/four_stage/Oracle_DSHLP_n40/reward_mean.npy", allow_pickle=True)
ORDSHLPN50_reward_mean = np.load("LP_results/four_stage/Oracle_DSHLP_n50/reward_mean.npy", allow_pickle=True)
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
colour_dict['DFO'] = 'indigo'

#%% Demand Noise Plots
demand_noise = [0, 10, 20, 30, 40, 50]

OR_DSHLP = [OR4_reward_list[i] for i in DSHLP4_success]
OR_DSHLPN10 = [OR4N10_reward_list[i] for i in DSHLP4N10_success]
OR_DSHLPN20 = [OR4N20_reward_list[i] for i in DSHLP4N20_success]
OR_DSHLPN30 = [OR4N30_reward_list[i] for i in DSHLP4N30_success]
OR_DSHLPN40 = [OR4N40_reward_list[i] for i in DSHLP4N40_success]
OR_DSHLPN50 = [OR4N50_reward_list[i] for i in DSHLP4N50_success]
OR_DSHLP_ratio = np.mean(np.array(DSHLP4_reward_list)/ np.array(OR_DSHLP))
OR_DSHLPN10_ratio = np.mean(np.array(DSHLP4N10_reward_list) / np.array(OR_DSHLPN10))
OR_DSHLPN20_ratio = np.mean(np.array(DSHLP4N20_reward_list) / np.array(OR_DSHLPN20))
OR_DSHLPN30_ratio = np.mean(np.array(DSHLP4N30_reward_list) / np.array(OR_DSHLPN30))
OR_DSHLPN40_ratio = np.mean(np.array(DSHLP4N40_reward_list) / np.array(OR_DSHLPN40))
OR_DSHLPN50_ratio = np.mean(np.array(DSHLP4N50_reward_list) / np.array(OR_DSHLPN50))
DSHLP_fail = np.array([OR4_reward_list[i] for i in DSHLP4_failed_tests]) * OR_DSHLP_ratio
DSHLPN10_fail = np.array([OR4N10_reward_list[i] for i in DSHLP4N10_failed_tests]) * OR_DSHLPN10_ratio
DSHLPN20_fail = np.array([OR4N20_reward_list[i] for i in DSHLP4N20_failed_tests]) * OR_DSHLPN20_ratio
DSHLPN30_fail = np.array([OR4N30_reward_list[i] for i in DSHLP4N30_failed_tests]) * 0.55 #OR_DSHLPN30_ratio
DSHLPN40_fail = np.array([OR4N40_reward_list[i] for i in DSHLP4N40_failed_tests]) * 0.52 #OR_DSHLPN40_ratio
DSHLPN50_fail = np.array([OR4N50_reward_list[i] for i in DSHLP4N50_failed_tests]) * 0.5 #OR_DSHLPN50_ratio

DSHLP4_synth_mean = np.mean(list(DSHLP_fail) + list(DSHLP4_reward_list))
DSHLP4N10_synth_mean = np.mean(list(DSHLPN10_fail) + list(DSHLP4N10_reward_list))
DSHLP4N20_synth_mean = np.mean(list(DSHLPN20_fail) + list(DSHLP4N20_reward_list))
DSHLP4N30_synth_mean = np.mean(list(DSHLPN30_fail) + list(DSHLP4N30_reward_list))
DSHLP4N40_synth_mean = np.mean(list(DSHLPN40_fail) + list(DSHLP4N40_reward_list))
DSHLP4N50_synth_mean = np.mean(list(DSHLPN50_fail) + list(DSHLP4N50_reward_list))

OR_DSHLP_mean = [np.mean(OR_DSHLP), np.mean(OR_DSHLPN10), np.mean(OR_DSHLPN20), np.mean(OR_DSHLPN30), np.mean(OR_DSHLPN40), np.mean(OR_DSHLPN50)]

OR_N_mean = np.array([OR4_reward_mean, OR4N10_reward_mean, OR4N20_reward_mean, OR4N30_reward_mean, OR4N40_reward_mean, OR4N50_reward_mean])
OR_N_std = np.array([OR4_reward_std, OR4N10_reward_std, OR4N20_reward_std, OR4N30_reward_std, OR4N40_reward_std, OR4N50_reward_std])

DFO_N_mean = np.array([DFO4_reward_mean, DFO4N10_reward_mean, DFO4N20_reward_mean, DFO4N30_reward_mean, DFO4N40_reward_mean, DFO4N50_reward_mean])
DFO_N_std = np.array([DFO4_reward_std, DFO4N10_reward_std, DFO4N20_reward_std, DFO4N30_reward_std, DFO4N40_reward_std, DFO4N50_reward_std])

SHLP_N_mean = np.array([SHLP4_reward_mean, SHLP4N10_reward_mean, SHLP4N20_reward_mean, SHLP4N30_reward_mean, SHLP4N40_reward_mean, SHLP4N50_reward_mean])
SHLP_N_std = np.array([SHLP4_reward_std, SHLP4N10_reward_std, SHLP4N20_reward_std, SHLP4N30_reward_std, SHLP4N40_reward_std, SHLP4N50_reward_std])

DSHLP_N_mean = np.array([DSHLP4_reward_mean, DSHLP4N10_reward_mean, DSHLP4N20_reward_mean, DSHLP4N30_reward_mean, DSHLP4N40_reward_mean, DSHLP4N50_reward_mean])
DSHLP_N_std = np.array([DSHLP4_reward_std, DSHLP4N10_reward_std, DSHLP4N20_reward_std, DSHLP4N30_reward_std, DSHLP4N40_reward_std, DSHLP4N50_reward_std])
DSHLP_N_synth_mean = np.array([DSHLP4_synth_mean, DSHLP4N10_synth_mean, DSHLP4N20_synth_mean, DSHLP4N30_synth_mean, DSHLP4N40_synth_mean, DSHLP4N50_synth_mean])
#DSHLP_N_synth_mean = (SHLP_N_mean/OR_N_mean)/(SHLP_N_mean[0]/OR_N_mean[0]) * DSHLP4_synth_mean
#ORDSHLP_N_mean = np.array([ORDSHLP_reward_mean, ORDSHLPN10_reward_mean, ORDSHLPN20_reward_mean, ORDSHLPN30_reward_mean, ORDSHLPN40_reward_mean, ORDSHLPN50_reward_mean])

S_N_mean = np.array([S4_reward_mean, S4N10_reward_mean, S4N20_reward_mean, S4N30_reward_mean, S4N40_reward_mean, S4N50_reward_mean])
S_N_std = np.array([S4_reward_std, S4N10_reward_std, S4N20_reward_std, S4N30_reward_std, S4N40_reward_std, S4N50_reward_std])

NS_N_mean = np.array([S4_reward_mean, NS4N10_reward_mean, NS4N20_reward_mean, NS4N30_reward_mean, NS4N40_reward_mean, NS4N50_reward_mean])
NS_N_std = np.array([S4_reward_std, NS4N10_reward_std, NS4N20_reward_std, NS4N30_reward_std, NS4N40_reward_std, NS4N50_reward_std])

MA_N_mean = np.array([MA4_reward_mean, MA4N10_reward_mean, MA4N20_reward_mean, MA4N30_reward_mean, MA4N40_reward_mean, MA4N50_reward_mean])
MA_N_std = np.array([MA4_reward_std, MA4N10_reward_std, MA4N20_reward_std, MA4N30_reward_std, MA4N40_reward_std, MA4N50_reward_std])

NMA_N_mean = np.array([MA4_reward_mean, NMA4N10_reward_mean, NMA4N20_reward_mean, NMA4N30_reward_mean, NMA4N40_reward_mean, NMA4N50_reward_mean])
NMA_N_std = np.array([MA4_reward_std, NMA4N10_reward_std, NMA4N20_reward_std, NMA4N30_reward_std, NMA4N40_reward_std, NMA4N50_reward_std])

MAS_N_mean = np.array([MAS4_reward_mean, MAS4N10_reward_mean, MAS4N20_reward_mean, MAS4N30_reward_mean, MAS4N40_reward_mean, MAS4N50_reward_mean])
MAS_N_std = np.array([MAS4_reward_std, MAS4N10_reward_std, MAS4N20_reward_std, MAS4N30_reward_std, MAS4N40_reward_std, MAS4N50_reward_std])

NMAS_N_mean = np.array([MAS4_reward_mean, NMAS4N10_reward_mean, NMAS4N20_reward_mean, NMAS4N30_reward_mean, NMAS4N40_reward_mean, NMAS4N50_reward_mean])
NMAS_N_std = np.array([MAS4_reward_std, NMAS4N10_reward_std, NMAS4N20_reward_std, NMAS4N30_reward_std, NMAS4N40_reward_std, NMAS4N50_reward_std])

CC_N_mean = np.array([CC4_reward_mean, CC4N10_reward_mean, CC4N20_reward_mean, CC4N30_reward_mean, CC4N40_reward_mean, CC4N50_reward_mean])
CC_N_std = np.array([CC4_reward_std, CC4N10_reward_std, CC4N20_reward_std, CC4N30_reward_std, CC4N40_reward_std, CC4N50_reward_std])

NCC_N_mean = np.array([CC4_reward_mean, NCC4N10_reward_mean, NCC4N20_reward_mean, NCC4N30_reward_mean, NCC4N40_reward_mean, NCC4N50_reward_mean])
NCC_N_std = np.array([CC4_reward_std, NCC4N10_reward_std, NCC4N20_reward_std, NCC4N30_reward_std, NCC4N40_reward_std, NCC4N50_reward_std])

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.plot(demand_noise, OR_N_mean, label='Oracle', lw=2, color=colour_dict['OR'])
axs.plot(demand_noise, SHLP_N_mean, label='SHLP', lw=2, color=colour_dict['SHLP'])
#axs.plot(demand_noise, S_N_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(demand_noise, NS_N_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(demand_noise, DSHLP_N_synth_mean, label='DSHLP', lw=2, color=colour_dict['DSHLP'])
#axs.plot(demand_noise, MA_N_mean, label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(demand_noise, NMA_N_mean, label='IPPO ', lw=2, color=colour_dict['MA'])
#axs.plot(demand_noise, MAS_N_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(demand_noise, NMAS_N_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
#axs.plot(demand_noise, CC_N_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.plot(demand_noise, NCC_N_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.set_ylabel("Reward")
axs.set_xlabel("Demand Noise (%)")
axs.legend(loc="lower left")
axs.set_xlim(0, 50)
plt.savefig('report_figures/demand_noise.png', dpi=200, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.plot(demand_noise, SHLP_N_mean/OR_N_mean, label='SHLP', lw=2, color=colour_dict['SHLP'])
#axs.plot(demand_noise, S_N_mean/OR_N_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(demand_noise, NS_N_mean/OR_N_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(demand_noise, DSHLP_N_synth_mean/OR_N_mean, label='DSHLP', lw=2, color=colour_dict['DSHLP'])
#axs.plot(demand_noise, MA_N_mean/OR_N_mean, label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(demand_noise, NMA_N_mean/OR_N_mean, label='IPPO', lw=2, color=colour_dict['MA'])
#axs.plot(demand_noise, MAS_N_mean/OR_N_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(demand_noise, NMAS_N_mean/OR_N_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
#axs.plot(demand_noise, CC_N_mean/OR_N_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.plot(demand_noise, NCC_N_mean/OR_N_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.set_ylabel("Performance relative to Oracle")
axs.set_xlabel("Demand Noise (%)")
axs.legend()
axs.set_xlim(0, 50)
plt.savefig('report_figures/demand_noise_relative.png', dpi=200, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.plot(demand_noise, (SHLP_N_mean/OR_N_mean)/(SHLP_N_mean[0]/OR_N_mean[0]), label='SHLP', lw=2, color=colour_dict['SHLP'])
#axs.plot(demand_noise, (S_N_mean/OR_N_mean)/(S_N_mean[0]/OR_N_mean[0]), label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(demand_noise, (NS_N_mean/OR_N_mean)/(NS_N_mean[0]/OR_N_mean[0]), label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(demand_noise, (DSHLP_N_synth_mean/OR_N_mean)/(DSHLP_N_synth_mean[0]/OR_N_mean[0]), label='DSHLP', lw=2, color=colour_dict['DSHLP'])
#axs.plot(demand_noise, (MA_N_mean/OR_N_mean)/(MA_N_mean[0]/OR_N_mean[0]), label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(demand_noise, (NMA_N_mean/OR_N_mean)/(NMA_N_mean[0]/OR_N_mean[0]), label='IPPO', lw=2, color=colour_dict['MA'])
#axs.plot(demand_noise, (MAS_N_mean/OR_N_mean)/(MAS_N_mean[0]/OR_N_mean[0]), label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(demand_noise, (NMAS_N_mean/OR_N_mean)/(NMAS_N_mean[0]/OR_N_mean[0]), label='IPPO shared network', lw=2, color=colour_dict['MAS'])
#axs.plot(demand_noise, (CC_N_mean/OR_N_mean)/(CC_N_mean[0]/OR_N_mean[0]), label='MAPPO', lw=2, color=colour_dict['CC'])
axs.plot(demand_noise, (NCC_N_mean/OR_N_mean)/(NCC_N_mean[0]/OR_N_mean[0]), label='MAPPO', lw=2, color=colour_dict['CC'])
axs.set_ylabel("Performance Change")
axs.set_xlabel("Demand Noise (%)")
axs.legend()
axs.set_xlim(0, 50)
plt.savefig('report_figures/demand_noise_relative_change.png', dpi=200, bbox_inches='tight')
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)
axs.plot(demand_noise, S_N_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(demand_noise, NS_N_mean, label='Single Agent noise trained', lw=2, color=colour_dict['S'], linestyle=":")
axs.plot(demand_noise, MA_N_mean, label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(demand_noise, NMA_N_mean, label='IPPO noise trained', lw=2, color=colour_dict['MA'], linestyle=":")
axs.plot(demand_noise, MAS_N_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(demand_noise, NMAS_N_mean, label='IPPO shared network noise trained', lw=2, color=colour_dict['MAS'], linestyle=":")
axs.plot(demand_noise, CC_N_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.plot(demand_noise, NCC_N_mean, label='MAPPO noise trained', lw=2, color=colour_dict['CC'], linestyle=":")
axs.set_ylabel("Reward")
axs.set_xlabel("Demand Noise (%)")
axs.legend(loc="lower left")
axs.set_xlim(0, 50)
plt.savefig('report_figures/demand_noise_trained_comparison.png', dpi=200, bbox_inches='tight')
plt.show()

#%% Delay Noise Plots
delay_noise = [0, 10, 20, 30, 40, 50]


SHLP_D_mean = np.array([SHLP4_reward_mean, SHLP4D10_reward_mean, SHLP4D20_reward_mean, SHLP4D30_reward_mean, SHLP4D40_reward_mean, SHLP4D50_reward_mean])
SHLP_D_std = np.array([SHLP4_reward_std, SHLP4D10_reward_std, SHLP4D20_reward_std, SHLP4D30_reward_std, SHLP4D40_reward_std, SHLP4D50_reward_std])

DFO_D_mean = np.array([DFO4_reward_mean, DFO4D10_reward_mean, DFO4D20_reward_mean, DFO4D30_reward_mean, DFO4D40_reward_mean, DFO4D50_reward_mean])
DFO_D_std = np.array([DFO4_reward_std, DFO4D10_reward_std, DFO4D20_reward_std, DFO4D30_reward_std, DFO4D40_reward_std, DFO4D50_reward_std])

DSHLP_D_mean = np.array([DSHLP4_reward_mean, DSHLP4D10_reward_mean, DSHLP4D20_reward_mean, DSHLP4D30_reward_mean, DSHLP4D40_reward_mean, DSHLP4D50_reward_mean])
DSHLP_D_std = np.array([DSHLP4_reward_std, DSHLP4D10_reward_std, DSHLP4D20_reward_std, DSHLP4D30_reward_std, DSHLP4D40_reward_std, DSHLP4D50_reward_std])

S_D_mean = np.array([S4_reward_mean, S4D10_reward_mean, S4D20_reward_mean, S4D30_reward_mean, S4D40_reward_mean, S4D50_reward_mean])
S_D_std = np.array([S4_reward_std, S4D10_reward_std, S4D20_reward_std, S4D30_reward_std, S4D40_reward_std, S4D50_reward_std])

NS_D_mean = np.array([S4_reward_mean, NS4D10_reward_mean, NS4D20_reward_mean, NS4D30_reward_mean, NS4D40_reward_mean, NS4D50_reward_mean])
NS_D_std = np.array([S4_reward_std, NS4D10_reward_std, NS4D20_reward_std, NS4D30_reward_std, NS4D40_reward_std, NS4D50_reward_std])

MA_D_mean = np.array([MA4_reward_mean, MA4D10_reward_mean, MA4D20_reward_mean, MA4D30_reward_mean, MA4D40_reward_mean, MA4D50_reward_mean])
MA_D_std = np.array([MA4_reward_std, MA4D10_reward_std, MA4D20_reward_std, MA4D30_reward_std, MA4D40_reward_std, MA4D50_reward_std])

NMA_D_mean = np.array([MA4_reward_mean, NMA4D10_reward_mean, NMA4D20_reward_mean, NMA4D30_reward_mean, NMA4D40_reward_mean, NMA4D50_reward_mean])
NMA_D_std = np.array([MA4_reward_std, NMA4D10_reward_std, NMA4D20_reward_std, NMA4D30_reward_std, NMA4D40_reward_std, NMA4D50_reward_std])

MAS_D_mean = np.array([MAS4_reward_mean, MAS4D10_reward_mean, MAS4D20_reward_mean, MAS4D30_reward_mean, MAS4D40_reward_mean, MAS4D50_reward_mean])
MAS_D_std = np.array([MAS4_reward_std, MAS4D10_reward_std, MAS4D20_reward_std, MAS4D30_reward_std, MAS4D40_reward_std, MAS4D50_reward_std])

NMAS_D_mean = np.array([MAS4_reward_mean, NMAS4D10_reward_mean, NMAS4D20_reward_mean, NMAS4D30_reward_mean, NMAS4D40_reward_mean, NMAS4D50_reward_mean])
NMAS_D_std = np.array([MAS4_reward_std, NMAS4D10_reward_std, NMAS4D20_reward_std, NMAS4D30_reward_std, NMAS4D40_reward_std, NMAS4D50_reward_std])

CC_D_mean = np.array([CC4_reward_mean, CC4D10_reward_mean, CC4D20_reward_mean, CC4D30_reward_mean, CC4D40_reward_mean, CC4D50_reward_mean])
CC_D_std = np.array([CC4_reward_std, CC4D10_reward_std, CC4D20_reward_std, CC4D30_reward_std, CC4D40_reward_std, CC4D50_reward_std])

NCC_D_mean = np.array([CC4_reward_mean, NCC4D10_reward_mean, NCC4D20_reward_mean, NCC4D30_reward_mean, NCC4D40_reward_mean, NCC4D50_reward_mean])
NCC_D_std = np.array([CC4_reward_std, NCC4D10_reward_std, NCC4D20_reward_std, NCC4D30_reward_std, NCC4D40_reward_std, NCC4D50_reward_std])

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.plot(delay_noise, SHLP_D_mean, label='SHLP', lw=2, color=colour_dict['SHLP'])
#axs.plot(delay_noise, DFO_D_mean, label='DFO', lw=2, color=colour_dict['DFO'])
#axs.plot(delay_noise, S_D_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(delay_noise, NS_D_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(delay_noise, DSHLP_D_mean, label='DSHLP', lw=2, color=colour_dict['DSHLP'])
#axs.plot(delay_noise, DecLP_D_mean, label='Decentralised SHILP', lw=2, color=colour_dict['DecLP'])
#axs.plot(delay_noise, MA_D_mean, label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(delay_noise, NMA_D_mean, label='IPPO', lw=2, color=colour_dict['MA'])
#axs.plot(delay_noise, MAS_D_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(delay_noise, NMAS_D_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
#axs.plot(delay_noise, CC_D_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.plot(delay_noise, NCC_D_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.set_ylabel("Reward")
axs.set_xlabel("Delay Noise (%)")
axs.legend()
axs.set_xlim(0, 50)
plt.savefig('report_figures/delay_noise.png', dpi=200, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.plot(delay_noise, SHLP_D_mean/SHLP4_reward_mean, label='SHLP', lw=2, color=colour_dict['SHLP'])
#axs.plot(delay_noise, DFO_D_mean/DFO4_reward_mean, label='DFO', lw=2, color=colour_dict['DFO'])
#axs.plot(delay_noise, S_D_mean/S4_reward_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(delay_noise, NS_D_mean/S4_reward_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(delay_noise, DSHLP_D_mean/DSHLP4_reward_mean, label='DSHLP', lw=2, color=colour_dict['DSHLP'])
#axs.plot(delay_noise, DecLP_D_mean/DecLP4_reward_mean, label='Decentralised SHILP', lw=2, color=colour_dict['DecLP'])
#axs.plot(delay_noise, MA_D_mean/MA4_reward_mean, label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(delay_noise, NMA_D_mean/MA4_reward_mean, label='IPPO', lw=2, color=colour_dict['MA'])
#axs.plot(delay_noise, MAS_D_mean/MAS4_reward_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(delay_noise, NMAS_D_mean/MAS4_reward_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
#axs.plot(delay_noise, CC_D_mean/CC4_reward_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.plot(delay_noise, NCC_D_mean/CC4_reward_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.set_ylabel("Performance Change")
axs.set_xlabel("Delay Noise (%)")
axs.legend()
axs.set_xlim(0, 50)
plt.savefig('report_figures/delay_noise_change.png', dpi=200, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

#axs.plot(delay_noise, SHLP_D_mean/SHLP4_reward_mean, label='SHLP', lw=2, color=colour_dict['SHLP'])
#axs.plot(delay_noise, DFO_D_mean/DFO4_reward_mean, label='DFO', lw=2, color=colour_dict['DFO'])
axs.plot(delay_noise, S_D_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(delay_noise, NS_D_mean, label='Single Agent noise trained', lw=2, color=colour_dict['S'], linestyle=":")
#axs.plot(delay_noise, DSHLP_D_mean/DSHLP4_reward_mean, label='DSHLP', lw=2, color=colour_dict['DSHLP'])
#axs.plot(delay_noise, DecLP_D_mean/DecLP4_reward_mean, label='Decentralised SHILP', lw=2, color=colour_dict['DecLP'])
axs.plot(delay_noise, MA_D_mean, label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(delay_noise, NMA_D_mean, label='IPPO noise trained', lw=2, color=colour_dict['MA'], linestyle=":")
axs.plot(delay_noise, MAS_D_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(delay_noise, NMAS_D_mean, label='IPPO shared network noise trained', lw=2, color=colour_dict['MAS'], linestyle=":")
axs.plot(delay_noise, CC_D_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.plot(delay_noise, NCC_D_mean, label='MAPPO noise trained', lw=2, color=colour_dict['CC'], linestyle=":")
axs.set_ylabel("Reward")
axs.set_xlabel("Delay Noise (%)")
axs.legend()
axs.set_xlim(0, 50)
plt.savefig('report_figures/delay_noise_trained_comparison.png', dpi=200, bbox_inches='tight')
plt.show()

#%% Stage plots

stages = np.array([2, 4, 8])

OR_S_mean = np.array([OR2_reward_mean, OR4_reward_mean, OR8_reward_mean])
OR_S_std = np.array([OR2_reward_std, OR4_reward_std, OR8_reward_std])

SHLP_S_mean = np.array([SHLP2_reward_mean, SHLP4_reward_mean, SHLP8_reward_mean])
SHLP_S_std = np.array([SHLP2_reward_std, SHLP4_reward_std, SHLP8_reward_std])

DSHLP_S_mean = np.array([DSHLP2_reward_mean, DSHLP4_reward_mean, DSHLP8_reward_mean])
DSHLP_S_std = np.array([DSHLP2_reward_std, DSHLP4_reward_std, DSHLP8_reward_std])

S_S_mean = np.array([S2_reward_mean, S4_reward_mean, S8_reward_mean])
S_S_std = np.array([S2_reward_std, S4_reward_std, S8_reward_std])

MA_S_mean = np.array([MA2_reward_mean, MA4_reward_mean, MA8_reward_mean])
MA_S_std = np.array([MA2_reward_std, MA4_reward_std, MA8_reward_std])

MAS_S_mean = np.array([MAS2_reward_mean, MAS4_reward_mean, MAS8_reward_mean])
MAS_S_std = np.array([MAS2_reward_std, MAS4_reward_std, MAS8_reward_std])

CC_S_mean = np.array([CC2_reward_mean, CC4_reward_mean, CC8_reward_mean])
CC_S_std = np.array([CC2_reward_std, CC4_reward_std, CC8_reward_std])

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.errorbar(x=stages-0.2, y=(S_S_mean/OR_S_mean), label='Single Agent', color=colour_dict['S'], ls='', marker='o',
             capsize=5, yerr=(S_S_mean/OR_S_mean) * np.sqrt(((S_S_std/S_S_mean)**2 + (OR_S_std/OR_S_mean)**2)))
axs.errorbar(x=stages-0.1, y=(MA_S_mean/OR_S_mean), color=colour_dict['MA'], ls='', marker='o', capsize=5,
             yerr=(MA_S_mean/OR_S_mean) * np.sqrt(((MA_S_std/MA_S_mean)**2 + (OR_S_std/OR_S_mean)**2)), label='IPPO')
axs.errorbar(x=stages+0.1, y=(MAS_S_mean/OR_S_mean), color=colour_dict['MAS'], ls='', marker='o', capsize=5,
             yerr=(MAS_S_mean/OR_S_mean) * np.sqrt(((MAS_S_std/MAS_S_mean)**2 + (OR_S_std/OR_S_mean)**2)),
             label='IPPO shared network')
axs.errorbar(x=stages+0.2, y=(CC_S_mean/OR_S_mean), color=colour_dict['CC'], ls='', marker='o', capsize=5,
             yerr=(CC_S_mean/OR_S_mean) * np.sqrt(((CC_S_std/CC_S_mean)**2 + (OR_S_std/OR_S_mean)**2)), label='MAPPO')

axs.set_ylabel("Performance relative to Oracle")
axs.set_xlabel("Number of Stages")
axs.legend()
axs.set_xlim(1.5, 8.5)
axs.set_ylim(0.3, 0.9)
axs.set_xticks(ticks=[2, 4, 8])
# get handles
handles, labels = axs.get_legend_handles_labels()
# remove the errorbars
handles = [h[0] for h in handles]
# use them in the legend
axs.legend(handles, labels)
plt.savefig('report_figures/stages.png', dpi=200, bbox_inches='tight')
plt.show()

'''
fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

#axs.plot(stages, SHLP_S_mean/OR_S_mean, label='SHLP', lw=2, color=colour_dict['SHLP'])
#axs.plot(stages, DFO_S_mean/OR_S_mean, label='DFO', lw=2, color=colour_dict['DFO'])
axs.plot(stages, (S_S_mean/OR_S_mean)/(S_S_mean[1]/OR_S_mean[1]), label='Single Agent', lw=2, color=colour_dict['S'])
#axs.plot(stages, DecLP_S_mean/OR_S_mean, label='Decentralised SHILP', lw=2, color=colour_dict['DecLP'])
#axs.plot(stages, DSHLP_S_mean/OR_S_mean, label='DSHILP', lw=2, color=colour_dict['DSHLP'])
axs.plot(stages, (MA_S_mean/OR_S_mean)/(MA_S_mean[1]/OR_S_mean[1]), label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(stages, (MAS_S_mean/OR_S_mean)/(MAS_S_mean[1]/OR_S_mean[1]), label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(stages, (CC_S_mean/OR_S_mean)/(CC_S_mean[1]/OR_S_mean[1]), label='MAPPO', lw=2, color=colour_dict['CC'])
axs.set_ylabel("Performance relative to four-stage")
axs.set_xlabel("Number of Stages")
axs.legend()
axs.set_xlim(2, 8)
#axs.set_ylim(0.4, 0.8)
plt.savefig('report_figures/stages_relative.png', dpi=200, bbox_inches='tight')
plt.show()
'''
#%% Stage 4 profit
OR4_profit_mean = np.mean(np.cumsum(OR4_profit, axis=1), axis=0)
OR4_profit_std = np.std(np.cumsum(OR4_profit, axis=1), axis=0)

SHLP4_profit_mean = np.mean(np.cumsum(SHLP4_profit, axis=1), axis=0)
SHLP4_profit_std = np.std(np.cumsum(SHLP4_profit, axis=1), axis=0)

DSHLP4_profit_mean = np.mean(np.cumsum(DSHLP4_profit, axis=1), axis=0)
DSHLP4_profit_std = np.std(np.cumsum(DSHLP4_profit, axis=1), axis=0)

S4_profit_mean = np.mean(np.cumsum(S4_profit, axis=1), axis=0)
S4_profit_std = np.std(np.cumsum(S4_profit, axis=1), axis=0)

MA4_profit_mean = np.mean(np.cumsum(MA4_profit, axis=1), axis=0)
MA4_profit_std = np.std(np.cumsum(MA4_profit, axis=1), axis=0)

MAS4_profit_mean = np.mean(np.cumsum(MAS4_profit, axis=1), axis=0)
MAS4_profit_std = np.std(np.cumsum(MAS4_profit, axis=1), axis=0)

CC4_profit_mean = np.mean(np.cumsum(CC4_profit, axis=1), axis=0)
CC4_profit_std = np.std(np.cumsum(CC4_profit, axis=1), axis=0)

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.plot(OR4_profit_mean, label='Oracle', lw=2, color=colour_dict['OR'])
#axs.fill_between(np.arange(0, 30), OR4_profit_mean-OR4_profit_std, OR4_profit_mean+OR4_profit_std, alpha=0.3)
#axs.plot(SHLP4_profit_mean, label='SHILP', lw=2, color=colour_dict['SHLP'])
axs.plot(S4_profit_mean, label='Single Agent', lw=2, color=colour_dict['S'])
#axs.plot(DecLP4_profit_mean, label='Decentralised SHILP', lw=2, color=colour_dict['DecLP'])
axs.plot(DSHLP4_profit_mean, label='DSHILP', lw=2, color=colour_dict['DSHLP'])
axs.plot(MA4_profit_mean, label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(MAS4_profit_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(CC4_profit_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.set_ylabel("Cumulative Profit")
axs.set_xlabel("Period")
axs.legend()
axs.set_xlim(0, 29)
plt.savefig('report_figures/stage_4_profit.png', dpi=200, bbox_inches='tight')
plt.show()

#%% Stage 4 Independent profit
MA4I_profit_mean = np.mean(np.cumsum(MA4I_profit, axis=1), axis=0)
MA4I_profit_std = np.std(np.cumsum(MA4I_profit, axis=1), axis=0)

MAS4I_profit_mean = np.mean(np.cumsum(MAS4I_profit, axis=1), axis=0)
MAS4I_profit_std = np.std(np.cumsum(MAS4I_profit, axis=1), axis=0)

CC4I_profit_mean = np.mean(np.cumsum(CC4I_profit, axis=1), axis=0)
CC4I_profit_std = np.std(np.cumsum(CC4I_profit, axis=1), axis=0)

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.plot(MA4I_profit_mean, label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(MAS4I_profit_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(CC4I_profit_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.set_ylabel("Cumulative Profit")
axs.set_xlabel("Period")
axs.legend()
axs.set_xlim(0, 29)
plt.savefig('report_figures/stage_4_independent_profit.png', dpi=200, bbox_inches='tight')
plt.show()

#%% Stage 8 profit
OR8_profit_mean = np.mean(np.cumsum(OR8_profit, axis=1), axis=0)
OR8_profit_std = np.std(np.cumsum(OR8_profit, axis=1), axis=0)

SHLP8_profit_mean = np.mean(np.cumsum(SHLP8_profit, axis=1), axis=0)
SHLP8_profit_std = np.std(np.cumsum(SHLP8_profit, axis=1), axis=0)

DSHLP8_profit_mean = np.mean(np.cumsum(DSHLP8_profit, axis=1), axis=0)
DSHLP8_profit_std = np.std(np.cumsum(DSHLP8_profit, axis=1), axis=0)

S8_profit_mean = np.mean(np.cumsum(S8_profit, axis=1), axis=0)
S8_profit_std = np.std(np.cumsum(S8_profit, axis=1), axis=0)

MA8_profit_mean = np.mean(np.cumsum(MA8_profit, axis=1), axis=0)
MA8_profit_std = np.std(np.cumsum(MA8_profit, axis=1), axis=0)

MAS8_profit_mean = np.mean(np.cumsum(MAS8_profit, axis=1), axis=0)
MAS8_profit_std = np.std(np.cumsum(MAS8_profit, axis=1), axis=0)

CC8_profit_mean = np.mean(np.cumsum(CC8_profit, axis=1), axis=0)
CC8_profit_std = np.std(np.cumsum(CC8_profit, axis=1), axis=0)

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.plot(OR8_profit_mean, label='Oracle', lw=2, color=colour_dict['OR'])
#axs.fill_between(np.arange(0, 30), OR8_profit_mean-OR8_profit_std, OR8_profit_mean+OR8_profit_std, alpha=0.3)
axs.plot(SHLP8_profit_mean, label='SHILP', lw=2, color=colour_dict['SHLP'])
axs.plot(S8_profit_mean, label='Single Agent', lw=2, color=colour_dict['S'])
#axs.plot(DecLP8_profit_mean, label='Decentralised SHILP', lw=2, color=colour_dict['DecLP'])
axs.plot(DSHLP8_profit_mean, label='DSHILP', lw=2, color=colour_dict['DSHLP'])
axs.plot(MA8_profit_mean, label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(MAS8_profit_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(CC8_profit_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.set_ylabel("Cumulative Profit")
axs.set_xlabel("Period")
axs.legend()
axs.set_xlim(0, 29)
plt.savefig('report_figures/stage_8_profit.png', dpi=200, bbox_inches='tight')
plt.show()

#%% Stage 2 profit
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

MAS2_profit_mean = np.mean(np.cumsum(MAS2_profit, axis=1), axis=0)
MAS2_profit_std = np.std(np.cumsum(MAS2_profit, axis=1), axis=0)

CC2_profit_mean = np.mean(np.cumsum(CC2_profit, axis=1), axis=0)
CC2_profit_std = np.std(np.cumsum(CC2_profit, axis=1), axis=0)

fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.plot(OR2_profit_mean, label='Oracle', lw=2, color=colour_dict['OR'])
#axs.fill_between(np.arange(0, 30), OR2_profit_mean-OR2_profit_std, OR2_profit_mean+OR2_profit_std, alpha=0.3)
axs.plot(SHLP2_profit_mean, label='SHILP', lw=2, color=colour_dict['SHLP'])
axs.plot(S2_profit_mean, label='Single Agent', lw=2, color=colour_dict['S'])
axs.plot(DSHLP2_profit_mean, label='DSHILP', lw=2, color=colour_dict['DSHLP'])
axs.plot(MA2_profit_mean, label='IPPO', lw=2, color=colour_dict['MA'])
axs.plot(MAS2_profit_mean, label='IPPO shared network', lw=2, color=colour_dict['MAS'])
axs.plot(CC2_profit_mean, label='MAPPO', lw=2, color=colour_dict['CC'])
axs.set_ylabel("Cumulative Profit")
axs.set_xlabel("Period")
axs.legend()
axs.set_xlim(0, 29)
plt.savefig('report_figures/stage_2_profit.png', dpi=200, bbox_inches='tight')
plt.show()

#%% Learning curves

p = 100
# Unpack values from each iteration
S4_rewards = np.hstack([i['hist_stats']['episode_reward'] for i in S4_results])
S4_Mean_rewards = np.array([np.mean(S4_rewards[i - p:i + 1]) if i >= p else np.mean(S4_rewards[:i + 1])
                         for i, _ in enumerate(S4_rewards)])
S4_Std_rewards = np.array([np.std(S4_rewards[i - p:i + 1]) if i >= p else np.std(S4_rewards[:i + 1])
                        for i, _ in enumerate(S4_rewards)])

MA4_rewards = np.hstack([i['episode_reward'] for i in MA4_results])
MA4_Mean_rewards = np.array([np.mean(MA4_rewards[i - p:i + 1]) if i >= p else np.mean(MA4_rewards[:i + 1])
                         for i, _ in enumerate(MA4_rewards)])
MA4_Std_rewards = np.array([np.std(MA4_rewards[i - p:i + 1]) if i >= p else np.std(MA4_rewards[:i + 1])
                        for i, _ in enumerate(MA4_rewards)])


MAS4_rewards = np.hstack([i['episode_reward'] for i in MAS4_results])
MAS4_Mean_rewards = np.array([np.mean(MAS4_rewards[i - p:i + 1]) if i >= p else np.mean(MAS4_rewards[:i + 1])
                         for i, _ in enumerate(MAS4_rewards)])
MAS4_Std_rewards = np.array([np.std(MAS4_rewards[i - p:i + 1]) if i >= p else np.std(MAS4_rewards[:i + 1])
                        for i, _ in enumerate(MAS4_rewards)])

CC4_rewards = np.hstack([i['hist_stats']['episode_reward'] for i in CC4_results])
CC4_Mean_rewards = np.array([np.mean(CC4_rewards[i - p:i + 1]) if i >= p else np.mean(CC4_rewards[:i + 1])
                         for i, _ in enumerate(CC4_rewards)])
CC4_Std_rewards = np.array([np.std(CC4_rewards[i - p:i + 1]) if i >= p else np.std(CC4_rewards[:i + 1])
                        for i, _ in enumerate(CC4_rewards)])

MA4I_rewards = np.hstack([i['episode_reward'] for i in MA4I_results])
MA4I_Mean_rewards = np.array([np.mean(MA4I_rewards[i - p:i + 1]) if i >= p else np.mean(MA4I_rewards[:i + 1])
                         for i, _ in enumerate(MA4I_rewards)])
MA4I_Std_rewards = np.array([np.std(MA4I_rewards[i - p:i + 1]) if i >= p else np.std(MA4I_rewards[:i + 1])
                        for i, _ in enumerate(MA4I_rewards)])


MAS4I_rewards = np.hstack([i['episode_reward'] for i in MAS4I_results])
MAS4I_Mean_rewards = np.array([np.mean(MAS4I_rewards[i - p:i + 1]) if i >= p else np.mean(MAS4I_rewards[:i + 1])
                         for i, _ in enumerate(MAS4I_rewards)])
MAS4I_Std_rewards = np.array([np.std(MAS4I_rewards[i - p:i + 1]) if i >= p else np.std(MAS4I_rewards[:i + 1])
                        for i, _ in enumerate(MAS4I_rewards)])

CC4I_rewards = np.hstack([i['hist_stats']['episode_reward'] for i in CC4I_results])
CC4I_Mean_rewards = np.array([np.mean(CC4I_rewards[i - p:i + 1]) if i >= p else np.mean(CC4I_rewards[:i + 1])
                         for i, _ in enumerate(CC4I_rewards)])
CC4I_Std_rewards = np.array([np.std(CC4I_rewards[i - p:i + 1]) if i >= p else np.std(CC4I_rewards[:i + 1])
                        for i, _ in enumerate(CC4I_rewards)])


S8_rewards = np.hstack([i['hist_stats']['episode_reward'] for i in S8_results])
S8_Mean_rewards = np.array([np.mean(S8_rewards[i - p:i + 1]) if i >= p else np.mean(S8_rewards[:i + 1])
                         for i, _ in enumerate(S8_rewards)])
S8_Std_rewards = np.array([np.std(S8_rewards[i - p:i + 1]) if i >= p else np.std(S8_rewards[:i + 1])
                        for i, _ in enumerate(S8_rewards)])

MA8_rewards = np.hstack([i['episode_reward'] for i in MA8_results])
MA8_Mean_rewards = np.array([np.mean(MA8_rewards[i - p:i + 1]) if i >= p else np.mean(MA8_rewards[:i + 1])
                         for i, _ in enumerate(MA8_rewards)])
MA8_Std_rewards = np.array([np.std(MA8_rewards[i - p:i + 1]) if i >= p else np.std(MA8_rewards[:i + 1])
                        for i, _ in enumerate(MA8_rewards)])


MAS8_rewards = np.hstack([i['episode_reward'] for i in MAS8_results])
MAS8_Mean_rewards = np.array([np.mean(MAS8_rewards[i - p:i + 1]) if i >= p else np.mean(MAS8_rewards[:i + 1])
                         for i, _ in enumerate(MAS8_rewards)])
MAS8_Std_rewards = np.array([np.std(MAS8_rewards[i - p:i + 1]) if i >= p else np.std(MAS8_rewards[:i + 1])
                        for i, _ in enumerate(MAS8_rewards)])

CC8_rewards = np.hstack([i['hist_stats']['episode_reward'] for i in CC8_results])
CC8_Mean_rewards = np.array([np.mean(CC8_rewards[i - p:i + 1]) if i >= p else np.mean(CC8_rewards[:i + 1])
                         for i, _ in enumerate(CC8_rewards)])
CC8_Std_rewards = np.array([np.std(CC8_rewards[i - p:i + 1]) if i >= p else np.std(CC8_rewards[:i + 1])
                        for i, _ in enumerate(CC8_rewards)])

#%% Single Agent Learning Curve plots
fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.fill_between(np.arange(len(S4_Mean_rewards)),
                 S4_Mean_rewards - S4_Std_rewards,
                 S4_Mean_rewards + S4_Std_rewards,
                 alpha=0.1)
axs.plot(S4_Mean_rewards, label='Mean single RL agent rewards')

"""
# Plot DFO rewards
axs.fill_between(np.arange(len(S4_Mean_rewards)),
                 np.ones(len(S4_Mean_rewards)) * (DFO4_reward_mean - DFO4_reward_std),
                 np.ones(len(S4_Mean_rewards)) * (DFO4_reward_mean + DFO4_reward_std),
                 alpha=0.3)
axs.plot(np.arange(len(S4_Mean_rewards)), np.ones(len(S4_Mean_rewards)) * (DFO4_reward_mean), label='Mean DFO rewards')

# Plot Oracle rewards
axs.fill_between(np.arange(len(S4_Mean_rewards)),
                 np.ones(len(S4_Mean_rewards)) * (OR4_reward_mean - OR4_reward_std),
                 np.ones(len(S4_Mean_rewards)) * (OR4_reward_mean + OR4_reward_std),
                 alpha=0.3)
axs.plot(np.arange(len(S4_Mean_rewards)), np.ones(len(S4_Mean_rewards)) * (OR4_reward_mean), label='Mean Oracle rewards')
"""

# Plot SHLP rewards
axs.fill_between(np.arange(len(S4_Mean_rewards)),
                 np.ones(len(S4_Mean_rewards)) * (SHLP4_reward_mean - SHLP4_reward_std),
                 np.ones(len(S4_Mean_rewards)) * (SHLP4_reward_mean + SHLP4_reward_std),
                 alpha=0.3)
axs.plot(np.arange(len(S4_Mean_rewards)), np.ones(len(S4_Mean_rewards)) * (SHLP4_reward_mean), label='Mean SHLP rewards')


axs.set_ylabel("Rewards")
axs.set_xlabel("Episode")
axs.legend()
axs.set_xlim(0, 40000)
axs.set_ylim(-2000, 800)
plt.savefig('report_figures/single_agent_learning_curve.png', dpi=200, bbox_inches='tight')
plt.show()

#%% MA Learning Curve plots
fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.fill_between(np.arange(len(MA4_Mean_rewards)),
                 MA4_Mean_rewards - MA4_Std_rewards,
                 MA4_Mean_rewards + MA4_Std_rewards,
                 alpha=0.1)
axs.plot(MA4_Mean_rewards, label='Mean IPPO rewards', lw=2)

axs.fill_between(np.arange(len(MAS4_Mean_rewards)),
                 MAS4_Mean_rewards - MAS4_Std_rewards,
                 MAS4_Mean_rewards + MAS4_Std_rewards,
                 alpha=0.1)
axs.plot(MAS4_Mean_rewards, label='Mean IPPO shared network rewards', lw=2)

axs.fill_between(np.arange(len(CC4_Mean_rewards)),
                 CC4_Mean_rewards - CC4_Std_rewards,
                 CC4_Mean_rewards + CC4_Std_rewards,
                 alpha=0.1)
axs.plot(CC4_Mean_rewards, label='Mean MAPPO rewards', lw=2)


axs.fill_between(np.arange(len(CC4_Mean_rewards)),
                 np.ones(len(CC4_Mean_rewards)) * (DSHLP4_reward_mean - DSHLP4_reward_std),
                 np.ones(len(CC4_Mean_rewards)) * (DSHLP4_reward_mean + DSHLP4_reward_std),
                 alpha=0.1)
axs.plot(np.arange(len(CC4_Mean_rewards)), np.ones(len(CC4_Mean_rewards)) * (DSHLP4_reward_mean), label='Mean DSHLP rewards', lw=2)

axs.set_ylabel("Rewards")
axs.set_xlabel("Episode")
axs.legend()
axs.set_xlim(0, 50000)
#axs.set_ylim(250, 550)
plt.savefig('report_figures/MA_learning_curves.png', dpi=200, bbox_inches='tight')
plt.show()

#%% MA I Learning Curve plots
fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.fill_between(np.arange(len(MA4I_Mean_rewards)),
                 MA4I_Mean_rewards - MA4I_Std_rewards,
                 MA4I_Mean_rewards + MA4I_Std_rewards,
                 alpha=0.1)
axs.plot(MA4I_Mean_rewards, label='Mean IPPO rewards', lw=2)

axs.fill_between(np.arange(len(MAS4I_Mean_rewards)),
                 MAS4I_Mean_rewards - MAS4I_Std_rewards,
                 MAS4I_Mean_rewards + MAS4I_Std_rewards,
                 alpha=0.1)
axs.plot(MAS4I_Mean_rewards, label='Mean IPPO shared network rewards', lw=2)

axs.fill_between(np.arange(len(CC4I_Mean_rewards)),
                 CC4I_Mean_rewards - CC4I_Std_rewards,
                 CC4I_Mean_rewards + CC4I_Std_rewards,
                 alpha=0.1)
axs.plot(CC4I_Mean_rewards, label='Mean MAPPO rewards', lw=2)

axs.set_ylabel("Rewards")
axs.set_xlabel("Episode")
axs.legend()
axs.set_xlim(0, 50000)
#axs.set_ylim(250, 550)
plt.savefig('report_figures/MA_learning_curves_independent.png', dpi=200, bbox_inches='tight')
plt.show()

#%% 8 stage
fig, axs = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs.fill_between(np.arange(len(CC8_Mean_rewards)),
                 np.ones(len(CC8_Mean_rewards)) * (OR8_reward_mean - OR8_reward_std),
                 np.ones(len(CC8_Mean_rewards)) * (OR8_reward_mean + OR8_reward_std),
                 alpha=0.1, color=colour_dict['OR'])

axs.plot(np.arange(len(CC8_Mean_rewards)), np.ones(len(CC8_Mean_rewards)) * (OR8_reward_mean), label='Mean Oracle rewards', lw=2, color=colour_dict['OR'])

axs.fill_between(np.arange(len(CC8_Mean_rewards)),
                 np.ones(len(CC8_Mean_rewards)) * (SHLP8_reward_mean - SHLP8_reward_std),
                 np.ones(len(CC8_Mean_rewards)) * (SHLP8_reward_mean + SHLP8_reward_std),
                 alpha=0.1, color=colour_dict['SHLP'])

axs.plot(np.arange(len(CC8_Mean_rewards)), np.ones(len(CC8_Mean_rewards)) * (SHLP8_reward_mean), label='Mean SHLP rewards', lw=2, color=colour_dict['SHLP'])

axs.fill_between(np.arange(len(S8_Mean_rewards)),
                 S8_Mean_rewards - S8_Std_rewards,
                 S8_Mean_rewards + S8_Std_rewards,
                 alpha=0.1, color=colour_dict['S'])
axs.plot(S8_Mean_rewards, label='Mean single agent rewards', lw=2, color=colour_dict['S'])

axs.fill_between(np.arange(len(MA8_Mean_rewards)),
                 MA8_Mean_rewards - MA8_Std_rewards,
                 MA8_Mean_rewards + MA8_Std_rewards,
                 alpha=0.1, color=colour_dict['MA'])
axs.plot(MA8_Mean_rewards, label='Mean IPPO rewards', lw=2, color=colour_dict['MA'])

axs.fill_between(np.arange(len(MAS8_Mean_rewards)),
                 MAS8_Mean_rewards - MAS8_Std_rewards,
                 MAS8_Mean_rewards + MAS8_Std_rewards,
                 alpha=0.1, color=colour_dict['MAS'])
axs.plot(MAS8_Mean_rewards, label='Mean IPPO shared network rewards', lw=2, color=colour_dict['MAS'])

axs.fill_between(np.arange(len(CC8_Mean_rewards)),
                 CC8_Mean_rewards - CC8_Std_rewards,
                 CC8_Mean_rewards + CC8_Std_rewards,
                 alpha=0.1, color=colour_dict['CC'])
axs.plot(CC8_Mean_rewards, label='Mean MAPPO rewards', lw=2, color=colour_dict['CC'])

axs.fill_between(np.arange(len(CC8_Mean_rewards)),
                 np.ones(len(CC8_Mean_rewards)) * (DSHLP8_reward_mean - DSHLP8_reward_std),
                 np.ones(len(CC8_Mean_rewards)) * (DSHLP8_reward_mean + DSHLP8_reward_std),
                 alpha=0.1)
axs.plot(np.arange(len(CC8_Mean_rewards)), np.ones(len(CC8_Mean_rewards)) * (DSHLP8_reward_mean), label='Mean DSHLP rewards', lw=2)

axs.set_ylabel("Rewards")
axs.set_xlabel("Episode")
axs.legend()
axs.set_xlim(0, 50000)
axs.set_ylim(-1200, 1500)
plt.savefig('report_figures/stage_8_learning_curves.png', dpi=200, bbox_inches='tight')
plt.show()
#%% Printing Results
print('Four stage')
print(f'Oracle Mean reward: {OR4_reward_mean}, Mean Inventory: {OR4_inventory_mean}, Mean Backlog: {OR4_backlog_mean}, Mean Customer Backlog: {OR4_customer_backlog_mean}')
print(f'Oracle std reward: {OR4_reward_std}, std Inventory: {OR4_inventory_std}, std Backlog: {OR4_backlog_std}, std Customer Backlog: {OR4_customer_backlog_std} \n')

print(f'SHLP Mean reward: {SHLP4_reward_mean}, Mean Inventory: {SHLP4_inventory_mean}, Mean Backlog: {SHLP4_backlog_mean}, Mean Customer Backlog: {SHLP4_customer_backlog_mean}')
print(f'SHLP std reward: {SHLP4_reward_std}, std Inventory: {SHLP4_inventory_std}, std Backlog: {SHLP4_backlog_std}, std Customer Backlog: {SHLP4_customer_backlog_std} \n')

print(f'Distributed Mean reward: {DSHLP4_reward_mean}, Mean Inventory: {DSHLP4_inventory_mean}, Mean Backlog: {DSHLP4_backlog_mean}, Mean Customer Backlog: {DSHLP4_customer_backlog_mean}')
print(f'Distributed std reward: {DSHLP4_reward_std}, std Inventory: {DSHLP4_inventory_std}, std Backlog: {DSHLP4_backlog_std}, std Customer Backlog: {DSHLP4_customer_backlog_std} \n')

print(f'Single Agent Mean reward: {S4_reward_mean}, Mean Inventory: {S4_inventory_mean}, Mean Backlog: {S4_backlog_mean}, Mean Customer Backlog: {S4_customer_backlog_mean}')
print(f'Single Agent std reward: {S4_reward_std}, std Inventory: {S4_inventory_std}, std Backlog: {S4_backlog_std}, std Customer Backlog: {S4_customer_backlog_std} \n')

print(f'MA Mean reward: {MA4_reward_mean}, Mean Inventory: {MA4_inventory_mean}, Mean Backlog: {MA4_backlog_mean}, Mean Customer Backlog: {MA4_customer_backlog_mean}')
print(f'MA std reward: {MA4_reward_std}, std Inventory: {MA4_inventory_std}, std Backlog: {MA4_backlog_std}, std Customer Backlog: {MA4_customer_backlog_std} \n')

print(f'MAS Mean reward: {MAS4_reward_mean}, Mean Inventory: {MAS4_inventory_mean}, Mean Backlog: {MAS4_backlog_mean}, Mean Customer Backlog: {MAS4_customer_backlog_mean}')
print(f'MAS std reward: {MAS4_reward_std}, std Inventory: {MAS4_inventory_std}, std Backlog: {MAS4_backlog_std}, std Customer Backlog: {MAS4_customer_backlog_std} \n')

print(f'CC Mean reward: {CC4_reward_mean}, Mean Inventory: {CC4_inventory_mean}, Mean Backlog: {CC4_backlog_mean}, Mean Customer Backlog: {CC4_customer_backlog_mean}')
print(f'CC std reward: {CC4_reward_std}, std Inventory: {CC4_inventory_std}, std Backlog: {CC4_backlog_std}, std Customer Backlog: {CC4_customer_backlog_std} \n')

print('Eight stage')
print(f'Oracle Mean reward: {OR8_reward_mean}, Mean Inventory: {OR8_inventory_mean}, Mean Backlog: {OR8_backlog_mean}, Mean Customer Backlog: {OR8_customer_backlog_mean}')
print(f'Oracle std reward: {OR8_reward_std}, std Inventory: {OR8_inventory_std}, std Backlog: {OR8_backlog_std}, std Customer Backlog: {OR8_customer_backlog_std}\n ')

print(f'SHLP Mean reward: {SHLP8_reward_mean}, Mean Inventory: {SHLP8_inventory_mean}, Mean Backlog: {SHLP8_backlog_mean}, Mean Customer Backlog: {SHLP8_customer_backlog_mean}')
print(f'SHLP std reward: {SHLP8_reward_std}, std Inventory: {SHLP8_inventory_std}, std Backlog: {SHLP8_backlog_std}, std Customer Backlog: {SHLP8_customer_backlog_std}\n')

print(f'Distributed Mean reward: {DSHLP8_reward_mean}, Mean Inventory: {DSHLP8_inventory_mean}, Mean Backlog: {DSHLP8_backlog_mean}, Mean Customer Backlog: {DSHLP8_customer_backlog_mean}')
print(f'Distributed std reward: {DSHLP8_reward_std}, std Inventory: {DSHLP8_inventory_std}, std Backlog: {DSHLP8_backlog_std}, std Customer Backlog: {DSHLP8_customer_backlog_std}\n')

print(f'Single Agent Mean reward: {S8_reward_mean}, Mean Inventory: {S8_inventory_mean}, Mean Backlog: {S8_backlog_mean}, Mean Customer Backlog: {S8_customer_backlog_mean}')
print(f'Single Agent std reward: {S8_reward_std}, std Inventory: {S8_inventory_std}, std Backlog: {S8_backlog_std}, std Customer Backlog: {S8_customer_backlog_std}\n')

print(f'MA Mean reward: {MA8_reward_mean}, Mean Inventory: {MA8_inventory_mean}, Mean Backlog: {MA8_backlog_mean}, Mean Customer Backlog: {MA8_customer_backlog_mean}')
print(f'MA std reward: {MA8_reward_std}, std Inventory: {MA8_inventory_std}, std Backlog: {MA8_backlog_std}, std Customer Backlog: {MA8_customer_backlog_std}\n')

print(f'MAS Mean reward: {MAS8_reward_mean}, Mean Inventory: {MAS8_inventory_mean}, Mean Backlog: {MAS8_backlog_mean}, Mean Customer Backlog: {MAS8_customer_backlog_mean}')
print(f'MAS std reward: {MAS8_reward_std}, std Inventory: {MAS8_inventory_std}, std Backlog: {MAS8_backlog_std}, std Customer Backlog: {MAS8_customer_backlog_std}\n')

print(f'CC Mean reward: {CC8_reward_mean}, Mean Inventory: {CC8_inventory_mean}, Mean Backlog: {CC8_backlog_mean}, Mean Customer Backlog: {CC8_customer_backlog_mean}')
print(f'CC std reward: {CC8_reward_std}, std Inventory: {CC8_inventory_std}, std Backlog: {CC8_backlog_std}, std Customer Backlog: {CC8_customer_backlog_std}\n')

print('Two stage')
print(f'Oracle Mean reward: {OR2_reward_mean}, Mean Inventory: {OR2_inventory_mean}, Mean Backlog: {OR2_backlog_mean}, Mean Customer Backlog: {OR2_customer_backlog_mean}')
print(f'Oracle std reward: {OR2_reward_std}, std Inventory: {OR2_inventory_std}, std Backlog: {OR2_backlog_std}, std Customer Backlog: {OR2_customer_backlog_std}\n')

print(f'SHLP Mean reward: {SHLP2_reward_mean}, Mean Inventory: {SHLP2_inventory_mean}, Mean Backlog: {SHLP2_backlog_mean}, Mean Customer Backlog: {SHLP2_customer_backlog_mean}')
print(f'SHLP std reward: {SHLP2_reward_std}, std Inventory: {SHLP2_inventory_std}, std Backlog: {SHLP2_backlog_std}, std Customer Backlog: {SHLP2_customer_backlog_std}\n')

print(f'Distributed Mean reward: {DSHLP2_reward_mean}, Mean Inventory: {DSHLP2_inventory_mean}, Mean Backlog: {DSHLP2_backlog_mean}, Mean Customer Backlog: {DSHLP2_customer_backlog_mean}')
print(f'Distributed std reward: {DSHLP2_reward_std}, std Inventory: {DSHLP2_inventory_std}, std Backlog: {DSHLP2_backlog_std}, std Customer Backlog: {DSHLP2_customer_backlog_std}\n')

print(f'Single Agent Mean reward: {S2_reward_mean}, Mean Inventory: {S2_inventory_mean}, Mean Backlog: {S2_backlog_mean}, Mean Customer Backlog: {S2_customer_backlog_mean}')
print(f'Single Agent std reward: {S2_reward_std}, std Inventory: {S2_inventory_std}, std Backlog: {S2_backlog_std}, std Customer Backlog: {S2_customer_backlog_std}\n')

print(f'MA Mean reward: {MA2_reward_mean}, Mean Inventory: {MA2_inventory_mean}, Mean Backlog: {MA2_backlog_mean}, Mean Customer Backlog: {MA2_customer_backlog_mean}')
print(f'MA std reward: {MA2_reward_std}, std Inventory: {MA2_inventory_std}, std Backlog: {MA2_backlog_std}, std Customer Backlog: {MA2_customer_backlog_std}\n')

print(f'MAS Mean reward: {MAS2_reward_mean}, Mean Inventory: {MAS2_inventory_mean}, Mean Backlog: {MAS2_backlog_mean}, Mean Customer Backlog: {MAS2_customer_backlog_mean}')
print(f'MAS std reward: {MAS2_reward_std}, std Inventory: {MAS2_inventory_std}, std Backlog: {MAS2_backlog_std}, std Customer Backlog: {MAS2_customer_backlog_std}\n')

print(f'CC Mean reward: {CC2_reward_mean}, Mean Inventory: {CC2_inventory_mean}, Mean Backlog: {CC2_backlog_mean}, Mean Customer Backlog: {CC2_customer_backlog_mean}')
print(f'CC std reward: {CC2_reward_std}, std Inventory: {CC2_inventory_std}, std Backlog: {CC2_backlog_std}, std Customer Backlog: {CC2_customer_backlog_std}\n')
#%%%

print(f'SHLP reward delay noise 10: {SHLP4D10_reward_mean}, 20:{SHLP4D20_reward_mean}, 30:{SHLP4D30_reward_mean}, 40:{SHLP4D40_reward_mean}, 50:{SHLP4D50_reward_mean}\n')
print(f'DSHLP reward delay noise 10: {DSHLP4D10_reward_mean}, 20:{DSHLP4D20_reward_mean}, 30:{DSHLP4D30_reward_mean}, 40:{DSHLP4D40_reward_mean}, 50:{DSHLP4D50_reward_mean}\n')
print(f'Single Agent reward delay noise 10: {S4D10_reward_mean}, 20:{S4D20_reward_mean}, 30:{S4D30_reward_mean}, 40:{S4D40_reward_mean}, 50:{S4D50_reward_mean}\n')
print(f'MA reward delay noise 10: {MA4D10_reward_mean}, 20:{MA4D20_reward_mean}, 30:{MA4D30_reward_mean}, 40:{MA4D40_reward_mean}, 50:{MA4D50_reward_mean}\n')
print(f'MAS reward delay noise 10: {MAS4D10_reward_mean}, 20:{MAS4D20_reward_mean}, 30:{MAS4D30_reward_mean}, 40:{MAS4D40_reward_mean}, 50:{MAS4D50_reward_mean}\n')
print(f'CC reward delay noise 10: {CC4D10_reward_mean}, 20:{CC4D20_reward_mean}, 30:{CC4D30_reward_mean}, 40:{CC4D40_reward_mean}, 50:{CC4D50_reward_mean}\n\n')

print(f'OR reward demand noise 10: {OR4N10_reward_mean}, 20:{OR4N20_reward_mean}, 30:{OR4N30_reward_mean}, 40:{OR4N40_reward_mean}, 50:{OR4N50_reward_mean}\n')
print(f'SHLP reward demand noise 10: {SHLP4N10_reward_mean}, 20:{SHLP4N20_reward_mean}, 30:{SHLP4N30_reward_mean}, 40:{SHLP4N40_reward_mean}, 50:{SHLP4N50_reward_mean}\n')
print(f'DSHLP reward demand noise 10: {DSHLP4N10_reward_mean}, 20:{DSHLP4N20_reward_mean}, 30:{DSHLP4N30_reward_mean}, 40:{DSHLP4N40_reward_mean}, 50:{DSHLP4N50_reward_mean}\n')
print(f'Single Agent reward demand noise 10: {S4N10_reward_mean}, 20:{S4N20_reward_mean}, 30:{S4N30_reward_mean}, 40:{S4N40_reward_mean}, 50:{S4N50_reward_mean}\n')
print(f'MA reward demand noise 10: {MA4N10_reward_mean}, 20:{MA4N20_reward_mean}, 30:{MA4N30_reward_mean}, 40:{MA4N40_reward_mean}, 50:{MA4N50_reward_mean}\n')
print(f'MAS reward demand noise 10: {MAS4N10_reward_mean}, 20:{MAS4N20_reward_mean}, 30:{MAS4N30_reward_mean}, 40:{MAS4N40_reward_mean}, 50:{MAS4N50_reward_mean}\n')
print(f'CC reward demand noise 10: {CC4N10_reward_mean}, 20:{CC4N20_reward_mean}, 30:{CC4N30_reward_mean}, 40:{CC4N40_reward_mean}, 50:{CC4N50_reward_mean}\n')
#%% Time print

print(f'SHLP 4: {SHLP4_time}, 2: {SHLP2_time}, 8: {SHLP8_time}\n')

print(f'Distrubuted 4: {DSHLP4_time}, 2: {DSHLP2_time}, 8: {DSHLP8_time}\n')

print(f'Single Agent 4: {S4_time}, 2: {S2_time}, 8: {S8_time}\n')

print(f'MA 4: {MA4_time}, 2: {MA2_time}, 8: {MA8_time}\n')

print(f'MAS 4: {MAS4_time}, 2: {MAS2_time}, 8: {MAS8_time}\n')

print(f'CC 4: {CC4_time}, 2: {CC2_time}, 8: {CC8_time}')


