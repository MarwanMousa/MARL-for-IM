from environments.IM_env import InvManagement
from ray import tune
import numpy as np
import pyomo.environ as pyo

import matplotlib.pyplot as plt
from matplotlib import rc
#%% Environment Configuration

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

# Define plot settings
rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 13})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["figure.dpi"] = 200

# Environment creator function for environment registration
def env_creator(configuration):
    env = InvManagement(configuration)
    return env

# Environment Configuration
num_stages = 4
num_periods = 30
customer_demand = np.ones(num_periods) * 5
mu = 5
lower_upper = (1, 5)
init_inv = np.ones(num_stages)*10
inv_target = np.ones(num_stages) * 0
inv_max = np.ones(num_stages) * 30
price = np.array([5, 4, 3, 2, 1])
stock_cost = np.array([0.35, 0.3, 0.4, 0.2])
backlog_cost = np.array([0.5, 0.7, 0.6, 0.9])
delay = np.array([1, 2, 3, 1], dtype=np.int8)
standardise_state = True
standardise_actions = True
a = -1
b = 1
time_dependency = False
use_lstm = False
prev_actions = False
prev_demand = False
prev_length = 1

demand_distribution = "poisson"

if demand_distribution == "custom":
    parameter = "customer_demand"
    parameter_value = customer_demand
elif demand_distribution == 'poisson':
    parameter = "mu"
    parameter_value = mu
elif demand_distribution == "uniform":
    parameter = "lower_upper"
    parameter_value = lower_upper

env_name = "InventoryManagement"
tune.register_env(env_name, env_creator)

env_config = {
    "num_stages": num_stages,
    "num_periods": num_periods,
    "init_inv": init_inv,
    "price": price,
    "stock_cost": stock_cost,
    "backlog_cost": backlog_cost,
    "demand_dist": demand_distribution,
    "inv_target": inv_target,
    "inv_max": inv_max,
    "seed": SEED,
    "delay": delay,
    parameter: parameter_value,
    "standardise_state": standardise_state,
    "standardise_actions": standardise_actions,
    "a": a,
    "b": b,
    "time_dependency": time_dependency,
    "prev_demand": prev_demand,
    "prev_actions": prev_actions,
    "prev_length": prev_length,
}
CONFIG = env_config.copy()
# Configuration for the base-restock policy with DFO that has no state-action standardisation
DFO_CONFIG = env_config.copy()
DFO_CONFIG["standardise_state"] = False
DFO_CONFIG["standardise_actions"] = False
DFO_CONFIG["time_dependency"] = False
DFO_CONFIG["prev_actions"] = False
DFO_CONFIG["prev_demand"] = False
# Test environment
test_env = InvManagement(env_config)
DFO_env = InvManagement(DFO_CONFIG)

#%% Linear Programming Pyomo
num_tests = 1000
test_seed = 420
np.random.seed(seed=test_seed)
LP_demand = DFO_env.dist.rvs(size=(num_tests, DFO_env.num_periods), **DFO_env.dist_param)
noisy_demand = True
noise_threshold = 40/100
if noisy_demand:
    for i in range(num_tests):
        for j in range(num_periods):
            double_demand = np.random.uniform(0, 1)
            zero_demand = np.random.uniform(0, 1)
            if double_demand <= noise_threshold:
                LP_demand[i, j] = 2 * LP_demand[i, j]
            if zero_demand <= noise_threshold:
                LP_demand[i, j] = 0

inventory_list = []
backlog_list = []
lp_reward_list = []
customer_backlog_list = []
profit = np.zeros((num_tests, num_periods))

array_obs = np.zeros((num_stages, 3, num_periods + 1))
array_actions = np.zeros((num_stages, num_periods))
array_profit = np.zeros((num_stages, num_periods))
array_profit_sum = np.zeros(num_periods)
array_demand = np.zeros((num_stages, num_periods))
array_ship = np.zeros((num_stages, num_periods))
array_acquisition = np.zeros((num_stages, num_periods))
array_rewards = np.zeros(num_periods)

for j in range(num_tests):
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(num_periods)
    d = {}
    for i in range(1, num_periods+1):
        d[i] = LP_demand[j, i-1]

    i10 = init_inv[0]  # initial inventory 1
    i20 = init_inv[1]  # initial inventory 2
    i30 = init_inv[2]  # initial inventory 3
    i40 = init_inv[3]  # initial inventory 4

    SC1 = stock_cost[0]  # inventory holding cost
    SC2 = stock_cost[1]  # inventory holding cost
    SC3 = stock_cost[2]  # inventory holding cost
    SC4 = stock_cost[3]  # inventory holding cost

    BC1 = backlog_cost[0]  # shortage cost 1
    BC2 = backlog_cost[1]  # shortage cost 2
    BC3 = backlog_cost[2]  # shortage cost 3
    BC4 = backlog_cost[3]  # shortage cost 4

    I1 = inv_max[0]  # maximum inventory 1
    I2 = inv_max[1]  # maximum inventory 2
    I3 = inv_max[2]  # maximum inventory 3
    I4 = inv_max[3]  # maximum inventory 4

    O1 = inv_max[0]  # maximum order 1
    O2 = inv_max[1]  # maximum order 2
    O3 = inv_max[2]  # maximum order 3
    O4 = inv_max[3]  # maximum order 4

    # Price of goods at each stage
    P1 = price[0]
    P2 = price[1]
    P3 = price[2]
    P4 = price[3]
    P5 = price[4]

    d1 = delay[0]
    d2 = delay[1]
    d3 = delay[2]
    d4 = delay[3]

    # create a block for a single time period
    def lotsizing_block_rule(b, t):
        # define the variables
        # Reorder Variables at each stage
        b.x1 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.x2 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.x3 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.x4 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # Inventory at each stage
        b.i1 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.i2 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.i3 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.i4 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # Initial Inventory at each time-step
        b.i10 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.i20 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.i30 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.i40 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # backlog
        b.bl1 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.bl2 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.bl3 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.bl4 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # Initial Backlog at each time-step
        b.bl10 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
        b.bl20 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
        b.bl30 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
        b.bl40 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)

        # Shipped goods/sales
        b.s1 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.s2 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.s3 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.s4 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # Acquisiton
        b.a1 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.a2 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.a3 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.a4 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # define the constraints
        b.inventory1 = pyo.Constraint(expr=b.i1 == b.i10 + b.a1 - b.s1)
        b.inventory2 = pyo.Constraint(expr=b.i2 == b.i20 + b.a2 - b.s2)
        b.inventory3 = pyo.Constraint(expr=b.i3 == b.i30 + b.a3 - b.s3)
        b.inventory4 = pyo.Constraint(expr=b.i4 == b.i40 + b.a4 - b.s4)

        # Inventory constraints
        b.inventorymax1 = pyo.Constraint(expr=b.i1 <= I1)
        b.inventorymax2 = pyo.Constraint(expr=b.i2 <= I2)
        b.inventorymax3 = pyo.Constraint(expr=b.i3 <= I3)
        b.inventorymax4 = pyo.Constraint(expr=b.i4 <= I4)

        # Order constraints
        b.ordermax1 = pyo.Constraint(expr=b.x1 <= O1)
        b.ordermax2 = pyo.Constraint(expr=b.x2 <= O2)
        b.ordermax3 = pyo.Constraint(expr=b.x3 <= O3)
        b.ordermax4 = pyo.Constraint(expr=b.x4 <= O4)

        # backlog constrains
        b.backlog1 = pyo.Constraint(expr=b.bl1 == b.bl10 - b.s1 + d[t])
        b.backlog2 = pyo.Constraint(expr=b.bl2 == b.bl20 - b.s2 + b.x1)
        b.backlog3 = pyo.Constraint(expr=b.bl3 == b.bl30 - b.s3 + b.x2)
        b.backlog4 = pyo.Constraint(expr=b.bl4 == b.bl40 - b.s4 + b.x3)

        #
        b.ship11 = pyo.Constraint(expr=b.s1 <= b.i10 + b.a1)
        b.ship12 = pyo.Constraint(expr=b.s1 <= b.bl10 + d[t])
        b.ship21 = pyo.Constraint(expr=b.s2 <= b.i20 + b.a2)
        b.ship22 = pyo.Constraint(expr=b.s2 <= b.bl20 + b.x1)
        b.ship31 = pyo.Constraint(expr=b.s3 <= b.i30 + b.a3)
        b.ship32 = pyo.Constraint(expr=b.s3 <= b.bl30 + b.x2)
        b.ship41 = pyo.Constraint(expr=b.s4 <= b.i40 + b.a4)
        b.ship42 = pyo.Constraint(expr=b.s4 <= b.bl40 + b.x3)

    model.lsb = pyo.Block(model.T, rule=lotsizing_block_rule)

    # link the inventory variables between blocks
    def i1_linking_rule(m, t):
        if t == m.T.first():
            return m.lsb[t].i10 == i10
        return m.lsb[t].i10 == m.lsb[t-1].i1


    def i2_linking_rule(m, t):
        if t == m.T.first():
            return m.lsb[t].i20 == i20
        return m.lsb[t].i20 == m.lsb[t-1].i2


    def i3_linking_rule(m, t):
        if t == m.T.first():
            return m.lsb[t].i30 == i30
        return m.lsb[t].i30 == m.lsb[t-1].i3


    def i4_linking_rule(m, t):
        if t == m.T.first():
            return m.lsb[t].i40 == i40
        return m.lsb[t].i40 == m.lsb[t-1].i4

    def bl1_linking_rule(m, t):
        if t == m.T.first():
            return m.lsb[t].bl10 == 0
        return m.lsb[t].bl10 == m.lsb[t-1].bl1

    def bl2_linking_rule(m, t):
        if t == m.T.first():
            return m.lsb[t].bl20 == 0
        return m.lsb[t].bl20 == m.lsb[t-1].bl2

    def bl3_linking_rule(m, t):
        if t == m.T.first():
            return m.lsb[t].bl30 == 0
        return m.lsb[t].bl30 == m.lsb[t-1].bl3

    def bl4_linking_rule(m, t):
        if t == m.T.first():
            return m.lsb[t].bl40 == 0
        return m.lsb[t].bl40 == m.lsb[t-1].bl4


    def a1_linking_rule(m, t):
        if t-d1 < 1:
            return m.lsb[t].a1 == 0
        return m.lsb[t].a1 == m.lsb[t-d1].s2

    def a2_linking_rule(m, t):
        if t-d2 < 1:
            return m.lsb[t].a2 == 0
        return m.lsb[t].a2 == m.lsb[t-d2].s3

    def a3_linking_rule(m, t):
        if t-d3 < 1:
            return m.lsb[t].a3 == 0
        return m.lsb[t].a3 == m.lsb[t-d3].s4

    def a4_linking_rule(m, t):
        if t-d4 < 1:
            return m.lsb[t].a4 == 0
        return m.lsb[t].a4 == m.lsb[t-d4].x4

    model.i_linking1 = pyo.Constraint(model.T, rule=i1_linking_rule)
    model.i_linking2 = pyo.Constraint(model.T, rule=i2_linking_rule)
    model.i_linking3 = pyo.Constraint(model.T, rule=i3_linking_rule)
    model.i_linking4 = pyo.Constraint(model.T, rule=i4_linking_rule)

    model.bl_linking1 = pyo.Constraint(model.T, rule=bl1_linking_rule)
    model.bl_linking2 = pyo.Constraint(model.T, rule=bl2_linking_rule)
    model.bl_linking3 = pyo.Constraint(model.T, rule=bl3_linking_rule)
    model.bl_linking4 = pyo.Constraint(model.T, rule=bl4_linking_rule)

    model.a_linking1 = pyo.Constraint(model.T, rule=a1_linking_rule)
    model.a_linking2 = pyo.Constraint(model.T, rule=a2_linking_rule)
    model.a_linking3 = pyo.Constraint(model.T, rule=a3_linking_rule)
    model.a_linking4 = pyo.Constraint(model.T, rule=a4_linking_rule)

    # construct the objective function over all the blocks
    def obj_rule(m):
        # Sum of Profit at each state at each timeperiod
        return sum(m.lsb[t].s1*P1 - m.lsb[t].x1*P2 - m.lsb[t].i1*SC1 - m.lsb[t].bl1*BC1
                   + m.lsb[t].s2*P2 - m.lsb[t].x2*P3 - m.lsb[t].i2*SC2 - m.lsb[t].bl2*BC2
                   + m.lsb[t].s3*P3 - m.lsb[t].x3*P4 - m.lsb[t].i3*SC3 - m.lsb[t].bl3*BC3
                   + m.lsb[t].s4*P4 - m.lsb[t].x4*P5 - m.lsb[t].i4*SC4 - m.lsb[t].bl4*BC4
                   for t in m.T)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    ### solve the problem
    solver = pyo.SolverFactory('gurobi', solver_io='python')
    results = solver.solve(model)

    s = DFO_env.reset(customer_demand=LP_demand[j, :])
    lp_reward = 0
    total_inventory = 0
    total_backlog = 0
    customer_backlog = 0
    done = False
    t = 1
    if j == 0:
        array_obs[:, :, 0] = s
    while not done:
        lp_action = [pyo.value(model.lsb[t].x1), pyo.value(model.lsb[t].x2), pyo.value(model.lsb[t].x3), pyo.value(model.lsb[t].x4)]
        s, r, done, info = DFO_env.step(lp_action)
        lp_reward += r
        total_inventory += sum(s[:, 0])
        total_backlog += sum(s[:, 1])
        customer_backlog += s[0, 1]
        profit[j, t - 1] = r
        if j == 0:
            array_obs[:, :, t] = s
            array_actions[:, t - 1] = lp_action
            array_actions[:, t - 1] = np.maximum(array_actions[:, t - 1], np.zeros(num_stages))
            array_rewards[t - 1] = r
            array_profit[:, t - 1] = info['profit']
            array_profit_sum[t - 1] = np.sum(info['profit'])
            array_demand[:, t - 1] = info['demand']
            array_ship[:, t - 1] = info['ship']
            array_acquisition[:, t - 1] = info['acquisition']
        t += 1

    lp_reward_list.append(lp_reward)
    inventory_list.append(total_inventory)
    backlog_list.append(total_backlog)
    customer_backlog_list.append(customer_backlog)
    if j % 10 == 0:
        print(f'reward at {j} is {lp_reward}')

lp_reward_mean = np.mean(lp_reward_list)
lp_reward_std = np.std(lp_reward_list)
inventory_level_mean = np.mean(inventory_list)
inventory_level_std = np.std(inventory_list)
backlog_level_mean = np.mean(backlog_list)
backlog_level_std = np.std(backlog_list)
customer_backlog_mean = np.mean(customer_backlog_list)
customer_backlog_std = np.std(customer_backlog_list)

print(f'Mean reward is: {lp_reward_mean} with std: {lp_reward_std}')
print(f'Mean inventory level is: {inventory_level_mean} with std: {inventory_level_std}')
print(f'Mean backlog level is: {backlog_level_mean} with std: {backlog_level_std}')
print(f'Mean customer backlog level is: {customer_backlog_mean } with std: {customer_backlog_std}')

path = 'LP_results/four_stage_noise_40/Oracle/'

np.save(path+'reward_mean.npy', lp_reward_mean)
np.save(path+'reward_std.npy', lp_reward_std)
np.save(path+'inventory_mean.npy', inventory_level_mean)
np.save(path+'inventory_std.npy', inventory_level_std)
np.save(path+'backlog_mean.npy', backlog_level_mean)
np.save(path+'backlog_std.npy', backlog_level_std)
np.save(path+'customer_backlog_mean', customer_backlog_mean)
np.save(path+'customer_backlog_std', customer_backlog_std)
np.save(path+'profit', profit)


#%% Test rollout plots
fig, axs = plt.subplots(3, num_stages, figsize=(18, 9), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.06, wspace=.16)

axs = axs.ravel()

for i in range(num_stages):
    axs[i].plot(array_obs[i, 0, :], label='Inventory', lw=2)
    axs[i].plot(array_obs[i, 1, :], label='Backlog', color='tab:red', lw=2)
    title = 'Stage ' + str(i+1)
    axs[i].set_title(title)
    axs[i].set_xlim(0, num_periods)
    axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i].legend()
        axs[i].set_ylabel('Products')

    #axs[i+num_stages].plot(np.arange(0, num_periods), array_acquisition[i, :], label='Acquisition')
    axs[i+num_stages].plot(np.arange(0, num_periods), array_actions[i, :], label='Replenishment order', color='k', lw=2)
    axs[i + num_stages].plot(np.arange(0, num_periods), array_demand[i, :], label='Demand', color='tab:orange', lw=2)
    axs[i+num_stages].set_xlim(0, num_periods)
    axs[i+num_stages].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i+num_stages].legend()
        axs[i + num_stages].set_ylabel('Products')

    #axs[i+num_stages*2].plot(np.arange(0, num_periods), array_demand[i, :], label='demand')
    #axs[i+num_stages*2].plot(np.arange(0, num_periods), array_ship[i, :], label='shipment')
    #axs[i+num_stages*2].set_xlim(0, num_periods)
    #axs[i+num_stages*2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    #if i == 0:
    #    axs[i+num_stages*2].legend()
    #    axs[i + num_stages * 2].set_ylabel('Products')

    axs[i+num_stages*2].plot(array_profit[i, :], label='Periodic profit', color='tab:green', lw=2)
    axs[i+num_stages*2].plot(np.cumsum(array_profit[i, :]), label='Cumulative profit', color='salmon', lw=2)
    axs[i+num_stages*2].plot([0, num_periods], [0, 0], color='k')
    axs[i+num_stages*2].set_xlabel('Period')
    axs[i+num_stages*2].set_xlim(0, num_periods)
    if i == 0:
        axs[i + num_stages * 2].legend()
        axs[i + num_stages * 2].set_ylabel('Profit')


test_name = path + '/test_rollout.png'
plt.savefig(test_name, dpi=200)
plt.show()

# Rewards plots
fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
#ax.plot(array_rewards, label='periodic reward')
#ax.plot(np.cumsum(array_rewards), label='cumulative reward')
ax.plot(array_profit_sum, label='periodic profit')
ax.plot(np.cumsum(array_profit_sum), label='cumulative profit')
ax.plot([0, num_periods], [0, 0], color='k')
ax.set_title('Aggregate Rewards')
ax.set_xlabel('Period')
ax.set_ylabel('Rewards/profit')
ax.legend()
ax.set_xlim(0, num_periods)


test_rewards_name = path + '/test_rollout_rewards.png'
plt.savefig(test_rewards_name, dpi=200)
plt.show()




