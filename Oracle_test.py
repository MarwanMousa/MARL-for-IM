from environments.IM_div_env import InvManagementDiv
from ray import tune
import numpy as np
import pyomo.environ as pyo
from utils import check_connections, create_network, ensure_dir

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
    env = InvManagementDiv(configuration)
    return env

# Environment Configuration
num_nodes = 2
connections = {
    0: [1],
    1: [],
    }
check_connections(connections)
network = create_network(connections)
num_periods = 30
customer_demand = np.ones(num_periods) * 5
mu = 5
lower_upper = (1, 5)
init_inv = np.ones(num_nodes)*10
inv_target = np.ones(num_nodes) * 0
inv_max = np.ones(num_nodes) * 30
stock_cost = np.array([0.4, 0.4])
backlog_cost = np.array([0.8, 0.7])
delay = np.array([1, 2], dtype=np.int8)
standardise_state = True
standardise_actions = True
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

env_name = "InventoryManagementDiv"
tune.register_env(env_name, env_creator)

env_config = {
    "num_nodes": num_nodes,
    "connections": connections,
    "num_periods": num_periods,
    "init_inv": init_inv,
    "stock_cost": stock_cost,
    "backlog_cost": backlog_cost,
    "demand_dist": demand_distribution,
    "inv_target": inv_target,
    "inv_max": inv_max,
    "seed": SEED,
    "delay": delay,
    parameter: parameter_value,
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
test_env = InvManagementDiv(env_config)
LP_env = InvManagementDiv(DFO_CONFIG)

#%% Linear Programming Pyomo
path = 'LP_results/test/Oracle/'
num_tests = 1
test_seed = 420
np.random.seed(seed=test_seed)
LP_demand = LP_env.dist.rvs(size=(num_tests, (len(LP_env.retailers)), LP_env.num_periods), **LP_env.dist_param)
noisy_demand = False
noise_threshold = 40/100
if noisy_demand:
    for i in range(num_tests):
        for k in range(len(LP_env.retailers)):
            for j in range(num_periods):
                double_demand = np.random.uniform(0, 1)
                zero_demand = np.random.uniform(0, 1)
                if double_demand <= noise_threshold:
                    LP_demand[i, k, j] = 2 * LP_demand[i, k, j]
                if zero_demand <= noise_threshold:
                    LP_demand[i, k, j] = 0

inventory_list = []
backlog_list = []
lp_reward_list = []
customer_backlog_list = []
profit = np.zeros((num_tests, num_periods))

array_obs = np.zeros((num_nodes, 3, num_periods + 1))
array_actions = np.zeros((num_nodes, num_periods))
array_profit = np.zeros((num_nodes, num_periods))
array_profit_sum = np.zeros(num_periods)
array_demand = np.zeros((num_nodes, num_periods))
array_ship = np.zeros((num_nodes, num_periods))
array_acquisition = np.zeros((num_nodes, num_periods))
array_rewards = np.zeros(num_periods)

for j in range(num_tests):
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(num_periods)
    d = {}
    for i in range(1, num_periods+1):
        d[i] = LP_demand[j, :, i-1]

    i10 = init_inv[0]  # initial inventory 1
    i20 = init_inv[1]  # initial inventory 2

    SC1 = stock_cost[0]  # inventory holding cost
    SC2 = stock_cost[1]  # inventory holding cost

    BC1 = backlog_cost[0]  # shortage cost 1
    BC2 = backlog_cost[1]  # shortage cost 2

    I1 = inv_max[0]  # maximum inventory 1
    I2 = inv_max[1]  # maximum inventory 2

    O1 = inv_max[0]  # maximum order 1
    O2 = inv_max[1]  # maximum order 2

    # Price of goods at each node
    P1 = 2
    P2 = 3

    C1 = 1
    C2 = 2

    d1 = delay[0]
    d2 = delay[1]

    # create a block for a single time period
    def lotsizing_block_rule(b, t):
        # define the variables
        # Re-order Variables at each node
        b.x1 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.x2 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # Inventory at each node
        b.i1 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.i2 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # Initial Inventory at each time-step
        b.i10 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.i20 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # backlog
        b.bl1 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.bl2 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # Initial Backlog at each time-step
        b.bl10 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
        b.bl20 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)

        # Shipped goods/sales
        b.s1 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.s2 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # Acquisiton
        b.a1 = pyo.Var(domain=pyo.NonNegativeIntegers)
        b.a2 = pyo.Var(domain=pyo.NonNegativeIntegers)

        # define the constraints
        b.inventory1 = pyo.Constraint(expr=b.i1 == b.i10 + b.a1 - b.s1)
        b.inventory2 = pyo.Constraint(expr=b.i2 == b.i20 + b.a2 - b.s2)

        # Inventory constraints
        b.inventorymax1 = pyo.Constraint(expr=b.i1 <= I1)
        b.inventorymax2 = pyo.Constraint(expr=b.i2 <= I2)

        # Order constraints
        b.ordermax1 = pyo.Constraint(expr=b.x1 <= O1)
        b.ordermax2 = pyo.Constraint(expr=b.x2 <= O2)

        # Backlog constraints
        b.backlog1 = pyo.Constraint(expr=b.bl1 == b.bl10 - b.s1 + b.x2)
        b.backlog3 = pyo.Constraint(expr=b.bl2 == b.bl20 - b.s2 + d[t][0])


        # Shipping Constraints
        b.ship1_inv = pyo.Constraint(expr=b.s1 <= b.i10 + b.a1)
        b.ship1_bl = pyo.Constraint(expr=b.s1 <= b.bl10 + b.x2)

        b.ship3_inv = pyo.Constraint(expr=b.s2 <= b.i20 + b.a2)
        b.ship3_bl = pyo.Constraint(expr=b.s2 <= b.bl20 + d[t][0])


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


    def bl1_linking_rule(m, t):
        if t == m.T.first():
            return m.lsb[t].bl10 == 0
        return m.lsb[t].bl10 == m.lsb[t-1].bl1


    def bl2_linking_rule(m, t):
        if t == m.T.first():
            return m.lsb[t].bl20 == 0
        return m.lsb[t].bl20 == m.lsb[t-1].bl2



    def a1_linking_rule(m, t):
        if t-d1 < 1:
            return m.lsb[t].a1 == 0
        return m.lsb[t].a1 == m.lsb[t-d1].x1

    def a2_linking_rule(m, t):
        if t-d2 < 1:
            return m.lsb[t].a2 == 0
        return m.lsb[t].a2 == m.lsb[t-d2].s1


    model.i_linking1 = pyo.Constraint(model.T, rule=i1_linking_rule)
    model.i_linking2 = pyo.Constraint(model.T, rule=i2_linking_rule)

    model.bl_linking1 = pyo.Constraint(model.T, rule=bl1_linking_rule)
    model.bl_linking2 = pyo.Constraint(model.T, rule=bl2_linking_rule)


    model.a_linking1 = pyo.Constraint(model.T, rule=a1_linking_rule)
    model.a_linking2 = pyo.Constraint(model.T, rule=a2_linking_rule)

    # construct the objective function over all the blocks
    def obj_rule(m):
        # Sum of Profit at each state at each timeperiod
        return sum(m.lsb[t].s1*P1 - m.lsb[t].x1*C1 - m.lsb[t].i1*SC1 - m.lsb[t].bl1*BC1
                   + m.lsb[t].s2*P2 - m.lsb[t].x2*C2 - m.lsb[t].i2*SC2 - m.lsb[t].bl2*BC2
                   for t in m.T)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    ### solve the problem
    solver = pyo.SolverFactory('gurobi', solver_io='python')
    results = solver.solve(model)

    s = LP_env.reset(customer_demand=LP_demand[j, :])
    lp_reward = 0
    total_inventory = 0
    total_backlog = 0
    customer_backlog = 0
    done = False
    t = 1
    if j == 0:
        array_obs[:, :, 0] = s
    while not done:
        lp_action = [pyo.value(model.lsb[t].x1), pyo.value(model.lsb[t].x2)]
        s, r, done, info = LP_env.step(lp_action)
        lp_reward += r
        total_inventory += sum(s[:, 0])
        total_backlog += sum(s[:, 1])
        customer_backlog += s[0, 1]
        profit[j, t - 1] = r
        if j == 0:
            array_obs[:, :, t] = s
            array_actions[:, t - 1] = lp_action
            array_actions[:, t - 1] = np.maximum(array_actions[:, t - 1], np.zeros(num_nodes))
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


ensure_dir(path)
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
fig, axs = plt.subplots(3, num_nodes, figsize=(18, 9), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.06, wspace=.16)

axs = axs.ravel()

for i in range(num_nodes):
    axs[i].plot(array_obs[i, 0, :], label='Inventory', lw=2)
    axs[i].plot(array_obs[i, 1, :], label='Backlog', color='tab:red', lw=2)
    title = 'Node ' + str(i)
    axs[i].set_title(title)
    axs[i].set_xlim(0, num_periods)
    axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i].legend()
        axs[i].set_ylabel('Products')

    axs[i + num_nodes].plot(np.arange(0, num_periods), array_actions[i, :], label='Replenishment order', color='k', lw=2)
    axs[i + num_nodes].plot(np.arange(0, num_periods), array_demand[i, :], label='Demand', color='tab:orange', lw=2)
    axs[i + num_nodes].set_xlim(0, num_periods)
    axs[i + num_nodes].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i + num_nodes].legend()
        axs[i + num_nodes].set_ylabel('Products')

    axs[i + num_nodes*2].plot(array_profit[i, :], label='Periodic profit', color='tab:green', lw=2)
    axs[i + num_nodes*2].plot(np.cumsum(array_profit[i, :]), label='Cumulative profit', color='salmon', lw=2)
    axs[i + num_nodes*2].plot([0, num_periods], [0, 0], color='k')
    axs[i + num_nodes*2].set_xlabel('Period')
    axs[i + num_nodes*2].set_xlim(0, num_periods)
    if i == 0:
        axs[i + num_nodes * 2].legend()
        axs[i + num_nodes * 2].set_ylabel('Profit')


test_name = path + '/test_rollout.png'
plt.savefig(test_name, dpi=200)
plt.show()

# Rewards plots
fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
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