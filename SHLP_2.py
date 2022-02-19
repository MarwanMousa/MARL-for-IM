from environments.IM_env import InvManagement
from ray import tune
import numpy as np
import pyomo.environ as pyo
import time
import matplotlib.pyplot as plt
from matplotlib import rc
#%% Environment Configuration

# Define plot settings
rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 13})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["figure.dpi"] = 200

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

# Environment creator function for environment registration
def env_creator(configuration):
    env = InvManagement(configuration)
    return env

# Environment Configuration
num_stages = 2
num_periods = 30
customer_demand = np.ones(num_periods) * 5
mu = 5
lower_upper = (1, 5)
init_inv = np.ones(num_stages)*10
inv_target = np.ones(num_stages) * 0
inv_max = np.ones(num_stages) * 30
price = np.array([3, 2, 1])
stock_cost = np.array([0.5, 0.2])
backlog_cost = np.array([0.6, 0.9])
delay = np.array([3, 1], dtype=np.int8)
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
# Test environment
test_env = InvManagement(env_config)
LP_env = InvManagement(DFO_CONFIG)

#%% Linear Programming Pyomo
num_tests = 200
test_seed = 420
np.random.seed(seed=test_seed)
LP_demand = LP_env.dist.rvs(size=(num_tests, LP_env.num_periods), **LP_env.dist_param)

# Initial Inventory
i10 = init_inv[0]
i20 = init_inv[1]

# Initial Backlog
bl10 = 0
bl20 = 0

# Initial Acquisition
a10 = 0
a11 = 0
a12 = 0
a20 = 0


# Inventory Holding Cost
SC1 = stock_cost[0]
SC2 = stock_cost[1]

# Backlog Cost
BC1 = backlog_cost[0]
BC2 = backlog_cost[1]

# Maximum Inventory
I1 = inv_max[0]
I2 = inv_max[1]

# Order Max
O1 = inv_max[0]
O2 = inv_max[1]

# Price of goods at each stage
P1 = price[0]
P2 = price[1]
P3 = price[2]

# Shipment Delay
d1 = delay[0]
d2 = delay[1]

# link the inventory variables between blocks
def i1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i10 == i10
    return m.lsb[t].i10 == m.lsb[t-1].i1

def i2_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i20 == i20
    return m.lsb[t].i20 == m.lsb[t-1].i2


# link the backlog variables between blocks
def bl1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl10 == bl10
    return m.lsb[t].bl10 == m.lsb[t-1].bl1

def bl2_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl20 == bl20
    return m.lsb[t].bl20 == m.lsb[t-1].bl2


# link the acquisition variables between blocks
def a1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a1 == a10
    # This condition is configuration specific
    if t - 1 == m.T.first() and not t - d1 < 0:
        return m.lsb[t].a1 == a11
    if t - 2 == m.T.first() and not t - d1 < 0:
        return m.lsb[t].a1 == a12
    if t - d1 < m.T.first() and t - d1 < 0:
        return m.lsb[t].a1 == 0
    return m.lsb[t].a1 == m.lsb[t-d1].s2

def a2_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a2 == a20
    if t - d2 < m.T.first() and t - d2 < 0:
        return m.lsb[t].a2 == 0
    # This condition is configuration specific
    return m.lsb[t].a2 == m.lsb[t-d2].x2


def d_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].demand == d
    return m.lsb[t].demand == mu


# create a block for a single time period
def lotsizing_block_rule(b, t):
    # define the variables
    # Reorder Variables at each stage
    b.x1 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.x2 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Inventory at each stage
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

    # Customer demand
    b.demand = pyo.Var(domain=pyo.NonNegativeIntegers)

    # define the constraints
    b.inventory1 = pyo.Constraint(expr=b.i1 == b.i10 + b.a1 - b.s1)
    b.inventory2 = pyo.Constraint(expr=b.i2 == b.i20 + b.a2 - b.s2)

    # Inventory constrainss
    b.inventorymax1 = pyo.Constraint(expr=b.i1 <= I1)
    b.inventorymax2 = pyo.Constraint(expr=b.i2 <= I2)

    # Order constraints
    b.ordermax1 = pyo.Constraint(expr=b.x1 <= O1)
    b.ordermax2 = pyo.Constraint(expr=b.x2 <= O2)

    # backlog constrains
    b.backlog1 = pyo.Constraint(expr=b.bl1 == b.bl10 - b.s1 + b.demand)
    b.backlog2 = pyo.Constraint(expr=b.bl2 == b.bl20 - b.s2 + b.x1)

    # Sales Constraints
    b.ship11 = pyo.Constraint(expr=b.s1 <= b.i10 + b.a1)
    b.ship12 = pyo.Constraint(expr=b.s1 <= b.bl10 + b.demand)
    b.ship21 = pyo.Constraint(expr=b.s2 <= b.i20 + b.a2)
    b.ship22 = pyo.Constraint(expr=b.s2 <= b.bl20 + b.x1)

# construct the objective function over all the blocks
def obj_rule(m):
    # Sum of Profit at each state at each timeperiod
    # Profit per stage per time period: sales - re-order cost - backlog cost  - inventory cost
    return sum((m.lsb[t].s1*P1 - m.lsb[t].x1*P2 - m.lsb[t].i1*SC1 - m.lsb[t].bl1*BC1
               + m.lsb[t].s2*P2 - m.lsb[t].x2*P3 - m.lsb[t].i2*SC2 - m.lsb[t].bl2*BC2)
               for t in m.T)

inventory_list = []
backlog_list = []
lp_reward_list = []
customer_backlog_list = []
profit = np.zeros((num_tests, num_periods))
start_time = time.time()

array_obs = np.zeros((num_stages, 3, num_periods + 1))
array_actions = np.zeros((num_stages, num_periods))
array_profit = np.zeros((num_stages, num_periods))
array_profit_sum = np.zeros(num_periods)
array_demand = np.zeros((num_stages, num_periods))
array_ship = np.zeros((num_stages, num_periods))
array_acquisition = np.zeros((num_stages, num_periods))
array_rewards = np.zeros(num_periods)

for j in range(num_tests):
    # Initial Inventory
    i10 = init_inv[0]
    i20 = init_inv[1]

    # Initial Backlog
    bl10 = 0
    bl20 = 0

    # Initial Acquisition
    a10 = 0
    a11 = 0
    a12 = 0
    a20 = 0

    SHLP_actions = np.zeros((num_periods, num_stages))
    SHLP_shipment = np.zeros((num_periods, num_stages))
    SHLP_inventory = np.zeros((num_periods, num_stages))
    SHLP_backlog = np.zeros((num_periods, num_stages))
    SHLP_acquisition = np.zeros((num_periods, num_stages))
    SHLP_sales = np.zeros(num_periods)
    for i in range(num_periods):

        # Get initial acquisition at each stage
        if i - d1 < 0:
            a10 = 0
            a11 = 0
            a12 = 0
        else:
            a10 = SHLP_shipment[i - d1, 0]
            a11 = SHLP_shipment[i - d1 + 1, 0]
            a12 = SHLP_shipment[i - d1 + 2, 0]

        if i - d2 < 0:
            a20 = 0
            a21 = 0
        else:
            a20 = SHLP_shipment[i - d2, 1]

        # Get real customer demand at current time-step
        d = LP_demand[j, i]
        # Create model over the horizon i:num_periods (shrinking horizon)
        model = pyo.ConcreteModel()
        model.T = pyo.RangeSet(i, num_periods-1)
        # Link all time-period blocks
        model.lsb = pyo.Block(model.T, rule=lotsizing_block_rule)

        # Inventory linking constraints
        model.i_linking1 = pyo.Constraint(model.T, rule=i1_linking_rule)
        model.i_linking2 = pyo.Constraint(model.T, rule=i2_linking_rule)

        # Backlog linking constraints
        model.bl_linking1 = pyo.Constraint(model.T, rule=bl1_linking_rule)
        model.bl_linking2 = pyo.Constraint(model.T, rule=bl2_linking_rule)

        # Acquisition linking constraints
        model.a_linking1 = pyo.Constraint(model.T, rule=a1_linking_rule)
        model.a_linking2 = pyo.Constraint(model.T, rule=a2_linking_rule)

        # Customer demand linking constraints
        model.d_linking = pyo.Constraint(model.T, rule=d_linking_rule)

        # Model obbjective function
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

        ### solve the problem
        solver = pyo.SolverFactory('gurobi', solver_io='python')
        results = solver.solve(model)

        # Log actions for testing in Gym environment
        SHLP_actions[i, :] = [pyo.value(model.lsb[i].x1), pyo.value(model.lsb[i].x2)]

        # Log shipments for updating the acquisition at next time-step
        SHLP_shipment[i, :] = [pyo.value(model.lsb[i].s2), pyo.value(model.lsb[i].x2)]


        SHLP_inventory[i, :] = [pyo.value(model.lsb[i].i10), pyo.value(model.lsb[i].i20)]


        SHLP_backlog[i, :] = [pyo.value(model.lsb[i].bl10), pyo.value(model.lsb[i].bl20)]

        SHLP_sales[i] = pyo.value(model.lsb[i].s1)

        SHLP_acquisition[i, :] = [pyo.value(model.lsb[i].a1), pyo.value(model.lsb[i].a2)]

        # Update initial inventory for next iteration
        i10 = pyo.value(model.lsb[i].i1)
        i20 = pyo.value(model.lsb[i].i2)

        # Update initial backlog for next iteration
        bl10 = pyo.value(model.lsb[i].bl1)
        bl20 = pyo.value(model.lsb[i].bl2)


    # Test Gym environment
    s = LP_env.reset(customer_demand=LP_demand[j, :])
    lp_reward = 0
    total_inventory = 0
    total_backlog = 0
    customer_backlog = 0
    done = False
    t = 0

    if j == 0:
        array_obs[:, :, 0] = s

    while not done:
        lp_action = SHLP_actions[t, :]
        s, r, done, info = LP_env.step(lp_action)
        profit[j, t] = r
        total_inventory += sum(s[:, 0])
        total_backlog += sum(s[:, 1])
        customer_backlog += s[0, 1]
        lp_reward += r
        if j == 0:
            array_obs[:, :, t + 1] = s
            array_actions[:, t] = lp_action
            array_actions[:, t] = np.maximum(array_actions[:, t], np.zeros(num_stages))
            array_rewards[t] = r
            array_profit[:, t] = info['profit']
            array_profit_sum[t] = np.sum(info['profit'])
            array_demand[:, t] = info['demand']
            array_ship[:, t] = info['ship']
            array_acquisition[:, t] = info['acquisition']
        t += 1

    lp_reward_list.append(lp_reward)
    inventory_list.append(total_inventory)
    backlog_list.append(total_backlog)
    customer_backlog_list.append(customer_backlog)
    if j % 10 == 0:
        print(f'reward at {j} is {lp_reward}')

shlp_time = time.time() - start_time
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

path = 'LP_results/two_stage/SHLP/'
np.save(path+'reward_mean.npy', lp_reward_mean)
np.save(path+'reward_std.npy', lp_reward_std)
np.save(path+'inventory_mean.npy', inventory_level_mean)
np.save(path+'inventory_std.npy', inventory_level_std)
np.save(path+'backlog_mean.npy', backlog_level_mean)
np.save(path+'backlog_std.npy', backlog_level_std)
np.save(path+'customer_backlog_mean', customer_backlog_mean)
np.save(path+'customer_backlog_std', customer_backlog_std)
np.save(path+'profit', profit)
np.save(path+'time', shlp_time)

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

    axs[i+num_stages].plot(np.arange(0, num_periods), array_actions[i, :], label='Replenishment order', color='k', lw=2)
    axs[i + num_stages].plot(np.arange(0, num_periods), array_demand[i, :], label='Demand', color='tab:orange', lw=2)
    axs[i+num_stages].set_xlim(0, num_periods)
    axs[i+num_stages].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i+num_stages].legend()
        axs[i + num_stages].set_ylabel('Products')

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
