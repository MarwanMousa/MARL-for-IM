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
# Test environment
test_env = InvManagement(env_config)
LP_env = InvManagement(DFO_CONFIG)

#%% Linear Programming Pyomo
num_tests = 1000
LP_demand = LP_env.dist.rvs(size=(num_tests, LP_env.num_periods), **LP_env.dist_param)
noisy_demand = True
noise_threshold = 40/100
noisy_delay = False
noisy_delay_threshold = 50/100
if noisy_demand:
    for i in range(num_tests):
        for j in range(num_periods):
            double_demand = np.random.uniform(0, 1)
            zero_demand = np.random.uniform(0, 1)
            if double_demand <= noise_threshold:
                LP_demand[i, j] = 2 * LP_demand[i, j]
            if zero_demand <= noise_threshold:
                LP_demand[i, j] = 0

# Initial Inventory
i10 = init_inv[0]
i20 = init_inv[1]
i30 = init_inv[2]
i40 = init_inv[3]

# Initial Backlog
bl10 = 0
bl20 = 0
bl30 = 0
bl40 = 0

# Initial Acquisition
a10 = 0
a20 = 0
a21 = 0
a30 = 0
a31 = 0
a32 = 0
a40 = 0

# Inventory Holding Cost
SC1 = stock_cost[0]
SC2 = stock_cost[1]
SC3 = stock_cost[2]
SC4 = stock_cost[3]

# Backlog Cost
BC1 = backlog_cost[0]
BC2 = backlog_cost[1]
BC3 = backlog_cost[2]
BC4 = backlog_cost[3]

# Maximum Inventory
I = inv_max[0]

# Order Max
O = inv_max[0]

# Price of goods at each stage
P1 = price[0]
P2 = price[1]
P3 = price[2]
P4 = price[3]
P5 = price[4]

# Shipment Delay
d1 = delay[0]
d2 = delay[1]
d3 = delay[2]
d4 = delay[3]

# link the inventory variables between blocks
def i1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i0 == i10
    return m.lsb[t].i0 == m.lsb[t-1].i

def i2_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i0 == i20
    return m.lsb[t].i0 == m.lsb[t-1].i

def i3_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i0 == i30
    return m.lsb[t].i0 == m.lsb[t-1].i

def i4_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i0 == i40
    return m.lsb[t].i0 == m.lsb[t-1].i

# link the backlog variables between blocks
def bl1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl0 == bl10
    return m.lsb[t].bl0 == m.lsb[t-1].bl

def bl2_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl0 == bl20
    return m.lsb[t].bl0 == m.lsb[t-1].bl

def bl3_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl0 == bl30
    return m.lsb[t].bl0 == m.lsb[t-1].bl

def bl4_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl0 == bl40
    return m.lsb[t].bl0 == m.lsb[t-1].bl


# link the acquisition variables between blocks
def a1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a == a10
    if t - d1 < m.T.first():
        return m.lsb[t].a == 0
    return m.lsb[t].a == m.lsb[t-d1].r

def a2_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a == a20
    # This condition is configuration specific
    if t - 1 == m.T.first() and not t - d2 < 0:
        return m.lsb[t].a == a21
    if t - d2 < m.T.first() and t - d2 < 0:
        return m.lsb[t].a == 0
    return m.lsb[t].a == m.lsb[t-d2].r

def a3_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a == a30
    # These conditions are configuration specific
    if t - 1 == m.T.first() and not t - d3 < 0:
        return m.lsb[t].a == a31
    if t - 2 == m.T.first() and not t - d3 < 0:
        return m.lsb[t].a == a32
    if t - d3 < m.T.first() and t - d3 < 0:
        return m.lsb[t].a == 0
    return m.lsb[t].a == m.lsb[t-d3].r

def a4_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a == a40
    if t - d4 < m.T.first():
        return m.lsb[t].a == 0
    return m.lsb[t].a == m.lsb[t-d4].r

# Linking demand variable for each stage
def d1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].demand == demand1
    return m.lsb[t].demand == mu

def d2_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].demand == demand2
    return m.lsb[t].demand == mu

def d3_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].demand == demand3
    return m.lsb[t].demand == mu

def d4_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].demand == demand4
    return m.lsb[t].demand == mu


# create a block for a single time period and single stage
def lotsizing_block_rule(b, t):
    # define the variables
    # Reorder Variables at each stage
    b.r = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Inventory at each stage
    b.i = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Initial Inventory at each time-step
    b.i0 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # backlog
    b.bl = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Initial Backlog at each time-step
    b.bl0 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)

    # Shipped goods/sales
    b.s = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Acquisiton
    b.a = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Customer demand
    b.demand = pyo.Var(domain=pyo.NonNegativeIntegers)

    # define the constraints
    b.inventory = pyo.Constraint(expr=b.i == b.i0 + b.a - b.s)

    # Inventory constrainss
    b.inventorymax = pyo.Constraint(expr=b.i <= I)

    # Order constraints
    b.ordermax = pyo.Constraint(expr=b.r <= O)
    b.ordermax2 = pyo.Constraint(expr=b.r <= -b.i0 + I)
    #b.ordermax3 = pyo.Constraint(expr=b.r <= b.demand)

    # backlog constrains
    b.backlog = pyo.Constraint(expr=b.bl == b.bl0 - b.s + b.demand)

    # Sales Constraints
    b.ship11 = pyo.Constraint(expr=b.s <= b.i0 + b.a)
    b.ship12 = pyo.Constraint(expr=b.s <= b.bl0 + b.demand)

coeff = 5e0
# construct the objective function over all the blocks for eac stage
def obj_rule_1(m):
    # Sum of Profit for all timeperiods
    return sum(m.lsb[t].s*P1 - m.lsb[t].r*P2 - m.lsb[t].i*SC1 - m.lsb[t].bl*BC1 - coeff*(m.lsb[t].demand - m.lsb[t].r)**2 for t in m.T)

def obj_rule_2(m):
    # Sum of Profit for all timeperiods
    return sum(m.lsb[t].s*P2 - m.lsb[t].r*P3 - m.lsb[t].i*SC2 - m.lsb[t].bl*BC2 - coeff*(m.lsb[t].demand - m.lsb[t].r)**2 for t in m.T)

def obj_rule_3(m):
    # Sum of Profit for all timeperiods
    return sum(m.lsb[t].s*P3 - m.lsb[t].r*P4 - m.lsb[t].i*SC3 - m.lsb[t].bl*BC3 - coeff*(m.lsb[t].demand - m.lsb[t].r)**2 for t in m.T)

def obj_rule_4(m):
    # Sum of Profit for all timeperiods
    return sum(m.lsb[t].s*P4 - m.lsb[t].r*P5 - m.lsb[t].i*SC4 - m.lsb[t].bl*BC4 - coeff*(m.lsb[t].demand - m.lsb[t].r)**2 for t in m.T)

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
    i30 = init_inv[2]
    i40 = init_inv[3]

    # Initial Backlog
    bl10 = 0
    bl20 = 0
    bl30 = 0
    bl40 = 0

    # Initial Acquisition
    a10 = 0
    a20 = 0
    a21 = 0
    a30 = 0
    a31 = 0
    a32 = 0
    a40 = 0

    # Initialise arrays for storing variables at each iteration
    SHLP_actions = np.zeros((num_periods, 4))
    SHLP_shipment = np.zeros((num_periods, 4))
    SHLP_inventory = np.zeros((num_periods, 4))
    SHLP_backlog = np.zeros((num_periods, 4))
    SHLP_acquisition = np.zeros((num_periods, 4))
    SHLP_demand = np.zeros((num_periods, 4))
    horizon_length = num_periods
    for i in range(num_periods):
        # Get initial acquisition at each stage
        # Stage 1 acquisition
        if i - d1 < 0:
            a10 = 0
        else:

            extra_delay = False
            extra_delay_prob = np.random.uniform(0, 1)
            if extra_delay_prob <= noisy_delay_threshold and noisy_delay and i < num_periods:
                extra_delay = True

            if extra_delay:
                a10 = 0
                SHLP_shipment[i - d1 + 1, 0] += SHLP_shipment[i - d1, 0]
            else:
                a10 = SHLP_shipment[i - d1, 0]

        # Stage 2 acquisition
        if i - d2 < 0:
            a20 = 0
            a21 = 0
        else:

            extra_delay = False
            extra_delay_prob = np.random.uniform(0, 1)
            if extra_delay_prob <= noisy_delay_threshold and noisy_delay and i < num_periods:
                extra_delay = True

            if extra_delay:
                a20 = 0
                SHLP_shipment[i - d2 + 1, 1] += SHLP_shipment[i - d2, 1]
                a21 = SHLP_shipment[i - d2 + 1, 1]  # Configuration specific
            else:
                a20 = SHLP_shipment[i - d2, 1]
                a21 = SHLP_shipment[i - d2 + 1, 1]  # Configuration specific

        # Stage 3 acquisition
        if i - d3 < 0:
            a30 = 0
            a31 = 0
            a32 = 0
        else:

            extra_delay = False
            extra_delay_prob = np.random.uniform(0, 1)
            if extra_delay_prob <= noisy_delay_threshold and noisy_delay and i < num_periods:
                extra_delay = True

            if extra_delay:
                a30 = 0
                SHLP_shipment[i - d3 + 1, 2] += SHLP_shipment[i - d3, 2]
                a31 = SHLP_shipment[i - d3 + 1, 2]  # Configuration specific
                a32 = SHLP_shipment[i - d3 + 2, 2]  # Configuration specific
            else:
                a30 = SHLP_shipment[i - d3, 2]
                a31 = SHLP_shipment[i - d3 + 1, 2]  # Configuration specific
                a32 = SHLP_shipment[i - d3 + 2, 2]  # Configuration specific

        if i - d4 < 0:
            a40 = 0
        else:

            extra_delay = False
            extra_delay_prob = np.random.uniform(0, 1)
            if extra_delay_prob <= noisy_delay_threshold and noisy_delay and i < num_periods:
                extra_delay = True

            if extra_delay:
                a40 = 0
                SHLP_shipment[i - d4 + 1, 3] += SHLP_shipment[i - d4, 3]
            else:
                a40 = SHLP_shipment[i - d4, 3]

        # Get solver
        solver = pyo.SolverFactory('gurobi', solver_io='python')

        # Get horizon to solve over (less than horizon length if i + horizon_length > episode T)
        if i + horizon_length < num_periods:
            horizon = i + horizon_length - 1
        else:
            horizon = num_periods - 1

        # Get real customer demand at current time-step
        demand1 = LP_demand[j, i]

        # Create model for stage 1 over the horizon i:num_periods (shrinking horizon)
        model_1 = pyo.ConcreteModel()
        model_1.T = pyo.RangeSet(i, horizon)
        model_1.lsb = pyo.Block(model_1.T, rule=lotsizing_block_rule)
        model_1.i_linking = pyo.Constraint(model_1.T, rule=i1_linking_rule)
        model_1.bl_linking = pyo.Constraint(model_1.T, rule=bl1_linking_rule)
        model_1.a_linking = pyo.Constraint(model_1.T, rule=a1_linking_rule)
        model_1.d_linking = pyo.Constraint(model_1.T, rule=d1_linking_rule)
        model_1.obj = pyo.Objective(rule=obj_rule_1, sense=pyo.maximize)
        results1 = solver.solve(model_1) # Optimise for stage 1

        # Create model for stage 2 over the horizon i:num_periods (shrinking horizon)
        demand2 = pyo.value(model_1.lsb[i].r) # Demand of stage 2 is re-order quantity of stage 1
        model_2 = pyo.ConcreteModel()
        model_2.T = pyo.RangeSet(i, horizon)
        model_2.lsb = pyo.Block(model_2.T, rule=lotsizing_block_rule)
        model_2.i_linking = pyo.Constraint(model_2.T, rule=i2_linking_rule)
        model_2.bl_linking = pyo.Constraint(model_2.T, rule=bl2_linking_rule)
        model_2.a_linking = pyo.Constraint(model_2.T, rule=a2_linking_rule)
        model_2.d_linking = pyo.Constraint(model_2.T, rule=d2_linking_rule)
        model_2.obj = pyo.Objective(rule=obj_rule_2, sense=pyo.maximize)
        results2 = solver.solve(model_2)

        # Create model for stage 3 over the horizon i:num_periods (shrinking horizon)
        demand3 = pyo.value(model_2.lsb[i].r) # Demand of stage 3 is re-order quantity of stage 2
        model_3 = pyo.ConcreteModel()
        model_3.T = pyo.RangeSet(i, horizon)
        model_3.lsb = pyo.Block(model_3.T, rule=lotsizing_block_rule)
        model_3.i_linking = pyo.Constraint(model_3.T, rule=i3_linking_rule)
        model_3.bl_linking = pyo.Constraint(model_3.T, rule=bl3_linking_rule)
        model_3.a_linking = pyo.Constraint(model_3.T, rule=a3_linking_rule)
        model_3.d_linking = pyo.Constraint(model_3.T, rule=d3_linking_rule)
        model_3.obj = pyo.Objective(rule=obj_rule_3, sense=pyo.maximize)
        results3 = solver.solve(model_3)

        # Create model for stage 1 over the horizon i:num_periods (shrinking horizon)
        demand4 = pyo.value(model_3.lsb[i].r) # Demand of stage 4 is re-order quantity of stage 3
        model_4 = pyo.ConcreteModel()
        model_4.T = pyo.RangeSet(i, horizon)
        model_4.lsb = pyo.Block(model_4.T, rule=lotsizing_block_rule)
        model_4.i_linking = pyo.Constraint(model_4.T, rule=i4_linking_rule)
        model_4.bl_linking = pyo.Constraint(model_4.T, rule=bl4_linking_rule)
        model_4.a_linking = pyo.Constraint(model_4.T, rule=a4_linking_rule)
        model_4.d_linking = pyo.Constraint(model_4.T, rule=d4_linking_rule)
        model_4.obj = pyo.Objective(rule=obj_rule_4, sense=pyo.maximize)
        results4 = solver.solve(model_4)


        # Log actions for test Gym environment
        SHLP_actions[i, :] = [pyo.value(model_1.lsb[i].r), pyo.value(model_2.lsb[i].r),
                              pyo.value(model_3.lsb[i].r), pyo.value(model_4.lsb[i].r)]

        SHLP_shipment[i, :] = [pyo.value(model_2.lsb[i].s), pyo.value(model_3.lsb[i].s),
                               pyo.value(model_4.lsb[i].s), pyo.value(model_4.lsb[i].r)]

        SHLP_inventory[i, :] = [pyo.value(model_1.lsb[i].i0), pyo.value(model_2.lsb[i].i0),
                               pyo.value(model_3.lsb[i].i0), pyo.value(model_4.lsb[i].i0)]

        SHLP_backlog[i, :] = [pyo.value(model_1.lsb[i].bl), pyo.value(model_2.lsb[i].bl),
                                pyo.value(model_3.lsb[i].bl), pyo.value(model_4.lsb[i].bl)]

        SHLP_demand[i, :] = [pyo.value(model_1.lsb[i].demand), pyo.value(model_2.lsb[i].demand),
                              pyo.value(model_3.lsb[i].demand), pyo.value(model_4.lsb[i].demand)]

        # Update initial inventory for next iteration
        i10 = pyo.value(model_1.lsb[i].i)
        i20 = pyo.value(model_2.lsb[i].i)
        i30 = pyo.value(model_3.lsb[i].i)
        i40 = pyo.value(model_4.lsb[i].i)

        # Update initial backlog for next iteration
        bl10 = pyo.value(model_1.lsb[i].bl)
        bl20 = pyo.value(model_2.lsb[i].bl)
        bl30 = pyo.value(model_3.lsb[i].bl)
        bl40 = pyo.value(model_4.lsb[i].bl)

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

decLP_time = time.time() - start_time
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

path = 'LP_results/four_stage_noise_40/DecLP/'
np.save(path+'reward_mean.npy', lp_reward_mean)
np.save(path+'reward_std.npy', lp_reward_std)
np.save(path+'inventory_mean.npy', inventory_level_mean)
np.save(path+'inventory_std.npy', inventory_level_std)
np.save(path+'backlog_mean.npy', backlog_level_mean)
np.save(path+'backlog_std.npy', backlog_level_std)
np.save(path+'customer_backlog_mean', customer_backlog_mean)
np.save(path+'customer_backlog_std', customer_backlog_std)
np.save(path+'profit', profit)
np.save(path+'time', decLP_time)

#%% Test rollout plots
fig, axs = plt.subplots(3, num_stages, figsize=(18, 9), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.06, wspace=.16)

axs = axs.ravel()

for i in range(num_stages):
    axs[i].plot(array_obs[i, 0, :], label='Inventory')
    axs[i].plot(array_obs[i, 1, :], label='Backlog', color='tab:red')
    title = 'Stage ' + str(i+1)
    axs[i].set_title(title)
    axs[i].set_xlim(0, num_periods)
    axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i].legend()
        axs[i].set_ylabel('Products')

    axs[i+num_stages].plot(np.arange(0, num_periods), array_actions[i, :], label='Replenishment order', color='k')
    axs[i + num_stages].plot(np.arange(0, num_periods), array_demand[i, :], label='Demand', color='tab:orange')
    axs[i+num_stages].set_xlim(0, num_periods)
    axs[i+num_stages].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i+num_stages].legend()
        axs[i + num_stages].set_ylabel('Products')

    axs[i+num_stages*2].plot(array_profit[i, :], label='Periodic profit', color='tab:green')
    axs[i+num_stages*2].plot(np.cumsum(array_profit[i, :]), label='Cumulative profit', color='salmon')
    axs[i+num_stages*2].plot([0, num_periods], [0, 0], color='k')
    axs[i+num_stages*2].set_xlabel('Period')
    axs[i+num_stages*2].set_xlim(0, num_periods)
    if i == 0:
        axs[i + num_stages * 2].legend()
        axs[i + num_stages * 2].set_ylabel('Profit')


test_name = path + '/test_rollout.png'
plt.savefig(test_name, dpi=200)
plt.show()