from environments.IM_div_env import InvManagementDiv
from ray import tune
import numpy as np
import pyomo.environ as pyo
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from utils import check_connections, create_network, ensure_dir
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
    env = InvManagementDiv(configuration)
    return env

# Environment Configuration
num_nodes = 4
connections = {
    0: [1],
    1: [2, 3],
    2: [],
    3: [],
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
stock_cost = np.array([0.35, 0.3, 0.4, 0.4])
backlog_cost = np.array([0.5, 0.7, 0.6, 0.6])
delay = np.array([1, 2, 1, 1], dtype=np.int8)
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

env_name = "InventoryManagement"
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
    "standardise_state": standardise_state,
    "standardise_actions": standardise_actions,
    "time_dependency": time_dependency,
    "prev_demand": prev_demand,
    "prev_actions": prev_actions,
    "prev_length": prev_length,
}
CONFIG = env_config.copy()
# Configuration for the base-restock policy with DFO that has no state-action standardisation
LP_CONFIG = env_config.copy()
LP_CONFIG["standardise_state"] = False
LP_CONFIG["standardise_actions"] = False
LP_CONFIG["time_dependency"] = False
# Test environment
test_env = InvManagementDiv(env_config)
LP_env = InvManagementDiv(LP_CONFIG)

#%% Linear Programming Pyomo
num_tests = 1000
test_seed = 420
np.random.seed(seed=test_seed)
LP_demand = test_env.dist.rvs(size=(num_tests, (len(test_env.retailers)), test_env.num_periods), **test_env.dist_param)
noisy_demand = False
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
bl230 = 0
bl240 = 0
bl30 = 0
bl40 = 0

# Initial Acquisition
a10 = 0
a20 = 0
a21 = 0
a30 = 0
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
I1 = inv_max[0]
I2 = inv_max[1]
I3 = inv_max[2]
I4 = inv_max[3]

# Order Max
O1 = inv_max[0]
O2 = inv_max[1]
O3 = inv_max[2]
O4 = inv_max[3]

# Price of goods at each stage
P1 = 2
P2 = 3
P3 = 4

C1 = 1
C2 = 2
C3 = 3

# Shipment Delay
d1 = delay[0]
d2 = delay[1]
d3 = delay[2]
d4 = delay[3]

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

# link the backlog variables between blocks
def bl1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl10 == bl10
    return m.lsb[t].bl10 == m.lsb[t-1].bl1


def bl23_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl230 == bl230
    return m.lsb[t].bl230 == m.lsb[t - 1].bl23


def bl24_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl240 == bl240
    return m.lsb[t].bl240 == m.lsb[t - 1].bl24


def bl3_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl30 == bl30
    return m.lsb[t].bl30 == m.lsb[t-1].bl3

def bl4_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl40 == bl40
    return m.lsb[t].bl40 == m.lsb[t-1].bl4


# link the acquisition variables between blocks
def a1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a1 == a10
    if t - d1 < m.T.first():
        return m.lsb[t].a1 == 0
    return m.lsb[t].a1 == m.lsb[t-d1].x1

def a2_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a2 == a20
    if t - d2 < m.T.first() and t - d2 < 0:
        return m.lsb[t].a2 == 0
    # This condition is configuration specific
    if t - 1 == m.T.first() and not t - d2 < 0:
        return m.lsb[t].a2 == a21
    return m.lsb[t].a2 == m.lsb[t-d2].s1

def a3_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a3 == a30
    if t - d3 < m.T.first():
        return m.lsb[t].a3 == 0
    return m.lsb[t].a3 == m.lsb[t-d3].s23

def a4_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a4 == a40
    if t - d4 < m.T.first():
        return m.lsb[t].a4 == 0
    return m.lsb[t].a4 == m.lsb[t-d4].s24

def d3_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].demand3 == dem3
    return m.lsb[t].demand3 == mu

def d4_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].demand4 == dem4
    return m.lsb[t].demand4 == mu


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
    b.bl23 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.bl24 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.bl3 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.bl4 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Initial Backlog at each time-step
    b.bl10 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl20 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl230 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl240 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl30 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl40 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)

    # Shipped goods/sales
    b.s1 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s2 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s23 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s24 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s3 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s4 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Acquisiton
    b.a1 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.a2 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.a3 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.a4 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Customer demand
    b.demand3 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.demand4 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # define the constraints
    b.inventory1 = pyo.Constraint(expr=b.i1 == b.i10 + b.a1 - b.s1)
    b.inventory2 = pyo.Constraint(expr=b.i2 == b.i20 + b.a2 - b.s2)
    b.inventory3 = pyo.Constraint(expr=b.i3 == b.i30 + b.a3 - b.s3)
    b.inventory4 = pyo.Constraint(expr=b.i4 == b.i40 + b.a4 - b.s4)

    # Inventory constrainss
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
    b.backlog1 = pyo.Constraint(expr=b.bl1 == b.bl10 - b.s1 + b.x2)

    b.backlog23 = pyo.Constraint(expr=b.bl23 == b.bl230 - b.s23 + b.x3)
    b.backlog24 = pyo.Constraint(expr=b.bl24 == b.bl240 - b.s24 + b.x4)
    b.backlog2 = pyo.Constraint(expr=b.bl2 == b.bl23 + b.bl24)

    b.backlog3 = pyo.Constraint(expr=b.bl3 == b.bl30 - b.s3 + b.demand3)
    b.backlog4 = pyo.Constraint(expr=b.bl4 == b.bl40 - b.s4 + b.demand4)

    # Sales Constraints
    # Shipping Constraints
    b.ship1_inv = pyo.Constraint(expr=b.s1 <= b.i10 + b.a1)
    b.ship1_bl = pyo.Constraint(expr=b.s1 <= b.bl10 + b.x2)

    # shipping from node 2 to node 3
    b.ship23_bl = pyo.Constraint(expr=b.s23 <= b.bl230 + b.x3)
    # shipping from node 2 to node 4
    b.ship24_bl = pyo.Constraint(expr=b.s24 <= b.bl240 + b.x4)
    # All shipping from node 2 is equal to sum shipped to both nodes 3 and 4
    b.ship2 = pyo.Constraint(expr=b.s2 == b.s23 + b.s24)
    # Sum of shipped goods from node 2 cannot exceed inventory
    b.ship2_inv = pyo.Constraint(expr=b.s2 <= b.i20 + b.a2)

    b.ship3_inv = pyo.Constraint(expr=b.s3 <= b.i30 + b.a3)
    b.ship3_bl = pyo.Constraint(expr=b.s3 <= b.bl30 + b.demand3)

    b.ship4_inv = pyo.Constraint(expr=b.s4 <= b.i40 + b.a4)
    b.ship4_bl = pyo.Constraint(expr=b.s4 <= b.bl40 + b.demand4)

# construct the objective function over all the blocks
def obj_rule(m):
    # Sum of Profit at each state at each timeperiod
    # Profit per stage per time period: sales - re-order cost - backlog cost  - inventory cost
    return sum(m.lsb[t].s1*P1 - m.lsb[t].x1*C1 - m.lsb[t].i1*SC1 - m.lsb[t].bl1*BC1
               + m.lsb[t].s2*P2 - m.lsb[t].x2*C2 - m.lsb[t].i2*SC2 - m.lsb[t].bl2*BC2
               + m.lsb[t].s3*P3 - m.lsb[t].x3*C3 - m.lsb[t].i3*SC3 - m.lsb[t].bl3*BC3
               + m.lsb[t].s4*P3 - m.lsb[t].x4*C3 - m.lsb[t].i4*SC4 - m.lsb[t].bl4*BC4
               for t in m.T)

inventory_list = []
backlog_list = []
lp_reward_list = []
customer_backlog_list = []
profit = np.zeros((num_tests, num_periods))
start_time = time.time()

array_obs = np.zeros((num_nodes, 3, num_periods + 1))
array_actions = np.zeros((num_nodes, num_periods))
array_profit = np.zeros((num_nodes, num_periods))
array_profit_sum = np.zeros(num_periods)
array_demand = np.zeros((num_nodes, num_periods))
array_ship = np.zeros((num_nodes, num_periods))
array_acquisition = np.zeros((num_nodes, num_periods))
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
    bl230 = 0
    bl240 = 0
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

    SHLP_actions = np.zeros((num_periods, 4))
    SHLP_shipment = np.zeros((num_periods, 4))
    SHLP_inventory = np.zeros((num_periods, 4))
    SHLP_backlog = np.zeros((num_periods, 4))
    SHLP_acquisition = np.zeros((num_periods, 4))
    for i in range(num_periods):

        # Get initial acquisition at each stage
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

        if i - d3 < 0:
            a30 = 0
        else:

            extra_delay = False
            extra_delay_prob = np.random.uniform(0, 1)
            if extra_delay_prob <= noisy_delay_threshold and noisy_delay and i < num_periods:
                extra_delay = True

            if extra_delay:
                a30 = 0
                SHLP_shipment[i - d3 + 1, 2] += SHLP_shipment[i - d3, 2]
            else:
                a30 = SHLP_shipment[i - d3, 2]

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
        # Get real customer demand at current time-step
        d = LP_demand[j, :, i]
        dem3 = d[0]
        dem4 = d[1]
        # Create model over the horizon i:num_periods (shrinking horizon)
        model = pyo.ConcreteModel()
        model.T = pyo.RangeSet(i, num_periods-1)
        # Link all time-period blocks
        model.lsb = pyo.Block(model.T, rule=lotsizing_block_rule)

        # Inventory linking constraints
        model.i_linking1 = pyo.Constraint(model.T, rule=i1_linking_rule)
        model.i_linking2 = pyo.Constraint(model.T, rule=i2_linking_rule)
        model.i_linking3 = pyo.Constraint(model.T, rule=i3_linking_rule)
        model.i_linking4 = pyo.Constraint(model.T, rule=i4_linking_rule)

        # Backlog linking constraints
        model.bl_linking1 = pyo.Constraint(model.T, rule=bl1_linking_rule)
        model.bl_linking23 = pyo.Constraint(model.T, rule=bl23_linking_rule)
        model.bl_linking24 = pyo.Constraint(model.T, rule=bl24_linking_rule)
        model.bl_linking3 = pyo.Constraint(model.T, rule=bl3_linking_rule)
        model.bl_linking4 = pyo.Constraint(model.T, rule=bl4_linking_rule)

        # Acquisition linking constraints
        model.a_linking1 = pyo.Constraint(model.T, rule=a1_linking_rule)
        model.a_linking2 = pyo.Constraint(model.T, rule=a2_linking_rule)
        model.a_linking3 = pyo.Constraint(model.T, rule=a3_linking_rule)
        model.a_linking4 = pyo.Constraint(model.T, rule=a4_linking_rule)

        # Customer demand linking constraints
        model.d3_linking = pyo.Constraint(model.T, rule=d3_linking_rule)
        model.d4_linking = pyo.Constraint(model.T, rule=d4_linking_rule)

        # Model obbjective function
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

        ### solve the problem
        solver = pyo.SolverFactory('gurobi', solver_io='python')
        results = solver.solve(model)

        # Log actions for testing in Gym environment
        SHLP_actions[i, :] = [pyo.value(model.lsb[i].x1), pyo.value(model.lsb[i].x2),
                              pyo.value(model.lsb[i].x3), pyo.value(model.lsb[i].x4)]

        # Log shipments for updating the acquisition at next time-step
        SHLP_shipment[i, :] = [pyo.value(model.lsb[i].x1), pyo.value(model.lsb[i].s1),
                               pyo.value(model.lsb[i].s23), pyo.value(model.lsb[i].s24)]

        # Log shipments for updating the acquisition at next time-step
        SHLP_inventory[i, :] = [pyo.value(model.lsb[i].i10), pyo.value(model.lsb[i].i20),
                               pyo.value(model.lsb[i].i30), pyo.value(model.lsb[i].i40)]


        # Update initial inventory for next iteration
        i10 = pyo.value(model.lsb[i].i1)
        i20 = pyo.value(model.lsb[i].i2)
        i30 = pyo.value(model.lsb[i].i3)
        i40 = pyo.value(model.lsb[i].i4)

        # Update initial backlog for next iteration
        bl10 = pyo.value(model.lsb[i].bl1)
        bl20 = pyo.value(model.lsb[i].bl2)
        bl230 = pyo.value(model.lsb[i].bl23)
        bl240 = pyo.value(model.lsb[i].bl24)
        bl30 = pyo.value(model.lsb[i].bl3)
        bl40 = pyo.value(model.lsb[i].bl4)


    # Test Gym environment
    s = LP_env.reset(customer_demand=LP_demand[j])
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
            array_actions[:, t] = np.maximum(array_actions[:, t], np.zeros(num_nodes))
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

path = 'LP_results/div_1/SHLP/'
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
np.save(path+'time', shlp_time)

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
        axs[i+num_nodes].legend()
        axs[i + num_nodes].set_ylabel('Products')

    axs[i+num_nodes*2].plot(array_profit[i, :], label='Periodic profit', color='tab:green', lw=2)
    axs[i+num_nodes*2].plot(np.cumsum(array_profit[i, :]), label='Cumulative profit', color='salmon', lw=2)
    axs[i+num_nodes*2].plot([0, num_periods], [0, 0], color='k')
    axs[i+num_nodes*2].set_xlabel('Period')
    axs[i+num_nodes*2].set_xlim(0, num_periods)
    if i == 0:
        axs[i + num_nodes * 2].legend()
        axs[i + num_nodes * 2].set_ylabel('Profit')


test_name = path + '/test_rollout.png'
plt.savefig(test_name, dpi=200)
plt.show()





