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
num_stages = 8
num_periods = 30
customer_demand = np.ones(num_periods) * 5
mu = 5
lower_upper = (1, 5)
init_inv = np.ones(num_stages)*10
inv_target = np.ones(num_stages) * 0
inv_max = np.ones(num_stages) * 30
price = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
stock_cost = np.array([0.35, 0.3, 0.4, 0.2, 0.35, 0.3, 0.4, 0.2])
backlog_cost = np.array([0.5, 0.7, 0.6, 0.9, 0.5, 0.7, 0.6, 0.9])
delay = np.array([1, 2, 3, 1, 4, 2, 3, 1], dtype=np.int8)
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
i30 = init_inv[2]
i40 = init_inv[3]
i50 = init_inv[4]
i60 = init_inv[5]
i70 = init_inv[6]
i80 = init_inv[7]

# Initial Backlog
bl10 = 0
bl20 = 0
bl30 = 0
bl40 = 0
bl50 = 0
bl60 = 0
bl70 = 0
bl80 = 0

# Initial Acquisition
a10 = 0
a20 = 0
a21 = 0
a30 = 0
a31 = 0
a32 = 0
a40 = 0
a50 = 0
a51 = 0
a52 = 0
a53 = 0
a60 = 0
a61 = 0
a70 = 0
a71 = 0
a72 = 0
a80 = 0

# Inventory Holding Cost
SC1 = stock_cost[0]
SC2 = stock_cost[1]
SC3 = stock_cost[2]
SC4 = stock_cost[3]
SC5 = stock_cost[4]
SC6 = stock_cost[5]
SC7 = stock_cost[6]
SC8 = stock_cost[7]

# Backlog Cost
BC1 = backlog_cost[0]
BC2 = backlog_cost[1]
BC3 = backlog_cost[2]
BC4 = backlog_cost[3]
BC5 = backlog_cost[4]
BC6 = backlog_cost[5]
BC7 = backlog_cost[6]
BC8 = backlog_cost[7]

# Maximum Inventory
I1 = inv_max[0]
I2 = inv_max[1]
I3 = inv_max[2]
I4 = inv_max[3]
I5 = inv_max[4]  # maximum inventory 4
I6 = inv_max[5]  # maximum inventory 4
I7 = inv_max[6]  # maximum inventory 4
I8 = inv_max[7]  # maximum inventory 4

# Order Max
O1 = inv_max[0]
O2 = inv_max[1]
O3 = inv_max[2]
O4 = inv_max[3]
O5 = inv_max[4]
O6 = inv_max[5]
O7 = inv_max[6]
O8 = inv_max[7]

# Price of goods at each stage
P1 = price[0]
P2 = price[1]
P3 = price[2]
P4 = price[3]
P5 = price[4]
P6 = price[5]
P7 = price[6]
P8 = price[7]
P9 = price[8]

# Shipment Delay
d1 = delay[0]
d2 = delay[1]
d3 = delay[2]
d4 = delay[3]
d5 = delay[4]
d6 = delay[5]
d7 = delay[6]
d8 = delay[7]

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

def i5_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i50 == i50
    return m.lsb[t].i50 == m.lsb[t-1].i5

def i6_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i60 == i60
    return m.lsb[t].i60 == m.lsb[t-1].i6

def i7_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i70 == i70
    return m.lsb[t].i70 == m.lsb[t-1].i7

def i8_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i80 == i80
    return m.lsb[t].i80 == m.lsb[t-1].i8

# link the backlog variables between blocks
def bl1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl10 == bl10
    return m.lsb[t].bl10 == m.lsb[t-1].bl1

def bl2_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl20 == bl20
    return m.lsb[t].bl20 == m.lsb[t-1].bl2

def bl3_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl30 == bl30
    return m.lsb[t].bl30 == m.lsb[t-1].bl3

def bl4_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl40 == bl40
    return m.lsb[t].bl40 == m.lsb[t-1].bl4

def bl5_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl50 == 0
    return m.lsb[t].bl50 == m.lsb[t-1].bl5

def bl6_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl60 == 0
    return m.lsb[t].bl60 == m.lsb[t-1].bl6

def bl7_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl70 == 0
    return m.lsb[t].bl70 == m.lsb[t-1].bl7

def bl8_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].bl80 == 0
    return m.lsb[t].bl80 == m.lsb[t-1].bl8


# link the acquisition variables between blocks
def a1_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a1 == a10
    if t - d1 < m.T.first():
        return m.lsb[t].a1 == 0
    return m.lsb[t].a1 == m.lsb[t-d1].s2

def a2_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a2 == a20
    if t - d2 < m.T.first() and t - d2 < 0:
        return m.lsb[t].a2 == 0
    # This condition is configuration specific
    if t - 1 == m.T.first() and not t - d2 < 0:
        return m.lsb[t].a2 == a21
    return m.lsb[t].a2 == m.lsb[t-d2].s3

def a3_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a3 == a30
    if t - d3 < m.T.first() and t - d3 < 0:
        return m.lsb[t].a3 == 0
    # These conditions are configuration specific
    if t - 1 == m.T.first() and not t - d3 < 0:
        return m.lsb[t].a3 == a31
    if t - 2 == m.T.first() and not t - d3 < 0:
        return m.lsb[t].a3 == a32
    return m.lsb[t].a3 == m.lsb[t-d3].s4

def a4_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a4 == a40
    if t - d4 < m.T.first():
        return m.lsb[t].a4 == 0
    return m.lsb[t].a4 == m.lsb[t-d4].s5

def a5_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a5 == a50
    if t - d5 < m.T.first() and t - d5 < 0:
        return m.lsb[t].a5 == 0
    # These conditions are configuration specific
    if t - 1 == m.T.first() and not t - d5 < 0:
        return m.lsb[t].a5 == a51
    if t - 2 == m.T.first() and not t - d5 < 0:
        return m.lsb[t].a5 == a52
    if t - 3 == m.T.first() and not t - d5 < 0:
        return m.lsb[t].a5 == a53
    return m.lsb[t].a5 == m.lsb[t-d5].s6

def a6_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a6 == a60
    if t - d6 < m.T.first() and t - d6 < 0:
        return m.lsb[t].a6 == 0
    # This condition is configuration specific
    if t - 1 == m.T.first() and not t - d6 < 0:
        return m.lsb[t].a6 == a61
    return m.lsb[t].a6 == m.lsb[t-d6].s7

def a7_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a7 == a70
    if t - d7 < m.T.first() and t - d7 < 0:
        return m.lsb[t].a7 == 0
    # These conditions are configuration specific
    if t - 1 == m.T.first() and not t - d7 < 0:
        return m.lsb[t].a7 == a71
    if t - 2 == m.T.first() and not t - d7 < 0:
        return m.lsb[t].a7 == a72
    return m.lsb[t].a7 == m.lsb[t-d7].s8

def a8_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].a8 == a80
    if t - d8 < m.T.first():
        return m.lsb[t].a8 == 0
    return m.lsb[t].a8 == m.lsb[t-d4].x8

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
    b.x3 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.x4 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.x5 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.x6 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.x7 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.x8 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Inventory at each stage
    b.i1 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i2 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i3 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i4 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i5 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i6 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i7 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i8 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Initial Inventory at each time-step
    b.i10 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i20 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i30 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i40 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i50 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i60 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i70 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.i80 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # backlog
    b.bl1 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.bl2 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.bl3 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.bl4 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.bl5 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.bl6 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.bl7 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.bl8 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Initial Backlog at each time-step
    b.bl10 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl20 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl30 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl40 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl50 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl60 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl70 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
    b.bl80 = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)

    # Shipped goods/sales
    b.s1 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s2 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s3 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s4 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s5 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s6 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s7 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.s8 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Acquisiton
    b.a1 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.a2 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.a3 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.a4 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.a5 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.a6 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.a7 = pyo.Var(domain=pyo.NonNegativeIntegers)
    b.a8 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # Customer demand
    b.demand = pyo.Var(domain=pyo.NonNegativeIntegers)

    # define the constraints
    b.inventory1 = pyo.Constraint(expr=b.i1 == b.i10 + b.a1 - b.s1)
    b.inventory2 = pyo.Constraint(expr=b.i2 == b.i20 + b.a2 - b.s2)
    b.inventory3 = pyo.Constraint(expr=b.i3 == b.i30 + b.a3 - b.s3)
    b.inventory4 = pyo.Constraint(expr=b.i4 == b.i40 + b.a4 - b.s4)
    b.inventory5 = pyo.Constraint(expr=b.i5 == b.i50 + b.a5 - b.s5)
    b.inventory6 = pyo.Constraint(expr=b.i6 == b.i60 + b.a6 - b.s6)
    b.inventory7 = pyo.Constraint(expr=b.i7 == b.i70 + b.a7 - b.s7)
    b.inventory8 = pyo.Constraint(expr=b.i8 == b.i80 + b.a8 - b.s8)

    # Inventory constrainss
    b.inventorymax1 = pyo.Constraint(expr=b.i1 <= I1)
    b.inventorymax2 = pyo.Constraint(expr=b.i2 <= I2)
    b.inventorymax3 = pyo.Constraint(expr=b.i3 <= I3)
    b.inventorymax4 = pyo.Constraint(expr=b.i4 <= I4)
    b.inventorymax5 = pyo.Constraint(expr=b.i5 <= I5)
    b.inventorymax6 = pyo.Constraint(expr=b.i6 <= I6)
    b.inventorymax7 = pyo.Constraint(expr=b.i7 <= I7)
    b.inventorymax8 = pyo.Constraint(expr=b.i8 <= I8)

    # Order constraints
    b.ordermax1 = pyo.Constraint(expr=b.x1 <= O1)
    b.ordermax2 = pyo.Constraint(expr=b.x2 <= O2)
    b.ordermax3 = pyo.Constraint(expr=b.x3 <= O3)
    b.ordermax4 = pyo.Constraint(expr=b.x4 <= O4)
    b.ordermax5 = pyo.Constraint(expr=b.x5 <= O5)
    b.ordermax6 = pyo.Constraint(expr=b.x6 <= O6)
    b.ordermax7 = pyo.Constraint(expr=b.x7 <= O7)
    b.ordermax8 = pyo.Constraint(expr=b.x8 <= O8)

    # backlog constrains
    b.backlog1 = pyo.Constraint(expr=b.bl1 == b.bl10 - b.s1 + b.demand)
    b.backlog2 = pyo.Constraint(expr=b.bl2 == b.bl20 - b.s2 + b.x1)
    b.backlog3 = pyo.Constraint(expr=b.bl3 == b.bl30 - b.s3 + b.x2)
    b.backlog4 = pyo.Constraint(expr=b.bl4 == b.bl40 - b.s4 + b.x3)
    b.backlog5 = pyo.Constraint(expr=b.bl5 == b.bl50 - b.s5 + b.x4)
    b.backlog6 = pyo.Constraint(expr=b.bl6 == b.bl60 - b.s6 + b.x5)
    b.backlog7 = pyo.Constraint(expr=b.bl7 == b.bl70 - b.s7 + b.x6)
    b.backlog8 = pyo.Constraint(expr=b.bl8 == b.bl80 - b.s8 + b.x7)

    # Sales Constraints
    b.ship11 = pyo.Constraint(expr=b.s1 <= b.i10 + b.a1)
    b.ship12 = pyo.Constraint(expr=b.s1 <= b.bl10 + b.demand)
    b.ship21 = pyo.Constraint(expr=b.s2 <= b.i20 + b.a2)
    b.ship22 = pyo.Constraint(expr=b.s2 <= b.bl20 + b.x1)
    b.ship31 = pyo.Constraint(expr=b.s3 <= b.i30 + b.a3)
    b.ship32 = pyo.Constraint(expr=b.s3 <= b.bl30 + b.x2)
    b.ship41 = pyo.Constraint(expr=b.s4 <= b.i40 + b.a4)
    b.ship42 = pyo.Constraint(expr=b.s4 <= b.bl40 + b.x3)
    b.ship51 = pyo.Constraint(expr=b.s5 <= b.i50 + b.a5)
    b.ship52 = pyo.Constraint(expr=b.s5 <= b.bl50 + b.x4)
    b.ship61 = pyo.Constraint(expr=b.s6 <= b.i60 + b.a6)
    b.ship62 = pyo.Constraint(expr=b.s6 <= b.bl60 + b.x5)
    b.ship71 = pyo.Constraint(expr=b.s7 <= b.i70 + b.a7)
    b.ship72 = pyo.Constraint(expr=b.s7 <= b.bl70 + b.x6)
    b.ship81 = pyo.Constraint(expr=b.s8 <= b.i80 + b.a8)
    b.ship82 = pyo.Constraint(expr=b.s8 <= b.bl80 + b.x7)

# construct the objective function over all the blocks
def obj_rule(m):
    # Sum of Profit at each state at each timeperiod
    # Profit per stage per time period: sales - re-order cost - backlog cost  - inventory cost
    return sum(m.lsb[t].s1*P1 - m.lsb[t].x1*P2 - m.lsb[t].i1*SC1 - m.lsb[t].bl1*BC1
               + m.lsb[t].s2*P2 - m.lsb[t].x2*P3 - m.lsb[t].i2*SC2 - m.lsb[t].bl2*BC2
               + m.lsb[t].s3*P3 - m.lsb[t].x3*P4 - m.lsb[t].i3*SC3 - m.lsb[t].bl3*BC3
               + m.lsb[t].s4*P4 - m.lsb[t].x4*P5 - m.lsb[t].i4*SC4 - m.lsb[t].bl4*BC4
               + m.lsb[t].s5*P5 - m.lsb[t].x5*P6 - m.lsb[t].i5*SC5 - m.lsb[t].bl5*BC5
               + m.lsb[t].s6*P6 - m.lsb[t].x6*P7 - m.lsb[t].i6*SC6 - m.lsb[t].bl6*BC6
               + m.lsb[t].s7*P7 - m.lsb[t].x7*P8 - m.lsb[t].i7*SC7 - m.lsb[t].bl7*BC7
               + m.lsb[t].s8*P8 - m.lsb[t].x8*P9 - m.lsb[t].i8*SC8 - m.lsb[t].bl8*BC8
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
    i30 = init_inv[2]
    i40 = init_inv[3]
    i50 = init_inv[4]
    i60 = init_inv[5]
    i70 = init_inv[6]
    i80 = init_inv[7]

    # Initial Backlog
    bl10 = 0
    bl20 = 0
    bl30 = 0
    bl40 = 0
    bl50 = 0
    bl60 = 0
    bl70 = 0
    bl80 = 0

    # Initial Acquisition
    a10 = 0
    a20 = 0
    a21 = 0
    a30 = 0
    a31 = 0
    a32 = 0
    a40 = 0
    a50 = 0
    a51 = 0
    a52 = 0
    a53 = 0
    a60 = 0
    a61 = 0
    a70 = 0
    a71 = 0
    a72 = 0
    a80 = 0

    SHLP_actions = np.zeros((num_periods, num_stages))
    SHLP_shipment = np.zeros((num_periods, num_stages))
    SHLP_inventory = np.zeros((num_periods, num_stages))
    SHLP_backlog = np.zeros((num_periods, num_stages))
    SHLP_acquisition = np.zeros((num_periods, num_stages))
    for i in range(num_periods):

        # Get initial acquisition at each stage
        if i - d1 < 0:
            a10 = 0
        else:
            a10 = SHLP_shipment[i - d1, 0]

        if i - d2 < 0:
            a20 = 0
            a21 = 0
        else:
            a20 = SHLP_shipment[i - d2, 1]
            a21 = SHLP_shipment[i - d2 + 1, 1]  # Configuration specific

        if i - d3 < 0:
            a30 = 0
            a31 = 0
            a32 = 0
        else:
            a30 = SHLP_shipment[i - d3, 2]
            a31 = SHLP_shipment[i - d3 + 1, 2]  # Configuration specific
            a32 = SHLP_shipment[i - d3 + 2, 2]  # Configuration specific

        if i - d4 < 0:
            a40 = 0
        else:
            a40 = SHLP_shipment[i - d4, 3]

        if i - d5 < 0:
            a50 = 0
            a51 = 0
            a52 = 0
            a533 = 0
        else:
            a50 = SHLP_shipment[i - d5, 4]
            a51 = SHLP_shipment[i - d5 + 1, 4]  # Configuration specific
            a52 = SHLP_shipment[i - d5 + 2, 4]  # Configuration specific
            a53 = SHLP_shipment[i - d5 + 2, 4]  # Configuration specific

        if i - d6 < 0:
            a60 = 0
            a61 = 0
        else:
            a60 = SHLP_shipment[i - d6, 5]
            a61 = SHLP_shipment[i - d6 + 1, 5]  # Configuration specific

        if i - d7 < 0:
            a70 = 0
            a71 = 0
            a72 = 0
        else:
            a70 = SHLP_shipment[i - d7, 6]
            a71 = SHLP_shipment[i - d7 + 1, 6]  # Configuration specific
            a72 = SHLP_shipment[i - d7 + 2, 6]  # Configuration specific

        if i - d8 < 0:
            a80 = 0
        else:
            a80 = SHLP_shipment[i - d8, 7]

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
        model.i_linking3 = pyo.Constraint(model.T, rule=i3_linking_rule)
        model.i_linking4 = pyo.Constraint(model.T, rule=i4_linking_rule)
        model.i_linking5 = pyo.Constraint(model.T, rule=i5_linking_rule)
        model.i_linking6 = pyo.Constraint(model.T, rule=i6_linking_rule)
        model.i_linking7 = pyo.Constraint(model.T, rule=i7_linking_rule)
        model.i_linking8 = pyo.Constraint(model.T, rule=i8_linking_rule)

        # Backlog linking constraints
        model.bl_linking1 = pyo.Constraint(model.T, rule=bl1_linking_rule)
        model.bl_linking2 = pyo.Constraint(model.T, rule=bl2_linking_rule)
        model.bl_linking3 = pyo.Constraint(model.T, rule=bl3_linking_rule)
        model.bl_linking4 = pyo.Constraint(model.T, rule=bl4_linking_rule)
        model.bl_linking5 = pyo.Constraint(model.T, rule=bl5_linking_rule)
        model.bl_linking6 = pyo.Constraint(model.T, rule=bl6_linking_rule)
        model.bl_linking7 = pyo.Constraint(model.T, rule=bl7_linking_rule)
        model.bl_linking8 = pyo.Constraint(model.T, rule=bl8_linking_rule)

        # Acquisition linking constraints
        model.a_linking1 = pyo.Constraint(model.T, rule=a1_linking_rule)
        model.a_linking2 = pyo.Constraint(model.T, rule=a2_linking_rule)
        model.a_linking3 = pyo.Constraint(model.T, rule=a3_linking_rule)
        model.a_linking4 = pyo.Constraint(model.T, rule=a4_linking_rule)
        model.a_linking5 = pyo.Constraint(model.T, rule=a5_linking_rule)
        model.a_linking6 = pyo.Constraint(model.T, rule=a6_linking_rule)
        model.a_linking7 = pyo.Constraint(model.T, rule=a7_linking_rule)
        model.a_linking8 = pyo.Constraint(model.T, rule=a8_linking_rule)

        # Customer demand linking constraints
        model.d_linking = pyo.Constraint(model.T, rule=d_linking_rule)

        # Model obbjective function
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

        ### solve the problem
        solver = pyo.SolverFactory('gurobi', solver_io='python')
        results = solver.solve(model)

        # Log actions for testing in Gym environment
        SHLP_actions[i, :] = [pyo.value(model.lsb[i].x1), pyo.value(model.lsb[i].x2),
                              pyo.value(model.lsb[i].x3), pyo.value(model.lsb[i].x4),
                              pyo.value(model.lsb[i].x5), pyo.value(model.lsb[i].x6),
                              pyo.value(model.lsb[i].x7), pyo.value(model.lsb[i].x8)]

        # Log shipments for updating the acquisition at next time-step
        SHLP_shipment[i, :] = [pyo.value(model.lsb[i].s2), pyo.value(model.lsb[i].s3),
                               pyo.value(model.lsb[i].s4), pyo.value(model.lsb[i].s5),
                               pyo.value(model.lsb[i].s6), pyo.value(model.lsb[i].s7),
                               pyo.value(model.lsb[i].s8), pyo.value(model.lsb[i].x8)
                               ]

        # Log shipments for updating the acquisition at next time-step
        SHLP_inventory[i, :] = [pyo.value(model.lsb[i].i10), pyo.value(model.lsb[i].i20),
                                pyo.value(model.lsb[i].i30), pyo.value(model.lsb[i].i40),
                                pyo.value(model.lsb[i].i50), pyo.value(model.lsb[i].i60),
                                pyo.value(model.lsb[i].i70), pyo.value(model.lsb[i].i80)
                                ]

        # Log shipments for updating the acquisition at next time-step
        SHLP_backlog[i, :] = [pyo.value(model.lsb[i].bl10), pyo.value(model.lsb[i].bl20),
                              pyo.value(model.lsb[i].bl30), pyo.value(model.lsb[i].bl40),
                              pyo.value(model.lsb[i].bl50), pyo.value(model.lsb[i].bl60),
                              pyo.value(model.lsb[i].bl70), pyo.value(model.lsb[i].bl80)
                              ]

        # Update initial inventory for next iteration
        i10 = pyo.value(model.lsb[i].i1)
        i20 = pyo.value(model.lsb[i].i2)
        i30 = pyo.value(model.lsb[i].i3)
        i40 = pyo.value(model.lsb[i].i4)
        i50 = pyo.value(model.lsb[i].i5)
        i60 = pyo.value(model.lsb[i].i6)
        i70 = pyo.value(model.lsb[i].i7)
        i80 = pyo.value(model.lsb[i].i8)

        # Update initial backlog for next iteration
        bl10 = pyo.value(model.lsb[i].bl1)
        bl20 = pyo.value(model.lsb[i].bl2)
        bl30 = pyo.value(model.lsb[i].bl3)
        bl40 = pyo.value(model.lsb[i].bl4)
        bl50 = pyo.value(model.lsb[i].bl5)
        bl60 = pyo.value(model.lsb[i].bl6)
        bl70 = pyo.value(model.lsb[i].bl7)
        bl80 = pyo.value(model.lsb[i].bl8)


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

path = 'LP_results/eight_stage/SHLP/'
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





