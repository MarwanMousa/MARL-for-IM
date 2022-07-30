import pyomo.environ as pyo
import numpy as np
from environments.IM_div_env import InvManagementDiv
from ray import tune
from utils import check_connections, create_network, ensure_dir
import matplotlib.pyplot as plt
import time
#%% Environment Configuration

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

# Environment creator function for environment registration
def env_creator(configuration):
    env = InvManagementDiv(configuration)
    return env

# Environment Configuration
num_nodes = 8
connections = {
    0: [1],
    1: [2],
    2: [3],
    3: [4],
    4: [5],
    5: [6],
    6: [7],
    7: []
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
stock_cost = np.array([0.35, 0.3, 0.4, 0.2, 0.35, 0.3, 0.4, 0.2])
backlog_cost = np.array([0.5, 0.7, 0.6, 0.9, 0.5, 0.7, 0.6, 0.9])
delay = np.array([1, 2, 3, 1, 4, 2, 3, 1], dtype=np.int8)
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

#%%
path = 'LP_results/eight_stage/DSHLP/'
save_results = True
num_tests = 200
test_seed = 420
np.random.seed(seed=test_seed)
LP_Customer_Demand = LP_env.dist.rvs(size=(num_tests, (len(LP_env.retailers)), LP_env.num_periods), **LP_env.dist_param)

noisy_demand = False
noise_threshold = 50/100
noisy_delay = False
noisy_delay_threshold = 50/100
if noisy_demand:
    for i in range(num_tests):
        for k in range(len(LP_env.retailers)):
            for j in range(num_periods):
                double_demand = np.random.uniform(0, 1)
                zero_demand = np.random.uniform(0, 1)
                if double_demand <= noise_threshold:
                    LP_Customer_Demand[i, k, j] = 2 * LP_Customer_Demand[i, k, j]
                if zero_demand <= noise_threshold:
                    LP_Customer_Demand[i, k, j] = 0

# Hyperparameters
rho = 1e-1
rho_tgt = rho * 1000
Rho = rho
N_iter = 120
use_variable_rho = False
use_scaled_rho = False
act_rho = np.linspace(rho, rho_tgt, N_iter)
ADMM = True


# Maximum inventory and Maximum order amount
I1 = 30
O1 = 30

I2 = 30
O2 = 30

I3 = 30
O3 = 30

I4 = 30
O4 = 30

I5 = 30
O5 = 30

I6 = 30
O6 = 30

I7 = 30
O7 = 30

I8 = 30
O8 = 30

# Delay
d1 = delay[0]
d2 = delay[1]
d3 = delay[2]
d4 = delay[3]
d5 = delay[4]
d6 = delay[5]
d7 = delay[6]
d8 = delay[7]

# Price of goods sold
P1 = 2
P2 = 3
P3 = 4
P4 = 5
P5 = 6
P6 = 7
P7 = 8
P8 = 9

# Cost of re-order goods
C1 = 1
C2 = 2
C3 = 3
C4 = 4
C5 = 5
C6 = 6
C7 = 7
C8 = 8

# Cost of Inventory
IC1 = stock_cost[0]
IC2 = stock_cost[1]
IC3 = stock_cost[2]
IC4 = stock_cost[3]
IC5 = stock_cost[4]
IC6 = stock_cost[5]
IC7 = stock_cost[6]
IC8 = stock_cost[7]

# Backlog Cost
BC1 = backlog_cost[0]
BC2 = backlog_cost[1]
BC3 = backlog_cost[2]
BC4 = backlog_cost[3]
BC5 = backlog_cost[4]
BC6 = backlog_cost[5]
BC7 = backlog_cost[6]
BC8 = backlog_cost[7]

# create a block for a single time period and single stage
# This is the block for stage 1
def node_block_rule(b, t):
    # define the variables
    # Reorder Variables at each stage
    b.r = pyo.Var(domain=pyo.NonNegativeReals)

    # Inventory at each stage
    b.i = pyo.Var(domain=pyo.NonNegativeReals)

    # Initial Inventory at each time-step
    b.i0 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # backlog
    b.bl = pyo.Var(domain=pyo.NonNegativeReals)

    # Initial Backlog at each time-step
    b.bl0 = pyo.Var(domain=pyo.NonNegativeReals)

    # Shipped goods/sales
    b.s = pyo.Var(domain=pyo.NonNegativeReals)

    # Acquisition
    b.a = pyo.Var(domain=pyo.NonNegativeReals)

    # Customer demand
    b.demand = pyo.Var(domain=pyo.NonNegativeReals)

    # define the constraints
    b.inventory = pyo.Constraint(expr=b.i == b.i0 + b.a - b.s)

    # backlog constrains
    b.backlog = pyo.Constraint(expr=b.bl == b.bl0 - b.s + b.demand)

    # Sales Constraints
    b.ship_inventory = pyo.Constraint(expr=b.s <= b.i0 + b.a)
    b.ship_backlog = pyo.Constraint(expr=b.s <= b.bl0 + b.demand)


# link inventory variables between time-step blocks
def i_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].i0 == m.I0
    return m.nb[t].i0 == m.nb[t-1].i


# link the backlog variables between time-step blocks
def bl_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].bl0 == m.B0
    return m.nb[t].bl0 == m.nb[t-1].bl


def demand_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].demand == m.Customer_Demand[t]
    return m.nb[t].demand == mu


# link the acquisition variables between blocks
def a1_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].a == a10
    return m.nb[t].a == m.nb[t-m.delay].r

def a2_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].a == a20
    if t - 1 == m.T.first():
         return m.nb[t].a == a21
    return m.nb[t].a <= m.I

def a3_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].a == a30
    if t - 1 == m.T.first():
         return m.nb[t].a == a31
    if t - 2 == m.T.first():
         return m.nb[t].a == a32
    return m.nb[t].a <= m.I

def a4_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].a == a40
    return m.nb[t].a <= m.I

def a5_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].a == a50
    if t - 1 == m.T.first():
         return m.nb[t].a == a51
    if t - 2 == m.T.first():
         return m.nb[t].a == a52
    if t - 3 == m.T.first():
         return m.nb[t].a == a53
    return m.nb[t].a <= m.I

def a6_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].a == a60
    if t - 1 == m.T.first():
         return m.nb[t].a == a61
    return m.nb[t].a <= m.I

def a7_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].a == a70
    if t - 1 == m.T.first():
         return m.nb[t].a == a71
    if t - 2 == m.T.first():
         return m.nb[t].a == a72
    return m.nb[t].a <= m.I

def a8_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].a == a80
    return m.nb[t].a <= m.I

def max_order_rule(m, t):
    return m.nb[t].r <= m.O

def max_inventory_rule(m,t):
    return m.nb[t].i <= m.I


def obj_rule_1(m):
    # Sum of Profit for all timeperiods
    return sum(m.nb[t].s*P1 - m.nb[t].r*C1 - m.nb[t].i*IC1 - m.nb[t].bl*BC1 -
               m.rho/2 * (m.nb[t].demand - m.z12_1[t] + m.u12_1[t])**2
               if (t-d2 < m.T.first() or m.TimeStep + m.delay2 > m.NumPeriods - 1) else
               m.nb[t].s * P1 - m.nb[t].r * C1 - m.nb[t].i * IC1 - m.nb[t].bl * BC1 -
               m.rho / 2 * (m.nb[t].demand - m.z12_1[t] + m.u12_1[t]) ** 2 -
               m.rho / 2 * (m.nb[t-m.delay2].s - m.z12_2[t] + m.u12_2[t]) ** 2
               for t in m.T)


def obj_rule_2(m):
    # Sum of Profit for all timeperiods
    obj = 0
    for t in m.T:
        obj += m.nb[t].s*P2 - m.nb[t].r*C2 - m.nb[t].i*IC2 - m.nb[t].bl*BC2
        obj -= m.rho/2 * (m.nb[t].r - m.z12_1[t] + m.u12_1[t])**2  # constraint 12 orders
        obj -= m.rho/2 * (m.nb[t].demand - m.z23_1[t] + m.u23_1[t])**2  # constraint 23 orders
        if t - m.delay >= m.T.first() and m.TimeStep + m.delay <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t].a - m.z12_2[t] + m.u12_2[t]) ** 2  # constraint 12 goods
        if t - m.delay3 >= m.T.first() and m.TimeStep + m.delay3 <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t - m.delay3].s - m.z23_2[t] + m.u23_2[t]) ** 2  # constraint 23 goods

    return obj


def obj_rule_3(m):
    # Sum of Profit for all timeperiods
    obj = 0
    for t in m.T:
        obj += m.nb[t].s * P3 - m.nb[t].r * C3 - m.nb[t].i * IC3 - m.nb[t].bl * BC3
        obj -= m.rho / 2 * (m.nb[t].r - m.z23_1[t] + m.u23_1[t]) ** 2  # constraint 23 orders
        obj -= m.rho / 2 * (m.nb[t].demand - m.z34_1[t] + m.u34_1[t]) ** 2  # constraint 34 orders
        if t - m.delay >= m.T.first() and m.TimeStep + m.delay <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t].a - m.z23_2[t] + m.u23_2[t]) ** 2  # constraint 23 goods
        if t - m.delay4 >= m.T.first() and m.TimeStep + m.delay4 <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t - m.delay4].s - m.z34_2[t] + m.u34_2[t]) ** 2  # constraint 34 goods

    return obj

def obj_rule_4(m):
    # Sum of Profit for all timeperiods
    obj = 0
    for t in m.T:
        obj += m.nb[t].s * P4 - m.nb[t].r * C4 - m.nb[t].i * IC4 - m.nb[t].bl * BC4
        obj -= m.rho / 2 * (m.nb[t].r - m.z34_1[t] + m.u34_1[t]) ** 2  # constraint 23 orders
        obj -= m.rho / 2 * (m.nb[t].demand - m.z45_1[t] + m.u45_1[t]) ** 2  # constraint 34 orders
        if t - m.delay >= m.T.first() and m.TimeStep + m.delay <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t].a - m.z34_2[t] + m.u34_2[t]) ** 2  # constraint 23 goods
        if t - m.delay5 >= m.T.first() and m.TimeStep + m.delay5 <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t - m.delay5].s - m.z45_2[t] + m.u45_2[t]) ** 2  # constraint 34 goods

    return obj

def obj_rule_5(m):
    # Sum of Profit for all timeperiods
    obj = 0
    for t in m.T:
        obj += m.nb[t].s * P5 - m.nb[t].r * C5 - m.nb[t].i * IC5 - m.nb[t].bl * BC5
        obj -= m.rho / 2 * (m.nb[t].r - m.z45_1[t] + m.u45_1[t]) ** 2  # constraint 23 orders
        obj -= m.rho / 2 * (m.nb[t].demand - m.z56_1[t] + m.u56_1[t]) ** 2  # constraint 34 orders
        if t - m.delay >= m.T.first() and m.TimeStep + m.delay <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t].a - m.z45_2[t] + m.u45_2[t]) ** 2  # constraint 23 goods
        if t - m.delay6 >= m.T.first() and m.TimeStep + m.delay6 <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t - m.delay6].s - m.z56_2[t] + m.u56_2[t]) ** 2  # constraint 34 goods

    return obj


def obj_rule_6(m):
    # Sum of Profit for all timeperiods
    obj = 0
    for t in m.T:
        obj += m.nb[t].s * P6 - m.nb[t].r * C6 - m.nb[t].i * IC6 - m.nb[t].bl * BC6
        obj -= m.rho / 2 * (m.nb[t].r - m.z56_1[t] + m.u56_1[t]) ** 2  # constraint 23 orders
        obj -= m.rho / 2 * (m.nb[t].demand - m.z67_1[t] + m.u67_1[t]) ** 2  # constraint 34 orders
        if t - m.delay >= m.T.first() and m.TimeStep + m.delay <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t].a - m.z56_2[t] + m.u56_2[t]) ** 2  # constraint 23 goods
        if t - m.delay7 >= m.T.first() and m.TimeStep + m.delay7 <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t - m.delay7].s - m.z67_2[t] + m.u67_2[t]) ** 2  # constraint 34 goods

    return obj

def obj_rule_7(m):
    # Sum of Profit for all timeperiods
    obj = 0
    for t in m.T:
        obj += m.nb[t].s * P7 - m.nb[t].r * C7 - m.nb[t].i * IC7 - m.nb[t].bl * BC7
        obj -= m.rho / 2 * (m.nb[t].r - m.z67_1[t] + m.u67_1[t]) ** 2  # constraint 23 orders
        obj -= m.rho / 2 * (m.nb[t].demand - m.z78_1[t] + m.u78_1[t]) ** 2  # constraint 34 orders
        if t - m.delay >= m.T.first() and m.TimeStep + m.delay <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t].a - m.z67_2[t] + m.u67_2[t]) ** 2  # constraint 23 goods
        if t - m.delay8 >= m.T.first() and m.TimeStep + m.delay8 <= m.NumPeriods - 1:
            obj -= m.rho / 2 * (m.nb[t - m.delay8].s - m.z78_2[t] + m.u78_2[t]) ** 2  # constraint 34 goods

    return obj


def obj_rule_8(m):
    # Sum of Profit for all timeperiods
    return sum(m.nb[t].s * P8 - m.nb[t].r * C8 - m.nb[t].i * IC8 - m.nb[t].bl * BC8 -
               m.rho/2 * (m.nb[t].r - m.z78_1[t] + m.u78_1[t])**2
               if (t - m.delay < m.T.first() or m.TimeStep + m.delay > m.NumPeriods - 1) else
               m.nb[t].s * P8 - m.nb[t].r * C8 - m.nb[t].i * IC8 - m.nb[t].bl * BC8 -
               m.rho / 2 * (m.nb[t].r - m.z78_1[t] + m.u78_1[t]) ** 2 -
               m.rho / 2 * (m.nb[t].a - m.z78_2[t] + m.u78_2[t]) ** 2
               for t in m.T)


# Get solver
solver = pyo.SolverFactory('gurobi', solver_io='python')

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

start_time = time.time()
failed_tests = list()
ft_prim_r_dict = dict()
ft_dual_r_dict = dict()


for j in range(num_tests):
    print(f"test no. {j + 1}")
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

    # Get solution results
    LP_inv = np.zeros((num_periods, num_nodes))
    LP_backlog = np.zeros((num_periods, num_nodes))
    LP_acquisition = np.zeros((num_periods, num_nodes))
    LP_shipment = np.zeros((num_periods, num_nodes))
    LP_actions = np.zeros((num_periods, num_nodes))

    # Reset boolean for skipping test
    skip_test = False
    prim_r_dict = {}
    dual_r_dict = {}
    for d in range(num_periods):

        # Get initial acquisition at each stage
        if d - d1 < 0:
            a10 = 0
        else:
            a10 = LP_shipment[d - d1, 0]

        if d - d2 < 0:
            a20 = 0
        else:
            a20 = LP_shipment[d - d2, 1]

        if d - d2 + 1 < 0:
            a21 = 0
        else:
            a21 = LP_shipment[d - d2 + 1, 1]  # Configuration specific

        if d - d3 < 0:
            a30 = 0
        else:
            a30 = LP_shipment[d - d3, 2]

        if d - d3 + 1 < 0:
            a31 = 0
        else:
            a31 = LP_shipment[d - d3 + 1, 2]  # Configuration specific

        if d - d3 + 2 < 0:
            a32 = 0
        else:
            a32 = LP_shipment[d - d3 + 2, 2]  # Configuration specific

        if d - d4 < 0:
            a40 = 0
        else:
            a40 = LP_shipment[d - d4, 3]

        if d - d5 < 0:
            a50 = 0
        else:
            a50 = LP_shipment[d - d5, 4]

        if d - d5 + 1 < 0:
            a51 = 0
        else:
            a51 = LP_shipment[d - d5 + 1, 4]  # Configuration specific

        if d - d5 + 2 < 0:
            a52 = 0
        else:
            a52 = LP_shipment[d - d5 + 2, 4]  # Configuration specific

        if d - d5 + 3 < 0:
            a53 = 0
        else:
            a53 = LP_shipment[d - d5 + 3, 4]  # Configuration specific

        if d - d6 < 0:
            a60 = 0
        else:
            a60 = LP_shipment[d - d6, 5]

        if d - d6 + 1 < 0:
            a61 = 0
        else:
            a61 = LP_shipment[d - d6 + 1, 5]  # Configuration specific

        if d - d7 < 0:
            a70 = 0
        else:
            a70 = LP_shipment[d - d7, 6]

        if d - d7 + 1 < 0:
            a71 = 0
        else:
            a71 = LP_shipment[d - d7 + 1, 6]  # Configuration specific

        if d - d7 + 2 < 0:
            a72 = 0
        else:
            a72 = LP_shipment[d - d7 + 2, 6]  # Configuration specific

        if d - d8 < 0:
            a80 = 0
        else:
            a80 = LP_shipment[d - d8, 7]

        # Node 1 Model
        model_1 = pyo.ConcreteModel()
        model_1.TimeStep = pyo.Param(default=d, mutable=False)
        model_1.NumPeriods = pyo.Param(default=num_periods, mutable=False)
        model_1.rho = pyo.Param(default=rho, mutable=True)
        model_1.I = pyo.Param(default=I1)
        model_1.O = pyo.Param(default=O1)
        model_1.I0 = pyo.Param(default=i10)
        model_1.B0 = pyo.Param(default=bl10)
        model_1.delay = pyo.Param(default=d1)
        model_1.delay2 = pyo.Param(default=d2)
        model_1.T = pyo.RangeSet(d, num_periods - 1)

        model_1.nb = pyo.Block(model_1.T, rule=node_block_rule)
        model_1.i_linking = pyo.Constraint(model_1.T, rule=i_linking_rule)
        model_1.bl_linking = pyo.Constraint(model_1.T, rule=bl_linking_rule)
        model_1.a_linking = pyo.Constraint(model_1.T, rule=a1_linking_rule)
        model_1.max_order = pyo.Constraint(model_1.T, rule=max_order_rule)
        model_1.max_inventory = pyo.Constraint(model_1.T, rule=max_inventory_rule)

        # Global variables between nodes 1 and 2
        model_1.z12_1 = pyo.Param(model_1.T, initialize=mu, mutable=True)
        model_1.z12_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_1.u12_1 = pyo.Param(model_1.T, initialize=0, mutable=True)

        if d + d2 <= num_periods - 1:
            model_1.T12 = pyo.RangeSet(d + d2, num_periods - 1)
            model_1.z12_2 = pyo.Param(model_1.T12, initialize=mu, mutable=True)
            model_1.z12_2[d + d2] = LP_Customer_Demand[j][0][d]  # <---
            model_1.u12_2 = pyo.Param(model_1.T12, initialize=0, mutable=True)

        model_1.obj = pyo.Objective(rule=obj_rule_1, sense=pyo.maximize)

        # Node 2 Model
        model_2 = pyo.ConcreteModel()
        model_2.TimeStep = pyo.Param(default=d, mutable=False)
        model_2.NumPeriods = pyo.Param(default=num_periods, mutable=False)
        model_2.rho = pyo.Param(default=rho, mutable=True)
        model_2.I = pyo.Param(default=I2)
        model_2.O = pyo.Param(default=O2)
        model_2.I0 = pyo.Param(default=i20)
        model_2.B0 = pyo.Param(default=bl20)
        model_2.delay = pyo.Param(default=d2)
        model_2.delay3 = pyo.Param(default=d3)
        model_2.T = pyo.RangeSet(d, num_periods - 1)

        model_2.nb = pyo.Block(model_2.T, rule=node_block_rule)
        model_2.i_linking = pyo.Constraint(model_2.T, rule=i_linking_rule)
        model_2.bl_linking = pyo.Constraint(model_1.T, rule=bl_linking_rule)
        model_2.a_linking = pyo.Constraint(model_2.T, rule=a2_linking_rule)
        model_2.max_order = pyo.Constraint(model_2.T, rule=max_order_rule)
        model_2.max_inventory = pyo.Constraint(model_2.T, rule=max_inventory_rule)

        # Global variables between nodes 1 and 2
        model_2.z12_1 = pyo.Param(model_2.T, initialize=mu, mutable=True)
        model_2.z12_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_2.u12_1 = pyo.Param(model_2.T, initialize=0, mutable=True)

        if d + d2 <= num_periods - 1:
            model_2.T12 = pyo.RangeSet(d + d2, num_periods - 1)
            model_2.z12_2 = pyo.Param(model_2.T12, initialize=mu, mutable=True)
            model_2.z12_2[d + d2] = LP_Customer_Demand[j][0][d]  # <---
            model_2.u12_2 = pyo.Param(model_2.T12, initialize=0, mutable=True)

        # Global variables between nodes 2 and 3
        model_2.z23_1 = pyo.Param(model_2.T, initialize=mu, mutable=True)
        model_2.z23_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_2.u23_1 = pyo.Param(model_2.T, initialize=0, mutable=True)

        if d + d3 <= num_periods - 1:
            model_2.T23 = pyo.RangeSet(d + d3, num_periods - 1)
            model_2.z23_2 = pyo.Param(model_2.T23, initialize=mu, mutable=True)
            model_2.z23_2[d + d3] = LP_Customer_Demand[j][0][d]  # <---
            model_2.u23_2 = pyo.Param(model_2.T23, initialize=0, mutable=True)

        model_2.obj = pyo.Objective(rule=obj_rule_2, sense=pyo.maximize)

        # Node 3 Model
        model_3 = pyo.ConcreteModel()
        model_3.TimeStep = pyo.Param(default=d, mutable=False)
        model_3.NumPeriods = pyo.Param(default=num_periods, mutable=False)
        model_3.rho = pyo.Param(default=rho, mutable=True)
        model_3.I = pyo.Param(default=I3)
        model_3.O = pyo.Param(default=O3)
        model_3.I0 = pyo.Param(default=i30)
        model_3.B0 = pyo.Param(default=bl30)
        model_3.delay = pyo.Param(default=d3)
        model_3.delay4 = pyo.Param(default=d4)
        model_3.T = pyo.RangeSet(d, num_periods - 1)


        model_3.nb = pyo.Block(model_3.T, rule=node_block_rule)
        model_3.i_linking = pyo.Constraint(model_3.T, rule=i_linking_rule)
        model_3.bl_linking = pyo.Constraint(model_3.T, rule=bl_linking_rule)
        model_3.a_linking = pyo.Constraint(model_3.T, rule=a3_linking_rule)
        model_3.max_order = pyo.Constraint(model_3.T, rule=max_order_rule)
        model_3.max_inventory = pyo.Constraint(model_3.T, rule=max_inventory_rule)

        # Global variables between nodes 2 and 3
        model_3.z23_1 = pyo.Param(model_3.T, initialize=mu, mutable=True)
        model_3.z23_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_3.u23_1 = pyo.Param(model_3.T, initialize=0, mutable=True)

        if d + d3 <= num_periods - 1:
            model_3.T23 = pyo.RangeSet(d + d3, num_periods - 1)
            model_3.z23_2 = pyo.Param(model_3.T23, initialize=mu, mutable=True)
            model_3.z23_2[d + d3] = LP_Customer_Demand[j][0][d]  # <---
            model_3.u23_2 = pyo.Param(model_3.T23, initialize=0, mutable=True)

        # Global variables between nodes 3 and 4
        model_3.z34_1 = pyo.Param(model_3.T, initialize=mu, mutable=True)
        model_3.z34_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_3.u34_1 = pyo.Param(model_3.T, initialize=0, mutable=True)

        if d + d4 <= num_periods - 1:
            model_3.T34 = pyo.RangeSet(d + d4, num_periods - 1)
            model_3.z34_2 = pyo.Param(model_3.T34, initialize=mu, mutable=True)
            model_3.z34_2[d + d4] = LP_Customer_Demand[j][0][d]  # <---
            model_3.u34_2 = pyo.Param(model_3.T34, initialize=0, mutable=True)

        model_3.obj = pyo.Objective(rule=obj_rule_3, sense=pyo.maximize)

        # Node 4 Model
        model_4 = pyo.ConcreteModel()
        model_4.TimeStep = pyo.Param(default=d, mutable=False)
        model_4.NumPeriods = pyo.Param(default=num_periods, mutable=False)
        model_4.rho = pyo.Param(default=rho, mutable=True)
        model_4.I = pyo.Param(default=I4)
        model_4.O = pyo.Param(default=O4)
        model_4.I0 = pyo.Param(default=i40)
        model_4.B0 = pyo.Param(default=bl40)
        model_4.delay = pyo.Param(default=d4)
        model_4.delay5 = pyo.Param(default=d5)
        model_4.T = pyo.RangeSet(d, num_periods - 1)

        model_4.nb = pyo.Block(model_4.T, rule=node_block_rule)
        model_4.i_linking = pyo.Constraint(model_4.T, rule=i_linking_rule)
        model_4.bl_linking = pyo.Constraint(model_4.T, rule=bl_linking_rule)
        model_4.a_linking = pyo.Constraint(model_4.T, rule=a4_linking_rule)
        model_4.max_order = pyo.Constraint(model_4.T, rule=max_order_rule)
        model_4.max_inventory = pyo.Constraint(model_4.T, rule=max_inventory_rule)

        # Global variables between nodes 3 and 4
        model_4.z34_1 = pyo.Param(model_4.T, initialize=mu, mutable=True)
        model_4.z34_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_4.u34_1 = pyo.Param(model_4.T, initialize=0, mutable=True)

        if d + d4 <= num_periods - 1:
            model_4.T34 = pyo.RangeSet(d + d4, num_periods - 1)
            model_4.z34_2 = pyo.Param(model_4.T34, initialize=mu, mutable=True)
            model_4.z34_2[d + d4] = LP_Customer_Demand[j][0][d]  # <---
            model_4.u34_2 = pyo.Param(model_4.T34, initialize=0, mutable=True)

        # Global variables between nodes 4 and 5
        model_4.z45_1 = pyo.Param(model_4.T, initialize=mu, mutable=True)
        model_4.z45_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_4.u45_1 = pyo.Param(model_4.T, initialize=0, mutable=True)

        if d + d5 <= num_periods - 1:
            model_4.T45 = pyo.RangeSet(d + d5, num_periods - 1)
            model_4.z45_2 = pyo.Param(model_4.T45, initialize=mu, mutable=True)
            model_4.z45_2[d + d5] = LP_Customer_Demand[j][0][d]  # <---
            model_4.u45_2 = pyo.Param(model_4.T45, initialize=0, mutable=True)

        model_4.obj = pyo.Objective(rule=obj_rule_4, sense=pyo.maximize)

        # Node 5 Model
        model_5 = pyo.ConcreteModel()
        model_5.TimeStep = pyo.Param(default=d, mutable=False)
        model_5.NumPeriods = pyo.Param(default=num_periods, mutable=False)
        model_5.rho = pyo.Param(default=rho, mutable=True)
        model_5.I = pyo.Param(default=I5)
        model_5.O = pyo.Param(default=O5)
        model_5.I0 = pyo.Param(default=i50)
        model_5.B0 = pyo.Param(default=bl50)
        model_5.delay = pyo.Param(default=d5)
        model_5.delay6 = pyo.Param(default=d6)
        model_5.T = pyo.RangeSet(d, num_periods - 1)

        model_5.nb = pyo.Block(model_5.T, rule=node_block_rule)
        model_5.i_linking = pyo.Constraint(model_5.T, rule=i_linking_rule)
        model_5.bl_linking = pyo.Constraint(model_5.T, rule=bl_linking_rule)
        model_5.a_linking = pyo.Constraint(model_5.T, rule=a5_linking_rule)
        model_5.max_order = pyo.Constraint(model_5.T, rule=max_order_rule)
        model_5.max_inventory = pyo.Constraint(model_5.T, rule=max_inventory_rule)

        # Global variables between nodes 4 and 5
        model_5.z45_1 = pyo.Param(model_5.T, initialize=mu, mutable=True)
        model_5.z45_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_5.u45_1 = pyo.Param(model_5.T, initialize=0, mutable=True)

        if d + d5 <= num_periods - 1:
            model_5.T45 = pyo.RangeSet(d + d5, num_periods - 1)
            model_5.z45_2 = pyo.Param(model_5.T45, initialize=mu, mutable=True)
            model_5.z45_2[d + d5] = LP_Customer_Demand[j][0][d]  # <---
            model_5.u45_2 = pyo.Param(model_5.T45, initialize=0, mutable=True)

        # Global variables between nodes 5 and 6
        model_5.z56_1 = pyo.Param(model_5.T, initialize=mu, mutable=True)
        model_5.z56_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_5.u56_1 = pyo.Param(model_5.T, initialize=0, mutable=True)

        if d + d6 <= num_periods - 1:
            model_5.T56 = pyo.RangeSet(d + d6, num_periods - 1)
            model_5.z56_2 = pyo.Param(model_5.T56, initialize=mu, mutable=True)
            model_5.z56_2[d + d6] = LP_Customer_Demand[j][0][d]  # <---
            model_5.u56_2 = pyo.Param(model_5.T56, initialize=0, mutable=True)

        model_5.obj = pyo.Objective(rule=obj_rule_5, sense=pyo.maximize)

        # Node 6 Model
        model_6 = pyo.ConcreteModel()
        model_6.TimeStep = pyo.Param(default=d, mutable=False)
        model_6.NumPeriods = pyo.Param(default=num_periods, mutable=False)
        model_6.rho = pyo.Param(default=rho, mutable=True)
        model_6.I = pyo.Param(default=I6)
        model_6.O = pyo.Param(default=O6)
        model_6.I0 = pyo.Param(default=i60)
        model_6.B0 = pyo.Param(default=bl60)
        model_6.delay = pyo.Param(default=d6)
        model_6.delay7 = pyo.Param(default=d7)
        model_6.T = pyo.RangeSet(d, num_periods - 1)

        model_6.nb = pyo.Block(model_6.T, rule=node_block_rule)
        model_6.i_linking = pyo.Constraint(model_6.T, rule=i_linking_rule)
        model_6.bl_linking = pyo.Constraint(model_6.T, rule=bl_linking_rule)
        model_6.a_linking = pyo.Constraint(model_6.T, rule=a6_linking_rule)
        model_6.max_order = pyo.Constraint(model_6.T, rule=max_order_rule)
        model_6.max_inventory = pyo.Constraint(model_6.T, rule=max_inventory_rule)

        # Global variables between nodes 5 and 6
        model_6.z56_1 = pyo.Param(model_6.T, initialize=mu, mutable=True)
        model_6.z56_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_6.u56_1 = pyo.Param(model_6.T, initialize=0, mutable=True)

        if d + d6 <= num_periods - 1:
            model_6.T56 = pyo.RangeSet(d + d6, num_periods - 1)
            model_6.z56_2 = pyo.Param(model_6.T56, initialize=mu, mutable=True)
            model_6.z56_2[d + d6] = LP_Customer_Demand[j][0][d]  # <---
            model_6.u56_2 = pyo.Param(model_6.T56, initialize=0, mutable=True)

        # Global variables between nodes 6 and 7
        model_6.z67_1 = pyo.Param(model_6.T, initialize=mu, mutable=True)
        model_6.z67_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_6.u67_1 = pyo.Param(model_6.T, initialize=0, mutable=True)

        if d + d7 <= num_periods - 1:
            model_6.T67 = pyo.RangeSet(d + d7, num_periods - 1)
            model_6.z67_2 = pyo.Param(model_6.T67, initialize=mu, mutable=True)
            model_6.z67_2[d + d7] = LP_Customer_Demand[j][0][d]  # <---
            model_6.u67_2 = pyo.Param(model_6.T67, initialize=0, mutable=True)

        model_6.obj = pyo.Objective(rule=obj_rule_6, sense=pyo.maximize)

        # Node 7 Model
        model_7 = pyo.ConcreteModel()
        model_7.TimeStep = pyo.Param(default=d, mutable=False)
        model_7.NumPeriods = pyo.Param(default=num_periods, mutable=False)
        model_7.rho = pyo.Param(default=rho, mutable=True)
        model_7.I = pyo.Param(default=I7)
        model_7.O = pyo.Param(default=O7)
        model_7.I0 = pyo.Param(default=i70)
        model_7.B0 = pyo.Param(default=bl70)
        model_7.delay = pyo.Param(default=d7)
        model_7.delay8 = pyo.Param(default=d8)
        model_7.T = pyo.RangeSet(d, num_periods - 1)

        model_7.nb = pyo.Block(model_7.T, rule=node_block_rule)
        model_7.i_linking = pyo.Constraint(model_7.T, rule=i_linking_rule)
        model_7.bl_linking = pyo.Constraint(model_7.T, rule=bl_linking_rule)
        model_7.a_linking = pyo.Constraint(model_7.T, rule=a7_linking_rule)
        model_7.max_order = pyo.Constraint(model_7.T, rule=max_order_rule)
        model_7.max_inventory = pyo.Constraint(model_7.T, rule=max_inventory_rule)

        # Global variables between nodes 6 and 7
        model_7.z67_1 = pyo.Param(model_7.T, initialize=mu, mutable=True)
        model_7.z67_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_7.u67_1 = pyo.Param(model_7.T, initialize=0, mutable=True)

        if d + d7 <= num_periods - 1:
            model_7.T67 = pyo.RangeSet(d + d7, num_periods - 1)
            model_7.z67_2 = pyo.Param(model_7.T67, initialize=mu, mutable=True)
            model_7.z67_2[d + d7] = LP_Customer_Demand[j][0][d]  # <---
            model_7.u67_2 = pyo.Param(model_7.T67, initialize=0, mutable=True)

        # Global variables between nodes 7 and 8
        model_7.z78_1 = pyo.Param(model_7.T, initialize=mu, mutable=True)
        model_7.z78_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_7.u78_1 = pyo.Param(model_7.T, initialize=0, mutable=True)

        if d + d8 <= num_periods - 1:
            model_7.T78 = pyo.RangeSet(d + d8, num_periods - 1)
            model_7.z78_2 = pyo.Param(model_7.T78, initialize=mu, mutable=True)
            model_7.z78_2[d + d8] = LP_Customer_Demand[j][0][d]  # <---
            model_7.u78_2 = pyo.Param(model_7.T78, initialize=0, mutable=True)

        model_7.obj = pyo.Objective(rule=obj_rule_7, sense=pyo.maximize)

        # Node 8 Model
        model_8 = pyo.ConcreteModel()
        model_8.TimeStep = pyo.Param(default=d, mutable=False)
        model_8.NumPeriods = pyo.Param(default=num_periods, mutable=False)
        model_8.rho = pyo.Param(default=rho, mutable=True)
        model_8.I = pyo.Param(default=I8)
        model_8.O = pyo.Param(default=O8)
        model_8.I0 = pyo.Param(default=i80)
        model_8.B0 = pyo.Param(default=bl80)
        model_8.delay = pyo.Param(default=d8)
        model_8.T = pyo.RangeSet(d, num_periods - 1)

        Customer_Demand8 = {t: LP_Customer_Demand[j][0][t] for t in model_8.T}
        model_8.Customer_Demand = pyo.Param(model_8.T, default=Customer_Demand8, mutable=False)

        model_8.nb = pyo.Block(model_8.T, rule=node_block_rule)
        model_8.i_linking = pyo.Constraint(model_8.T, rule=i_linking_rule)
        model_8.bl_linking = pyo.Constraint(model_8.T, rule=bl_linking_rule)
        model_8.d_linking = pyo.Constraint(model_8.T, rule=demand_linking_rule)
        model_8.a_linking = pyo.Constraint(model_8.T, rule=a8_linking_rule)
        model_8.max_order = pyo.Constraint(model_8.T, rule=max_order_rule)
        model_8.max_inventory = pyo.Constraint(model_8.T, rule=max_inventory_rule)

        model_8.z78_1 = pyo.Param(model_8.T, initialize=mu, mutable=True)
        model_8.z78_1[d] = LP_Customer_Demand[j][0][d]  # <---
        model_8.u78_1 = pyo.Param(model_8.T, initialize=0, mutable=True)

        if d + d8 <= num_periods - 1:
            model_8.T78 = pyo.RangeSet(d + d8, num_periods - 1)
            model_8.z78_2 = pyo.Param(model_8.T78, initialize=mu, mutable=True)
            model_8.z78_2[d + d8] = LP_Customer_Demand[j][0][d]  # <---
            model_8.u78_2 = pyo.Param(model_8.T78, initialize=0, mutable=True)

        model_8.obj = pyo.Objective(rule=obj_rule_8, sense=pyo.maximize)


        # Iterations
        prim_r = []
        dual_r = []
        for i in range(N_iter):

            # set rho
            if use_variable_rho:
                model_1.rho = act_rho[i]
                model_2.rho = act_rho[i]
                model_3.rho = act_rho[i]
                model_4.rho = act_rho[i]
                model_5.rho = act_rho[i]
                model_6.rho = act_rho[i]
                model_7.rho = act_rho[i]
                model_8.rho = act_rho[i]

            # Solve Sub-problems
            sub_problem1 = solver.solve(model_1)
            sub_problem2 = solver.solve(model_2)
            sub_problem3 = solver.solve(model_3)
            sub_problem4 = solver.solve(model_4)
            sub_problem5 = solver.solve(model_5)
            sub_problem6 = solver.solve(model_6)
            sub_problem7 = solver.solve(model_7)
            sub_problem8 = solver.solve(model_8)

            # check if optimal solution found
            solution = [sub_problem1.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded,
                        sub_problem2.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded,
                        sub_problem3.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded,
                        sub_problem4.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded,
                        sub_problem5.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded,
                        sub_problem6.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded,
                        sub_problem7.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded,
                        sub_problem8.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded]

            if any(solution):
                skip_test = True
                failed_tests.append(j)
                break


            old_z12_1 = np.array([pyo.value(model_1.z12_1[t]) for t in model_1.T])
            old_z23_1 = np.array([pyo.value(model_2.z23_1[t]) for t in model_2.T])
            old_z34_1 = np.array([pyo.value(model_3.z34_1[t]) for t in model_3.T])
            old_z45_1 = np.array([pyo.value(model_4.z45_1[t]) for t in model_4.T])
            old_z56_1 = np.array([pyo.value(model_5.z56_1[t]) for t in model_5.T])
            old_z67_1 = np.array([pyo.value(model_6.z67_1[t]) for t in model_6.T])
            old_z78_1 = np.array([pyo.value(model_7.z78_1[t]) for t in model_7.T])
            if d + d2 <= num_periods - 1:
                old_z12_2 = np.array([pyo.value(model_1.z12_2[t]) for t in model_1.T12])
            if d + d3 <= num_periods - 1:
                old_z23_2 = np.array([pyo.value(model_2.z23_2[t]) for t in model_2.T23])
            if d + d4 <= num_periods - 1:
                old_z34_2 = np.array([pyo.value(model_3.z34_2[t]) for t in model_3.T34])
            if d + d5 <= num_periods - 1:
                old_z45_2 = np.array([pyo.value(model_4.z45_2[t]) for t in model_4.T45])
            if d + d6 <= num_periods - 1:
                old_z56_2 = np.array([pyo.value(model_5.z56_2[t]) for t in model_5.T56])
            if d + d7 <= num_periods - 1:
                old_z67_2 = np.array([pyo.value(model_6.z67_2[t]) for t in model_6.T67])
            if d + d8 <= num_periods - 1:
                old_z78_2 = np.array([pyo.value(model_7.z78_2[t]) for t in model_7.T78])


            # calculate new z
            # 12 Global variables
            reorder_2 = [pyo.value(model_2.nb[t].r)for t in model_2.T]
            demand_1 = [pyo.value(model_1.nb[t].demand) for t in model_1.T]
            z12_1_arr = (np.array(reorder_2) + np.array(demand_1)) / 2
            z12_1 = {t: z12_1_arr[i] for i, t in enumerate(model_1.T)}

            if d + d2 <= num_periods - 1:
                shipping_1 = [pyo.value(model_1.nb[t - d2].s) for t in model_1.T12]
                acquisition_2 = [pyo.value(model_2.nb[t].a) for t in model_2.T12]
                z12_2_arr = (np.array(shipping_1) + np.array(acquisition_2)) / 2
                z12_2 = {t: z12_2_arr[i] for i, t in enumerate(model_1.T12)}


            # 23 Global variables
            reorder_3 = [pyo.value(model_3.nb[t].r) for t in model_3.T]
            demand_23 = [pyo.value(model_2.nb[t].demand) for t in model_2.T]
            z23_1_arr = (np.array(reorder_3) + np.array(demand_23)) / 2
            z23_1 = {t: z23_1_arr[i] for i, t in enumerate(model_3.T)}

            if d + d3 <= num_periods - 1:
                shipping_23 = [pyo.value(model_2.nb[t - d3].s) for t in model_2.T23]
                acquisition_3 = [pyo.value(model_3.nb[t].a) for t in model_3.T23]
                z23_2_arr = (np.array(shipping_23) + np.array(acquisition_3)) / 2
                z23_2 = {t: z23_2_arr[i] for i, t in enumerate(model_3.T23)}

            # 34 Global variables
            reorder_4 = [pyo.value(model_4.nb[t].r) for t in model_4.T]
            demand_34 = [pyo.value(model_3.nb[t].demand) for t in model_3.T]
            z34_1_arr = (np.array(reorder_4) + np.array(demand_34)) / 2
            z34_1 = {t: z34_1_arr[i] for i, t in enumerate(model_4.T)}

            if d + d4 <= num_periods - 1:
                shipping_34 = [pyo.value(model_3.nb[t - d4].s) for t in model_3.T34]
                acquisition_4 = [pyo.value(model_4.nb[t].a) for t in model_4.T34]
                z34_2_arr = (np.array(shipping_34) + np.array(acquisition_4)) / 2
                z34_2 = {t: z34_2_arr[i] for i, t in enumerate(model_4.T34)}

            # 45 Global variables
            reorder_5 = [pyo.value(model_5.nb[t].r) for t in model_5.T]
            demand_45 = [pyo.value(model_4.nb[t].demand) for t in model_4.T]
            z45_1_arr = (np.array(reorder_5) + np.array(demand_45)) / 2
            z45_1 = {t: z45_1_arr[i] for i, t in enumerate(model_5.T)}

            if d + d5 <= num_periods - 1:
                shipping_45 = [pyo.value(model_4.nb[t - d5].s) for t in model_4.T45]
                acquisition_5 = [pyo.value(model_5.nb[t].a) for t in model_5.T45]
                z45_2_arr = (np.array(shipping_45) + np.array(acquisition_5)) / 2
                z45_2 = {t: z45_2_arr[i] for i, t in enumerate(model_5.T45)}

            # 56 Global variables
            reorder_6 = [pyo.value(model_6.nb[t].r) for t in model_6.T]
            demand_56 = [pyo.value(model_5.nb[t].demand) for t in model_5.T]
            z56_1_arr = (np.array(reorder_6) + np.array(demand_56)) / 2
            z56_1 = {t: z56_1_arr[i] for i, t in enumerate(model_6.T)}

            if d + d6 <= num_periods - 1:
                shipping_56 = [pyo.value(model_5.nb[t - d6].s) for t in model_5.T56]
                acquisition_6 = [pyo.value(model_6.nb[t].a) for t in model_6.T56]
                z56_2_arr = (np.array(shipping_56) + np.array(acquisition_6)) / 2
                z56_2 = {t: z56_2_arr[i] for i, t in enumerate(model_6.T56)}

            # 67 Global variables
            reorder_7 = [pyo.value(model_7.nb[t].r) for t in model_7.T]
            demand_67 = [pyo.value(model_6.nb[t].demand) for t in model_6.T]
            z67_1_arr = (np.array(reorder_7) + np.array(demand_67)) / 2
            z67_1 = {t: z67_1_arr[i] for i, t in enumerate(model_7.T)}

            if d + d7 <= num_periods - 1:
                shipping_67 = [pyo.value(model_6.nb[t - d7].s) for t in model_6.T67]
                acquisition_7 = [pyo.value(model_7.nb[t].a) for t in model_7.T67]
                z67_2_arr = (np.array(shipping_67) + np.array(acquisition_7)) / 2
                z67_2 = {t: z67_2_arr[i] for i, t in enumerate(model_7.T67)}

            # 78 Global variables
            reorder_8 = [pyo.value(model_8.nb[t].r) for t in model_8.T]
            demand_78 = [pyo.value(model_7.nb[t].demand) for t in model_7.T]
            z78_1_arr = (np.array(reorder_8) + np.array(demand_78)) / 2
            z78_1 = {t: z78_1_arr[i] for i, t in enumerate(model_8.T)}

            if d + d8 <= num_periods - 1:
                shipping_78 = [pyo.value(model_7.nb[t - d8].s) for t in model_7.T78]
                acquisition_8 = [pyo.value(model_8.nb[t].a) for t in model_8.T78]
                z78_2_arr = (np.array(shipping_78) + np.array(acquisition_8)) / 2
                z78_2 = {t: z78_2_arr[i] for i, t in enumerate(model_8.T78)}

            # update z
            for t in model_1.T:
                model_1.z12_1[t] = z12_1[t]
                model_2.z12_1[t] = z12_1[t]
                model_2.z23_1[t] = z23_1[t]
                model_3.z23_1[t] = z23_1[t]
                model_3.z34_1[t] = z34_1[t]
                model_4.z34_1[t] = z34_1[t]
                model_4.z45_1[t] = z45_1[t]
                model_5.z45_1[t] = z45_1[t]
                model_5.z56_1[t] = z56_1[t]
                model_6.z56_1[t] = z56_1[t]
                model_6.z67_1[t] = z67_1[t]
                model_7.z67_1[t] = z67_1[t]
                model_7.z78_1[t] = z78_1[t]
                model_8.z78_1[t] = z78_1[t]
            if d + d2 <= num_periods - 1:
                for t in model_1.T12:
                    model_1.z12_2[t] = z12_2[t]
                    model_2.z12_2[t] = z12_2[t]
            if d + d3 <= num_periods - 1:
                for t in model_2.T23:
                    model_2.z23_2[t] = z23_2[t]
                    model_3.z23_2[t] = z23_2[t]
            if d + d4 <= num_periods - 1:
                for t in model_3.T34:
                    model_3.z34_2[t] = z34_2[t]
                    model_4.z34_2[t] = z34_2[t]
            if d + d5 <= num_periods - 1:
                for t in model_4.T45:
                    model_4.z45_2[t] = z45_2[t]
                    model_5.z45_2[t] = z45_2[t]
            if d + d6 <= num_periods - 1:
                for t in model_5.T56:
                    model_5.z56_2[t] = z56_2[t]
                    model_6.z56_2[t] = z56_2[t]
            if d + d7 <= num_periods - 1:
                for t in model_6.T67:
                    model_6.z67_2[t] = z67_2[t]
                    model_7.z67_2[t] = z67_2[t]
            if d + d8 <= num_periods - 1:
                for t in model_7.T78:
                    model_7.z78_2[t] = z78_2[t]
                    model_8.z78_2[t] = z78_2[t]

            # calculate new u
            # 12
            u12_1_model1 = {t: pyo.value(model_1.u12_1[t]) + pyo.value(model_1.nb[t].demand - z12_1[t])
                            for t in model_1.T}
            u12_1_model2 = {t: pyo.value(model_2.u12_1[t]) + pyo.value(model_2.nb[t].r - z12_1[t])
                            for t in model_2.T}

            if d + d2 <= num_periods - 1:
                u12_2_model1 = {t: pyo.value(model_1.u12_2[t]) + pyo.value(model_1.nb[t - d2].s - z12_2[t])
                                for t in model_1.T12}
                u12_2_model2 = {t: pyo.value(model_2.u12_2[t]) + pyo.value(model_2.nb[t].a - z12_2[t])
                                for t in model_2.T12}

            # 23
            u23_1_model2 = {t: pyo.value(model_2.u23_1[t]) + pyo.value(model_2.nb[t].demand - z23_1[t])
                            for t in model_2.T}
            u23_1_model3 = {t: pyo.value(model_3.u23_1[t]) + pyo.value(model_3.nb[t].r - z23_1[t])
                            for t in model_3.T}

            if d + d3 <= num_periods - 1:
                u23_2_model2 = {t: pyo.value(model_2.u23_2[t]) + pyo.value(model_2.nb[t - d3].s - z23_2[t])
                                for t in model_2.T23}
                u23_2_model3 = {t: pyo.value(model_3.u23_2[t]) + pyo.value(model_3.nb[t].a - z23_2[t])
                                for t in model_3.T23}

            # 34
            u34_1_model3 = {t: pyo.value(model_3.u34_1[t]) + pyo.value(model_3.nb[t].demand - z34_1[t])
                            for t in model_3.T}
            u34_1_model4 = {t: pyo.value(model_4.u34_1[t]) + pyo.value(model_4.nb[t].r - z34_1[t])
                            for t in model_4.T}

            if d + d4 <= num_periods - 1:
                u34_2_model3 = {t: pyo.value(model_3.u34_2[t]) + pyo.value(model_3.nb[t - d4].s - z34_2[t])
                                for t in model_3.T34}
                u34_2_model4 = {t: pyo.value(model_4.u34_2[t]) + pyo.value(model_4.nb[t].a - z34_2[t])
                                for t in model_4.T34}

            # 45
            u45_1_model4 = {t: pyo.value(model_4.u45_1[t]) + pyo.value(model_4.nb[t].demand - z45_1[t])
                            for t in model_4.T}
            u45_1_model5 = {t: pyo.value(model_5.u45_1[t]) + pyo.value(model_5.nb[t].r - z45_1[t])
                            for t in model_5.T}

            if d + d5 <= num_periods - 1:
                u45_2_model4 = {t: pyo.value(model_4.u45_2[t]) + pyo.value(model_4.nb[t - d5].s - z45_2[t])
                                for t in model_4.T45}
                u45_2_model5 = {t: pyo.value(model_5.u45_2[t]) + pyo.value(model_5.nb[t].a - z45_2[t])
                                for t in model_5.T45}

            # 56
            u56_1_model5 = {t: pyo.value(model_5.u56_1[t]) + pyo.value(model_5.nb[t].demand - z56_1[t])
                            for t in model_5.T}
            u56_1_model6 = {t: pyo.value(model_6.u56_1[t]) + pyo.value(model_6.nb[t].r - z56_1[t])
                            for t in model_6.T}

            if d + d6 <= num_periods - 1:
                u56_2_model5 = {t: pyo.value(model_5.u56_2[t]) + pyo.value(model_5.nb[t - d6].s - z56_2[t])
                                for t in model_5.T56}
                u56_2_model6 = {t: pyo.value(model_6.u56_2[t]) + pyo.value(model_6.nb[t].a - z56_2[t])
                                for t in model_6.T56}

            # 67
            u67_1_model6 = {t: pyo.value(model_6.u67_1[t]) + pyo.value(model_6.nb[t].demand - z67_1[t])
                            for t in model_6.T}
            u67_1_model7 = {t: pyo.value(model_7.u67_1[t]) + pyo.value(model_7.nb[t].r - z67_1[t])
                            for t in model_7.T}

            if d + d7 <= num_periods - 1:
                u67_2_model6 = {t: pyo.value(model_6.u67_2[t]) + pyo.value(model_6.nb[t - d7].s - z67_2[t])
                                for t in model_6.T67}
                u67_2_model7 = {t: pyo.value(model_7.u67_2[t]) + pyo.value(model_7.nb[t].a - z67_2[t])
                                for t in model_7.T67}

            # 78
            u78_1_model7 = {t: pyo.value(model_7.u78_1[t]) + pyo.value(model_7.nb[t].demand - z78_1[t])
                            for t in model_7.T}
            u78_1_model8 = {t: pyo.value(model_8.u78_1[t]) + pyo.value(model_8.nb[t].r - z78_1[t])
                            for t in model_8.T}

            if d + d8 <= num_periods - 1:
                u78_2_model7 = {t: pyo.value(model_7.u78_2[t]) + pyo.value(model_7.nb[t - d8].s - z78_2[t])
                                for t in model_7.T78}
                u78_2_model8 = {t: pyo.value(model_8.u78_2[t]) + pyo.value(model_8.nb[t].a - z78_2[t])
                                for t in model_8.T78}

            # update u
            if ADMM:
                for t in model_1.T:
                    model_1.u12_1[t] = u12_1_model1[t]
                    model_2.u12_1[t] = u12_1_model2[t]
                    model_2.u23_1[t] = u23_1_model2[t]
                    model_3.u23_1[t] = u23_1_model3[t]
                    model_3.u34_1[t] = u34_1_model3[t]
                    model_4.u34_1[t] = u34_1_model4[t]
                    model_4.u45_1[t] = u45_1_model4[t]
                    model_5.u45_1[t] = u45_1_model5[t]
                    model_5.u56_1[t] = u56_1_model5[t]
                    model_6.u56_1[t] = u56_1_model6[t]
                    model_6.u67_1[t] = u67_1_model6[t]
                    model_7.u67_1[t] = u67_1_model7[t]
                    model_7.u78_1[t] = u78_1_model7[t]
                    model_8.u78_1[t] = u78_1_model8[t]
                if d + d2 <= num_periods - 1:
                    for t in model_1.T12:
                        model_1.u12_2[t] = u12_2_model1[t]
                        model_2.u12_2[t] = u12_2_model2[t]
                if d + d3 <= num_periods - 1:
                    for t in model_2.T23:
                        model_2.u23_2[t] = u23_2_model2[t]
                        model_3.u23_2[t] = u23_2_model3[t]
                if d + d4 <= num_periods - 1:
                    for t in model_3.T34:
                        model_3.u34_2[t] = u34_2_model3[t]
                        model_4.u34_2[t] = u34_2_model4[t]
                if d + d5 <= num_periods - 1:
                    for t in model_4.T45:
                        model_4.u45_2[t] = u45_2_model4[t]
                        model_5.u45_2[t] = u45_2_model5[t]
                if d + d6 <= num_periods - 1:
                    for t in model_5.T56:
                        model_5.u56_2[t] = u56_2_model5[t]
                        model_6.u56_2[t] = u56_2_model6[t]
                if d + d7 <= num_periods - 1:
                    for t in model_6.T67:
                        model_6.u67_2[t] = u67_2_model6[t]
                        model_7.u67_2[t] = u67_2_model7[t]
                if d + d8 <= num_periods - 1:
                    for t in model_7.T78:
                        model_7.u78_2[t] = u78_2_model7[t]
                        model_8.u78_2[t] = u78_2_model8[t]

            # residual calculation
            old_z = np.concatenate((old_z12_1, old_z23_1, old_z34_1, old_z45_1, old_z56_1, old_z67_1, old_z78_1))
            z = np.concatenate((z12_1_arr, z23_1_arr, z34_1_arr, z45_1_arr, z56_1_arr, z67_1_arr, z78_1_arr))

            prim_r_z = []
            prim_r_z12_1 = [
                np.linalg.norm([np.array(reorder_2)[i] - z12_1_arr[i], np.array(demand_1)[i] - z12_1_arr[i]]) for i in
                range(len(z12_1_arr))]
            prim_r_z.append(prim_r_z12_1)

            prim_r_z23_1 = [
                np.linalg.norm([np.array(reorder_3)[i] - z23_1_arr[i], np.array(demand_23)[i] - z23_1_arr[i]]) for i in
                range(len(z23_1_arr))]
            prim_r_z.append(prim_r_z23_1)

            prim_r_z34_1 = [
                np.linalg.norm([np.array(reorder_4)[i] - z34_1_arr[i], np.array(demand_34)[i] - z34_1_arr[i]]) for i in
                range(len(z34_1_arr))]
            prim_r_z.append(prim_r_z34_1)

            prim_r_z45_1 = [
                np.linalg.norm([np.array(reorder_5)[i] - z45_1_arr[i], np.array(demand_45)[i] - z45_1_arr[i]]) for i in
                range(len(z45_1_arr))]
            prim_r_z.append(prim_r_z45_1)

            prim_r_z56_1 = [
                np.linalg.norm([np.array(reorder_6)[i] - z56_1_arr[i], np.array(demand_56)[i] - z56_1_arr[i]]) for i in
                range(len(z56_1_arr))]
            prim_r_z.append(prim_r_z56_1)

            prim_r_z67_1 = [
                np.linalg.norm([np.array(reorder_7)[i] - z67_1_arr[i], np.array(demand_67)[i] - z67_1_arr[i]]) for i in
                range(len(z67_1_arr))]
            prim_r_z.append(prim_r_z67_1)

            prim_r_z78_1 = [
                np.linalg.norm([np.array(reorder_8)[i] - z78_1_arr[i], np.array(demand_78)[i] - z78_1_arr[i]]) for i in
                range(len(z78_1_arr))]
            prim_r_z.append(prim_r_z78_1)

            if d + d2 <= num_periods - 1:
                old_z = np.concatenate((old_z, old_z12_2))
                z = np.concatenate((z, z12_2_arr))

                prim_r_z12_2 = [
                    np.linalg.norm([np.array(acquisition_2)[i] - z12_2_arr[i], np.array(shipping_1)[i] - z12_2_arr[i]])
                    for i in range(len(z12_2_arr))]
                prim_r_z.append(prim_r_z12_2)
            if d + d3 <= num_periods - 1:
                old_z = np.concatenate((old_z, old_z23_2))
                z = np.concatenate((z, z23_2_arr))

                prim_r_z23_2 = [
                    np.linalg.norm([np.array(acquisition_3)[i] - z23_2_arr[i], np.array(shipping_23)[i] - z23_2_arr[i]])
                    for i in range(len(z23_2_arr))]
                prim_r_z.append(prim_r_z23_2)
            if d + d4 <= num_periods - 1:
                old_z = np.concatenate((old_z, old_z34_2))
                z = np.concatenate((z, z34_2_arr))

                prim_r_z34_2 = [
                    np.linalg.norm([np.array(acquisition_4)[i] - z34_2_arr[i], np.array(shipping_34)[i] - z34_2_arr[i]])
                    for i in range(len(z34_2_arr))]
                prim_r_z.append(prim_r_z34_2)

            if d + d5 <= num_periods - 1:
                old_z = np.concatenate((old_z, old_z45_2))
                z = np.concatenate((z, z45_2_arr))

                prim_r_z45_2 = [
                    np.linalg.norm([np.array(acquisition_5)[i] - z45_2_arr[i], np.array(shipping_45)[i] - z45_2_arr[i]])
                    for i in range(len(z45_2_arr))]
                prim_r_z.append(prim_r_z45_2)

            if d + d6 <= num_periods - 1:
                old_z = np.concatenate((old_z, old_z56_2))
                z = np.concatenate((z, z56_2_arr))

                prim_r_z56_2 = [
                    np.linalg.norm([np.array(acquisition_6)[i] - z56_2_arr[i], np.array(shipping_56)[i] - z56_2_arr[i]])
                    for i in range(len(z56_2_arr))]
                prim_r_z.append(prim_r_z56_2)

            if d + d7 <= num_periods - 1:
                old_z = np.concatenate((old_z, old_z67_2))
                z = np.concatenate((z, z67_2_arr))

                prim_r_z67_2 = [
                    np.linalg.norm([np.array(acquisition_7)[i] - z67_2_arr[i], np.array(shipping_67)[i] - z67_2_arr[i]])
                    for i in range(len(z67_2_arr))]
                prim_r_z.append(prim_r_z67_2)

            if d + d8 <= num_periods - 1:
                old_z = np.concatenate((old_z, old_z78_2))
                z = np.concatenate((z, z78_2_arr))

                prim_r_z78_2 = [
                    np.linalg.norm([np.array(acquisition_8)[i] - z78_2_arr[i], np.array(shipping_78)[i] - z78_2_arr[i]])
                    for i in range(len(z78_2_arr))]
                prim_r_z.append(prim_r_z78_2)

            # primal residual
            prim_r_z = [item for sublist in prim_r_z for item in sublist]
            prim_r += [np.linalg.norm(prim_r_z)]
            # dual residual
            dual_r += [rho * np.linalg.norm(z-old_z)]

        # If current test failed at a time-step go to next test
        if skip_test:
            ft_prim_r_dict[j] = prim_r_dict
            ft_dual_r_dict[j] = dual_r_dict
            break


        # Update initial inventory for next real-time iteration
        i10 = pyo.value(model_1.nb[d].i)
        i20 = pyo.value(model_2.nb[d].i)
        i30 = pyo.value(model_3.nb[d].i)
        i40 = pyo.value(model_4.nb[d].i)
        i50 = pyo.value(model_5.nb[d].i)
        i60 = pyo.value(model_6.nb[d].i)
        i70 = pyo.value(model_7.nb[d].i)
        i80 = pyo.value(model_8.nb[d].i)

        # Update initial backlog for next real-time iteration
        bl10 = pyo.value(model_1.nb[d].bl)
        bl20 = pyo.value(model_2.nb[d].bl)
        bl30 = pyo.value(model_3.nb[d].bl)
        bl40 = pyo.value(model_4.nb[d].bl)
        bl50 = pyo.value(model_5.nb[d].bl)
        bl60 = pyo.value(model_6.nb[d].bl)
        bl70 = pyo.value(model_7.nb[d].bl)
        bl80 = pyo.value(model_8.nb[d].bl)


        LP_actions[d, :] = [pyo.value(model_1.nb[d].r), pyo.value(model_2.nb[d].r),
                            pyo.value(model_3.nb[d].r), pyo.value(model_4.nb[d].r),
                            pyo.value(model_5.nb[d].r), pyo.value(model_6.nb[d].r),
                            pyo.value(model_7.nb[d].r), pyo.value(model_8.nb[d].r)]
        LP_inv[d, :] = [pyo.value(model_1.nb[d].i0), pyo.value(model_2.nb[d].i0),
                        pyo.value(model_3.nb[d].i0), pyo.value(model_4.nb[d].i0),
                        pyo.value(model_5.nb[d].i0), pyo.value(model_6.nb[d].i0),
                        pyo.value(model_7.nb[d].i0), pyo.value(model_8.nb[d].i0)]
        LP_backlog[d, :] = [pyo.value(model_1.nb[d].bl0), pyo.value(model_2.nb[d].bl0),
                            pyo.value(model_3.nb[d].bl0), pyo.value(model_4.nb[d].bl0),
                            pyo.value(model_5.nb[d].bl0), pyo.value(model_6.nb[d].bl0),
                            pyo.value(model_7.nb[d].bl0), pyo.value(model_8.nb[d].bl0)]
        LP_acquisition[d, :] = [pyo.value(model_1.nb[d].a), pyo.value(model_2.nb[d].a),
                                pyo.value(model_3.nb[d].a), pyo.value(model_4.nb[d].a),
                                pyo.value(model_5.nb[d].a), pyo.value(model_6.nb[d].a),
                                pyo.value(model_7.nb[d].a), pyo.value(model_8.nb[d].a)]
        LP_shipment[d, :] = [pyo.value(model_1.nb[d].r), pyo.value(model_1.nb[d].s),
                             pyo.value(model_2.nb[d].s), pyo.value(model_3.nb[d].s),
                             pyo.value(model_4.nb[d].s), pyo.value(model_5.nb[d].s),
                             pyo.value(model_6.nb[d].s), pyo.value(model_7.nb[d].s)]


        if use_scaled_rho:
            rho = Rho * ((num_periods - d - 1)/(num_periods))

        dual_r_dict[d] = dual_r
        prim_r_dict[d] = prim_r

#%% Testing actions in RL environment
    if skip_test:
        continue

    s = LP_env.reset(customer_demand=LP_Customer_Demand[j])
    lp_reward = 0
    total_inventory = 0
    total_backlog = 0
    customer_backlog = 0
    done = False
    t = 0

    #if j == 0:
    array_obs[:, :, 0] = s

    while not done:
        lp_action = np.round(LP_actions[t, :], 0)
        s, r, done, info = LP_env.step(lp_action)
        profit[j, t] = r
        total_inventory += sum(s[:, 0])
        total_backlog += sum(s[:, 1])
        customer_backlog += s[0, 1]
        lp_reward += r
        #if j == 0:
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

dshlp_time = time.time() - start_time
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
if save_results:
    np.save(path+'reward_mean.npy', lp_reward_mean)
    np.save(path + 'reward_list.npy', lp_reward_list)
    np.save(path+'reward_std.npy', lp_reward_std)
    np.save(path+'inventory_mean.npy', inventory_level_mean)
    np.save(path+'inventory_std.npy', inventory_level_std)
    np.save(path+'backlog_mean.npy', backlog_level_mean)
    np.save(path+'backlog_std.npy', backlog_level_std)
    np.save(path+'customer_backlog_mean', customer_backlog_mean)
    np.save(path+'customer_backlog_std', customer_backlog_std)
    np.save(path+'failed_tests', failed_tests)
    np.save(path + 'ft_dual_r', ft_dual_r_dict)
    np.save(path + 'ft_prim_r', ft_prim_r_dict)
    np.save(path+'profit', profit)
    np.save(path+'time', dshlp_time)

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

print(f"Total profit: {sum(array_profit_sum)}")