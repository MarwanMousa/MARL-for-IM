import pyomo.environ as pyo
import numpy as np
from environments.IM_div_env import InvManagementDiv
from ray import tune
from utils import check_connections, create_network, ensure_dir
import matplotlib.pyplot as plt
#%% Environment Configuration

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
    3: []
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

#%%
num_tests = 1
test_seed = 420
np.random.seed(seed=test_seed)
LP_Customer_Demand = LP_env.dist.rvs(size=(num_tests, (len(LP_env.retailers)), LP_env.num_periods), **LP_env.dist_param)
LP_Customer_Demand = np.array([[[ 0,  5,  0,  6,  0,  3,  0,  4,  0,  8,  0,  8,  4,  8,  4,  0,
          5,  0, 12,  5,  0,  4,  3,  0, 12,  0,  0,  0, 16,  8],
        [ 8,  4,  4,  4,  4,  0,  0, 11,  6,  4,  6,  6,  7, 10,  9,  0,
         20,  0,  8,  0,  0,  0,  4,  0,  0,  0,  0,  0,  6,  6]]])
# Maximum inventory and Maximum order amount
I1 = 30
O1 = 30

I2 = 30
O2 = 30

I3 = 30
O3 = 30

I4 = 30
O4 = 30


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

# Delay
d1 = delay[0]
d2 = delay[1]
d3 = delay[2]
d4 = delay[3]

# Price of goods sold
P1 = 2
P2 = 3
P3 = 4

# Cost of re-order goods
C1 = 1
C2 = 2
C3 = 3

# Cost of Inventory
IC1 = stock_cost[0]
IC2 = stock_cost[1]
IC3 = stock_cost[2]
IC4 = stock_cost[3]

# Backlog Cost
BC1 = backlog_cost[0]
BC2 = backlog_cost[1]
BC3 = backlog_cost[2]
BC4 = backlog_cost[3]


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

def node_div_block_rule(b, t):
    # define the variables
    # Reorder Variables at each stage
    b.r = pyo.Var(domain=pyo.NonNegativeReals)

    # Inventory at each stage
    b.i = pyo.Var(domain=pyo.NonNegativeReals)

    # Initial Inventory at each time-step
    b.i0 = pyo.Var(domain=pyo.NonNegativeIntegers)

    # backlog
    b.bl3 = pyo.Var(domain=pyo.NonNegativeReals)
    b.bl4 = pyo.Var(domain=pyo.NonNegativeReals)
    b.bl = pyo.Var(domain=pyo.NonNegativeReals)

    # Initial Backlog at each time-step
    b.bl30 = pyo.Var(domain=pyo.NonNegativeReals)
    b.bl40 = pyo.Var(domain=pyo.NonNegativeReals)
    b.bl0 = pyo.Var(domain=pyo.NonNegativeReals)

    # Shipped goods/sales
    b.s3 = pyo.Var(domain=pyo.NonNegativeReals)
    b.s4 = pyo.Var(domain=pyo.NonNegativeReals)
    b.s = pyo.Var(domain=pyo.NonNegativeReals)

    # Acquisition
    b.a = pyo.Var(domain=pyo.NonNegativeReals)

    # Customer demand
    b.demand3 = pyo.Var(domain=pyo.NonNegativeReals)
    b.demand4 = pyo.Var(domain=pyo.NonNegativeReals)
    b.demand = pyo.Var(domain=pyo.NonNegativeReals)

    # define the constraints
    b.inventory = pyo.Constraint(expr=b.i == b.i0 + b.a - b.s)

    # backlog constrains
    b.backlog3 = pyo.Constraint(expr=b.bl3 == b.bl30 - b.s3 + b.demand3)
    b.backlog4 = pyo.Constraint(expr=b.bl4 == b.bl40 - b.s4 + b.demand4)
    b.backlog = pyo.Constraint(expr=b.bl == b.bl3 + b.bl4)
    b.backlog0 = pyo.Constraint(expr=b.bl0 == b.bl30 + b.bl40)

    # Sales Constraints
    b.ship_sum = pyo.Constraint(expr=b.s == b.s3 + b.s4)
    b.ship_inventory = pyo.Constraint(expr=b.s <= b.i0 + b.a)
    b.ship_backlog3 = pyo.Constraint(expr=b.s3 <= b.bl30 + b.demand3)
    b.ship_backlog4 = pyo.Constraint(expr=b.s4 <= b.bl40 + b.demand4)


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


def bl23_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].bl30 == m.B30
    return m.nb[t].bl30 == m.nb[t-1].bl3


def bl24_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].bl40 == m.B40
    return m.nb[t].bl40 == m.nb[t-1].bl4


def demand_linking_rule(m, t):
    #if t == m.T.first():
    #    return m.nb[t].demand == Customer_Demand[t]
    return m.nb[t].demand == m.Customer_Demand[t]


# link the acquisition variables between blocks
def a1_linking_rule(m, t):
    if t - m.delay < m.T.first():
        return m.nb[t].a == 0
    return m.nb[t].a == m.nb[t-d1].r

def a_linking_rule(m, t):
    if t - m.delay < m.T.first():
        return m.nb[t].a == 0
    return m.nb[t].a <= m.I0

def max_order_rule(m, t):
    return m.nb[t].r <= m.O

def max_inventory_rule(m,t):
    return m.nb[t].i <= m.I


def obj_rule_1(m):
    # Sum of Profit for all timeperiods
    return sum(m.nb[t].s*P1 - m.nb[t].r*C1 - m.nb[t].i*IC1 - m.nb[t].bl*BC1 -
               m.rho/2 * (m.nb[t].demand - m.z12_1[t] + m.u12_1[t])**2
               if t-d2 < m.T.first() else
               m.nb[t].s * P1 - m.nb[t].r * C1 - m.nb[t].i * IC1 - m.nb[t].bl * BC1 -
               m.rho / 2 * (m.nb[t].demand - m.z12_1[t] + m.u12_1[t]) ** 2 -
               m.rho / 2 * (m.nb[t-d2].s - m.z12_2[t] + m.u12_2[t]) ** 2
               for t in m.T)


def obj_rule_2(m):
    # Sum of Profit for all timeperiods
    obj = 0
    for t in m.T:
        obj += m.nb[t].s*P2 - m.nb[t].r*C2 - m.nb[t].i*IC2 - m.nb[t].bl*BC2
        obj -= m.rho/2 * (m.nb[t].r - m.z12_1[t] + m.u12_1[t])**2  # constraint 12 orders
        obj -= m.rho/2 * (m.nb[t].demand3 - m.z23_1[t] + m.u23_1[t])**2  # constraint 23 orders
        obj -= m.rho/2 * (m.nb[t].demand4 - m.z24_1[t] + m.u24_1[t])**2  # constraint 24 orders
        if t-d2 >= m.T.first():
            obj -= m.rho / 2 * (m.nb[t].a - m.z23_2[t] + m.u23_2[t]) ** 2  # constraint 12 goods
        if t-d3 >= m.T.first():
            obj -= m.rho / 2 * (m.nb[t - d3].s3 - m.z23_2[t] + m.u23_2[t]) ** 2  # constraint 23 goods
        if t-d4 >= m.T.first():
            obj -= m.rho / 2 * (m.nb[t - d4].s4 - m.z24_2[t] + m.u24_2[t]) ** 2  # constraint 14 goods

    return obj

def obj_rule_3(m):
    # Sum of Profit for all timeperiods
    return sum(m.nb[t].s*P3 - m.nb[t].r*C3 - m.nb[t].i*IC3 - m.nb[t].bl*BC3 -
               m.rho/2 * (m.nb[t].r - m.z23_1[t] + m.u23_1[t])**2
               if t-d3 < m.T.first() else
               m.nb[t].s * P3 - m.nb[t].r * C3 - m.nb[t].i * IC3 - m.nb[t].bl * BC3 -
               m.rho / 2 * (m.nb[t].r - m.z23_1[t] + m.u23_1[t]) ** 2 -
               m.rho / 2 * (m.nb[t].a - m.z23_2[t] + m.u23_2[t]) ** 2
               for t in m.T)

def obj_rule_4(m):
    # Sum of Profit for all timeperiods
    return sum(m.nb[t].s*P3 - m.nb[t].r*C3 - m.nb[t].i*IC4 - m.nb[t].bl*BC4 -
               m.rho/2 * (m.nb[t].r - m.z24_1[t] + m.u24_1[t])**2
               if t-d4 < m.T.first() else
               m.nb[t].s * P3 - m.nb[t].r * C3 - m.nb[t].i * IC4 - m.nb[t].bl * BC4 -
               m.rho / 2 * (m.nb[t].r - m.z24_1[t] + m.u24_1[t]) ** 2 -
               m.rho / 2 * (m.nb[t].a - m.z24_2[t] + m.u24_2[t]) ** 2
               for t in m.T)


period = 30
rho = 2e-1
N_iter = 50
# Get solver
solver = pyo.SolverFactory('gurobi', solver_io='python')


# Node 1 Model
model_1 = pyo.AbstractModel()
model_1.rho = pyo.Param(default=rho)
model_1.I = pyo.Param(default=I1)
model_1.O = pyo.Param(default=O1)
model_1.I0 = pyo.Param(default=i10)
model_1.B0 = pyo.Param(default=bl10)
model_1.delay = pyo.Param(default=d1)
model_1.T = pyo.RangeSet(period)
model_1.T12 = pyo.RangeSet(1 + d2, period)

model_1.nb = pyo.Block(model_1.T, rule=node_block_rule)
model_1.i_linking = pyo.Constraint(model_1.T, rule=i_linking_rule)
model_1.bl_linking = pyo.Constraint(model_1.T, rule=bl_linking_rule)
model_1.a_linking = pyo.Constraint(model_1.T, rule=a1_linking_rule)
model_1.max_order = pyo.Constraint(model_1.T, rule=max_order_rule)
model_1.max_inventory = pyo.Constraint(model_1.T, rule=max_inventory_rule)


# Global variables between nodes 1 and 2
model_1.z12_1 = pyo.Param(model_1.T, initialize=mu, mutable=True)
model_1.u12_1 = pyo.Param(model_1.T, initialize=0, mutable=True)
model_1.z12_2 = pyo.Param(model_1.T12, initialize=mu, mutable=True)
model_1.u12_2 = pyo.Param(model_1.T12, initialize=0, mutable=True)

model_1.obj = pyo.Objective(rule=obj_rule_1, sense=pyo.maximize)

# Node 2 Model
model_2 = pyo.AbstractModel()
model_2.rho = pyo.Param(default=rho)
model_2.I = pyo.Param(default=I2)
model_2.O = pyo.Param(default=O2)
model_2.I0 = pyo.Param(default=i20)
model_2.B30 = pyo.Param(default=bl20)
model_2.B40 = pyo.Param(default=bl20)
model_2.delay = pyo.Param(default=d2)
model_2.T = pyo.RangeSet(period)
model_2.T12 = pyo.RangeSet(1 + d2, period)
model_2.T23 = pyo.RangeSet(1 + d3, period)
model_2.T24 = pyo.RangeSet(1 + d4, period)

model_2.nb = pyo.Block(model_2.T, rule=node_div_block_rule)
model_2.i_linking = pyo.Constraint(model_2.T, rule=i_linking_rule)
model_2.bl23_linking = pyo.Constraint(model_2.T, rule=bl23_linking_rule)
model_2.bl24_linking = pyo.Constraint(model_2.T, rule=bl24_linking_rule)
model_2.a_linking = pyo.Constraint(model_2.T, rule=a_linking_rule)
model_2.max_order = pyo.Constraint(model_2.T, rule=max_order_rule)
model_2.max_inventory = pyo.Constraint(model_2.T, rule=max_inventory_rule)

# Global variables between nodes 1 and 2
model_2.z12_1 = pyo.Param(model_2.T, initialize=mu, mutable=True)
model_2.u12_1 = pyo.Param(model_2.T, initialize=0, mutable=True)
model_2.z12_2 = pyo.Param(model_2.T12, initialize=mu, mutable=True)
model_2.u12_2 = pyo.Param(model_2.T12, initialize=0, mutable=True)

# Global variables between nodes 2 and 3
model_2.z23_1 = pyo.Param(model_2.T, initialize=mu, mutable=True)
model_2.u23_1 = pyo.Param(model_2.T, initialize=0, mutable=True)
model_2.z23_2 = pyo.Param(model_2.T23, initialize=mu, mutable=True)
model_2.u23_2 = pyo.Param(model_2.T23, initialize=0, mutable=True)

# Global variables between nodes 2 and 4
model_2.z24_1 = pyo.Param(model_2.T, initialize=mu, mutable=True)
model_2.u24_1 = pyo.Param(model_2.T, initialize=0, mutable=True)
model_2.z24_2 = pyo.Param(model_2.T24, initialize=mu, mutable=True)
model_2.u24_2 = pyo.Param(model_2.T24, initialize=0, mutable=True)

model_2.obj = pyo.Objective(rule=obj_rule_2, sense=pyo.maximize)

# Node 3 Model
model_3 = pyo.AbstractModel()
model_3.rho = pyo.Param(default=rho)
model_3.I = pyo.Param(default=I3)
model_3.O = pyo.Param(default=O3)
model_3.I0 = pyo.Param(default=i30)
model_3.B0 = pyo.Param(default=bl30)
model_3.delay = pyo.Param(default=d3)
model_3.T = pyo.RangeSet(period)
model_3.T23 = pyo.RangeSet(1 + d3, period)

Customer_Demand3 = {t: LP_Customer_Demand[0][0][i] for i, t in enumerate(model_3.T)}
model_3.Customer_Demand = pyo.Param(model_3.T, default=Customer_Demand3, mutable=False)

model_3.nb = pyo.Block(model_3.T, rule=node_block_rule)
model_3.i_linking = pyo.Constraint(model_3.T, rule=i_linking_rule)
model_3.bl_linking = pyo.Constraint(model_3.T, rule=bl_linking_rule)
model_3.d_linking = pyo.Constraint(model_3.T, rule=demand_linking_rule)
model_3.a_linking = pyo.Constraint(model_3.T, rule=a_linking_rule)
model_3.max_order = pyo.Constraint(model_3.T, rule=max_order_rule)
model_3.max_inventory = pyo.Constraint(model_3.T, rule=max_inventory_rule)

model_3.z23_1 = pyo.Param(model_3.T, initialize=mu, mutable=True)
model_3.u23_1 = pyo.Param(model_3.T, initialize=0, mutable=True)

model_3.z23_2 = pyo.Param(model_3.T23, initialize=mu, mutable=True)
model_3.u23_2 = pyo.Param(model_3.T23, initialize=0, mutable=True)

model_3.obj = pyo.Objective(rule=obj_rule_3, sense=pyo.maximize)

# Node 4 Model
model_4 = pyo.AbstractModel()
model_4.rho = pyo.Param(default=rho)
model_4.I = pyo.Param(default=I4)
model_4.O = pyo.Param(default=O4)
model_4.I0 = pyo.Param(default=i40)
model_4.B0 = pyo.Param(default=bl40)
model_4.delay = pyo.Param(default=d4)
model_4.T = pyo.RangeSet(period)
model_4.T24 = pyo.RangeSet(1 + d4, period)

Customer_Demand4 = {t: LP_Customer_Demand[0][1][i] for i, t in enumerate(model_4.T)}
model_4.Customer_Demand = pyo.Param(model_4.T, default=Customer_Demand4, mutable=False)

model_4.nb = pyo.Block(model_4.T, rule=node_block_rule)
model_4.i_linking = pyo.Constraint(model_4.T, rule=i_linking_rule)
model_4.bl_linking = pyo.Constraint(model_4.T, rule=bl_linking_rule)
model_4.d_linking = pyo.Constraint(model_4.T, rule=demand_linking_rule)
model_4.a_linking = pyo.Constraint(model_4.T, rule=a_linking_rule)
model_4.max_order = pyo.Constraint(model_4.T, rule=max_order_rule)
model_4.max_inventory = pyo.Constraint(model_4.T, rule=max_inventory_rule)

model_4.z24_1 = pyo.Param(model_4.T, initialize=mu, mutable=True)
model_4.u24_1 = pyo.Param(model_4.T, initialize=0, mutable=True)

model_4.z24_2 = pyo.Param(model_4.T24, initialize=mu, mutable=True)
model_4.u24_2 = pyo.Param(model_4.T24, initialize=0, mutable=True)

model_4.obj = pyo.Objective(rule=obj_rule_4, sense=pyo.maximize)


# Iterations
instance_1 = model_1.create_instance()
instance_2 = model_2.create_instance()
instance_3 = model_3.create_instance()
instance_4 = model_4.create_instance()
ADMM = True

for i in range(N_iter):
    # Solve Sub-problems
    sub_problem1 = solver.solve(instance_1)
    sub_problem2 = solver.solve(instance_2)
    sub_problem3 = solver.solve(instance_3)
    sub_problem4 = solver.solve(instance_4)

    # Debugging
    print(f"d1: {pyo.value(instance_1.nb[1].demand)}")
    print(f"r2: {pyo.value(instance_2.nb[1].r)}")
    print(f"z1: {pyo.value(instance_1.z12_1[1])}")

    # calculate new z
    # 12 Global variables
    reorder_2 = [pyo.value(instance_2.nb[t].r)for t in instance_2.T]
    demand_1 = [pyo.value(instance_1.nb[t].demand) for t in instance_1.T]
    z12_1 = (np.array(reorder_2) + np.array(demand_1)) / 2
    z12_1 = {t: z12_1[i] for i, t in enumerate(model_1.T)}

    shipping_1 = [pyo.value(instance_1.nb[t - d2].s) for t in instance_1.T12]
    acquisition_2 = [pyo.value(instance_2.nb[t].a) for t in instance_2.T12]
    z12_2 = (np.array(shipping_1) + np.array(acquisition_2)) / 2
    z12_2 = {t: z12_2[i] for i, t in enumerate(model_1.T12)}

    # 23 Global variables
    reorder_3 = [pyo.value(instance_3.nb[t].r) for t in instance_3.T]
    demand_23 = [pyo.value(instance_2.nb[t].demand3) for t in instance_2.T]
    z23_1 = (np.array(reorder_3) + np.array(demand_23)) / 2
    z23_1 = {t: z23_1[i] for i, t in enumerate(model_3.T)}

    shipping_23 = [pyo.value(instance_2.nb[t - d3].s3) for t in instance_2.T23]
    acquisition_3 = [pyo.value(instance_3.nb[t].a) for t in instance_3.T23]
    z23_2 = (np.array(shipping_23) + np.array(acquisition_3)) / 2
    z23_2 = {t: z23_2[i] for i, t in enumerate(model_3.T23)}

    # 24 Global variables
    reorder_4 = [pyo.value(instance_4.nb[t].r) for t in instance_4.T]
    demand_24 = [pyo.value(instance_2.nb[t].demand4) for t in instance_2.T]
    z24_1 = (np.array(reorder_3) + np.array(demand_24)) / 2
    z24_1 = {t: z24_1[i] for i, t in enumerate(model_4.T)}

    shipping_24 = [pyo.value(instance_2.nb[t - d4].s4) for t in instance_2.T24]
    acquisition_4 = [pyo.value(instance_4.nb[t].a) for t in instance_4.T24]
    z24_2 = (np.array(shipping_24) + np.array(acquisition_4)) / 2
    z24_2 = {t: z24_2[i] for i, t in enumerate(model_4.T24)}

    # update z
    for t in model_1.T:
        instance_1.z12_1[t] = z12_1[t]
        instance_2.z12_1[t] = z12_1[t]
        instance_2.z23_1[t] = z23_1[t]
        instance_2.z24_1[t] = z24_1[t]
        instance_3.z23_1[t] = z23_1[t]
        instance_4.z24_1[t] = z24_1[t]
    for t in model_1.T12:
        instance_1.z12_2[t] = z12_2[t]
        instance_2.z12_2[t] = z12_2[t]
    for t in model_2.T23:
        instance_2.z23_2[t] = z23_2[t]
        instance_3.z23_2[t] = z23_2[t]
    for t in model_2.T24:
        instance_2.z24_2[t] = z24_2[t]
        instance_4.z24_2[t] = z24_2[t]

    # calculate new u
    # 12
    u12_1_model1 = {t: pyo.value(instance_1.u12_1[t]) + pyo.value(instance_1.nb[t].demand - z12_1[t])
                      for t in model_1.T}
    u12_1_model2 = {t: pyo.value(instance_2.u12_1[t]) + pyo.value(instance_2.nb[t].r - z12_1[t])
                      for t in model_2.T}

    u12_2_model1 = {t: pyo.value(instance_1.u12_2[t]) + pyo.value(instance_1.nb[t - d2].s - z12_2[t])
                      for t in model_1.T12}
    u12_2_model2 = {t: pyo.value(instance_2.u12_2[t]) + pyo.value(instance_2.nb[t].a - z12_2[t])
                      for t in model_2.T12}

    # 23
    u23_1_model2 = {t: pyo.value(instance_2.u23_1[t]) + pyo.value(instance_2.nb[t].demand3 - z23_1[t])
                    for t in model_2.T}
    u23_1_model3 = {t: pyo.value(instance_3.u23_1[t]) + pyo.value(instance_3.nb[t].r - z23_1[t])
                    for t in model_3.T}

    u23_2_model2 = {t: pyo.value(instance_2.u23_2[t]) + pyo.value(instance_2.nb[t - d3].s3 - z23_2[t])
                    for t in model_2.T23}
    u23_2_model3 = {t: pyo.value(instance_3.u23_2[t]) + pyo.value(instance_3.nb[t].a - z23_2[t])
                    for t in model_3.T23}

    # 24
    u24_1_model2 = {t: pyo.value(instance_2.u24_1[t]) + pyo.value(instance_2.nb[t].demand4 - z24_1[t])
                    for t in model_2.T}
    u24_1_model4 = {t: pyo.value(instance_4.u24_1[t]) + pyo.value(instance_4.nb[t].r - z24_1[t])
                    for t in model_4.T}

    u24_2_model2 = {t: pyo.value(instance_2.u24_2[t]) + pyo.value(instance_2.nb[t - d4].s4 - z24_2[t])
                    for t in model_2.T24}
    u24_2_model4 = {t: pyo.value(instance_4.u24_2[t]) + pyo.value(instance_4.nb[t].a - z24_2[t])
                    for t in model_4.T24}

    # update u
    if ADMM:
        for t in model_1.T:
            instance_1.u12_1[t] = u12_1_model1[t]
            instance_2.u12_1[t] = u12_1_model2[t]
            instance_2.u23_1[t] = u23_1_model2[t]
            instance_2.u24_1[t] = u24_1_model2[t]
            instance_3.u23_1[t] = u23_1_model3[t]
            instance_4.u24_1[t] = u24_1_model4[t]
        for t in model_1.T12:
            instance_1.u12_2[t] = u12_2_model1[t]
            instance_2.u12_2[t] = u12_2_model2[t]
        for t in model_2.T23:
            instance_2.u23_2[t] = u23_2_model2[t]
            instance_3.u23_2[t] = u23_2_model3[t]
        for t in model_2.T24:
            instance_2.u24_2[t] = u24_2_model2[t]
            instance_4.u24_2[t] = u24_2_model4[t]

# Get solution results
LP_inv = np.zeros((num_periods, num_nodes))
LP_backlog = np.zeros((num_periods, num_nodes))
LP_acquisition = np.zeros((num_periods, num_nodes))
LP_shipment = np.zeros((num_periods, num_nodes))
LP_actions = np.zeros((num_periods, num_nodes))
for i in range(1, num_periods + 1):
    LP_actions[i - 1, :] = [pyo.value(instance_1.nb[i].r), pyo.value(instance_2.nb[i].r),
                            pyo.value(instance_3.nb[i].r), pyo.value(instance_4.nb[i].r)]
    LP_inv[i - 1, :] = [pyo.value(instance_1.nb[i].i0), pyo.value(instance_2.nb[i].i0),
                        pyo.value(instance_3.nb[i].i0), pyo.value(instance_4.nb[i].i0)]
    LP_backlog[i - 1, :] = [pyo.value(instance_1.nb[i].bl0), pyo.value(instance_2.nb[i].bl0),
                            pyo.value(instance_3.nb[i].bl0), pyo.value(instance_4.nb[i].bl0)]
    LP_acquisition[i - 1, :] = [pyo.value(instance_1.nb[i].a), pyo.value(instance_2.nb[i].a),
                                pyo.value(instance_3.nb[i].a), pyo.value(instance_4.nb[i].a)]
    LP_shipment[i - 1, :] = [pyo.value(instance_1.nb[i].s), pyo.value(instance_2.nb[i].s),
                             pyo.value(instance_3.nb[i].s), pyo.value(instance_4.nb[i].s)]

#%% Testing actions in RL environment
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

j = 0
s = LP_env.reset(customer_demand=LP_Customer_Demand[j])
lp_reward = 0
total_inventory = 0
total_backlog = 0
customer_backlog = 0
done = False
t = 0

if j == 0:
    array_obs[:, :, 0] = s

while not done:
    lp_action = np.round(LP_actions[t, :], 0)
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
plt.show()

print(f"Total profit: {sum(array_profit_sum)}")