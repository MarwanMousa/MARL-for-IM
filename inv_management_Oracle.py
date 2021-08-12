from environments.IM_env import InvManagement
import ray
from ray import tune
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import os
from utils import get_config, get_trainer
from base_restock_policy import optimize_inventory_policy, dfo_func, base_stock_policy
from models.RNN_Model import RNNModel, SharedRNNModel
from ray.rllib.models import ModelCatalog
from pyomo.environ import *
#%% Environment Configuration

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
DFO_env = InvManagement(DFO_CONFIG)

#%% Linear Programming Pyomo
LP_demand = DFO_env.dist.rvs(size=(DFO_env.num_periods), **DFO_env.dist_param)
model = ConcreteModel()
model.T = RangeSet(num_periods)
d = {}
for i in range(1, num_periods+1):
    d[i] = LP_demand[i-1]

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
    b.x1 = Var(domain=NonNegativeIntegers)
    b.x2 = Var(domain=NonNegativeIntegers)
    b.x3 = Var(domain=NonNegativeIntegers)
    b.x4 = Var(domain=NonNegativeIntegers)

    # Inventory at each stage
    b.i1 = Var(domain=NonNegativeIntegers)
    b.i2 = Var(domain=NonNegativeIntegers)
    b.i3 = Var(domain=NonNegativeIntegers)
    b.i4 = Var(domain=NonNegativeIntegers)

    # Initial Inventory at each time-step
    b.i10 = Var(domain=NonNegativeIntegers)
    b.i20 = Var(domain=NonNegativeIntegers)
    b.i30 = Var(domain=NonNegativeIntegers)
    b.i40 = Var(domain=NonNegativeIntegers)

    # backlog
    b.bl1 = Var(domain=NonNegativeIntegers)
    b.bl2 = Var(domain=NonNegativeIntegers)
    b.bl3 = Var(domain=NonNegativeIntegers)
    b.bl4 = Var(domain=NonNegativeIntegers)

    # Initial Backlog at each time-step
    b.bl10 = Var(domain=NonNegativeIntegers, initialize=0)
    b.bl20 = Var(domain=NonNegativeIntegers, initialize=0)
    b.bl30 = Var(domain=NonNegativeIntegers, initialize=0)
    b.bl40 = Var(domain=NonNegativeIntegers, initialize=0)

    # Shipped goods/sales
    b.s1 = Var(domain=NonNegativeIntegers)
    b.s2 = Var(domain=NonNegativeIntegers)
    b.s3 = Var(domain=NonNegativeIntegers)
    b.s4 = Var(domain=NonNegativeIntegers)

    # Acquisiton
    b.a1 = Var(domain=NonNegativeIntegers)
    b.a2 = Var(domain=NonNegativeIntegers)
    b.a3 = Var(domain=NonNegativeIntegers)
    b.a4 = Var(domain=NonNegativeIntegers)

    # define the constraints
    b.inventory1 = Constraint(expr=b.i1 == b.i10 + b.a1 - b.s1)
    b.inventory2 = Constraint(expr=b.i2 == b.i20 + b.a2 - b.s2)
    b.inventory3 = Constraint(expr=b.i3 == b.i30 + b.a3 - b.s3)
    b.inventory4 = Constraint(expr=b.i4 == b.i40 + b.a4 - b.s4)

    # Inventory constrainss
    b.inventorymax1 = Constraint(expr=b.i1 <= I1)
    b.inventorymax2 = Constraint(expr=b.i2 <= I2)
    b.inventorymax3 = Constraint(expr=b.i3 <= I3)
    b.inventorymax4 = Constraint(expr=b.i4 <= I4)

    # Order constraints
    b.ordermax1 = Constraint(expr=b.x1 <= O1)
    b.ordermax2 = Constraint(expr=b.x2 <= O2)
    b.ordermax3 = Constraint(expr=b.x3 <= O3)
    b.ordermax4 = Constraint(expr=b.x4 <= O4)

    # backlog constrains
    b.backlog1 = Constraint(expr=b.bl1 == b.bl10 - b.s1 + d[t])
    b.backlog2 = Constraint(expr=b.bl2 == b.bl20 - b.s2 + b.x1)
    b.backlog3 = Constraint(expr=b.bl3 == b.bl30 - b.s3 + b.x2)
    b.backlog4 = Constraint(expr=b.bl4 == b.bl40 - b.s4 + b.x3)

    #
    b.ship11 = Constraint(expr=b.s1 <= b.i10)
    b.ship12 = Constraint(expr=b.s1 <= b.bl10 + d[t])
    b.ship21 = Constraint(expr=b.s2 <= b.i20)
    b.ship22 = Constraint(expr=b.s2 <= b.bl20 + b.x1)
    b.ship31 = Constraint(expr=b.s3 <= b.i30)
    b.ship32 = Constraint(expr=b.s3 <= b.bl30 + b.x2)
    b.ship41 = Constraint(expr=b.s4 <= b.i40)
    b.ship42 = Constraint(expr=b.s4 <= b.bl40 + b.x3)

model.lsb = Block(model.T, rule=lotsizing_block_rule)

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

model.i_linking1 = Constraint(model.T, rule=i1_linking_rule)
model.i_linking2 = Constraint(model.T, rule=i2_linking_rule)
model.i_linking3 = Constraint(model.T, rule=i3_linking_rule)
model.i_linking4 = Constraint(model.T, rule=i4_linking_rule)

model.bl_linking1 = Constraint(model.T, rule=bl1_linking_rule)
model.bl_linking2 = Constraint(model.T, rule=bl2_linking_rule)
model.bl_linking3 = Constraint(model.T, rule=bl3_linking_rule)
model.bl_linking4 = Constraint(model.T, rule=bl4_linking_rule)

model.a_linking1 = Constraint(model.T, rule=a1_linking_rule)
model.a_linking2 = Constraint(model.T, rule=a2_linking_rule)
model.a_linking3 = Constraint(model.T, rule=a3_linking_rule)
model.a_linking4 = Constraint(model.T, rule=a4_linking_rule)

# construct the objective function over all the blocks
def obj_rule(m):
    # Sum of Profit at each state at each timeperiod
    return sum(m.lsb[t].s1*P1 - m.lsb[t].x1*P2 - m.lsb[t].i1*SC1 - m.lsb[t].bl1*BC1
               + m.lsb[t].s2*P2 - m.lsb[t].x2*P3 - m.lsb[t].i2*SC2 - m.lsb[t].bl2*BC2
               + m.lsb[t].s3*P3 - m.lsb[t].x3*P4 - m.lsb[t].i3*SC3 - m.lsb[t].bl3*BC3
               + m.lsb[t].s4*P4 - m.lsb[t].x4*P5 - m.lsb[t].i4*SC4 - m.lsb[t].bl4*BC4
               for t in m.T)

model.obj = Objective(rule=obj_rule, sense=maximize)

### solve the problem
solver = SolverFactory('gurobi', solver_io='python')
results = solver.solve(model)
# print the results
for t in model.T:
    print(f'Period: {t}, demand: {d[t]}')
    print(f'O1: {value(model.lsb[t].x1)}, O2: {value(model.lsb[t].x2)}, O3: {value(model.lsb[t].x3)}, O4: {value(model.lsb[t].x4)}')
    print(f'I1: {value(model.lsb[t].i1)}, I2: {value(model.lsb[t].i2)}, I3: {value(model.lsb[t].i3)}, I4: {value(model.lsb[t].i4)}')
    print(f'B1: {value(model.lsb[t].bl1)}, B2: {value(model.lsb[t].bl2)}, B3: {value(model.lsb[t].bl3)}, B4: {value(model.lsb[t].bl4)}')
    print(f'S1: {value(model.lsb[t].s1)}, S2: {value(model.lsb[t].s2)}, S3: {value(model.lsb[t].s3)}, S4: {value(model.lsb[t].s4)}')
    print(f'A1: {value(model.lsb[t].a1)}, A2: {value(model.lsb[t].a2)}, A3: {value(model.lsb[t].a3)}, A4: {value(model.lsb[t].a4)}')


DFO_env.reset(customer_demand=LP_demand)
lp_reward = 0
done = False
t = 1
while not done:
    dfo_action = [value(model.lsb[t].x1), value(model.lsb[t].x2), value(model.lsb[t].x3), value(model.lsb[t].x4)]
    s, r, done, _ = DFO_env.step(dfo_action)
    print(s)
    print(r)
    print(value(model.lsb[t].x1))
    print(LP_demand[t-1])
    lp_reward += r
    t += 1

print(lp_reward)
