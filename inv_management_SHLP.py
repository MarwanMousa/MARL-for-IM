from environments.IM_env import InvManagement
from ray import tune
import numpy as np
import pyomo.environ as pyo
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
    return m.lsb[t].a4 == m.lsb[t-d4].x4

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

    # Customer demand
    b.demand = pyo.Var(domain=pyo.NonNegativeIntegers)

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
    b.backlog1 = pyo.Constraint(expr=b.bl1 == b.bl10 - b.s1 + b.demand)
    b.backlog2 = pyo.Constraint(expr=b.bl2 == b.bl20 - b.s2 + b.x1)
    b.backlog3 = pyo.Constraint(expr=b.bl3 == b.bl30 - b.s3 + b.x2)
    b.backlog4 = pyo.Constraint(expr=b.bl4 == b.bl40 - b.s4 + b.x3)

    # Sales Constraints
    b.ship11 = pyo.Constraint(expr=b.s1 <= b.i10)
    b.ship12 = pyo.Constraint(expr=b.s1 <= b.bl10 + b.demand)
    b.ship21 = pyo.Constraint(expr=b.s2 <= b.i20)
    b.ship22 = pyo.Constraint(expr=b.s2 <= b.bl20 + b.x1)
    b.ship31 = pyo.Constraint(expr=b.s3 <= b.i30)
    b.ship32 = pyo.Constraint(expr=b.s3 <= b.bl30 + b.x2)
    b.ship41 = pyo.Constraint(expr=b.s4 <= b.i40)
    b.ship42 = pyo.Constraint(expr=b.s4 <= b.bl40 + b.x3)

# construct the objective function over all the blocks
def obj_rule(m):
    # Sum of Profit at each state at each timeperiod
    return sum(m.lsb[t].s1*P1 - m.lsb[t].x1*P2 - m.lsb[t].i1*SC1 - m.lsb[t].bl1*BC1
               + m.lsb[t].s2*P2 - m.lsb[t].x2*P3 - m.lsb[t].i2*SC2 - m.lsb[t].bl2*BC2
               + m.lsb[t].s3*P3 - m.lsb[t].x3*P4 - m.lsb[t].i3*SC3 - m.lsb[t].bl3*BC3
               + m.lsb[t].s4*P4 - m.lsb[t].x4*P5 - m.lsb[t].i4*SC4 - m.lsb[t].bl4*BC4
               for t in m.T)

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

    # Get real customer demand at current time-step
    d = LP_demand[i]
    # Create model over the horizon i:num_periods (shrinking horizon)
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(i, num_periods-1)
    model.lsb = pyo.Block(model.T, rule=lotsizing_block_rule)

    # Inventory linking constraints
    model.i_linking1 = pyo.Constraint(model.T, rule=i1_linking_rule)
    model.i_linking2 = pyo.Constraint(model.T, rule=i2_linking_rule)
    model.i_linking3 = pyo.Constraint(model.T, rule=i3_linking_rule)
    model.i_linking4 = pyo.Constraint(model.T, rule=i4_linking_rule)

    # Backlog linking constraints
    model.bl_linking1 = pyo.Constraint(model.T, rule=bl1_linking_rule)
    model.bl_linking2 = pyo.Constraint(model.T, rule=bl2_linking_rule)
    model.bl_linking3 = pyo.Constraint(model.T, rule=bl3_linking_rule)
    model.bl_linking4 = pyo.Constraint(model.T, rule=bl4_linking_rule)

    # Acquisition linking constraints
    model.a_linking1 = pyo.Constraint(model.T, rule=a1_linking_rule)
    model.a_linking2 = pyo.Constraint(model.T, rule=a2_linking_rule)
    model.a_linking3 = pyo.Constraint(model.T, rule=a3_linking_rule)
    model.a_linking4 = pyo.Constraint(model.T, rule=a4_linking_rule)

    # Customer demand linking constraints
    model.d_linking = pyo.Constraint(model.T, rule=d_linking_rule)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    ### solve the problem
    solver = pyo.SolverFactory('gurobi', solver_io='python')
    results = solver.solve(model)
    SHLP_actions[i, :] = [pyo.value(model.lsb[i].x1), pyo.value(model.lsb[i].x2),
                          pyo.value(model.lsb[i].x3), pyo.value(model.lsb[i].x4)]

    SHLP_shipment[i, :] = [pyo.value(model.lsb[i].s2), pyo.value(model.lsb[i].s3),
                           pyo.value(model.lsb[i].s4), pyo.value(model.lsb[i].x4)]

    i10 = pyo.value(model.lsb[i].i1)
    i20 = pyo.value(model.lsb[i].i2)
    i30 = pyo.value(model.lsb[i].i3)
    i40 = pyo.value(model.lsb[i].i4)

    bl10 = pyo.value(model.lsb[i].bl1)
    bl20 = pyo.value(model.lsb[i].bl2)
    bl30 = pyo.value(model.lsb[i].bl3)
    bl40 = pyo.value(model.lsb[i].bl4)


s = DFO_env.reset(customer_demand=LP_demand)
lp_reward = 0
done = False
t = 0
while not done:
    lp_action = SHLP_actions[t, :]
    s, r, done, _ = DFO_env.step(lp_action)

    lp_reward += r
    t += 1

print(lp_reward)
