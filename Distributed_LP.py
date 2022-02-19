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
delay = np.array([1, 1], dtype=np.int8)
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
LP_Customer_Demand = np.array([[[6,  5,  0,  3, 14,  0, 14,  0,  0,  8,  6,  0,  0,  4,  8,  8,
          5,  0,  6, 10,  8,  0,  3,  4, 12,  2,  2,  8,  0,  8]]])
# Maximum inventory and Maximum order amount
I1 = 30
O1 = 30

I2 = 30
O2 = 30

# Initial Inventory
i10 = init_inv[0]
i20 = init_inv[1]

# Initial Backlog
bl10 = 0
bl20 = 0

# Delay
d1 = delay[0]
d2 = delay[1]

# Price of goods sold
P1 = 2
P2 = 3

# Cost of re-order goods
C1 = 1
C2 = 2

# Cost of Inventory
IC1 = stock_cost[0]
IC2 = stock_cost[1]

# Backlog Cost
BC1 = backlog_cost[0]
BC2 = backlog_cost[1]


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


def demand2_linking_rule(m, t):
    #if t == m.T.first():
    #    return m.nb[t].demand == Customer_Demand[t]
    return m.nb[t].demand == Customer_Demand[t]

# link the acquisition variables between blocks
def a1_linking_rule(m, t):
    if t - d1 < m.T.first():
        return m.nb[t].a == 0
    return m.nb[t].a == m.nb[t-d1].r

def a2_linking_rule(m, t):
    if t - d2 < m.T.first():
        return m.nb[t].a == 0
    return m.nb[t].a <= m.I0

def max_order_rule(m, t):
    return m.nb[t].r <= m.O

def max_inventory_rule(m,t):
    return m.nb[t].i <= m.I


def obj_rule_1(m):
    # Sum of Profit for all timeperiods
    return sum(m.nb[t].s*P1 - m.nb[t].r*C1 - m.nb[t].i*IC1 - m.nb[t].bl*BC1 -
               m.rho/2 * (m.nb[t].demand - m.z1[t] + m.u1[t])**2
               if t-d2 < m.T.first() else
               m.nb[t].s * P1 - m.nb[t].r * C1 - m.nb[t].i * IC1 - m.nb[t].bl * BC1 -
               m.rho / 2 * (m.nb[t].demand - m.z1[t] + m.u1[t]) ** 2 -
               m.rho / 2 * (m.nb[t-d1].s - m.z2[t] + m.u2[t]) ** 2
               for t in m.T)


def obj_rule_2(m):
    # Sum of Profit for all timeperiods
    return sum(m.nb[t].s*P2 - m.nb[t].r*C2 - m.nb[t].i*IC2 - m.nb[t].bl*BC2 -
               m.rho/2 * (m.nb[t].r - m.z1[t] + m.u1[t])**2
               if t-d2 < m.T.first() else
               m.nb[t].s * P2 - m.nb[t].r * C2 - m.nb[t].i * IC2 - m.nb[t].bl * BC2 -
               m.rho / 2 * (m.nb[t].r - m.z1[t] + m.u1[t]) ** 2 -
               m.rho / 2 * (m.nb[t].a - m.z2[t] + m.u2[t]) ** 2
               for t in m.T)


period = 30
rho = 2e-1
N_iter = 50
# Get solver
solver = pyo.SolverFactory('gurobi', solver_io='python')
#solver.options['NonConvex'] = 2
#solver = pyo.SolverFactory('ipopt')

#      _____ <--- order <------ _____ <---- Customer demand
#      | 1 |                    | 2 |
#      |___| ----> goods -----> |___| ----->

Optimal_z1 = [10.,  0., -0.,  8., -0., 14.,  0.,  0.,  8.,  6.,  0., -0.,  4.,
        8.,  8.,  5.,  0.,  6., 10.,  8., -0.,  3.,  4., 12.,  2.,  2.,
        8., -0.,  8., -0.]
Optimal_z2 = [10.,  0.,  0.,  8.,  0., 14.,  0.,  0.,  8.,  6.,  0.,  0.,  4.,
        8.,  8.,  5.,  0.,  6., 10.,  8.,  0.,  3.,  4., 12.,  2.,  2.,
        8.,  0.,  8.]

# Node 1 Model
model_1 = pyo.AbstractModel()
model_1.rho = pyo.Param(default=rho)
model_1.I = pyo.Param(default=I1)
model_1.O = pyo.Param(default=O1)
model_1.I0 = pyo.Param(default=i10)
model_1.B0 = pyo.Param(default=bl10)
model_1.T = pyo.RangeSet(period)
model_1.T12 = pyo.RangeSet(1 + d2, period)

Customer_Demand = {t: LP_Customer_Demand[0][0][i] for i, t in enumerate(model_1.T)}
Opt_z1_init = {t: Optimal_z1[t-1] for t in model_1.T}
Opt_z2_init = {t: Optimal_z2[i] for i, t in enumerate(model_1.T12)}


model_1.z1 = pyo.Param(model_1.T, initialize=mu, mutable=True)
model_1.u1 = pyo.Param(model_1.T, initialize=0, mutable=True)
model_1.z2 = pyo.Param(model_1.T12, initialize=mu, mutable=True)
model_1.u2 = pyo.Param(model_1.T12, initialize=0, mutable=True)
model_1.nb = pyo.Block(model_1.T, rule=node_block_rule)
model_1.i_linking = pyo.Constraint(model_1.T, rule=i_linking_rule)
model_1.bl_linking = pyo.Constraint(model_1.T, rule=bl_linking_rule)
model_1.a_linking = pyo.Constraint(model_1.T, rule=a1_linking_rule)
model_1.max_order = pyo.Constraint(model_1.T, rule=max_order_rule)
model_1.max_inventory = pyo.Constraint(model_1.T, rule=max_inventory_rule)
model_1.obj = pyo.Objective(rule=obj_rule_1, sense=pyo.maximize)

# Node 2 Model
model_2 = pyo.AbstractModel()
model_2.rho = pyo.Param(default=rho)
model_2.I = pyo.Param(default=I2)
model_2.O = pyo.Param(default=O2)
model_2.I0 = pyo.Param(default=i20)
model_2.B0 = pyo.Param(default=bl20)
model_2.T = pyo.RangeSet(period)
model_2.T12 = pyo.RangeSet(1 + d2, period)

model_2.z1 = pyo.Param(model_2.T, initialize=mu, mutable=True)
model_2.u1 = pyo.Param(model_2.T, initialize=0, mutable=True)

model_2.z2 = pyo.Param(model_2.T12, initialize=mu, mutable=True)
model_2.u2 = pyo.Param(model_2.T12, initialize=0, mutable=True)

model_2.nb = pyo.Block(model_2.T, rule=node_block_rule)
model_2.i_linking = pyo.Constraint(model_2.T, rule=i_linking_rule)
model_2.bl_linking = pyo.Constraint(model_2.T, rule=bl_linking_rule)
model_2.d_linking = pyo.Constraint(model_2.T, rule=demand2_linking_rule)
model_2.a_linking = pyo.Constraint(model_2.T, rule=a2_linking_rule)
model_2.max_order = pyo.Constraint(model_2.T, rule=max_order_rule)
model_2.max_inventory = pyo.Constraint(model_2.T, rule=max_inventory_rule)
model_2.obj = pyo.Objective(rule=obj_rule_2, sense=pyo.maximize)

# Iterations
instance_1 = model_1.create_instance()
instance_2 = model_2.create_instance()
ADMM = True
for i in range(N_iter):
    # Solve Sub-problems
    sub_problem1 = solver.solve(instance_1)
    sub_problem2 = solver.solve(instance_2)

    # Debugging
    print(f"d1: {pyo.value(instance_1.nb[1].demand)}")
    print(f"r2: {pyo.value(instance_2.nb[1].r)}")
    print(f"z1: {pyo.value(instance_1.z1[1])}")


    # calculate new z
    reorder_2 = [pyo.value(instance_2.nb[t].r)for t in instance_2.T]
    demand_1 = [pyo.value(instance_1.nb[t].demand) for t in instance_1.T]
    new_z1 = (np.array(reorder_2) + np.array(demand_1)) / 2
    new_z1_dict = {t: new_z1[i] for i, t in enumerate(model_1.T)}

    shipping_1 = [pyo.value(instance_1.nb[t - d2].s) for t in instance_1.T12]
    acquisition_2 = [pyo.value(instance_2.nb[t].a) for t in instance_2.T12]
    new_z2 = (np.array(shipping_1) + np.array(acquisition_2)) / 2
    new_z2_dict = {t: new_z2[i] for i, t in enumerate(model_1.T12)}

    # update z
    for t in model_1.T:
        instance_1.z1[t] = new_z1_dict[t]
        instance_2.z1[t] = new_z1_dict[t]
    for t in model_1.T12:
        instance_1.z2[t] = new_z2_dict[t]
        instance_2.z2[t] = new_z2_dict[t]

    # calculate new u
    new_u1_model_1 = {t: pyo.value(instance_1.u1[t]) + pyo.value(instance_1.nb[t].demand - new_z1_dict[t])
                      for t in model_1.T}
    new_u1_model_2 = {t: pyo.value(instance_2.u1[t]) + pyo.value(instance_2.nb[t].r - new_z1_dict[t])
                      for t in model_2.T}

    new_u2_model_1 = {t: pyo.value(instance_1.u2[t]) + pyo.value(instance_1.nb[t - 1].s - new_z2_dict[t])
                      for t in model_1.T12}
    new_u2_model_2 = {t: pyo.value(instance_2.u2[t]) + pyo.value(instance_2.nb[t].a - new_z2_dict[t])
                      for t in model_2.T12}

    # update u
    if ADMM:
        for t in model_1.T:
            instance_1.u1[t] = new_u1_model_1[t]
            instance_2.u1[t] = new_u1_model_2[t]
        for t in model_1.T12:
            instance_1.u2[t] = new_u2_model_1[t]
            instance_2.u2[t] = new_u2_model_2[t]

# Get solution results
LP_inv = np.zeros((num_periods, num_nodes))
LP_backlog = np.zeros((num_periods, num_nodes))
LP_acquisition = np.zeros((num_periods, num_nodes))
LP_shipment = np.zeros((num_periods, num_nodes))
LP_actions = np.zeros((num_periods, num_nodes))
for i in range(1, num_periods + 1):
    LP_actions[i - 1, :] = [pyo.value(instance_1.nb[i].r), pyo.value(instance_2.nb[i].r)]
    LP_inv[i - 1, :] = [pyo.value(instance_1.nb[i].i0), pyo.value(instance_2.nb[i].i0)]
    LP_backlog[i - 1, :] = [pyo.value(instance_1.nb[i].bl0), pyo.value(instance_2.nb[i].bl0)]
    LP_acquisition[i - 1, :] = [pyo.value(instance_1.nb[i].a), pyo.value(instance_2.nb[i].a)]
    LP_shipment[i - 1, :] = [pyo.value(instance_1.nb[i].s), pyo.value(instance_2.nb[i].s)]

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