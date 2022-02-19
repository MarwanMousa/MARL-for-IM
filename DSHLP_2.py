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
stock_cost = np.array([0.5, 0.2])
backlog_cost = np.array([0.6, 0.9])
delay = np.array([3, 1], dtype=np.int8)
standardise_state = False
standardise_actions = False
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
path = 'LP_results/two_stage/DSHLP/'
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
N_iter = 50
use_variable_rho = False
use_scaled_rho = False
act_rho = np.linspace(rho, rho_tgt, N_iter)
ADMM = True


# Maximum inventory and Maximum order amount
I1 = 30
O1 = 30

I2 = 30
O2 = 30

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


def demand_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].demand == m.Customer_Demand[t]
    return m.nb[t].demand == mu


# link the acquisition variables between blocks
def a1_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].a == a10
    if t - 1 == m.T.first():
         return m.nb[t].a == a11
    if t - 2 == m.T.first():
         return m.nb[t].a == a12
    return m.nb[t].a == m.nb[t-m.delay].r

def a2_linking_rule(m, t):
    if t == m.T.first():
        return m.nb[t].a == a20
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
    return sum(m.nb[t].s * P2 - m.nb[t].r * C2 - m.nb[t].i * IC2 - m.nb[t].bl * BC2 -
               m.rho/2 * (m.nb[t].r - m.z12_1[t] + m.u12_1[t])**2
               if (t - m.delay < m.T.first() or m.TimeStep + m.delay > m.NumPeriods - 1) else
               m.nb[t].s * P2 - m.nb[t].r * C2 - m.nb[t].i * IC2 - m.nb[t].bl * BC2 -
               m.rho / 2 * (m.nb[t].r - m.z12_1[t] + m.u12_1[t]) ** 2 -
               m.rho / 2 * (m.nb[t].a - m.z12_2[t] + m.u12_2[t]) ** 2
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

    # Initial Backlog
    bl10 = 0
    bl20 = 0

    # Initial Acquisition
    a10 = 0
    a11 = 0
    a12 = 0
    a20 = 0

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

            extra_delay = False
            extra_delay_prob = np.random.uniform(0, 1)
            if extra_delay_prob <= noisy_delay_threshold and noisy_delay and d < num_periods:
                extra_delay = True

            if extra_delay:
                a10 = 0
                LP_shipment[d - d1 + 1, 0] += LP_shipment[d - d1, 0]
                LP_shipment[d - d1, 0] = 0
            else:
                a10 = LP_shipment[d - d1, 0]

        if d - d1 + 1 < 0:
            a11 = 0
        else:
            a11 = LP_shipment[d - d1 + 1, 0]  # Configuration specific

        if d - d1 + 2 < 0:
            a12 = 0
        else:
            a12 = LP_shipment[d - d1 + 2, 0]  # Configuration specific

        if d - d2 < 0:
            a20 = 0
        else:

            extra_delay = False
            extra_delay_prob = np.random.uniform(0, 1)
            if extra_delay_prob <= noisy_delay_threshold and noisy_delay and d < num_periods:
                extra_delay = True

            if extra_delay:
                a20 = 0
                LP_shipment[d - d2 + 1, 1] += LP_shipment[d - d2, 1]
                LP_shipment[d - d2, 1] = 0
            else:
                a20 = LP_shipment[d - d2, 1]


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
        model_1.u12_1 = pyo.Param(model_1.T, initialize=0, mutable=True)

        if d + d2 <= num_periods - 1:
            model_1.T12 = pyo.RangeSet(d + d2, num_periods - 1)
            model_1.z12_2 = pyo.Param(model_1.T12, initialize=mu, mutable=True)
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
        model_2.T = pyo.RangeSet(d, num_periods - 1)

        Customer_Demand2 = {t: LP_Customer_Demand[j][0][t] for t in model_2.T}
        model_2.Customer_Demand = pyo.Param(model_2.T, default=Customer_Demand2, mutable=False)

        model_2.nb = pyo.Block(model_2.T, rule=node_block_rule)
        model_2.i_linking = pyo.Constraint(model_2.T, rule=i_linking_rule)
        model_2.bl_linking = pyo.Constraint(model_2.T, rule=bl_linking_rule)
        model_2.d_linking = pyo.Constraint(model_2.T, rule=demand_linking_rule)
        model_2.a_linking = pyo.Constraint(model_2.T, rule=a2_linking_rule)
        model_2.max_order = pyo.Constraint(model_2.T, rule=max_order_rule)
        model_2.max_inventory = pyo.Constraint(model_2.T, rule=max_inventory_rule)

        model_2.z12_1 = pyo.Param(model_2.T, initialize=mu, mutable=True)
        model_2.u12_1 = pyo.Param(model_2.T, initialize=0, mutable=True)

        if d + d2 <= num_periods - 1:
            model_2.T12 = pyo.RangeSet(d + d2, num_periods - 1)
            model_2.z12_2 = pyo.Param(model_2.T12, initialize=mu, mutable=True)
            model_2.u12_2 = pyo.Param(model_2.T12, initialize=0, mutable=True)

        model_2.obj = pyo.Objective(rule=obj_rule_2, sense=pyo.maximize)


        # Iterations
        prim_r = []
        dual_r = []
        for i in range(N_iter):

            # set rho
            if use_variable_rho:
                model_1.rho = act_rho[i]
                model_2.rho = act_rho[i]

            # Solve Sub-problems
            sub_problem1 = solver.solve(model_1)
            sub_problem2 = solver.solve(model_2)

            # check if optimal solution found
            solution = [sub_problem1.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded,
                        sub_problem2.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded]

            if any(solution):
                skip_test = True
                failed_tests.append(j)
                break


            old_z12_1 = np.array([pyo.value(model_1.z12_1[t]) for t in model_1.T])
            if d + d2 <= num_periods - 1:
                old_z12_2 = np.array([pyo.value(model_1.z12_2[t]) for t in model_1.T12])


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

            # update z
            for t in model_1.T:
                model_1.z12_1[t] = z12_1[t]
                model_2.z12_1[t] = z12_1[t]
            if d + d2 <= num_periods - 1:
                for t in model_1.T12:
                    model_1.z12_2[t] = z12_2[t]
                    model_2.z12_2[t] = z12_2[t]

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

            # update u
            if ADMM:
                for t in model_1.T:
                    model_1.u12_1[t] = u12_1_model1[t]
                    model_2.u12_1[t] = u12_1_model2[t]
                if d + d2 <= num_periods - 1:
                    for t in model_1.T12:
                        model_1.u12_2[t] = u12_2_model1[t]
                        model_2.u12_2[t] = u12_2_model2[t]

            # residual calculation
            old_z = old_z12_1
            z = z12_1_arr

            prim_r_z = []
            prim_r_z12_1 = [
                np.linalg.norm([np.array(reorder_2)[i] - z12_1_arr[i], np.array(demand_1)[i] - z12_1_arr[i]]) for i in
                range(len(z12_1_arr))]
            prim_r_z.append(prim_r_z12_1)

            if d + d2 <= num_periods - 1:
                old_z = np.concatenate((old_z, old_z12_2))
                z = np.concatenate((z, z12_2_arr))

                prim_r_z12_2 = [
                    np.linalg.norm([np.array(acquisition_2)[i] - z12_2_arr[i], np.array(shipping_1)[i] - z12_2_arr[i]])
                    for i in range(len(z12_2_arr))]
                prim_r_z.append(prim_r_z12_2)


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

        # Update initial backlog for next real-time iteration
        bl10 = pyo.value(model_1.nb[d].bl)
        bl20 = pyo.value(model_2.nb[d].bl)

        LP_actions[d, :] = [pyo.value(model_1.nb[d].r), pyo.value(model_2.nb[d].r)]
        LP_inv[d, :] = [pyo.value(model_1.nb[d].i0), pyo.value(model_2.nb[d].i0)]
        LP_backlog[d, :] = [pyo.value(model_1.nb[d].bl0), pyo.value(model_2.nb[d].bl0)]
        LP_acquisition[d, :] = [pyo.value(model_1.nb[d].a), pyo.value(model_2.nb[d].a)]
        LP_shipment[d, :] = [pyo.value(model_1.nb[d].r), pyo.value(model_1.nb[d].s)]

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