import numpy as np
from scipy.optimize import minimize

def base_stock_policy(policy, env):
    '''
    Implements a re-order up-to policy. This means that for
    each node in the network, if the inventory at that node
    falls below the level denoted by the policy, we will
    re-order inventory to bring it to the policy level.
    '''

    inv_ech = env.inv[env.period, :] + env.order_u[env.period, :] - env.backlog[env.period, :]

    # Get unconstrained actions
    unc_actions = policy - inv_ech

    # Ensure that actions can be fulfilled by checking
    # constraints
    actions = np.minimum(env.order_max,
                         np.maximum(unc_actions, np.zeros(env.num_stages)))
    return actions


def dfo_func(policy, env, demand=None, *args):
    '''
    Runs an episode based on current base-stock model
    settings. This allows us to use our environment for the
    DFO optimizer.
    '''
    if demand is None:
        env.reset()  # Ensure env is fresh
    else:
        env.reset(customer_demand=demand)  # Ensure env is fresh
    rewards = []
    done = False
    while not done:
        action = base_stock_policy(policy, env)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)

    rewards = np.array(rewards)
    prob = env.dist.pmf(env.customer_demand, **env.dist_param)

    # Return negative of expected profit
    return -1 / env.num_periods * np.sum(prob * rewards)


def optimize_inventory_policy(env, fun, init_policy=None, method='Powell', demand=None):

    if init_policy is None:
        init_policy = np.ones(env.num_stages)*env.mu

    # Optimize policy
    if demand is None:
        out = minimize(fun=fun, x0=init_policy, args=env, method=method)
    else:
        out = minimize(fun=fun, x0=init_policy, args=(env, demand), method=method)
    policy = out.x.copy()

    # Policy must be positive integer
    policy = np.round(np.maximum(policy, 0), 0).astype(int)

    return policy, out