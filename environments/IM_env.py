import gym
import numpy as np
from scipy.stats import poisson, randint


class InvManagement(gym.Env):
    def __init__(self, config):

        # Number of Periods in Episode
        self.num_periods = config.pop("num_periods", 50)

        # Structure
        self.num_stages = config.pop("num_stages", 3)
        self.inv_init = config.pop("init_inv", np.ones(self.num_stages) * 20)
        self.delay = config.pop("delay", np.ones(self.num_stages, dtype=np.int8))

        # Price of goods
        self.price = config.pop("price", np.flip(np.arange(self.num_stages + 1) + 1))

        # Stock Holding and Backlog cost
        self.inv_target = config.pop("inv_target", np.ones(self.num_stages) * 10)
        self.stock_cost = config.pop("stock_cost", np.ones(self.num_stages) * 0.5)
        self.backlog_cost = config.pop("backlog_cost", np.ones(self.num_stages))

        # Customer demand
        self.demand_dist = config.pop("demand_dist", "custom")
        self.SEED = config.pop("seed", 52)
        np.random.seed(seed=int(self.SEED))

        # Custom customer demand
        if self.demand_dist == "custom":
            self.customer_demand_init = config.pop("customer_demand", np.ones(self.num_periods, dtype=np.int16) * 5)
        # Poisson distribution
        elif self.demand_dist == "poisson":
            mu = config.pop("mu", 5)
            self.dist = poisson
            self.dist_param = {'mu': mu}
            self.customer_demand_init = self.dist.rvs(size=self.num_periods, **self.dist_param)
        # Uniform distribution
        elif self.demand_dist == "uniform":
            lower_upper = config.pop("lower_upper", (1, 5))
            lower = lower_upper[0]
            upper = lower_upper[1]
            self.dist = randint
            self.dist_param = {'low': lower, 'high': upper}
            if lower >= upper:
                raise Exception('Lower bound cannot be larger than upper bound')
            self.customer_demand_init = self.dist.rvs(size=self.num_periods, **self.dist_param)
        else:
            raise Exception('Unrecognised, Distribution Not Implemented')

        # Capacity
        self.inv_max = config.pop("inv_max", np.ones(self.num_stages, dtype=np.int16) * 100)
        order_max = np.zeros(self.num_stages)
        for i in range(self.num_stages - 1):
            order_max[i] = self.inv_max[i + 1]
        order_max[self.num_stages - 1] = self.inv_max[self.num_stages - 1]
        self.order_max = config.pop("order_max", order_max)
        inv_max_obs = np.max(self.inv_max)

        self.done = set()

        self.action_space = gym.spaces.Box(
            low=np.int32(np.zeros(self.num_stages)),
            high=np.int32(self.inv_max),
            dtype=np.int32,
            shape=(self.num_stages,)
        )

        # observation space (Inventory position at each echelon, which is any integer value)
        self.observation_space = gym.spaces.Box(
            low=-np.zeros((self.num_stages, 4)),
            high=np.tile(
                np.array([inv_max_obs, np.inf, inv_max_obs, inv_max_obs]),
                (self.num_stages, 1)
            ),
            dtype=np.float,
            shape=(self.num_stages, 4)
        )

        self.state = np.zeros((self.num_stages, 3))

        self.reset()

    def reset(self, customer_demand=None):
        """
        Create and initialize all variables.
        Nomenclature:
            inv = On hand inventory at the start of each period at each stage (except last one).
            pipe_inv = Pipeline inventory at the start of each period at each stage (except last one).
            order_r = Replenishment order placed at each period at each stage (except last one).
            demand = demand at each stage
            ship = Sales performed at each period at each stage.
            backlog = Backlog at each period at each stage.
            profit = Total profit at each stage.
        """

        periods = self.num_periods
        num_stages = self.num_stages

        if customer_demand is not None:
            self.customer_demand = customer_demand
        else:
            self.customer_demand = self.customer_demand_init

        #print(self.customer_demand[0])
        # simulation result lists
        self.inv = np.zeros([periods + 1, num_stages])  # inventory at the beginning of each period
        self.order_r = np.zeros([periods, num_stages])  # replenishment order (last stage places no replenishment orders)
        self.order_u = np.zeros([periods + 1, num_stages])  # Unfulfilled order
        self.ship = np.zeros([periods, num_stages])  # units sold
        self.acquisition = np.zeros([periods, num_stages])
        self.backlog = np.zeros([periods + 1, num_stages])  # backlog
        self.demand = np.zeros([periods + 1, num_stages])

        # initialization
        self.period = 0  # initialize time
        self.demand[self.period, 0] = self.customer_demand[self.period]
        self.inv[self.period, :] = self.inv_init  # initial inventory

        # set state
        self._update_state()

        return self.state

    def _update_state(self):
        t = self.period
        m = self.num_stages
        demand = np.zeros(m)
        demand[0] = self.demand[t, 0]
        if t >= 1:
            demand[1:m] = self.demand[t - 1, 1:m]
        obs = np.stack((self.inv[t, :], self.backlog[t, :], self.order_u[t, :], demand), axis=1)

        self.state = obs.copy()

    def step(self, action):
        """
        Update state, transition to next state/period/time-step
        :param action_dict:
        :return:
        """
        t = self.period
        m = self.num_stages

        # Get replenishment order at each stage

        self.order_r[t, :] = np.round(np.minimum(np.squeeze(action), self.order_max), 0).astype(int)


        # Demand of goods at each stage
        # Demand at first (retailer stage) is customer demand
        self.demand[t, 0] = self.customer_demand[t]
        # Demand at other stages is the replenishment order of the downstream stage
        self.demand[t, 1:m] = self.order_r[t, :m - 1]

        # Amount shipped by each stage to downstream stage at each time-step. This is backlog from previous time-steps
        # And demand from current time-step, This cannot be more than the current inventory at each stage
        self.ship[t, :] = np.minimum(self.backlog[t, :] + self.demand[t, :], self.inv[t, :])

        # Update backlog demand increases backlog while fulfilling demand reduces it
        self.backlog[t + 1, :] = self.backlog[t, :] + self.demand[t, :] - self.ship[t, :]

        # Update acquisition, i.e. goods received from previous stage
        self.update_acquisition()

        # Update unfulfilled orders/ pipeline inventory
        self.order_u[t + 1, :] = np.minimum(
            np.maximum(
                self.order_u[t, :] + self.order_r[t, :] - self.acquisition[t, :],
                np.zeros(self.num_stages)),
            self.inv_max)


        # Update inventory
        self.inv[t + 1, :] = np.minimum(
            np.maximum(
                self.inv[t, :] + self.acquisition[t, :] - self.ship[t, :],
                np.zeros(self.num_stages)),
            self.inv_max)

        # Calculate rewards
        rewards, profit = self.get_rewards()

        info = {}
        info['period'] = self.period
        info['demand'] = self.demand[t, :]
        info['ship'] = self.ship[t, :]
        info['acquisition'] = self.acquisition[t, :]
        info['profit'] = profit

        # Update period
        self.period += 1
        # Update state
        self._update_state()

        # determine if simulation should terminate
        done = self.period >= self.num_periods

        return self.state, rewards, done, info

    def get_rewards(self):
        m = self.num_stages
        t = self.period
        profit = self.price[0:m] * self.ship[t, :] - self.price[1:m+1] * self.order_r[t, :] \
            - self.stock_cost * np.abs(self.inv[t + 1, :] - self.inv_target)\
            - self.backlog_cost * self.backlog[t + 1, :]

        reward = - self.stock_cost * np.abs(self.inv[t + 1, :] - self.inv_target) \
                 - self.backlog_cost * self.backlog[t + 1, :]

        reward_sum = np.sum(profit)

        return reward_sum, profit

    def update_acquisition(self):
        """
        Get acquisition at each stage
        :return: None
        """
        m = self.num_stages
        t = self.period

        # Acquisition at stage m is unique since delay is manufacturing delay instead of shipment delay
        if t - self.delay[m - 1] >= 0:
            self.acquisition[t, m - 1] = self.order_r[t - self.delay[m - 1], m - 1]
        else:
            self.acquisition[t, m - 1] = self.acquisition[t, m - 1]

        # Acquisition at subsequent stage is the delayed shipment of the upstream stage
        for i in range(m - 1):
            if t - self.delay[i] >= 0:
                self.acquisition[t, i] = self.ship[t - self.delay[i], i + 1]
            else:
                self.acquisition[t, i] = self.acquisition[t, i]

