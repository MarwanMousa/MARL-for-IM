import gym
import numpy as np
from scipy.stats import poisson, randint


class InvManagement(gym.Env):
    def __init__(self, config):

        self.config = config.copy()

        # Number of Periods in Episode
        self.num_periods = config.pop("num_periods", 50)

        # Structure
        self.num_stages = config.pop("num_stages", 3)
        self.inv_init = config.pop("init_inv", np.ones(self.num_stages) * 20)
        self.delay = config.pop("delay", np.ones(self.num_stages, dtype=np.int8))
        self.standardise_state = config.pop("standardise_state", True)
        self.standardise_actions = config.pop("standardise_actions", True)
        self.a = config.pop("a", -1)
        self.b = config.pop("b", 1)
        self.time_dependency = config.pop("time_dependency", False)
        self.prev_actions = config.pop("prev_actions", False)
        self.prev_demand = config.pop("prev_demand", False)
        self.prev_length = config.pop("prev_length", 1)
        self.max_delay = np.max(self.delay)
        if self.max_delay == 0:
            self.time_dependency = False

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


        # Capacity
        self.inv_max = config.pop("inv_max", np.ones(self.num_stages, dtype=np.int16) * 100)
        order_max = np.zeros(self.num_stages)
        for i in range(self.num_stages - 1):
            order_max[i] = self.inv_max[i + 1]
        order_max[self.num_stages - 1] = self.inv_max[self.num_stages - 1]
        self.order_max = config.pop("order_max", order_max)
        inv_max_obs = np.max(self.inv_max)

        self.done = set()
        # Action space (Re-order amount at every tage)
        if self.standardise_actions:
            self.action_space = gym.spaces.Box(
                low=np.ones(self.num_stages, dtype=np.float64)*self.a,
                high=np.ones(self.num_stages, dtype=np.float64)*self.b,
                dtype=np.float64,
                shape=(self.num_stages,)
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.zeros(self.num_stages, dtype=np.int32),
                high=np.int32(self.order_max),
                dtype=np.int32,
                shape=(self.num_stages,)
            )

        # observation space (Inventory position at each echelon, which is any integer value)
        if self.standardise_state:
            if self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_stages, 3 + self.max_delay))*self.a,
                    high=np.ones((self.num_stages, 3 + self.max_delay))*self.b,
                    dtype=np.float64,
                    shape=(self.num_stages, 3 + self.max_delay)
                )
            elif self.time_dependency and self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_stages, 3 + self.max_delay + self.prev_length)) * self.a,
                    high=np.ones((self.num_stages, 3 + self.max_delay + self.prev_length)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_stages, 3 + self.max_delay + self.prev_length)
                )
            elif self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_stages, 3 + self.max_delay + self.prev_length*2)) * self.a,
                    high=np.ones((self.num_stages, 3 + self.max_delay + self.prev_length*2)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_stages, 3 + self.max_delay + self.prev_length*2)
                )
            elif self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_stages, 3 + self.max_delay + self.prev_length)) * self.a,
                    high=np.ones((self.num_stages, 3 + self.max_delay + self.prev_length)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_stages, 3 + self.max_delay + self.prev_length)
                )
            elif not self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_stages, 3 + self.prev_length * 2)) * self.a,
                    high=np.ones((self.num_stages, 3 + self.prev_length * 2)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_stages, 3 + self.prev_length * 2)
                )
            elif not self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_stages, 3 + self.prev_length)) * self.a,
                    high=np.ones((self.num_stages, 3 + self.prev_length)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_stages, 3 + self.prev_length)
                )
            elif not self.time_dependency and self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_stages, 3 + self.prev_length)) * self.a,
                    high=np.ones((self.num_stages, 3 + self.prev_length)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_stages, 3 + self.prev_length)
                )
            elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_stages, 3)) * self.a,
                    high=np.ones((self.num_stages, 3)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_stages, 3)
                )
        else:
            if not self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.zeros((self.num_stages, 3)),
                    high=np.tile(
                        np.array([inv_max_obs, np.inf, inv_max_obs]),
                        (self.num_stages, 1)
                    ),
                    dtype=np.float64,
                    shape=(self.num_stages, 3)
                )

        if self.time_dependency and not self.prev_actions and not self.prev_demand:
            self.state = np.zeros((self.num_stages, 3 + self.max_delay))
        elif self.time_dependency and self.prev_actions and not self.prev_demand:
            self.state = np.zeros((self.num_stages, 3 + self.max_delay + self.prev_length))
        elif self.time_dependency and not self.prev_actions and self.prev_demand:
            self.state = np.zeros((self.num_stages, 3 + self.max_delay + self.prev_length))
        elif self.time_dependency and self.prev_actions and self.prev_demand:
            self.state = np.zeros((self.num_stages, 3 + self.max_delay + self.prev_length*2))
        elif not self.time_dependency and self.prev_actions and self.prev_demand:
            self.state = np.zeros((self.num_stages, 3 + self.prev_length*2))
        elif not self.time_dependency and not self.prev_actions and self.prev_demand:
            self.state = np.zeros((self.num_stages, 3 + self.prev_length))
        elif not self.time_dependency and self.prev_actions and not self.prev_demand:
            self.state = np.zeros((self.num_stages, 3 + self.prev_length))
        elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
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
            # Custom customer demand
            if self.demand_dist == "custom":
                self.customer_demand = self.config.pop("customer_demand", np.ones(self.num_periods, dtype=np.int16) * 5)
            # Poisson distribution
            elif self.demand_dist == "poisson":
                self.mu = self.config.pop("mu", 5)
                self.dist = poisson
                self.dist_param = {'mu': self.mu}
                self.customer_demand = self.dist.rvs(size=self.num_periods, **self.dist_param)
            # Uniform distribution
            elif self.demand_dist == "uniform":
                lower_upper = self.config.pop("lower_upper", (1, 5))
                lower = lower_upper[0]
                upper = lower_upper[1]
                self.dist = randint
                self.dist_param = {'low': lower, 'high': upper}
                if lower >= upper:
                    raise Exception('Lower bound cannot be larger than upper bound')
                self.customer_demand = self.dist.rvs(size=self.num_periods, **self.dist_param)
            else:
                raise Exception('Unrecognised, Distribution Not Implemented')


        # simulation result lists
        self.inv = np.zeros([periods + 1, num_stages])  # inventory at the beginning of each period
        self.order_r = np.zeros([periods, num_stages])  # replenishment order (last stage places no replenishment orders)
        self.order_u = np.zeros([periods + 1, num_stages])  # Unfulfilled order
        self.ship = np.zeros([periods, num_stages])  # units sold
        self.acquisition = np.zeros([periods, num_stages])
        self.backlog = np.zeros([periods + 1, num_stages])  # backlog
        self.demand = np.zeros([periods + 1, num_stages])
        if self.time_dependency:
            self.time_dependent_state = np.zeros([periods, num_stages, self.max_delay])

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
        if self.prev_demand:
            demand_history = np.zeros((m, self.prev_length))
            for i in range(self.prev_length):
                if i < t:
                    demand_history[:, i] = self.demand[t - 1 - i, :]
            demand_history = self.rescale(demand_history, np.zeros((m, self.prev_length)),
                                          np.tile(self.inv_max.reshape((-1, 1)), (1, self.prev_length)),
                                          self.a, self.b)

        if self.prev_actions:
            order_history = np.zeros((m, self.prev_length))
            for i in range(self.prev_length):
                if i < t:
                    order_history[:, i] = self.order_r[t - 1 - i, :]

            order_history = self.rescale(order_history, np.zeros((m, self.prev_length)),
                                          np.tile(self.order_max.reshape((-1, 1)), (1, self.prev_length)),
                                          self.a, self.b)
        if self.time_dependency:
            time_dependent_state = np.zeros((m, self.max_delay))
        if t >= 1 and self.time_dependency:
            time_dependent_state = self.time_dependent_state[t - 1, :, :]

        if self.standardise_state and self.time_dependency:
            time_dependent_state = self.rescale(time_dependent_state, np.zeros((m, self.max_delay)),
                                                np.tile(self.inv_max.reshape((-1, 1)), (1, self.max_delay)),
                                                self.a, self.b)

        if self.standardise_state:
            inv = self.rescale(self.inv[t, :], np.zeros(self.num_stages), self.inv_max, self.a, self.b)
            backlog = self.rescale(self.backlog[t, :], np.zeros(self.num_stages), self.inv_max, self.a, self.b)
            order_u = self.rescale(self.order_u[t, :], np.zeros(self.num_stages), self.inv_max, self.a, self.b)
            obs = np.stack((inv, backlog, order_u), axis=1)
        else:
            obs = np.stack((self.inv[t, :], self.backlog[t, :], self.order_u[t, :]), axis=1)

        if self.time_dependency and not self.prev_actions and not self.prev_demand:
            obs = np.concatenate((obs, time_dependent_state), axis=1)
        elif self.time_dependency and self.prev_actions and not self.prev_demand:
            obs = np.concatenate((obs, order_history, time_dependent_state), axis=1)
        elif self.time_dependency and not self.prev_actions and self.prev_demand:
            obs = np.concatenate((obs, demand_history, time_dependent_state), axis=1)
        elif self.time_dependency and self.prev_actions and self.prev_demand:
            obs = np.concatenate((obs, demand_history, order_history, time_dependent_state), axis=1)
        elif not self.time_dependency and not self.prev_actions and self.prev_demand:
            obs = np.concatenate((obs, demand_history), axis=1)
        elif not self.time_dependency and self.prev_actions and not self.prev_demand:
            obs = np.concatenate((obs, order_history), axis=1)
        elif not self.time_dependency and self.prev_actions and self.prev_demand:
            obs = np.concatenate((obs, demand_history, order_history), axis=1)

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

        if self.standardise_actions:
            self.order_r[t, :] = self.rev_scale(np.squeeze(action), np.zeros(self.num_stages), self.order_max, self.a, self.b)
            self.order_r[t, :] = np.round(np.minimum(np.maximum(self.order_r[t, :], np.zeros(self.num_stages)), self.order_max), 0).astype(int)
        else:
            self.order_r[t, :] = np.round(np.minimum(np.maximum(np.squeeze(action), np.zeros(self.num_stages)), self.order_max), 0).astype(int)

        # Demand of goods at each stage
        # Demand at first (retailer stage) is customer demand
        self.demand[t, 0] = np.minimum(self.customer_demand[t], self.inv_max[0])
        # Demand at other stages is the replenishment order of the downstream stage
        self.demand[t, 1:m] = self.order_r[t, :m - 1]

        # Update acquisition, i.e. goods received from previous stage
        self.update_acquisition()

        # Amount shipped by each stage to downstream stage at each time-step. This is backlog from previous time-steps
        # And demand from current time-step, This cannot be more than the current inventory at each stage
        self.ship[t, :] = np.minimum(self.backlog[t, :] + self.demand[t, :], self.inv[t, :] + self.acquisition[t, :])

        # Update backlog demand increases backlog while fulfilling demand reduces it
        self.backlog[t + 1, :] = self.backlog[t, :] + self.demand[t, :] - self.ship[t, :]
        if self.standardise_state:
            self.backlog[t + 1, :] = np.minimum(self.backlog[t + 1, :], self.inv_max)


        # Update time-dependent states
        if self.time_dependency:
            self.time_dependent_acquisition()

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

    def time_dependent_acquisition(self):
        """
        Get time-dependent states
        :return: None
        """
        m = self.num_stages
        t = self.period

        # Shift delay down with every time-step
        if self.max_delay > 1 and t >= 1:
            self.time_dependent_state[t, :, 0:self.max_delay - 1] = self.time_dependent_state[t - 1, :, 1:self.max_delay]

        # Delayed states of final stage
        self.time_dependent_state[t, m - 1, self.delay[m - 1] - 1] = self.order_r[t, m - 1]
        # Delayed states of rest of stages:
        for i in range(m - 1):
            self.time_dependent_state[t, i, self.delay[i] - 1] = self.ship[t, i + 1]


    def rescale(self, val, min_val, max_val, A=-1, B=1):
        if isinstance(val, np.ndarray):
            a = np.ones(np.shape(val)) * A
            b = np.ones(np.shape(val)) * B
        else:
            a = A
            b = B
        val_scaled = a + (((val - min_val) * (b - a)) / (max_val - min_val))

        return val_scaled

    def rev_scale(self, val_scaled, min_val, max_val, A=-1, B=1):
        if isinstance(val_scaled, np.ndarray):
            a = np.ones(np.shape(val_scaled)) * A
            b = np.ones(np.shape(val_scaled)) * B
        else:
            a = A
            b = B

        val = (((val_scaled - a) * (max_val - min_val)) / (b - a)) + min_val

        return val



