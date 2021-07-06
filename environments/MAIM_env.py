from ray.rllib import MultiAgentEnv
import gym
import numpy as np
from scipy.stats import poisson, randint


class MultiAgentInvManagement(MultiAgentEnv):
    def __init__(self, config):

        # Number of Periods in Episode
        self.num_periods = config.pop("num_periods", 50)

        # Structure
        self.independent = config.pop("independent", True)
        self.num_stages = config.pop("num_stages", 3)
        if self.num_stages == 4:
            self.stage_names = ["retailer", "wholesaler", "distributor", "factory"]
        elif self.num_stages == 3:
            self.stage_names = ["retailer", "distributor", "factory"]
        else:
            raise Exception('Not Implemented')

        self.num_agents = config.pop("num_agents", self.num_stages)
        self.inv_init = config.pop("init_inv", np.ones(self.num_stages)*100)
        self.inv_target = config.pop("inv_target", np.ones(self.num_stages) * 10)
        self.delay = config.pop("delay", np.ones(self.num_stages, dtype=np.int8))

        # Price of goods
        self.price = config.pop("price", np.flip(np.arange(self.num_stages + 1) + 1))

        # Stock Holding and Backlog cost
        self.stock_cost = config.pop("stock_cost", np.ones(self.num_stages)*0.5)
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
        self.inv_max = config.pop("inv_max", np.ones(self.num_stages, dtype=np.int8)*200)
        order_max = np.zeros(self.num_stages)
        for i in range(self.num_stages - 1):
            order_max[i] = self.inv_max[i + 1]
        order_max[self.num_stages - 1] = self.inv_max[self.num_stages - 1]
        self.order_max = config.pop("order_max", order_max)
        inv_max_obs = np.max(self.inv_max)
        order_max_obs = np.max(self.order_max)

        self.done = set()

        self.action_space = gym.spaces.Box(
            low=0,
            high=order_max_obs,
            dtype=np.int16,
            shape=(1,))

        # observation space (Inventory position at each echelon, which is any integer value)
        self.observation_space = gym.spaces.Box(
            low=-np.zeros(3),
            high=np.array([inv_max_obs, np.inf,  inv_max_obs]),
            dtype=np.float,
            shape=(3,))

        self.state = {}

        # Error catching
        assert isinstance(self.num_periods, int)

        # Check maximum possible order is less than inventory capacity for each stage
        for i in range(len(self.order_max) - 1):
            if self.order_max[i] > self.inv_max[i + 1]:
                break
                raise Exception('Maximum order cannot exceed maximum inventory of upstream stage')

        # Check sell price of a stage product is more than sell price of upstream stage product
        for i in range(len(self.price) - 1):
            assert self.price[i] > self.price[i + 1]

        # Maximum order of last stage cannot exceed its own inventory
        assert self.order_max[self.num_stages - 1] <= self.inv_max[self.num_stages - 1]

        self.reset()


    def reset(self, customer_demand=None):
        """
        Create and initialize all variables.
        Nomenclature:
            inv = On hand inventory at the start of each period at each stage (except last one).
            order_u = Pipeline inventory at the start of each period at each stage (except last one).
            order_r = Replenishment order placed at each period at each stage (except last one).
            demand = demand at each stage
            ship = Sales performed at each period at each stage.
            backlog = Backlog at each period at each stage.
            profit = Total profit at each stage.
        """

        periods = self.num_periods
        num = self.num_stages

        if customer_demand is not None:
            self.customer_demand = customer_demand
        else:
            self.customer_demand = self.customer_demand_init

        # simulation result lists
        self.inv = np.zeros([periods + 1, num])  # inventory at the beginning of each period
        self.order_r = np.zeros([periods, num])  # replenishment order (last stage places no replenishment orders)
        self.order_u = np.zeros([periods + 1, num])  # Unfulfilled order
        self.ship = np.zeros([periods, num])  # units sold
        self.acquisition = np.zeros([periods, num])
        self.backlog = np.zeros([periods + 1, num])  # backlog
        self.demand = np.zeros([periods, num])

        # initialization
        self.period = 0  # initialize time
        self.demand[self.period, 0] = self.customer_demand[self.period]
        self.inv[self.period, :] = self.inv_init  # initial inventory

        # set state
        self._update_state()

        return self.state

    def _update_state(self):
        # Dictionary containing observation of each agent
        obs = {}

        t = self.period
        m = self.num_stages

        for i in range(m):
            # Each agent observes five things at every time-step
            # Their inventory, backlog, demand received, acquired inventory from upstream stage
            # and inventory sent to downstream stage which forms an observation/state vecto
            agent = self.stage_names[i] # Get agent name
            obs_vector = np.zeros(3) # Initialise state vector
            obs_vector[0] = self.inv[t, i]
            obs_vector[1] = self.backlog[t, i]
            obs_vector[2] = self.order_u[t, i]
            obs[agent] = obs_vector

        self.state = obs.copy()

    def step(self, action_dict):
        """
        Update state, transition to next state/period/time-step
        :param action_dict:
        :return:
        """
        t = self.period
        m = self.num_stages

        # Get replenishment order at each stage
        if self.num_stages == 4:
            self.order_r[t, :] = np.minimum(
                np.squeeze(
                    np.array([int(action_dict["retailer"]), int(action_dict["wholesaler"]),
                              int(action_dict["distributor"]), int(action_dict["factory"])]
                             )
                )
            , self.order_max)

        elif self.num_stages == 3:
            self.order_r[t, :] = np.minimum(
                np.squeeze(
                    np.array([int(action_dict["retailer"]), int(action_dict["distributor"]),
                              int(action_dict["factory"])]
                    )
                )
            , self.order_max)

        self.order_r[t, :] = np.round(self.order_r[t, :], 0).astype(int)

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

        # Update period
        self.period += 1
        # Update state
        self._update_state()

        # Calculate rewards
        rewards, profit = self.get_rewards()

        # determine if simulation should terminate
        done = {
            "__all__": self.period >= self.num_periods,
        }

        info = {}
        for i in range(m):
            meta_info = dict()
            meta_info['period'] = self.period
            meta_info['demand'] = self.demand[t, i]
            meta_info['ship'] = self.ship[t, i]
            meta_info['acquisition'] = self.acquisition[t, i]
            meta_info['actual order'] = self.order_r[t, i]
            meta_info['profit'] = profit[i]
            stage = self.stage_names[i]
            info[stage] = meta_info

        return self.state, rewards, done, info

    def get_rewards(self):
        rewards = {}
        profit = np.zeros(self.num_stages)
        m = self.num_stages
        t = self.period
        reward_sum = 0
        for i in range(m):
            agent = self.stage_names[i]
            reward = self.price[i] * self.ship[t - 1, i] \
                - self.price[i + 1] * self.order_r[t - 1, i] \
                - self.stock_cost[i] * np.abs(self.inv[t, i] - self.inv_target[i]) \
                - self.backlog_cost[i] * self.backlog[t, i]

            reward_sum += reward
            profit[i] = reward
            if self.independent:
                rewards[agent] = reward

        if not self.independent:
            for i in range(m):
                agent = self.stage_names[i]
                rewards[agent] = reward_sum/self.num_stages

        return rewards, profit

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

