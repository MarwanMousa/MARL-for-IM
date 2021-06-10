from ray.rllib import MultiAgentEnv
import gym
import numpy as np


class MultiAgentInvManagement(MultiAgentEnv):
    def __init__(self, config):

        # Number of Periods in Episode
        self.num_periods = config.pop("periods", 50)

        # Structure
        self.num_stages = 4
        self.stage_names = ["retailer", "wholesaler", "distributor", "factory"]
        self.num_agents = config.pop("num_agents", 4)
        self.inv_init = config.pop("init_inv", np.array([100, 100, 100, 100]))
        self.delay = config.pop("delay", np.array([1, 1, 1, 1]))

        # Price of goods
        self.price = config.pop("price", np.array([1, 1, 1, 1]))

        # Stock Holding and Backlog cost
        self.stock_cost = config.pop("stock_cost", np.array([1, 1, 1, 1]))
        self.backlog_cost = config.pop("backlog_cost", np.array([1, 1, 1, 1]))

        # Customer demand
        self.customer_demand = config.pop("customer_demand", np.ones(self.num_periods) * 80)

        # Capacity
        self.inv_max = config.pop("inv_max", np.array([200, 200, 200, 200]))
        self.order_max = config.pop("order_max", np.array([200, 200, 200, 200]))
        inv_max_obs = np.max(self.inv_max)
        order_max_obs = np.max(self.order_max)
        demand_max = np.max(self.customer_demand)

        self.done = set()

        self.action_space = gym.spaces.Box(
            low=0,
            high=order_max_obs,
            dtype=np.int16,
            shape=(1,))

        # observation space (Inventory position at each echelon, which is any integer value)
        self.observation_space = gym.spaces.Box(
            low=-np.zeros(3),
            high=np.array([inv_max_obs, inv_max_obs,  inv_max_obs]),
            dtype=np.int32,
            shape=(3,))

        self.state = {}

        # Error catching
        assert isinstance(self.num_periods, int)

        # Check maximum possible order is less than inventory capacity for each stage
        for i in range(len(self.order_max)):
            if self.order_max[i] > self.inv_max[i]:
                break
                raise Exception('Maximum order cannot exceed maximum inventory')



    def _RESET(self):
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
        num = self.num_stages

        # simulation result lists
        self.inv = np.zeros([periods, num])  # inventory at the beginning of each period
        self.order_r = np.zeros([periods, num])  # replenishment order (last stage places no replenishment orders)
        self.order_u = np.zeros([periods, num])  # Unfulfilled order
        # self.demand = np.zeros(periods)  # demand at retailer
        self.ship = np.zeros([periods, num])  # units sold
        self.acquisition = np.zeros([periods, num])
        self.backlog = np.zeros([periods, num])  # backlog
        self.profit = np.zeros(periods)  # profit
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

    def _STEP(self, action_dict):
        """
        Update state, transition to next state/period/time-step
        :param action_dict:
        :return:
        """
        t = self.period
        m = self.num_stages

        # Get replenishment order at each stage
        self.order_r[t, :] = np.array([action_dict["retailer"], action_dict["wholesaler"],
                                       action_dict["distributor"], action_dict["factory"]])

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
                [0, 0, 0, 0]),
            self.inv_max)

        # Update inventory
        self.inv[t + 1, :] = np.minimum(
            np.maximum(
                self.inv[t, :] + self.acquisition[t, :] - self.ship[t, :],
                [0, 0, 0, 0]),
            self.inv_max)

        # Update period
        self.period += 1

        # Update state
        self._update_state()

        # Calculate rewards
        rewards = self.get_rewards()

        # determine if simulation should terminate
        done = {
            "__all__": self.period >= self.num_periods,
        }

        info = {}
        info['period'] = self.period
        info['demand'] = self.demand[t, :]
        info['ship'] = self.ship[t, :]
        info['acquisition'] = self.acquisition[t, :]


        return self.state, rewards, done, info

    def get_rewards(self):
        rewards = {}
        m = self.num_stages
        t = self.period
        for i in range(m):
            agent = self.stage_names[i]
            reward = self.price[i] * self.ship[t, i] \
                     - self.stock_cost[i] * self.inv[t, i] \
                     - self.backlog_cost[i] * self.backlog[t, i]

            rewards[agent] = reward

        return rewards

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

    def reset(self):
        return self._RESET()

    def step(self, action_dict):
        return self._STEP(action_dict)
