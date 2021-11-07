import gym
import numpy as np
import copy
from scipy.stats import poisson, randint
from utils import get_stage, get_retailers, create_network


class InvManagementDiv(gym.Env):
    def __init__(self, config):

        self.config = config.copy()

        # Number of Periods in Episode
        self.num_periods = config.pop("num_periods", 50)

        # Structure
        self.num_nodes = config.pop("num_nodes", 3)
        self.connections = config.pop("connections", {0: [1], 1: [2], 2: []})
        self.network = create_network(self.connections)
        self.retailers = get_retailers(self.network)
        self.order_network = np.transpose(self.network)
        self.num_stages = get_stage(node=int(self.num_nodes - 1), network=self.network) + 1
        self.inv_init = config.pop("init_inv", np.ones(self.num_nodes) * 20)
        self.delay = config.pop("delay", np.ones(self.num_nodes, dtype=np.int8))
        self.standardise_state = config.pop("standardise_state", True)
        self.standardise_actions = config.pop("standardise_actions", True)
        self.a = -1
        self.b = 1
        self.time_dependency = config.pop("time_dependency", False)
        self.prev_actions = config.pop("prev_actions", False)
        self.prev_demand = config.pop("prev_demand", False)
        self.prev_length = config.pop("prev_length", 1)
        self.max_delay = np.max(self.delay)
        if self.max_delay == 0:
            self.time_dependency = False

        # Delay uncertainty
        self.noisy_delay = False
        self.noisy_delay_threshold = 0

        # Price of goods
        stage_price = np.arange(self.num_stages) + 2
        stage_cost = np.arange(self.num_stages) + 1
        self.node_price = np.zeros(self.num_nodes)
        self.node_cost = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            self.node_price[i] = stage_price[get_stage(i, self.network)]
            self.node_cost[i] = stage_cost[get_stage(i, self.network)]

        # Stock Holding and Backlog cost
        self.inv_target = config.pop("inv_target", np.ones(self.num_nodes) * 10)
        self.stock_cost = config.pop("stock_cost", np.ones(self.num_nodes) * 0.5)
        self.backlog_cost = config.pop("backlog_cost", np.ones(self.num_nodes))

        # Customer demand
        self.demand_dist = config.pop("demand_dist", "custom")
        self.SEED = config.pop("seed", 52)
        np.random.seed(seed=int(self.SEED))


        # Capacity
        self.inv_max = config.pop("inv_max", np.ones(self.num_nodes, dtype=np.int16) * 100)
        order_max = np.zeros(self.num_nodes)
        for i in range(1, self.num_nodes):
            order_max[i] = self.inv_max[np.where(self.order_network[i] == 1)]
        order_max[0] = self.inv_max[0]
        self.order_max = config.pop("order_max", order_max)
        inv_max_obs = np.max(self.inv_max)

        # Number of downstream nodes of a given node
        self.num_downstream = dict()
        self.demand_max = copy.deepcopy(self.inv_max)
        for i in range(self.num_nodes):
            self.num_downstream[i] = np.sum(self.network[i])
            downstream_max_demand = 0
            for j in range(len(self.network[i])):
                if self.network[i][j] == 1:
                    downstream_max_demand += self.order_max[j]
            if downstream_max_demand > self.demand_max[i]:
                self.demand_max[i] = downstream_max_demand


        self.done = set()
        # Action space (Re-order amount at every tage)
        if self.standardise_actions:
            self.action_space = gym.spaces.Box(
                low=np.ones(self.num_nodes, dtype=np.float64)*self.a,
                high=np.ones(self.num_nodes, dtype=np.float64)*self.b,
                dtype=np.float64,
                shape=(self.num_nodes,)
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.zeros(self.num_nodes, dtype=np.int32),
                high=np.int32(self.order_max),
                dtype=np.int32,
                shape=(self.num_nodes,)
            )

        # observation space (Inventory position at each echelon, which is any integer value)
        if self.standardise_state:
            if self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_nodes, 3 + self.max_delay))*self.a,
                    high=np.ones((self.num_nodes, 3 + self.max_delay))*self.b,
                    dtype=np.float64,
                    shape=(self.num_nodes, 3 + self.max_delay)
                )
            elif self.time_dependency and self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_nodes, 3 + self.max_delay + self.prev_length)) * self.a,
                    high=np.ones((self.num_nodes, 3 + self.max_delay + self.prev_length)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_nodes, 3 + self.max_delay + self.prev_length)
                )
            elif self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_nodes, 3 + self.max_delay + self.prev_length*2)) * self.a,
                    high=np.ones((self.num_nodes, 3 + self.max_delay + self.prev_length*2)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_nodes, 3 + self.max_delay + self.prev_length*2)
                )
            elif self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_nodes, 3 + self.max_delay + self.prev_length)) * self.a,
                    high=np.ones((self.num_nodes, 3 + self.max_delay + self.prev_length)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_nodes, 3 + self.max_delay + self.prev_length)
                )
            elif not self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_nodes, 3 + self.prev_length * 2)) * self.a,
                    high=np.ones((self.num_nodes, 3 + self.prev_length * 2)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_nodes, 3 + self.prev_length * 2)
                )
            elif not self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_nodes, 3 + self.prev_length)) * self.a,
                    high=np.ones((self.num_nodes, 3 + self.prev_length)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_nodes, 3 + self.prev_length)
                )
            elif not self.time_dependency and self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_nodes, 3 + self.prev_length)) * self.a,
                    high=np.ones((self.num_nodes, 3 + self.prev_length)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_nodes, 3 + self.prev_length)
                )
            elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones((self.num_nodes, 3)) * self.a,
                    high=np.ones((self.num_nodes, 3)) * self.b,
                    dtype=np.float64,
                    shape=(self.num_nodes, 3)
                )
        else:
            if not self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.zeros((self.num_nodes, 3)),
                    high=np.tile(
                        np.array([inv_max_obs, np.inf, inv_max_obs]),
                        (self.num_nodes, 1)
                    ),
                    dtype=np.float64,
                    shape=(self.num_nodes, 3)
                )

        if self.time_dependency and not self.prev_actions and not self.prev_demand:
            self.state = np.zeros((self.num_nodes, 3 + self.max_delay))
        elif self.time_dependency and self.prev_actions and not self.prev_demand:
            self.state = np.zeros((self.num_nodes, 3 + self.max_delay + self.prev_length))
        elif self.time_dependency and not self.prev_actions and self.prev_demand:
            self.state = np.zeros((self.num_nodes, 3 + self.max_delay + self.prev_length))
        elif self.time_dependency and self.prev_actions and self.prev_demand:
            self.state = np.zeros((self.num_nodes, 3 + self.max_delay + self.prev_length*2))
        elif not self.time_dependency and self.prev_actions and self.prev_demand:
            self.state = np.zeros((self.num_nodes, 3 + self.prev_length*2))
        elif not self.time_dependency and not self.prev_actions and self.prev_demand:
            self.state = np.zeros((self.num_nodes, 3 + self.prev_length))
        elif not self.time_dependency and self.prev_actions and not self.prev_demand:
            self.state = np.zeros((self.num_nodes, 3 + self.prev_length))
        elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
            self.state = np.zeros((self.num_nodes, 3))

        self.reset()

    def reset(self, customer_demand=None, noisy_delay=False, noisy_delay_threshold=0):
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
        num_nodes = self.num_nodes

        if noisy_delay:
            self.noisy_delay = noisy_delay
            self.noisy_delay_threshold = noisy_delay_threshold

        if customer_demand is not None:
            self.customer_demand = customer_demand
        else:
            # Custom customer demand
            if self.demand_dist == "custom":
                self.customer_demand = self.config.pop("customer_demand", np.ones((len(self.retailers), self.num_periods), dtype=np.int16) * 5)
            # Poisson distribution
            elif self.demand_dist == "poisson":
                self.mu = self.config.pop("mu", 5)
                self.dist = poisson
                self.dist_param = {'mu': self.mu}
                self.customer_demand = self.dist.rvs(size=(len(self.retailers), self.num_periods), **self.dist_param)
            # Uniform distribution
            elif self.demand_dist == "uniform":
                lower_upper = self.config.pop("lower_upper", (1, 5))
                lower = lower_upper[0]
                upper = lower_upper[1]
                self.dist = randint
                self.dist_param = {'low': lower, 'high': upper}
                if lower >= upper:
                    raise Exception('Lower bound cannot be larger than upper bound')
                self.customer_demand = self.dist.rvs(size=(len(self.retailers), self.num_periods), **self.dist_param)
            else:
                raise Exception('Unrecognised, Distribution Not Implemented')

        # Assign customer demand to each retailer
        self.retailer_demand = dict()
        for i in range(self.customer_demand.shape[0]):
            self.retailer_demand[self.retailers[i]] = self.customer_demand[i]

        # Simulation result lists
        self.inv = np.zeros([periods + 1, num_nodes])  # inventory at the beginning of each period
        self.order_r = np.zeros([periods, num_nodes])  # replenishment order (last stage places no replenishment orders)
        self.order_u = np.zeros([periods + 1, num_nodes])  # Unfulfilled order
        self.ship = np.zeros([periods, num_nodes])  # units sold
        self.acquisition = np.zeros([periods, num_nodes])
        self.backlog = np.zeros([periods + 1, num_nodes])  # backlog
        self.demand = np.zeros([periods + 1, num_nodes])
        if self.time_dependency:
            self.time_dependent_state = np.zeros([periods, num_nodes, self.max_delay])

        # Initialise list of dicts tracking goods shipped from one node to another
        self.ship_to_list = []
        for i in range(self.num_periods):
            # Shipping dict
            ship_to = dict()
            self.backlog_to = dict()
            for i in range(self.num_nodes):
                if len(self.connections[i]) > 0:
                    ship_to[i] = dict()
                    for node in self.connections[i]:
                        ship_to[i][node] = 0
                if len(self.connections[i]) > 1:
                    self.backlog_to[i] = dict()
                    for node in self.connections[i]:
                        self.backlog_to[i][node] = 0

            self.ship_to_list.append(ship_to)


        # initialization
        self.period = 0  # initialize time
        for node in self.retailers:
            self.demand[self.period, node] = self.retailer_demand[node][self.period]
        self.inv[self.period, :] = self.inv_init  # initial inventory

        # set state
        self._update_state()

        return self.state

    def _update_state(self):
        t = self.period
        m = self.num_nodes

        if self.prev_demand:
            demand_history = np.zeros((m, self.prev_length))
            for i in range(self.prev_length):
                if i < t:
                    demand_history[:, i] = self.demand[t - 1 - i, :]
            demand_history = self.rescale(demand_history, np.zeros((m, self.prev_length)),
                                          np.tile(self.demand_max.reshape((-1, 1)), (1, self.prev_length)),
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
            inv = self.rescale(self.inv[t, :], np.zeros(self.num_nodes), self.inv_max, self.a, self.b)
            backlog = self.rescale(self.backlog[t, :], np.zeros(self.num_nodes), self.demand_max, self.a, self.b)
            order_u = self.rescale(self.order_u[t, :], np.zeros(self.num_nodes), self.inv_max, self.a, self.b)
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
        m = self.num_nodes

        # Get replenishment order at each stage

        if self.standardise_actions:
            self.order_r[t, :] = self.rev_scale(np.squeeze(action), np.zeros(self.num_nodes), self.order_max, self.a, self.b)
            self.order_r[t, :] = np.round(np.minimum(np.maximum(self.order_r[t, :], np.zeros(self.num_nodes)), self.order_max), 0).astype(int)
        else:
            self.order_r[t, :] = np.round(np.minimum(np.maximum(np.squeeze(action), np.zeros(self.num_nodes)), self.order_max), 0).astype(int)

        # Demand of goods at each stage
        # Demand at first (retailer stage) is customer demand
        for node in self.retailers:
            self.demand[t, node] = np.minimum(self.retailer_demand[node][t], self.inv_max[node])  # min for re-scaling
        # Demand at other stages is the replenishment order of the downstream stage
        for i in range(self.num_nodes):
            if i not in self.retailers:
                for j in range(i, len(self.network[i])):
                    if self.network[i][j] == 1:
                        self.demand[t, i] += self.order_r[t, j]
        #print(t)
        #print(f"order: {self.order_r[t, :]}")
        #print(f"demand: {self.demand[t, :]}")

        # Update acquisition, i.e. goods received from previous stage
        self.update_acquisition()

        # Amount shipped by each stage to downstream stage at each time-step. This is backlog from previous time-steps
        # And demand from current time-step, This cannot be more than the current inventory at each stage
        self.ship[t, :] = np.minimum(self.backlog[t, :] + self.demand[t, :], self.inv[t, :] + self.acquisition[t, :])

        # Get amount shipped to downstream nodes
        for i in range(self.num_nodes):
            # If shipping to only one downstream node, the total amount shipped is equivalent to amount shipped to
            # downstream node
            if self.num_downstream[i] == 1:
                self.ship_to_list[t][i][self.connections[i][0]] = self.ship[t, i]
            # If node has more than one downstream nodes, then the amount shipped needs to be split appropriately
            elif self.num_downstream[i] > 1:
                # Extract the total amount shipped in this period
                ship_amount = self.ship[t, i]
                # If shipment equal to or more than demand, send ordered amount to each downstream node
                if self.ship[t, i] >= self.demand[t, i]:
                    # If there is backlog, fulfill it first then fulfill demand
                    if self.backlog[t, i] > 0:
                        # Fulfill backlog first
                        while_counter = 0  # to exit infinite loops if error
                        # Keep distributing shipment across downstream nodes until there is no backlog or no goods left
                        while sum(list(self.backlog_to[i].values())) > 0 and ship_amount > 0:
                            # Keep distributing shipped goods to downstream nodes
                            for node in self.connections[i]:
                                # If there is a backlog towards a downstream node ship a unit of product to that node
                                if self.backlog_to[i][node] > 0:
                                    self.ship_to_list[t][i][node] += 1  # increase amount shipped to node
                                    self.backlog_to[i][node] -= 1  # decrease its corresponding backlog
                                    ship_amount -= 1  # reduce amount of shipped goods left

                            # Counter to escape while loop with error if infinite
                            while_counter += 1
                            if while_counter > self.demand_max[i]:
                                raise Exception("Infinite Loop 1")

                        # If there is still left-over shipped goods fulfill current demand if any
                        if ship_amount > 0 and self.demand[t, i] > 0:
                            # Create a dict of downstream nodes' demand/orders
                            outstanding_order = dict()
                            for node in self.connections[i]:
                                outstanding_order[node] = self.order_r[t, node]

                            while_counter = 0
                            # Keep distributing shipment across downstream nodes until there is no backlog or no
                            # outstanding orders left
                            while ship_amount > 0 and sum(list(outstanding_order.values())) > 0:
                                for node in self.connections[i]:
                                    if outstanding_order[node] > 0:
                                        self.ship_to_list[t][i][node] += 1  # increase amount shipped to node
                                        outstanding_order[node] -= 1  # decrease its corresponding outstanding order
                                        ship_amount -= 1  # reduce amount of shipped goods left

                                # Counter to escape while loop with error if infinite
                                while_counter += 1
                                if while_counter > self.demand_max[i]:
                                    raise Exception("Infinite Loop 2")

                            # Update backlog if some outstanding order unfulfilled
                            for node in self.connections[i]:
                                self.backlog_to[i][node] += outstanding_order[node]

                    # If there is no backlog
                    else:
                        for node in self.connections[i]:
                            self.ship_to_list[t][i][node] += self.order_r[t, node]
                            ship_amount = ship_amount - self.order_r[t, node]
                        if ship_amount > 0:
                            print("WTF")

                # If shipment is insufficient to meet downstream demand
                elif self.ship[t, i] < self.demand[t, i]:
                    while_counter = 0
                    # Distribute amount shipped to downstream nodes
                    if self.backlog[t, i] > 0:
                        # Fulfill backlog first
                        while_counter = 0  # to exit infinite loops if error
                        # Keep distributing shipment across downstream nodes until there is no backlog or no goods left
                        while sum(list(self.backlog_to[i].values())) > 0 and ship_amount > 0:
                            # Keep distributing shipped goods to downstream nodes
                            for node in self.connections[i]:
                                # If there is a backlog towards a downstream node ship a unit of product to that node
                                if self.backlog_to[i][node] > 0:
                                    self.ship_to_list[t][i][node] += 1  # increase amount shipped to node
                                    self.backlog_to[i][node] -= 1  # decrease its corresponding backlog
                                    ship_amount -= 1  # reduce amount of shipped goods left

                            # Counter to escape while loop with error if infinite
                            while_counter += 1
                            if while_counter > self.demand_max[i]:
                                raise Exception("Infinite Loop 3")

                    else:
                        # Keep distributing shipped goods to downstream nodes until no goods left
                        while ship_amount > 0:
                            for node in self.connections[i]:
                                # If amount being shipped less than amount ordered
                                if self.ship_to_list[t][i][node] < self.order_r[t, node] + self.backlog_to[i][node]:
                                    self.ship_to_list[t][i][node] += 1  # increase amount shipped to node
                                    ship_amount -= 1  # reduce amount of shipped goods left

                            # Counter to escape while loop with error if infinite
                            while_counter += 1
                            if while_counter > self.demand_max[i]:
                                raise Exception("Infinite Loop 4")

                    # Log unfulfilled order amount as backlog
                    for node in self.connections[i]:
                        self.backlog_to[i][node] += self.order_r[t, node] - self.ship_to_list[t][i][node]

        # Update backlog demand increases backlog while fulfilling demand reduces it
        self.backlog[t + 1, :] = self.backlog[t, :] + self.demand[t, :] - self.ship[t, :]
        # Cap backlog to standardise state <--------------------------
        # ------------------------------------------------------------------------- #
        if self.standardise_state:
            self.backlog[t + 1, :] = np.minimum(self.backlog[t + 1, :], self.demand_max)
        # ------------------------------------------------------------------------- #

        # Update time-dependent states
        if self.time_dependency:
            self.time_dependent_acquisition()

        # Update unfulfilled orders/ pipeline inventory
        self.order_u[t + 1, :] = np.minimum(
            np.maximum(
                self.order_u[t, :] + self.order_r[t, :] - self.acquisition[t, :],
                np.zeros(self.num_nodes)),
            self.inv_max)


        # Update inventory
        self.inv[t + 1, :] = np.minimum(
            np.maximum(
                self.inv[t, :] + self.acquisition[t, :] - self.ship[t, :],
                np.zeros(self.num_nodes)),
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
        m = self.num_nodes
        t = self.period
        profit = self.node_price * self.ship[t, :] - self.node_cost * self.order_r[t, :] \
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
        m = self.num_nodes
        t = self.period

        # Acquisition at stage 0 is unique since delay is manufacturing delay instead of shipment delay
        if t - self.delay[0] >= 0:
            extra_delay = False
            if self.noisy_delay:
                delay_percent = np.random.uniform(0, 1)
                if delay_percent <= self.noisy_delay_threshold:
                    extra_delay = True

            self.acquisition[t, 0] += self.order_r[t - self.delay[0], 0]
            if extra_delay and t < self.num_periods - 1:
                self.acquisition[t + 1, 0] += self.acquisition[t, 0]
                self.acquisition[t, 0] = 0
        else:
            self.acquisition[t, 0] = self.acquisition[t, 0]

        # Acquisition at subsequent stage is the delayed shipment of the upstream stage
        for i in range(1, m):
            if t - self.delay[i] >= 0:
                extra_delay = False
                if self.noisy_delay:
                    delay_percent = np.random.uniform(0, 1)
                    if delay_percent <= self.noisy_delay_threshold:
                        extra_delay = True
                self.acquisition[t, i] += self.ship_to_list[t - self.delay[i]][np.where(self.order_network[i] == 1)[0][0]][i]
                if extra_delay and t < self.num_periods - 1:
                    self.acquisition[t + 1, i] += self.acquisition[t, i]
                    self.acquisition[t, i] = 0

            else:
                self.acquisition[t, i] = self.acquisition[t, i]

    def time_dependent_acquisition(self):
        """
        Get time-dependent states
        :return: None
        """
        m = self.num_nodes
        t = self.period

        # Shift delay down with every time-step
        if self.max_delay > 1 and t >= 1:
            self.time_dependent_state[t, :, 0:self.max_delay - 1] = self.time_dependent_state[t - 1, :, 1:self.max_delay]

        # Delayed states of first stage/node
        self.time_dependent_state[t, 0, self.delay[0] - 1] = self.order_r[t, 0]
        # Delayed states of rest of stages:
        for i in range(1, m):
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