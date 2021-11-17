from ray.rllib import MultiAgentEnv
import copy
import gym
import numpy as np
from scipy.stats import poisson, randint
from utils import get_stage, get_retailers, create_network

class MultiAgentInvManagementDiv(MultiAgentEnv):
    def __init__(self, config):

        self.config = config.copy()

        # Number of Periods in Episode
        self.num_periods = config.pop("num_periods", 50)

        # Structure
        self.independent = config.pop("independent", True)
        self.share_network = config.pop("share_network", False)
        self.num_nodes = config.pop("num_nodes", 3)
        self.node_names = []
        for i in range(self.num_nodes):
            node_name = "node_" + str(i)
            self.node_names.append(node_name)

        self.connections = config.pop("connections", {0: [1], 1: [2], 2: []})
        self.network = create_network(self.connections)
        self.order_network = np.transpose(self.network)
        self.retailers = get_retailers(self.network)
        self.non_retailers = list()
        for i in range(self.num_nodes):
            if i not in self.retailers:
                self.non_retailers.append(i)
        self.upstream_node = dict()
        for i in range(1, self.num_nodes):
            self.upstream_node[i] = np.where(self.order_network[i] == 1)[0][0]

        self.num_stages = get_stage(node=int(self.num_nodes - 1), network=self.network) + 1
        self.a = config.pop("a", -1)
        self.b = config.pop("b", 1)

        self.num_agents = config.pop("num_agents", self.num_nodes)
        self.inv_init = config.pop("init_inv", np.ones(self.num_nodes)*100)
        self.inv_target = config.pop("inv_target", np.ones(self.num_nodes) * 0)
        self.delay = config.pop("delay", np.ones(self.num_nodes, dtype=np.int32))
        self.time_dependency = config.pop("time_dependency", False)
        self.prev_actions = config.pop("prev_actions", False)
        self.prev_demand = config.pop("prev_demand", False)
        self.prev_length = config.pop("prev_length", 1)
        self.max_delay = np.max(self.delay)
        if self.max_delay == 0:
            self.time_dependency = False


        # Price of goods
        stage_price = np.arange(self.num_stages) + 2
        stage_cost = np.arange(self.num_stages) + 1
        self.node_price = np.zeros(self.num_nodes)
        self.node_cost = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            self.node_price[i] = stage_price[get_stage(i, self.network)]
            self.node_cost[i] = stage_cost[get_stage(i, self.network)]
        self.price = config.pop("price", np.flip(np.arange(self.num_stages + 1) + 1))

        # Stock Holding and Backlog cost
        self.stock_cost = config.pop("stock_cost", np.ones(self.num_nodes)*0.5)
        self.backlog_cost = config.pop("backlog_cost", np.ones(self.num_nodes))

        # Customer demand
        self.demand_dist = config.pop("demand_dist", "custom")
        self.SEED = config.pop("seed", 52)
        np.random.seed(seed=int(self.SEED))

        # Delay uncertainty
        self.noisy_delay = False
        self.noisy_delay_threshold = 0

        # Capacity
        self.inv_max = config.pop("inv_max", np.ones(self.num_nodes, dtype=np.int16) * 100)
        order_max = np.zeros(self.num_nodes)
        for i in range(1, self.num_nodes):
            order_max[i] = self.inv_max[np.where(self.order_network[i] == 1)]
        order_max[0] = self.inv_max[0]
        self.order_max = config.pop("order_max", order_max)

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

        self.action_space = gym.spaces.Box(
            low=np.ones(1)*self.a,
            high=np.ones(1)*self.b,
            dtype=np.float64,
            shape=(1,)
        )


        # observation space (Inventory position at each echelon, which is any integer value)
        if not self.share_network:
            if self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.max_delay, dtype=np.float64)*self.a,
                    high=np.ones(3 + self.max_delay, dtype=np.float64)*self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay,)
                )
            elif self.time_dependency and self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length + self.max_delay, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length + self.max_delay, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length,)
                )
            elif self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length + self.max_delay, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length + self.max_delay, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length,)
                )
            elif self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length*2 + self.max_delay, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length*2 + self.max_delay, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length*2,)
                )
            elif not self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length*2, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length*2, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.prev_length*2,)
                )

            elif not self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.prev_length,)
                )

            elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3, dtype=np.float64) * self.a,
                    high=np.ones(3, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3,)
                )
            else:
                raise Exception('Not Implemented')
        else:
            if self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.max_delay + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.max_delay + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + 1,)
                )
            elif self.time_dependency and self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length + self.max_delay + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length + self.max_delay + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length + 1,)
                )
            elif self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length + self.max_delay + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length + self.max_delay + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length + 1,)
                )
            elif self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length * 2 + self.max_delay + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length * 2 + self.max_delay + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length * 2 + 1,)
                )
            elif not self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length * 2 + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length * 2 + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.prev_length * 2 + 1,)
                )

            elif not self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.prev_length + 1,)
                )

            elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + 1,)
                )
            else:
                raise Exception('Not Implemented')

        self.state = {}

        # Error catching
        assert isinstance(self.num_periods, int)

        # Check maximum possible order is less than inventory capacity for each node
        for i in range(len(self.order_max) - 1):
            if self.order_max[i] > self.inv_max[i + 1]:
                break
                raise Exception('Maximum order cannot exceed maximum inventory of upstream node')



        # Maximum order of first node cannot exceed its own inventory
        assert self.order_max[0] <= self.inv_max[0]

        self.reset()


    def reset(self, customer_demand=None, noisy_delay=False, noisy_delay_threshold=0):
        """
        Create and initialize all variables.
        Nomenclature:
            inv = On hand inventory at the start of each period at each node (except last one).
            order_u = Pipeline inventory at the start of each period at each node (except last one).
            order_r = Replenishment order placed at each period at each node (except last one).
            demand = demand at each node
            ship = Sales performed at each period at each node.
            backlog = Backlog at each period at each node.
            profit = Total profit at each node.
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
                self.customer_demand = self.config.pop("customer_demand",
                                                       np.ones((len(self.retailers), self.num_periods),
                                                               dtype=np.int16) * 5)
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

        # simulation result lists
        self.inv = np.zeros([periods + 1, num_nodes])  # inventory at the beginning of each period
        self.order_r = np.zeros([periods, num_nodes])  # replenishment order (last node places no replenishment orders)
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
            for node in self.non_retailers:
                ship_to[node] = dict()
                for d_node in self.connections[node]:
                    ship_to[node][d_node] = 0

            self.ship_to_list.append(ship_to)

        self.backlog_to = dict()
        for i in range(self.num_nodes):
            if len(self.connections[i]) > 1:
                self.backlog_to[i] = dict()
                for node in self.connections[i]:
                    self.backlog_to[i][node] = 0

        # initialization
        self.period = 0  # initialize time
        for node in self.retailers:
            self.demand[self.period, node] = self.retailer_demand[node][self.period]
        self.inv[self.period, :] = self.inv_init  # initial inventory

        # set state
        self._update_state()

        return self.state

    def _update_state(self):
        # Dictionary containing observation of each agent
        obs = {}

        t = self.period
        m = self.num_nodes

        for i in range(m):
            # Each agent observes five things at every time-step
            # Their inventory, backlog, demand received, acquired inventory from upstream node
            # and inventory sent to downstream node which forms an observation/state vecto
            agent = self.node_names[i] # Get agent name
            # Initialise state vector
            if not self.share_network:
                if self.time_dependency and not self.prev_actions and not self.prev_demand:
                    obs_vector = np.zeros(3 + self.max_delay)
                elif self.time_dependency and self.prev_actions and not self.prev_demand:
                    obs_vector = np.zeros(3 + self.prev_length + self.max_delay)
                elif self.time_dependency and not self.prev_actions and self.prev_demand:
                    obs_vector = np.zeros(3 + self.prev_length + self.max_delay)
                elif self.time_dependency and self.prev_actions and self.prev_demand:
                    obs_vector = np.zeros(3 + self.prev_length*2 + self.max_delay)
                elif not self.time_dependency and self.prev_actions and self.prev_demand:
                    obs_vector = np.zeros(3 + self.prev_length*2)
                elif not self.time_dependency and not self.prev_actions and self.prev_demand:
                    obs_vector = np.zeros(3 + self.prev_length)
                elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
                    obs_vector = np.zeros(3)
            else:
                if self.time_dependency and not self.prev_actions and not self.prev_demand:
                    obs_vector = np.zeros(3 + self.max_delay + 1)
                elif self.time_dependency and self.prev_actions and not self.prev_demand:
                    obs_vector = np.zeros(3 + self.prev_length + self.max_delay + 1)
                elif self.time_dependency and not self.prev_actions and self.prev_demand:
                    obs_vector = np.zeros(3 + self.prev_length + self.max_delay + 1)
                elif self.time_dependency and self.prev_actions and self.prev_demand:
                    obs_vector = np.zeros(3 + self.prev_length*2 + self.max_delay + 1)
                elif not self.time_dependency and self.prev_actions and self.prev_demand:
                    obs_vector = np.zeros(3 + self.prev_length*2 + 1)
                elif not self.time_dependency and not self.prev_actions and self.prev_demand:
                    obs_vector = np.zeros(3 + self.prev_length + 1)
                elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
                    obs_vector = np.zeros(3 + 1)

            if self.prev_demand:
                demand_history = np.zeros(self.prev_length)
                for j in range(self.prev_length):
                    if j < t:
                        demand_history[j] = self.demand[t - 1 - j, i]
                demand_history = self.rescale(demand_history, np.zeros(self.prev_length),
                                              np.ones(self.prev_length)*self.demand_max[i],
                                              self.a, self.b)

            if self.prev_actions:
                order_history = np.zeros(self.prev_length)
                for j in range(self.prev_length):
                    if j < t:
                        order_history[j] = self.order_r[t - 1 - j, i]
                order_history = self.rescale(order_history, np.zeros(self.prev_length),
                                              np.ones(self.prev_length)*self.order_max[i],
                                              self.a, self.b)

            if self.time_dependency:
                delay_states = np.zeros(self.max_delay)
                if t >= 1:
                    delay_states = self.time_dependent_state[t - 1, i, :]
                delay_states = self.rescale(delay_states, np.zeros(self.max_delay),
                                                                    np.ones(self.max_delay)*self.inv_max[i]*2,  # <<<<<<
                                                                    self.a, self.b)

            obs_vector[0] = self.rescale(self.inv[t, i], 0, self.inv_max[i], self.a, self.b)
            obs_vector[1] = self.rescale(self.backlog[t, i], 0, self.demand_max[i], self.a, self.b)
            obs_vector[2] = self.rescale(self.order_u[t, i], 0, self.order_max[i], self.a, self.b)
            if self.time_dependency and not self.prev_actions and not self.prev_demand:
                obs_vector[3:3+self.max_delay] = delay_states
            elif self.time_dependency and self.prev_actions and not self.prev_demand:
                obs_vector[3:3+self.prev_length] = order_history
                obs_vector[3+self.prev_length:3+self.prev_length+self.max_delay] = delay_states
            elif self.time_dependency and not self.prev_actions and self.prev_demand:
                obs_vector[3:3+self.prev_length] = demand_history
                obs_vector[3+self.prev_length:3+self.prev_length+self.max_delay] = delay_states
            elif self.time_dependency and self.prev_actions and self.prev_demand:
                obs_vector[3:3+self.prev_length] = demand_history
                obs_vector[3+self.prev_length:3+self.prev_length*2] = order_history
                obs_vector[3+self.prev_length*2:3+self.prev_length*2+self.max_delay] = delay_states
            elif not self.time_dependency and self.prev_actions and not self.prev_demand:
                obs_vector[3:3 + self.prev_length] = demand_history
            elif not self.time_dependency and self.prev_actions and self.prev_demand:
                obs_vector[3:3 + self.prev_length] = demand_history
                obs_vector[3 + self.prev_length:3 + self.prev_length * 2] = order_history

            if self.share_network:
                obs_vector[len(obs_vector) - 1] = self.rescale(i, 0, self.num_nodes, self.a, self.b)

            obs[agent] = obs_vector

        self.state = obs.copy()

    def step(self, action_dict):
        """
        Update state, transition to next state/period/time-step
        :param action_dict:
        :return:
        """
        t = self.period
        m = self.num_nodes

        # Get replenishment order at each node
        for i in range(self.num_nodes):
            node_name = "node_" + str(i)
            self.order_r[t, i] = self.rev_scale(action_dict[node_name], 0, self.order_max[i], self.a, self.b)
            self.order_r[t, i] = np.round(self.order_r[t, i], 0).astype(int)

        self.order_r[t, :] = np.minimum(np.maximum(self.order_r[t, :], np.zeros(self.num_nodes)), self.order_max)

        # Demand of goods at each stage
        # Demand at last (retailer stages) is customer demand
        for node in self.retailers:
            self.demand[t, node] = np.minimum(self.retailer_demand[node][t], self.inv_max[node])  # min for re-scaling
        # Demand at other stages is the replenishment order of the downstream stage
        for i in range(self.num_nodes):
            if i not in self.retailers:
                for j in range(i, len(self.network[i])):
                    if self.network[i][j] == 1:
                        self.demand[t, i] += self.order_r[t, j]

        # Update acquisition, i.e. goods received from previous node
        self.update_acquisition()

        # Amount shipped by each node to downstream node at each time-step. This is backlog from previous time-steps
        # And demand from current time-step, This cannot be more than the current inventory at each node
        self.ship[t, :] = np.minimum(self.backlog[t, :] + self.demand[t, :], self.inv[t, :] + self.acquisition[t, :])

        # Get amount shipped to downstream nodes
        for i in self.non_retailers:
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
                            if while_counter > self.demand_max[i] * 2:
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
        # Capping backlog to allow re-scaling
        self.backlog[t + 1, :] = np.minimum(self.backlog[t + 1, :], self.demand_max)

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

        # Update period
        self.period += 1
        # Update state
        self._update_state()

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
            node = self.node_names[i]
            info[node] = meta_info

        return self.state, rewards, done, info

    def get_rewards(self):
        rewards = {}
        profit = np.zeros(self.num_nodes)
        m = self.num_nodes
        t = self.period
        reward_sum = 0
        for i in range(m):
            agent = self.node_names[i]
            reward = self.node_price[i] * self.ship[t, i] \
                - self.node_cost[i] * self.order_r[t, i] \
                - self.stock_cost[i] * np.abs(self.inv[t + 1, i] - self.inv_target[i]) \
                - self.backlog_cost[i] * self.backlog[t + 1, i]

            reward_sum += reward
            profit[i] = reward
            if self.independent:
                rewards[agent] = reward

        if not self.independent:
            for i in range(m):
                agent = self.node_names[i]
                rewards[agent] = reward_sum/self.num_nodes

        return rewards, profit

    def update_acquisition(self):
        """
        Get acquisition at each node
        :return: None
        """
        m = self.num_nodes
        t = self.period

        # Acquisition at node 0 is unique since delay is manufacturing delay instead of shipment delay
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
                self.acquisition[t, i] += \
                self.ship_to_list[t - self.delay[i]][self.upstream_node[i]][i]
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
            self.time_dependent_state[t, :, 0:self.max_delay - 1] = self.time_dependent_state[t - 1, :,
                                                                    1:self.max_delay]

        # Delayed states of first node
        self.time_dependent_state[t, 0, self.delay[0] - 1] = self.order_r[t, 0]
        # Delayed states of rest of n:
        for i in range(1, m):
            self.time_dependent_state[t, i, self.delay[i] - 1] = \
                self.ship_to_list[t][self.upstream_node[i]][i]

    def rescale(self, val, min_val, max_val, A=-1, B=1):
        if isinstance(val, np.ndarray):
            a = np.ones(np.size(val)) * A
            b = np.ones(np.size(val)) * B
        else:
            a = A
            b = B

        val_scaled = a + (((val - min_val) * (b - a)) / (max_val - min_val))

        return val_scaled

    def rev_scale(self, val_scaled, min_val, max_val, A=-1, B=1):
        if isinstance(val_scaled, np.ndarray):
            a = np.ones(np.size(val_scaled)) * A
            b = np.ones(np.size(val_scaled)) * B
        else:
            a = A
            b = B

        val = (((val_scaled - a) * (max_val - min_val)) / (b - a)) + min_val

        return val
