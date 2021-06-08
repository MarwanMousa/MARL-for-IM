from ray.rllib import MultiAgentEnv
import gym
import numpy as np


class MultiAgentInvMangement(MultiAgentEnv):
    def __init__(self, config):

        # Number of Periods in Episode
        self.num_periods = config.pop("periods", 50)
        self.num_stages = 4
        self.stage_names = ["retailer", "wholesaler", "distributor", "factory"]

        # Structure
        self.num_agents = config.pop("num_agents", 4)
        self.init_inv = config.pop("init_inv", [100, 100, 100, 100])

        # Price of goods
        self.p_final = config.pop("p_final", 1)
        self.p_inter = config.pop("p_inter", [1, 1, 1])

        # Holding and Backlog cost
        self.stock_cost = config.pop("stock_cost", [1, 1, 1, 1])
        self.backlog_cost = config.pop("backlog_cost", [1, 1, 1, 1])

        self.delay = config.pop("delay", [1, 1, 1, 1])

        self.demand = config.pop("demand", np.ones(self.num_periods) * 80)

        if isinstance(env_name_or_creator, str):
            self.agents = [
                gym.make(env_name_or_creator) for _ in range(num_agents)
            ]
        else:
            self.agents = [env_name_or_creator(config) for _ in range(num_agents)]

        self.dones = set()
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space

    def _RESET(self):
        """
        Create and initialize all variables and containers.
        Nomenclature:
            inv = On hand inventory at the start of each period at each stage (except last one).
            pipe_inv = Pipeline inventory at the start of each period at each stage (except last one).
            order_r = Replenishment order placed at each period at each stage (except last one).
            demand = Customer demand at each period (at the retailer)
            shipped = Sales performed at each period at each stage.
            backlog = Backlog at each period at each stage.
            profit = Total profit at each stage.
        """

        periods = self.num_periods
        m = self.num_stages
        inv0 = self.init_inv

        # simulation result lists
        self.inv = np.zeros([periods, m])  # inventory at the beginning of each period
        self.pipe_inv = np.zeros([periods, m])  # pipeline inventory at the beginning of each period
        self.order_r = np.zeros([periods, m])  # replenishment order (last stage places no replenishment orders)
        self.order_u = np.zeros([periods, m])  # Unfullfilled order
        # self.demand = np.zeros(periods)  # demand at retailer
        self.shipped = np.zeros([periods, m])  # units sold
        self.acquisition = np.zeros([periods, m])
        self.backlog = np.zeros([periods, m])  # backlog
        self.profit = np.zeros(periods)  # profit
        self.p = np.zeros(m)

        # initialization
        self.period = 0  # initialize time
        self.inv[self.period, :] = np.array(inv0)  # initial inventory
        self.pipe_inv[self.period, :] = np.zeros(m)  # initial pipeline inventory

        self.p[0] = self.p_final
        self.p[1:m] = self.p_inter

        # set state
        self._update_state()

        return self.state

    def _update_state(self):
        obs = {}
        t = self.period
        m = self.num_stages
        for i in range(self.num_agents):
            obs[i] = np.array([self.water])

        lt_max = self.lead_time.max()
        state = np.zeros(m * (lt_max + 1))
        # state = np.zeros(m)
        if t == 0:
            state[:m] = self.I0
        else:
            state[:m] = self.I[t]

        if t == 0:
            pass
        elif t >= lt_max:
            state[-m * lt_max:] += self.action_log[t - lt_max:t].flatten()
        else:
            state[-m * (t):] += self.action_log[:t].flatten()

        self.state = state.copy()

    def _STEP(self, action_dict):
        t = self.period
        m = self.num_stages
        self.order_r[t, :] = np.array([action_dict["retailer"], action_dict["wholesaler"],
                                       action_dict["distributor"], action_dict["factory"]])

        demand = np.zeros(m)
        demand[0] = self.demand[t]
        demand[1:m] = self.order_r[t, :m - 1]

        self.shipped[t, :] = np.minimum(self.backlog[t, :] + demand, self.inv[t, :])

        self.backlog[t + 1, :] = self.backlog[t, :] + demand - self.shipped[t, :]

        # Get acquisition at each stage

        # Acquisition at stage m is unique since delay is manufacturing delay instead of shipment delay
        if t - self.delay[m - 1] >= 0:
            self.acquisition[t, m - 1] = self.order_r[t - self.delay[m - 1], m-1]
        else:
            self.acquisition[t, m - 1] = self.acquisition[t, m - 1]

        for i in range(m - 1):
            if t - self.delay[i] >= 0:
                self.acquisition[t, i] = self.shipped[t - self.delay[i], i+1]
            else:
                self.acquisition[t, i] = self.acquisition[t, i]

        self.order_u[t + 1, :] = self.order_u[t, :] + self.order_r[t, :] - self.acquisition[t, :]

        self.inv[t + 1, :] = np.maximum(self.inv[t, :] + self.acquisition[t, :] - self.shipped[t, :], [0, 0, 0, 0])


        # update period
        self.period += 1

        # update state
        self._update_state()

        # set reward (profit from current time-step)
        #reward = P

        # determine if simulation should terminate
        if self.period >= self.num_periods:
            done = True
        else:
            done = False

        return done


    def reset(self):
        return self._RESET()

    def step(self, action_dict):
        return self._STEP(action_dict)
