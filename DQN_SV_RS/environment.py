import numpy as np
from config import args, log
from typing import Literal


class Env:
    def __init__(self,
                 args,
                 member: Literal["wholesaler", "retailer"] = "wholesaler",
                 type_: Literal["train", "test"] = "train"):
        self.member = member
        self.type = type_

        # load the dataset
        if self.type == "train":
            self.dm_dataset = np.load(f"{args.dataset_parent_dir}/dm_train.npy")
            self.lr_dataset = np.load(f"{args.dataset_parent_dir}/lr_train.npy")
            self.lw_dataset = np.load(f"{args.dataset_parent_dir}/lw_train.npy")
        elif self.type == "test":
            self.dm_dataset = np.load(f"{args.dataset_parent_dir}/dm_test.npy")
            self.lr_dataset = np.load(f"{args.dataset_parent_dir}/lr_test.npy")
            self.lw_dataset = np.load(f"{args.dataset_parent_dir}/lw_test.npy")

        # Weibull distribution: simulate the deterioration of the produce
        self.weibull = []
        for j in range(-args.k, 0):
            if j <= -args.k:
                self.weibull.append(1)
            elif j >= -args.c:
                self.weibull.append(0)
            else:
                self.weibull.append(
                    1 - np.exp(-args.a * ((-j-args.c+1)**args.b - (-j-args.c)**args.b)))
        self.weibull = np.array(self.weibull)

        # Initialize episode and time
        self.episode = 1
        self.t = 1

    def reset(self):
        # Initialize
        self.dm = -np.ones(args.time_window_length)  # demand of market
        self.lr = -np.ones(args.time_window_length, dtype=int)  # lead time of retailer

        if self.member == "retailer":
            self.l_max = args.lr_max
            self.I_init = args.Ir_init
        elif self.member == "wholesaler":
            self.l_max = args.lw_max
            self.I_init = args.Iw_init
            self.dr = -np.ones(args.time_window_length)  # demand of retailer
            # lead time of wholesaler
            self.lw = -np.ones(args.time_window_length, dtype=int)

        # I_mat and I_mat_tmp in this code are represented as i_{t,j} and i_{t,j}^{'} in our paper, respectively
        self.I_mat = np.zeros((args.time_window_length, args.time_window_length))
        self.I_mat[1, 1] = self.I_init
        self.I_mat_tmp = np.zeros((args.time_window_length, args.time_window_length))

        self.Q_mat = np.zeros(args.time_window_length+self.l_max+1)

        # zero-inventory gap, namely, the inventory after fulfilling the demand
        self.zero_inventory_gap = np.zeros(args.time_window_length)

        self.NI = None
        self.W = None

        self.order = np.zeros(args.time_window_length)
        self.net_inventory = np.zeros(args.time_window_length)
        self.lost_sales = np.zeros(args.time_window_length)
        self.waste = np.zeros(args.time_window_length)

        self.cost = np.zeros(args.time_window_length)

    def sync_time(self, episode: int, t: int):
        self.episode = episode
        self.t = t

    def get_state(self, t: int):
        # state: I
        I_mat_copy = np.copy(self.I_mat)  # copy the I_mat to avoid changing the original I_mat
        if t < args.k:
            # range is [1, t], other elements are -1 for padding
            state_I = np.hstack(
                (-np.ones(args.k-t), I_mat_copy[t, 1:t+1]))
        else:
            # range is [t-k+1, t]
            state_I = I_mat_copy[t, t-args.k+1:t+1]
        state_I[-1] += self.Q_mat[t]

        # state: Q
        state_Q = self.Q_mat[t: t+self.l_max+2].sum()

        # state: Lw, Lr, Dr, Dm
        if args.model != "DQN":
            # The agent can only observe the lead times of products received in
            # the current period. These products were ordered at the latest in
            # the period t - l_min - 1. Therefore, the lead times of these products
            # and those ordered even earlier are known. The currently observable
            # lead time corresponds to the lead time of the products ordered at
            # last_ = t - l_min - 1. Even if the products ordered at last_ have not
            # yet been received, meaning their lead times are still uncertain, they
            # are initialized with a value of -1. Thus, the agent recognizes that
            # the lead times of these products are unknown.
            # However, the value of "last_" must be greater than or equal to 1, since
            # the first order is placed at period 1.
            last_r = t-args.lr_min-1
            last_w = t-args.lw_min-1
            if self.member == "wholesaler" and last_r >= 1 and last_w >= 1:
                # update the lead time in IoT
                # the range of retailer's lead time is [t-lr_max-1, t-2]
                for j in range(max(t-args.lr_max-1, 1), last_r+1):
                    # Because the lead time is not known until the produce is received
                    if j+self.lr_dataset[self.episode, j]+1 == t:
                        self.lr[j] = self.lr_dataset[self.episode, j].astype(int)

                # the range of wholesaler's lead time is [t-lw_max-1, t-2]
                for j in range(max(t-args.lw_max-1, 1), last_w+1):
                    # Because the lead time is not known until the produce is received
                    if j+self.lw_dataset[self.episode, j]+1 == t:
                        self.lw[j] = self.lw_dataset[self.episode, j].astype(int)

            elif self.member == "retailer" and last_r >= 1:
                # the range of retailer's lead time is [t-lr_max-1, t-2]
                for j in range(max(t-args.lr_max-1, 1), last_r+1):
                    # Because the lead time is not known until the produce is received
                    if j+self.lr_dataset[self.episode, j]+1 == t:
                        self.lr[j] = self.lr_dataset[self.episode, j]

            if last_w < args.tau:
                state_Lw = np.hstack(
                    (-np.ones(args.tau-len(self.lw[1: last_w+1])), self.lw[1: last_w+1]))
                state_Lr = np.hstack(
                    (-np.ones(args.tau-len(self.lr[1: last_r+1])), self.lr[1: last_r+1]))
            else:
                state_Lw = self.lw[t-args.tau-1: last_w+1]
                state_Lr = self.lr[t-args.tau-1: last_w+1]

            if self.t-1 < args.tau:
                # range is [1, t-1], other elements are -1 for padding
                state_Dr = np.hstack(
                    (-np.ones(args.tau-len(self.dr[1: self.t])), self.dr[1: self.t]))
                state_Dm = np.hstack(
                    (-np.ones(args.tau-len(self.dm[1: self.t])), self.dm[1: self.t]))
            else:
                state_Dr = self.dr[self.t-args.tau: self.t]
                state_Dm = self.dm[self.t-args.tau: self.t]

            # state = (I, Q, Lw, Lr, Dr, Dm) if the model is not DQN
            state = [state_I, state_Q, state_Lw, state_Lr, state_Dr, state_Dm]
        else:
            # state = (I, Q) if the model is DQN
            state = [state_I, state_Q]
        return state

    def reception_stage(self):
        self.I_mat[self.t, self.t] += self.Q_mat[self.t]

    def serving_stage(self, demand: int):
        # calculate the lost sales
        self.lost_sales[self.t] = max(
            demand - np.sum(self.I_mat[self.t, max(self.t-args.k+1, 1): self.t+1]),
            0)

        # log.debug(f"member: {self.member} - t: {self.t} | demand: {demand}")
        # log.debug(f"member: {self.member} - t: {self.t} | inventory sum: {np.sum(self.I_mat[self.t, max(self.t-args.k+1, 1): self.t+1])}")
        # log.debug(f"member: {self.member} - t: {self.t} | lost_sales: {self.lost_sales[self.t]}")

        # calculate the inventory after fulfilling the demand
        for j in range(max(self.t-args.k+1, 1), self.t+1):
            if j == max(self.t-args.k+1, 1):
                self.I_mat_tmp[self.t, j] = max(self.I_mat[self.t, j] - demand, 0)
            else:
                self.I_mat_tmp[self.t, j] = max(
                    self.I_mat[self.t, j] - max(demand - np.sum(self.I_mat[self.t, max(self.t-args.k+1, 1):j]), 0),
                    0)
        # log.debug(f'member: {self.member} - t: {self.t} | I_mat_tmp: {self.I_mat_tmp[self.t, max(self.t-args.k+1, 1): self.t+1]}')

        # zero-inventory gap = max(on-hand inventory, lost sales)
        if self.lost_sales[self.t] > 0:
            self.zero_inventory_gap[self.t] = -self.lost_sales[self.t]
        else:
            self.zero_inventory_gap[self.t] = np.sum(self.I_mat_tmp[self.t, max(self.t-args.k+1, 1): self.t+1])

        # log.debug(f'member: {self.member} - t: {self.t} | lost_sales > 0: {self.lost_sales[self.t] > 0}')
        # log.debug(f'member: {self.member} - t: {self.t} | zero_inventory_gap: {self.zero_inventory_gap[self.t]}')

        # update the demand in IoT
        if self.member == "retailer":
            self.dm[self.t] = demand

        elif self.member == "wholesaler":
            self.dm[self.t] = self.dm_dataset[self.episode, self.t]
            self.dr[self.t] = demand

    def deterioration_stage(self):
        # the simulation of IoT for waste detection
        self.W = self.I_mat_tmp[self.t, max(self.t-args.k+1, 1): self.t+1] * self.weibull[-(self.t-max(self.t-args.k+1, 1)+1):]
        self.waste[self.t] = np.sum(self.W)
        # log.debug(f'member: {self.member} - t: {self.t} | W: {self.W}')
        # log.debug(f'member: {self.member} - t: {self.t} | waste: {self.waste[self.t]}')

    def sourcing_stage(self,
                       order: int,
                       available_I_upstream: int | float):
        # the simulation of IoT for inventory review. NI = the end-of-period net_inventory vector.
        self.NI = self.I_mat_tmp[self.t, max(self.t-args.k+1, 1): self.t+1] - self.W
        self.net_inventory[self.t] = np.sum(self.NI)

        # transfer the I_mat to the next period
        self.I_mat[self.t+1, max(self.t-args.k+1, 1): self.t+1] = self.NI
        # log.debug(f'member: {self.member} - t: {self.t} | next I_mat: {self.I_mat[self.t+1, max(self.t-args.k+1, 1): self.t+1]}')

        self.order[self.t] = order
        if self.member == "retailer":
            self.Q_mat[self.t+self.lr_dataset[self.episode, self.t]+1] += min(order, available_I_upstream)
        elif self.member == "wholesaler":
            self.Q_mat[self.t+self.lw_dataset[self.episode, self.t]+1] += min(order, available_I_upstream)
        # log.debug(f'member: {self.member} - t: {self.t} | Q_mat: {self.Q_mat[self.t: self.t+self.l_max+2]}')

    def compute_cost(self):
        # calculate the cost
        self.cost[self.t] = args.cost_order * self.order[self.t] + args.cost_holding * self.net_inventory[self.t] + args.cost_lost_sales * self.lost_sales[self.t] + args.cost_waste * self.waste[self.t]
        # log.debug(f'member: {self.member} - t: {self.t} | cost: {self.cost[self.t]}, order: {self.order[self.t]}, net_inventory: {self.net_inventory[self.t]}, lost_sales: {self.lost_sales[self.t]}, waste: {self.waste[self.t]}')

    def step(self,
             order: int,
             demand_downstream: int,
             available_I_upstream: int | float):
        self.reception_stage()
        self.serving_stage(demand_downstream)
        self.deterioration_stage()
        self.sourcing_stage(order, available_I_upstream)
        self.compute_cost()

class General_Replenishiment_Agent:
    def __init__(
        self,
        member: Literal["wholesaler", "retailer"] = "retailer"
    ):
        self.member = member
        self.d_forecast = np.zeros(args.time_window_length)
        self.alpha = 0.9
        self.beta = 0.1
        self.episode = 1
        self.t = 1

    def sync_time(self, episode: int, t: int):
        self.episode = episode
        self.t = t

    def order(self, self_env: Env, demand_downstream: int):
        d_f = self.alpha * demand_downstream + \
            (1-self.alpha) * self.d_forecast[self.t-1]
        self.d_forecast[self.t] = d_f

        self.SS = 3 * d_f
        self.DWIP = self_env.lr_dataset[self_env.episode, self.t] * d_f
        self.WIP = self_env.Q_mat[self.t: self.t+self_env.l_max+2].sum()
        self.NS = self_env.I_mat[self.t, max(self.t-args.k+1, 1): self.t+1].sum() + self_env.Q_mat[self.t]
        order = np.round(d_f + self.beta * (self.SS - self.NS) + self.beta * (self.DWIP - self.WIP))
        return order
