import copy
import numpy as np

from config import args
from typing import Literal

from DQN_SV_RS.environment import Env, General_Replenishiment_Agent

import torch
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


class Demand_LSTM(nn.Module):
    def __init__(self):
        super(Demand_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=args.input_D_size,
            hidden_size=args.hidden_D_size,
            num_layers=1,
            batch_first=True,
        )

        self.linear = nn.Linear(args.hidden_D_size, args.out_D_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        out, _ = self.lstm(input)
        out = self.linear(out)
        out = self.relu(out)
        return out

class Leadtime_LSTM(nn.Module):
    def __init__(self):
        super(Leadtime_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=args.input_L_size,
            hidden_size=args.hidden_L_size,
            num_layers=1,
            batch_first=True,
        )

        self.linear = nn.Linear(args.hidden_L_size, args.out_L_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        out, _ = self.lstm(input)
        out = self.linear(out)
        out = self.relu(out)
        return out


class Rolling_BlE_Replenishment_Agent:
    def __init__(
        self,
        member: Literal["wholesaler", "retailer"] = "wholesaler"
    ):
        self.member = member

        # setups
        self.history_demand_pool = np.array([])
        self.history_leadtime_pool = np.array([])
        self.label_demand_pool = np.array([])
        self.label_leadtime_pool = np.array([])

        self.history_demand_batch = None
        self.history_leadtime_batch = None
        self.label_demand_batch = None
        self.label_leadtime_batch = None

        # define the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        # set up the model
        self.demand_model = Demand_LSTM().to(self.device)
        self.leadtime_model = Leadtime_LSTM().to(self.device)

        # set up the optimizer, loss function and learning rate scheduler
        self.demand_optimizer = optim.Adam(self.demand_model.parameters(), lr=args.lr)
        self.leadtime_optimizer = optim.Adam(self.leadtime_model.parameters(), lr=args.lr)

        self.demand_loss_function = nn.MSELoss()
        self.leadtime_loss_function = nn.MSELoss()

        self.reset_losses()

    def reset_losses(self):
        self.losses = []

    def sync_time(self, episode: int, t: int):
        self.episode = episode
        self.t = t

    def get_history_data(self, demand_data, leadtime_data, t):
        if t <= args.tau:
                # if t is less than tau, pad with zeros
            history_demand = np.hstack(
                    (-np.zeros(args.tau-len(demand_data[1: t])), demand_data[1: t])
                )
            history_leadtime = np.hstack(
                    (-np.zeros(args.tau-len(leadtime_data[1: t])), leadtime_data[1: t])
                )
        else:
            history_demand = demand_data[t-args.tau: t]
            history_leadtime = leadtime_data[t-args.tau: t]
        return history_demand, history_leadtime

    def process_data(
        self,
        demand_data: np.array,
        leadtime_data: np.array,
    ):
        for t in range(1, args.T+1):
            history_demand, history_leadtime = self.get_history_data(demand_data, leadtime_data, t)

            if t + args.out_D_size <= args.T:
                label_demand = demand_data[t: t + args.out_D_size]
            else:
                # if t + out_D_size is greater than T, pad with zeros
                label_demand = np.hstack((demand_data[t: args.T + 1], -np.zeros(args.out_D_size - len(demand_data[t: args.T + 1]))))

            if t + args.out_L_size <= args.T:
                label_leadtime = leadtime_data[t: t + args.out_L_size]
            else:
                # if t + out_L_size is greater than T, pad with zeros
                label_leadtime = np.hstack((leadtime_data[t: args.T + 1], -np.zeros(args.out_L_size - len(leadtime_data[t: args.T + 1]))))

            if self.history_demand_pool.size == 0:
                self.history_demand_pool = history_demand
                self.history_leadtime_pool = history_leadtime
                self.label_demand_pool = label_demand
                self.label_leadtime_pool = label_leadtime

            self.history_demand_pool = np.vstack((self.history_demand_pool, history_demand))
            self.history_leadtime_pool = np.vstack((self.history_leadtime_pool, history_leadtime))
            self.label_demand_pool = np.vstack((self.label_demand_pool, label_demand))
            self.label_leadtime_pool = np.vstack((self.label_leadtime_pool, label_leadtime))

            # keep the last args.buffer_size samples
            if self.history_demand_pool.shape[0] > args.buffer_size:
                self.history_demand_pool = self.history_demand_pool[-args.buffer_size:]
                self.history_leadtime_pool = self.history_leadtime_pool[-args.buffer_size:]
                self.label_demand_pool = self.label_demand_pool[-args.buffer_size:]
                self.label_leadtime_pool = self.label_leadtime_pool[-args.buffer_size:]

    def sample(self):
        total_samples = self.history_demand_pool.shape[0]
        # Randomly sample args.batch_size samples from total_samples samples, samples can be drawn with replacement.
        sample_indices = np.random.choice(total_samples, args.batch_size, replace=True)
        self.history_demand_batch = self.history_demand_pool[sample_indices, :]
        self.history_leadtime_batch = self.history_leadtime_pool[sample_indices, :]
        self.label_demand_batch = self.label_demand_pool[sample_indices, :]
        self.label_leadtime_batch = self.label_leadtime_pool[sample_indices, :]

    def train(
        self,
        env: Env
    ):
        if self.member == 'retailer':
            demand_data = env.dm
            leadtime_data = env.lr
        elif self.member == 'wholesaler':
            demand_data = env.dr
            leadtime_data = env.lw

        self.process_data(demand_data, leadtime_data)

        self.demand_model.train()
        self.leadtime_model.train()
        # train 5 times
        for _ in range(5):
            # sample data
            self.sample()

            # convert data to torch.Tensor
            history_demand_batch = torch.Tensor(self.history_demand_batch).to(self.device)
            history_leadtime_batch = torch.Tensor(self.history_leadtime_batch).to(self.device)
            label_demand_batch = torch.Tensor(self.label_demand_batch).to(self.device)
            label_leadtime_batch = torch.Tensor(self.label_leadtime_batch).to(self.device)

            # zero the parameter gradients
            self.demand_optimizer.zero_grad()
            self.leadtime_optimizer.zero_grad()

            # forward + backward + optimize
            demand_outputs = self.demand_model(history_demand_batch)
            leadtime_outputs = self.leadtime_model(history_leadtime_batch)

            demand_loss = self.demand_loss_function(demand_outputs, label_demand_batch)
            leadtime_loss = self.leadtime_loss_function(leadtime_outputs, label_leadtime_batch)

            demand_loss.backward()
            leadtime_loss.backward()

            self.demand_optimizer.step()
            self.leadtime_optimizer.step()

            self.losses.append((demand_loss.item()+leadtime_loss.item())/2)

    def eval(
        self,
        history_demand: np.array,
        history_leadtime: np.array
    ):
        self.demand_model.eval()
        self.leadtime_model.eval()

        history_demand = torch.Tensor(history_demand).unsqueeze(0).to(self.device)
        history_demand[history_demand == -1] = 0

        history_leadtime = torch.Tensor(history_leadtime).unsqueeze(0).to(self.device)
        history_leadtime[history_leadtime == -1] = 0

        demand_outputs = self.demand_model(history_demand)
        leadtime_outputs = self.leadtime_model(history_leadtime)

        return demand_outputs.detach().numpy().flatten(), leadtime_outputs.detach().numpy().flatten()[0]

    def BlE(
        self,
        t: int,
        S1_star: int,
        S2_star: int,
        B: int,
        env: Env,
    ):
        A = 1 - (S2_star-S1_star)/B
        IP = env.I_mat[t, :].sum() + env.Q_mat[t]
        W = env.waste[t]

        if IP < B:
            order = round(max(S1_star - A*IP + W, 0))
        else:
            order = round(max(S2_star - IP + W, 0))

        return order

    def Rolling(
        self,
        retailer,
        retailer_env_list,
        wholesaler,
        wholesaler_env_list,
    ):
        # copy the environment
        t_ = max(self.t - 5, 1)
        retailer_env_copy  = copy.deepcopy(retailer_env_list[t_])
        wholesaler_env_copy  = copy.deepcopy(wholesaler_env_list[t_])

        if self.member == 'retailer':
            demand_data = retailer_env_list[self.t].dm[:self.t]
            leadtime_data = retailer_env_list[self.t].lr[:self.t]
            member = retailer
            member_env = retailer_env_copy
        elif self.member == 'wholesaler':
            demand_data = wholesaler_env_list[self.t].dr[:self.t]
            leadtime_data = wholesaler_env_list[self.t].lw[:self.t]
            member = wholesaler
            member_env = wholesaler_env_copy

        # deepcopy these values for reset
        episode_copy = copy.deepcopy(member_env.episode)
        t_copy = copy.deepcopy(member_env.t)
        I_mat_copy = copy.deepcopy(member_env.I_mat)
        I_mat_tmp_copy = copy.deepcopy(member_env.I_mat_tmp)
        Q_mat_copy = copy.deepcopy(member_env.Q_mat)
        dm_copy = copy.deepcopy(member_env.dm)
        lr_copy = copy.deepcopy(member_env.lr)
        if self.member == 'wholesaler':
            dr_copy = copy.deepcopy(member_env.dr)
            lw_copy = copy.deepcopy(member_env.lw)
        order_copy = copy.deepcopy(member_env.order)
        net_inventory_copy = copy.deepcopy(member_env.net_inventory)
        lost_sales_copy = copy.deepcopy(member_env.lost_sales)
        waste_copy = copy.deepcopy(member_env.waste)
        cost_copy = copy.deepcopy(member_env.cost)

        # get the history data
        history_demand, history_leadtime = self.get_history_data(demand_data, leadtime_data, self.t)

        # predict the demand and leadtime
        predicted_demand, predicted_leadtime = self.eval(history_demand, history_leadtime)

        # concatenate the historical and predicted demand
        historical_and_predicted_demand = np.hstack((demand_data, predicted_demand))

        S1_star, S2_star, B_star, cost_min = 0, 0, 0, float('inf')
        S_max = int(4*(args.dm_max+args.dm_min)/2)

        for middle in range(1, S_max+1):
            for S1 in range(1, middle+1):
                for S2 in range(middle+1, S_max+1):
                    for B in range(1, middle+1):
                        # reset the environment
                        member_env.I_mat = copy.deepcopy(I_mat_copy)
                        member_env.I_mat_tmp = copy.deepcopy(I_mat_tmp_copy)
                        member_env.Q_mat = copy.deepcopy(Q_mat_copy)
                        member_env.dm = copy.deepcopy(dm_copy)
                        member_env.lr = copy.deepcopy(lr_copy)
                        if self.member == 'wholesaler':
                            member_env.dr = copy.deepcopy(dr_copy)
                            member_env.lw = copy.deepcopy(lw_copy)
                        member_env.order = copy.deepcopy(order_copy)
                        member_env.net_inventory = copy.deepcopy(net_inventory_copy)
                        member_env.lost_sales = copy.deepcopy(lost_sales_copy)
                        member_env.waste = copy.deepcopy(waste_copy)
                        member_env.cost = copy.deepcopy(cost_copy)

                        cost_list = []
                        for t in range(max(t_copy - 5, 1), min(t_copy+int(predicted_leadtime)+1, args.T)+1):
                            # load the demand
                            demand_upsteam = historical_and_predicted_demand[t]
                            # wholesaler replenishment
                            # wholesaler syncnize the episode and time
                            member.sync_time(episode_copy, t)
                            member_env.sync_time(episode_copy, t)

                            member_env.get_state(t)

                            member_env.reception_stage()
                            member_env.serving_stage(demand_upsteam)
                            member_env.deterioration_stage()

                            order = member.BlE(t, S1, S2, B, member_env)
                            member_env.sourcing_stage(order, 1e6)
                            member_env.compute_cost()
                            cost_list.append(member_env.cost[t])

                        if np.sum(cost_list) <= cost_min:
                            S1_star, S2_star, B_star = S1, S2, B
                            cost_min = np.sum(cost_list)
        # because the environment is changed, we need to reset the period t
        self.t = t_copy

        return S1_star, S2_star, B_star

    def order(
        self,
        retailer,
        retailer_env_list,
        wholesaler,
        wholesaler_env_list,
    ):
        S1_star, S2_star, B_star = self.Rolling(retailer, retailer_env_list, wholesaler, wholesaler_env_list)

        if self.member == 'retailer':
            order = self.BlE(self.t, S1_star, S2_star, B_star, retailer_env_list[-1])
        elif self.member == 'wholesaler':
            order = self.BlE(self.t, S1_star, S2_star, B_star, wholesaler_env_list[-1])

        return order

    def save_model(self, episode):
        torch.save(self.demand_model.state_dict(), f"model/seed={args.seed}/{args.tuning_param_middle_path}/state_dict/Rolling_BlE_demand_{episode}.pth")
        torch.save(self.leadtime_model.state_dict(), f"model/seed={args.seed}/{args.tuning_param_middle_path}/state_dict/Rolling_BlE_leadtime_{episode}.pth")

        torch.save(self.demand_model, f"model/seed={args.seed}/{args.tuning_param_middle_path}/complete_model/Rolling_BlE_demand_{episode}.pth")
        torch.save(self.leadtime_model, f"model/seed={args.seed}/{args.tuning_param_middle_path}/complete_model/Rolling_BlE_leadtime_{episode}.pth")

    def load_model(self, episode):
        self.demand_model.load_state_dict(torch.load(f"model/seed={args.seed}/{args.tuning_param_middle_path}/state_dict/Rolling_BlE_demand_{episode}.pth"))
        self.leadtime_model.load_state_dict(torch.load(f"model/seed={args.seed}/{args.tuning_param_middle_path}/state_dict/Rolling_BlE_leadtime_{episode}.pth"))

