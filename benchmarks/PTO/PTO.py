import numpy as np

from config import args
from typing import Literal

from DQN_SV_RS.environment import Env

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


class PTO_Replenishment_Agent:
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

        return demand_outputs.detach().numpy().flatten(), leadtime_outputs.detach().numpy().flatten()

    def predict(self, env):
        if self.member == 'retailer':
            demand_data = env.dm[:self.t]
            leadtime_data = env.lr[:self.t]
        elif self.member == 'wholesaler':
            demand_data = env.dr[:self.t]
            leadtime_data = env.lw[:self.t]

        # get the history data
        history_demand, history_leadtime = self.get_history_data(demand_data, leadtime_data, self.t)

        # predict the demand and leadtime
        predicted_demand, predicted_leadtime = self.eval(history_demand, history_leadtime)

        # concatenate the historical and predicted demand
        historical_and_predicted_demand = np.hstack((demand_data, predicted_demand))
        return historical_and_predicted_demand, predicted_leadtime

    def optimize(
        self,
        env: Env,
        historical_and_predicted_demand,
        predicted_leadtime
    ):
        first_order_arrival_time = env.t + predicted_leadtime[0] + 1
        second_order_arrival_time = (env.t+1) + predicted_leadtime[1] + 1
        t1 = round(first_order_arrival_time)
        t2 = round(min(second_order_arrival_time, args.T+1) if second_order_arrival_time >= first_order_arrival_time else min(first_order_arrival_time+1, args.T+1))

        NI = env.I_mat[env.t, :].sum() + env.Q_mat[env.t]

        order_best, cost_min = None, float('inf')

        for order in range(args.min_action, args.max_action+1):
            cost = 0

            for t in range(t1, t2):
                cost += args.cost_order*order \
                      + args.cost_holding*max((NI+order-historical_and_predicted_demand[env.t:t+1].sum()), 0) \
                      + args.cost_lost_sales*max((historical_and_predicted_demand[env.t:t+1].sum()-NI-order), 0)

            if cost < cost_min:
                cost_min = cost
                order_best = order
        return order_best

    def order(
        self,
        env,
    ):
        # predict the demand and leadtime
        historical_and_predicted_demand, predicted_leadtime = self.predict(env)
        order = self.optimize(env, historical_and_predicted_demand, predicted_leadtime)
        return order

    def save_model(self, episode):
        torch.save(self.demand_model.state_dict(), f"model/seed={args.seed}/{args.tuning_param_middle_path}/state_dict/Rolling_BlE_demand_{episode}.pth")
        torch.save(self.leadtime_model.state_dict(), f"model/seed={args.seed}/{args.tuning_param_middle_path}/state_dict/Rolling_BlE_leadtime_{episode}.pth")

        torch.save(self.demand_model, f"model/seed={args.seed}/{args.tuning_param_middle_path}/complete_model/Rolling_BlE_demand_{episode}.pth")
        torch.save(self.leadtime_model, f"model/seed={args.seed}/{args.tuning_param_middle_path}/complete_model/Rolling_BlE_leadtime_{episode}.pth")

    def load_model(self, episode):
        self.demand_model.load_state_dict(torch.load(f"model/seed={args.seed}/{args.tuning_param_middle_path}/state_dict/Rolling_BlE_demand_{episode}.pth"))
        self.leadtime_model.load_state_dict(torch.load(f"model/seed={args.seed}/{args.tuning_param_middle_path}/state_dict/Rolling_BlE_leadtime_{episode}.pth"))

