from config import args, log
from environment import Env
from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as torch_fn

import numpy as np

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

np.random.seed(args.seed)


# Define the neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # bn: batch normalization
        self.input_bn = nn.BatchNorm1d(args.input_size)

        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Create the first hidden layer
        self.hidden_layers.append(nn.Linear(args.input_size, args.hidden_size))
        self.bn_layers.append(nn.BatchNorm1d(args.hidden_size))

        # Create additional hidden layers
        for _ in range(args.num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(args.hidden_size, args.hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(args.hidden_size))

        self.output = nn.Linear(args.hidden_size, args.output_size)

    def forward(self, input_value):
        input_value = self.input_bn(input_value)
        x = input_value

        for hidden_layer, bn_layer in zip(self.hidden_layers, self.bn_layers):
            x = hidden_layer(x)
            x = bn_layer(x)
            x = torch_fn.relu(x)

        output_value = self.output(x)
        return output_value


# Define the DQN agent
class DQN_Replenishment_Agent:
    def __init__(self,
                 member: Literal["wholesaler", "retailer"] = "wholesaler"
                 ):
        self.member = member

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # state_len is number of state variables
        self.state_len = len(args.state_size)

        # 2 networks: main network and target network
        self.main_network = Network().to(self.device)
        self.target_network = Network().to(self.device)

        # loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=args.lr)

        # initialize replay buffer
        self.state_buffer = np.zeros((args.buffer_size, np.sum(args.state_size)))
        self.action_buffer = np.zeros((args.buffer_size, 1))
        self.reward_buffer = np.zeros((args.buffer_size, 1))
        self.next_state_buffer = np.zeros((args.buffer_size, np.sum(args.state_size)))
        self.done_buffer = np.zeros((args.buffer_size, 1))

        # initialize episode replay buffer
        self.reset_episode_buffer()

        self.buffer_counter = 1

        # initialize losses list
        self.losses = np.zeros(args.time_window_length)

        # initialize episode
        self.train_step_counter = 0

        # initialize episode and time step
        self.episode = 1
        self.t = 1

        # initialize epsilon
        self.epsilon = args.epsilon_min

    def sync_time(self, episode: int, t: int):
        self.episode = episode
        self.t = t

    def epsilon_decay(self):
        self.epsilon = min(args.epsilon_max, args.epsilon_min + (args.epsilon_max - args.epsilon_min) * self.episode / args.epsilon_decay_steps)
        # log.debug(f"epsilon: {self.epsilon}")
        return self.epsilon

    def choose_action(self, state):
        # self.epsilon = self.epsilon_decay()

        if np.random.uniform() > self.epsilon:
            action = np.random.randint(args.min_action, args.max_action+1)
        else:
            state = torch.unsqueeze(torch.FloatTensor(np.hstack(state)), 0).to(self.device)
            self.main_network.eval()
            q_value = self.main_network.forward(state)
            self.main_network.train()
            action = torch.argmax(q_value).item()

        # log.debug(f"member: wholesaler - t: {self.t} | action: {action}")
        return action

    def store_to_episode_buffer(self, state, action, reward, next_state, done):
        self.state_buffer_episode[self.t] = np.hstack(state)
        self.action_buffer_episode[self.t] = action
        self.reward_buffer_episode[self.t] = reward
        self.next_state_buffer_episode[self.t] = np.hstack(next_state)
        self.done_buffer_episode[self.t] = done

    def reset_episode_buffer(self):
        # reset episode buffer
        self.state_buffer_episode = np.zeros((args.time_window_length, np.sum(args.state_size)))
        self.action_buffer_episode = np.zeros((args.time_window_length, 1))
        self.reward_buffer_episode = np.zeros((args.time_window_length, 1))
        self.next_state_buffer_episode = np.zeros((args.time_window_length, np.sum(args.state_size)))
        self.done_buffer_episode = np.zeros((args.time_window_length, 1))
        self.losses = np.zeros(args.time_window_length)

    def store_to_buffer(self):
        for t in range(1, args.T+1):
            # "-1" because the buffer is 0-indexed, whereas buffer_counter is 1-indexed.
            index = self.buffer_counter % args.buffer_size - 1
            self.state_buffer[index] = self.state_buffer_episode[t]
            self.action_buffer[index] = self.action_buffer_episode[t]
            self.reward_buffer[index] = self.reward_buffer_episode[t]
            self.next_state_buffer[index] = self.next_state_buffer_episode[t]
            self.done_buffer[index] = self.done_buffer_episode[t]
            self.buffer_counter += 1

    def experience_replay(self):
        if self.buffer_counter <= args.buffer_size:
            batch_index = np.random.choice(self.buffer_counter, args.batch_size)
        else:
            batch_index = np.random.choice(args.buffer_size, args.batch_size)

        state_batch = torch.FloatTensor(self.state_buffer[batch_index]).to(self.device)
        action_batch = torch.LongTensor(self.action_buffer[batch_index]).to(self.device)
        reward_batch = torch.FloatTensor(self.reward_buffer[batch_index]).to(self.device)
        next_state_batch = torch.FloatTensor(self.next_state_buffer[batch_index]).to(self.device)
        done_batch = torch.FloatTensor(self.done_buffer[batch_index]).to(self.device)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def train(self):
        # update target network
        if self.train_step_counter % args.target_network_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        self.train_step_counter += 1

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.experience_replay()
        q_eval = self.main_network.forward(state_batch).gather(1, action_batch)
        q_next = self.target_network.forward(next_state_batch).detach()
        q_target = reward_batch + args.gamma * q_next.max(1)[0].unsqueeze(1) * (1 - done_batch)

        loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses[self.t] = loss.item()

    def reward_shaping(self, env: Env):
        if self.member == "wholesaler":
            l_max = args.lw_max
            l = env.lw_dataset[self.episode, :]
        elif self.member == "retailer":
            l_max = args.lr_max
            l = env.lr_dataset[self.episode, :]

        # The calculation of the zero-inventory gap (\left| \sum I_{t'} - d_{t'} \right|)
        # has been completed in environment.py. Here, it is directly invoked and denoted by zero_inventory_gap.
        zero_inventory_gap = env.zero_inventory_gap
        # let the order list be represented as [0, a1, a2, ..., aT, 0]
        # such that its dimensions correspond to those of env.I_after_meeting_demand
        order_list = np.vstack([0, self.action_buffer_episode, 0])

        # zero-inventory gap
        def P(t_):
            if np.abs(zero_inventory_gap[t_]) >= args.omega:
                # log.debug(f"t_: {t_} | zero_inventory_gap[t_]: {zero_inventory_gap[t_]}")
                return -np.abs(zero_inventory_gap[t_])
            else:
                return 0

        # penalty weight
        def Lambda(t, t_):
            # indicator function
            indicator = lambda x: 1 if x else 0
            Tau_ = lambda x: range(x+l[x]+1, x+l[x]+args.k + 1)

            denominator = 0
            # log.debug(f"t: {t}, t_: {t_} | zero_inventory_gap[t_]: {zero_inventory_gap[t_]}")
            if zero_inventory_gap[t_] >= 0:
                numerator = order_list[t] if order_list[t] != 0 else args.e
                denominator =  np.sum(np.array([(order_list[j] if order_list[j] != 0 else args.e) * indicator(t_ in Tau_(j)) for j in range(1, t_+1)], dtype=object))
            else:
                o_max = np.max([order_list[x]*indicator(t_ in Tau_(x)) for x in range(1, t_+1)])
                numerator = o_max-order_list[t] if o_max-order_list[t] != 0 else args.e
                denominator = np.sum(np.array([(o_max-order_list[j] if o_max-order_list[j] != 0 else args.e) * indicator(t_ in Tau_(j)) for j in range(1, t_+1)], dtype=object))
            # There are two cases where denominator = 0: (1) period t_ did not receive produce,
            # resulting in indicator(j+l[j]+1 == t_) = 0, (2) np.sum([]) = 0, which means that
            # the range 'for j in range(max(t_-l_max-1, 1), t_-2+1)' did not include any elements j,
            # also indicating that period t_ did not receive produce. Both cases are caused by
            # the lead time of these orders and are unrelated to the previous order behavior, so their
            # penalty weight lambda(t, t_) = 0.
            # log.debug(f"t: {t}, t_: {t_} | numerator: {numerator}, denominator: {denominator}")
            return numerator/denominator if denominator != 0 else 0

        # the time-related zero-inventory penalty
        def Phi(t):
            phi = 0
            # log.debug(f"t: {t}, t_: {t_} | Lambda(t, t_): {Lambda(t, t_)}, P(t_): {P(t_)}")
            for t_ in range(t+l[t]+1, min(t+l[t]+args.k+1, (args.T)+1)):
                phi += args.rho * args.gamma**(t_-t) * Lambda(t, t_) * P(t_)  # Period (args.T+1) is greater 1 than the max period T, becuase the next state of max periid occurs at period (T+1).

            return phi

        # cumulative penalty-based shaping function
        def f(t):
            return Phi(t) - args.gamma*Phi(t+1)


        for t in range(1, args.T+1):
            self.reward_buffer_episode[t] += f(t)
