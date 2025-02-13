from config import args
from typing import Literal, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# class MQRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_quantiles=args.num_quantiles):
#         super(MQRNN, self).__init__()
#         self.rnn = nn.RNN(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=1,
#             batch_first=True,
#         )

#         self.linear = nn.Linear(hidden_size, num_quantiles)

#     def forward(self, x):
#         rnn_out, _ = self.rnn(x)
#         # Get the output from the last time step
#         rnn_out = rnn_out[:, -1, :]
#         quantile_out = self.linear(rnn_out)
#         return quantile_out

class DF_submodule(nn.Module):
    def __init__(self):
        super(DF_submodule, self).__init__()
        self.rnn = nn.RNN(
            input_size=args.input_DF_size,
            hidden_size=args.hidden_DF_size,
            num_layers=1,
            batch_first=True,
        )

        # self.mqrnn = MultiQuantileRNN(
        #     input_size=args.hidden_DF_size,
        #     hidden_size=args.hidden_DF_size,
        #     num_quantiles=args.num_quantiles
        # )

        self.linear = nn.Linear(args.hidden_DF_size, args.out_2_size)
        self.relu = nn.ReLU()

    def forward(self, input_DF):
        self.rnn.flatten_parameters()
        out_2, _ = self.rnn(input_DF)
        out_2 = self.linear(out_2[:, -1, :])
        return out_2.view(-1)

def quantile_loss(y_pred, y_true, quantiles):
    errors = y_true - y_pred
    loss = torch.max((quantiles - 1) * errors, quantiles * errors)
    return torch.mean(loss)

class VLT_submodule(nn.Module):
    def __init__(self):
        super(VLT_submodule, self).__init__()
        self.vlt_layer_1 = nn.Linear(args.input_VLT_size, args.hidden_VLT_size)
        self.vlt_layer_2 = nn.Linear(args.hidden_VLT_size, args.out_3_size)
        self.dropout = nn.Dropout(p=args.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, input_VLT):
        out_3 = self.dropout(self.relu(self.vlt_layer_1(input_VLT)))
        out_3 = self.vlt_layer_2(out_3)
        return out_3

class Integration_module(nn.Module):
    def __init__(
        self,
        member: Literal["wholesaler", "retailer"] = "wholesaler"
    ):
        super(Integration_module, self).__init__()
        self.member = member

        if self.member == "retailer":
            self.DF_submodule_dm = DF_submodule()
            self.VLT_submodule_lr = VLT_submodule()
            self.layer_3 = nn.Linear(args.input_I_size, args.hidden_I_size)
            self.layer_4 = nn.Linear(args.hidden_I_size+args.init_inventory_size+1, args.out_1_size)
        elif self.member == "wholesaler":
            self.DF_submodule_dm = DF_submodule()
            self.DF_submodule_dr = DF_submodule()
            self.VLT_submodule_lr = VLT_submodule()
            self.VLT_submodule_lw = VLT_submodule()
            self.layer_3 = nn.Linear(args.input_I_size*2, args.hidden_I_size)
            self.layer_4 = nn.Linear(args.hidden_I_size+args.init_inventory_size+1, args.out_1_size)

        self.dropout = nn.Dropout(p=args.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, input_DF_dm, input_VLT_lr, init_inventory, input_Q, input_DF_dr=None, input_VLT_lw=None):
        if self.member == "retailer":
            out_2 = self.DF_submodule_dm(input_DF_dm)
            out_3 = self.VLT_submodule_lr(input_VLT_lr)
            layer_3_input = torch.cat((out_2, out_3))
            layer_3_output = self.dropout(self.relu(self.layer_3(layer_3_input)))
            layer_4_input = torch.cat((layer_3_output, init_inventory, input_Q))
            out_1 = self.layer_4(layer_4_input)
        elif self.member == "wholesaler":
            out_2_dm = self.DF_submodule_dm(input_DF_dm)
            out_2_dr = self.DF_submodule_dr(input_DF_dr)
            out_3_lr = self.VLT_submodule_lr(input_VLT_lr)
            out_3_lw = self.VLT_submodule_lw(input_VLT_lw)
            layer_3_input = torch.cat((out_2_dm, out_2_dr, out_3_lr, out_3_lw))
            layer_3_output = self.dropout(self.relu(self.layer_3(layer_3_input)))
            layer_4_input = torch.cat((layer_3_output, init_inventory, input_Q))
            out_1 = self.layer_4(layer_4_input)
            out_2 = out_2_dr
            out_3 = out_3_lw
        return out_1, out_2, out_3

class E2E_Replenishment_Agent:
    def __init__(
        self,
        member: Literal["wholesaler", "retailer"] = "wholesaler"
    ):
        self.member = member

        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2

        # define the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.intg_model = Integration_module(self.member).to(self.device)

        # define the loss function and optimizer
        self.loss_df = nn.MSELoss()
        self.loss_vlt = nn.MSELoss()
        self.loss_intg = nn.MSELoss()
        # self.quantiles = torch.tensor([0.1, 0.5, 0.9])
        # self.loss_df = quantile_loss

        self.optim = optim.Adam(self.intg_model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
        self.lr_sched = lr_scheduler.ExponentialLR(self.optim, gamma=1-args.learning_rate_decay)

        # reset the losses
        self.reset_losses()

    def reset_losses(self):
        self.losses = np.zeros(args.time_window_length)

    def sync_time(self, episode: int, t: int):
        self.episode = episode
        self.t = t

    def train(
        self,
        label_DF: np.array,
        label_VLT: np.array,
        label_order: np.array,
        input_DF_dm: np.array,
        input_VLT_lr: np.array,
        init_inventory: np.array,
        input_Q: np.array,
        input_DF_dr: np.array = np.array([]),
        input_VLT_lw: np.array = np.array([])
    ):
        label_DF = torch.Tensor([label_DF]).to(self.device)
        # label_DF[label_DF == -1] = 0
        label_VLT = torch.Tensor([label_VLT]).to(self.device)
        # label_VLT[label_VLT == -1] = 0
        label_order = torch.Tensor([label_order]).to(self.device)
        # label_order[label_order == -1] = 0

        input_DF_dm = torch.Tensor(input_DF_dm).view(1, 1, -1).to(self.device)
        # input_DF_dm[input_DF_dm == -1] = 0
        input_VLT_lr = torch.Tensor(input_VLT_lr).to(self.device)
        # input_VLT_lr[input_VLT_lr == -1] = 0
        init_inventory = torch.Tensor(init_inventory).to(self.device)
        # init_inventory[init_inventory == -1] = 0
        input_Q = torch.Tensor([input_Q]).to(self.device)
        # input_Q[input_Q == -1] = 0

        if self.member == "wholesaler":
            input_DF_dr = torch.Tensor(input_DF_dr).view(1, 1, -1).to(self.device)
            input_DF_dr[input_DF_dr == -1] = 0
            input_VLT_lw = torch.Tensor(input_VLT_lw).to(self.device)
            input_VLT_lw[input_VLT_lw == -1] = 0

        # forward pass
        if self.member == "retailer":
            output_order, output_DF, output_VLT = self.intg_model.forward(input_DF_dm, input_VLT_lr, init_inventory, input_Q)
        elif self.member == "wholesaler":
            output_order, output_DF, output_VLT = self.intg_model.forward(input_DF_dm, input_VLT_lr, init_inventory, input_Q, input_DF_dr, input_VLT_lw)

        # compute the loss
        loss_DF = self.loss_df(output_DF, label_DF)
        # loss_DF = self.loss_df(output_DF, [label_DF for _ in range(args.num_quantiles)], self.quantiles)
        loss_VLT = self.loss_vlt(output_VLT, label_VLT)
        loss_order = self.loss_intg(output_order, label_order)
        loss = self.lambda_1 * loss_DF + self.lambda_2 * loss_VLT + loss_order

        # backward pass
        self.optim.zero_grad()
        loss.backward()

        # update the weights
        self.optim.step()

        # update lr scheduler
        self.lr_sched.step()

        self.loss_value = loss.item()
        self.losses[self.t] = self.loss_value

    def eval(
        self,
        input_DF: np.array,
        input_VLT: np.array,
        init_inventory: np.array,
        input_Q: Union[np.array, float],
        input_DF_dr: np.array = np.array([]),
        input_VLT_lw: np.array = np.array([])
    ):
        # In MQRNN, it is better to set missing values to 0 than to -1, so we replace them.
        # To unify, we also replace the data used by the fully connected layer.
        input_DF = torch.Tensor(input_DF).view(1, 1, -1).to(self.device)
        # input_DF[input_DF == -1] = 0
        input_VLT = torch.Tensor(input_VLT).to(self.device)
        # input_VLT[input_VLT == -1] = 0
        init_inventory = torch.Tensor(init_inventory).to(self.device)
        # init_inventory[init_inventory == -1] = 0
        input_Q = torch.Tensor([input_Q]).to(self.device)
        # input_Q[input_Q == -1] = 0

        if self.member == "wholesaler":
            input_DF_dr = torch.Tensor(input_DF_dr).view(1, 1, -1).to(self.device)
            # input_DF_dr[input_DF_dr == -1] = 0
            input_VLT_lw = torch.Tensor(input_VLT_lw).to(self.device)
            # input_VLT_lw[input_VLT_lw == -1] = 0

        self.intg_model.eval()

        if self.member == "retailer":
            output_order, output_DF, output_VLT = self.intg_model(input_DF, input_VLT, init_inventory, input_Q)
        elif self.member == "wholesaler":
            output_order, output_DF, output_VLT = self.intg_model(input_DF, input_VLT, init_inventory, input_Q, input_DF_dr, input_VLT_lw)

        output_order = max(round(output_order.item()), 0)
        output_DF = max(round(output_DF.item()), 0)
        output_VLT = max(round(output_VLT.item()), 0)

        self.intg_model.train()
        return output_order, output_DF, output_VLT

    def save_model(self, episode):
        torch.save(self.intg_model.state_dict(), f"model/seed={args.seed}/{args.tuning_param_middle_path}/state_dict/E2E_{episode}.pth")
        torch.save(self.intg_model, f"model/seed={args.seed}/{args.tuning_param_middle_path}/complete_model/E2E_{episode}.pth")

    def load_model(self, episode):
        self.intg_model.load_state_dict(torch.load(f"model/seed={args.seed}/{args.tuning_param_middle_path}/state_dict/E2E_{episode}.pth"))
