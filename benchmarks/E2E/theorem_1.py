import copy
import numpy as np
from config import args
from DQN_SV_RS.environment import Env, General_Replenishiment_Agent
from E2E import E2E_Replenishment_Agent


def get_order_label(
    episode: int,
    t: int,
    dm: np.array,
    lw: np.array,
    retailer: General_Replenishiment_Agent,
    retailer_env_train: Env,
    wholesaler: E2E_Replenishment_Agent,
    wholesaler_env_train: Env,
) -> float | None:
    # $s^{*} = \left \lfloor \frac{b(v_{m+1}-v_m)}{h+b} \right \rfloor + v_m$
    v_m = t + lw[episode, t] + 1
    v_m_plus_1 = (t+1) + lw[episode, t+1] + 1
    b = args.cost_lost_sales
    h = args.cost_holding
    s_star = int(np.floor((b * (v_m_plus_1 - v_m)) / (h + b)) + v_m)

    # copy retailer, wholesaler, and their environments to avoid changing the original objects
    retailer_copy = copy.deepcopy(retailer)
    wholesaler_copy = copy.deepcopy(wholesaler)

    retailer_env_train_copy = copy.deepcopy(retailer_env_train)
    wholesaler_env_train_copy = copy.deepcopy(wholesaler_env_train)

    # calculate the order label: $a_m^{**}=\max\{d_{[v_m,s^*]}-I_{v_m},0\}$, where $d_{[i,j]}:=\sum_{t=i}^jd_t$
    if s_star <= args.T:
        d_sum = 0
        I_sum = 0

        for t_ in range(t, s_star+1):
            # load the market demand
            demand_market = dm[episode, t]

            # retailer replenishment
            # retailer syncnize the episode and time
            retailer_copy.sync_time(episode, t_)
            retailer_env_train_copy.sync_time(episode, t_)
            # retailer order decision
            retailer_order = retailer_copy.order(retailer_env_train_copy, demand_market)
            # retailer environment accept the order and update itself
            retailer_env_train_copy.step(
                retailer_order,
                demand_market,
                wholesaler_env_train_copy.I_mat[t_, :].sum()
            )

            # load the retailer demand
            demand_retailer = retailer_env_train_copy.order[t_]

            # wholesaler replenishment
            # wholesaler syncnize the episode and time
            wholesaler_copy.sync_time(episode, t_)
            wholesaler_env_train_copy.sync_time(episode, t_)

            # input
            state = wholesaler_env_train_copy.get_state(t_)
            input_DF_dm = state[5]
            input_VLT_lr = state[3]
            init_inventory = state[0]
            input_Q = state[1]
            input_DF_dr = state[4]
            input_VLT_lw = state[2]

            # output
            wholesaler_order, _, _ = wholesaler_copy.eval(input_DF_dm, input_VLT_lr, init_inventory, input_Q, input_DF_dr, input_VLT_lw)

            # wholesaler environment accept the order and update itself
            wholesaler_env_train_copy.step(
                wholesaler_order,
                demand_retailer,
                1e6  # Assume that the available inventories of producers are sufficient (so we set at 10^6) for wholesalers to order any quantity, as stated in Assumption (2) of our paper.
            )

            if t_ >= t+lw[episode, t]+1:
                d_sum += demand_retailer

            if t_ == t+lw[episode, t]+1:
                I_sum = init_inventory.sum()

        label_order = max(d_sum - I_sum, 0)
    else:
        label_order = None
    return label_order