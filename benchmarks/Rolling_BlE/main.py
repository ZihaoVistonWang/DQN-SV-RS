import sys
import os

# add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tqdm import tqdm
from config import args, log, Data_Store, time
from DQN_SV_RS.dateset_generator import Cosine_Dataset_Generator
from DQN_SV_RS.environment import Env, General_Replenishiment_Agent
from Rolling_BlE import Rolling_BlE_Replenishment_Agent
from autodl_msg import send_msg


# generate the dataset
data_gen = Cosine_Dataset_Generator()

train_d_t, train_lr_t, train_lw_t = data_gen.generate_train_dataset()

# initialize the agents and environments for wholesaler and retailer
retailer = General_Replenishiment_Agent("retailer")
wholesaler = Rolling_BlE_Replenishment_Agent("wholesaler")

retailer_env = Env(args, "retailer", "train")

wholesaler_env = Env(args, "wholesaler", "train")

# initialize the data store
data_store = Data_Store(
    type_="test",
    whether_to_record_retailer_loss=False,
    whether_to_record_wholesaler_loss=True,
    whether_to_record_retailer_env_data=True,
    whether_to_record_wholeshaler_env_data=True,
    )

log.info_('The wholesaler and retailer and their environments have been initialized.')

log.info(f"{'Start':*^75}")

for episode in tqdm(
    range(1, args.train_episodes+1),
    desc="Training",
    leave=True,
    dynamic_ncols=True,
    position=0,
    colour="green",
    postfix=f"{args.tuning_param_middle_path:<15}",
    ):
    # reset the environment and update agent epsilon
    retailer_env.reset()
    wholesaler_env.reset()

    retailer_env_list = [None]
    wholesaler_env_list = [None]

    for t in tqdm(
        range(1, args.T+1),
        position=1,
        leave=False,
        dynamic_ncols=True,
        colour="yellow"
        ):
        # load the market demand
        demand_market = train_d_t[episode, t]

        # retailer replenishment
        # retailer syncnize the episode and time
        retailer.sync_time(episode, t)
        retailer_env.sync_time(episode, t)

        retailer_env_list.append(retailer_env)

        # retailer order decision
        retailer_order = retailer.order(retailer_env, demand_market)
        # retailer environment accept the order and update itself
        retailer_env.step(
            retailer_order,
            demand_market,
            wholesaler_env.I_mat[t, :].sum()
        )

        # load the retailer demand
        demand_retailer = retailer_env.order[t]

        # wholesaler replenishment
        # wholesaler syncnize the episode and time
        wholesaler.sync_time(episode, t)
        wholesaler_env.sync_time(episode, t)

        wholesaler_env.get_state(t)
        wholesaler_env_list.append(wholesaler_env)

        wholesaler_env.reception_stage()
        wholesaler_env.serving_stage(demand_retailer)
        wholesaler_env.deterioration_stage()

        wholesaler_order = wholesaler.order(
            retailer,
            retailer_env_list,
            wholesaler,
            wholesaler_env_list
        )
        wholesaler_env.sourcing_stage(wholesaler_order, 1e6)
        wholesaler_env.compute_cost()

    # a series of operations executed after the conclusion of each episode
    # train
    wholesaler.train(wholesaler_env)

    # store data during this episode
    data_store.record(
        episode,
        episode,
        retailer,
        wholesaler,
        retailer_env,
        wholesaler_env,
    )
    # save the data about cost, net inventory, lost sale, waste and loss
    data_store.save()

    # reset the losses
    wholesaler.reset_losses()
    # save the model
    wholesaler.save_model(episode)

log.info(f"{'End':*^75}")
log.info_(f"Runing time: {(time.time()-args.start_time)/60:.2f} mins.")

# send message to WeChat by Autodl.com
send_msg()