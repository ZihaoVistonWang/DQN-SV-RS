import sys
import os

# add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tqdm import tqdm
from config import args, log, Data_Store, time
from DQN_SV_RS.dateset_generator import Cosine_Dataset_Generator
from DQN_SV_RS.environment import Env, General_Replenishiment_Agent
from PTO import PTO_Replenishment_Agent
from autodl_msg import send_msg


# generate the dataset
data_gen = Cosine_Dataset_Generator()

train_d_t, train_lr_t, train_lw_t = data_gen.generate_train_dataset()
test_d_t, test_lr_t, test_lw_t = data_gen.generate_test_dataset()

# initialize the agents and environments for wholesaler and retailer
retailer = General_Replenishiment_Agent("retailer")
wholesaler = PTO_Replenishment_Agent("wholesaler")

retailer_env_train = Env(args, "retailer", "train")
retailer_env_test = Env(args, "retailer", "test")

wholesaler_env_train = Env(args, "wholesaler", "train")
wholesaler_env_test = Env(args, "wholesaler", "test")

# initialize the data store
train_data_store = Data_Store(
    type_="train",
    whether_to_record_retailer_loss=False,
    whether_to_record_wholesaler_loss=True,
    whether_to_record_retailer_env_data=True,
    whether_to_record_wholeshaler_env_data=True,
    )
test_data_store = Data_Store(
    type_="test",
    whether_to_record_retailer_loss=False,
    whether_to_record_wholesaler_loss=False,
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
    retailer_env_train.reset()

    wholesaler_env_train.reset()

    for t in range(1, args.T+1):
        # load the market demand
        demand_market = train_d_t[episode, t]

        # retailer replenishment
        # retailer syncnize the episode and time
        retailer.sync_time(episode, t)
        retailer_env_train.sync_time(episode, t)
        # retailer order decision
        retailer_order = retailer.order(retailer_env_train, demand_market)
        # retailer environment accept the order and update itself
        retailer_env_train.step(
            retailer_order,
            demand_market,
            wholesaler_env_train.I_mat[t, :].sum()
        )

        # load the retailer demand
        demand_retailer = retailer_env_train.order[t]

        # wholesaler replenishment
        # wholesaler syncnize the episode and time
        wholesaler.sync_time(episode, t)
        wholesaler_env_train.sync_time(episode, t)

        wholesaler_env_train.get_state(t)

        # order decision
        wholesaler_order = wholesaler.order(wholesaler_env_train)

        wholesaler_env_train.step(
            wholesaler_order,
            demand_retailer,
            1e6
        )

    # a series of operations executed after the conclusion of each episode
    # train
    wholesaler.train(wholesaler_env_train)

    # store data during this episode
    train_data_store.record(
        episode,
        0,
        retailer,
        wholesaler,
        retailer_env_train,
        wholesaler_env_train,
    )

    # reset the losses
    wholesaler.reset_losses()
    # save the model
    wholesaler.save_model(episode)

    if episode % args.test_interval == 0:
        # log.info_("Start testing the wholesaler DQN agent")
        for episode in tqdm(
                range(1, args.test_episodes+1),
                desc="Testing",
                leave=False,
                dynamic_ncols=True,
                position=1,
                colour="blue",
                postfix=f"{args.tuning_param_middle_path:<15}",
            ):
            # reset the environment
            retailer_env_test.reset()
            wholesaler_env_test.reset()

            for t in range(1, args.T+1):
                # load the market demand
                demand_market = test_d_t[episode, t]

                # retailer replenishment
                # retailer syncnize the episode and time
                retailer.sync_time(episode, t)
                retailer_env_test.sync_time(episode, t)
                # retailer order decision
                retailer_order = retailer.order(retailer_env_test, demand_market)
                # retailer environment accept the order and update itself
                retailer_env_test.step(
                    retailer_order,
                    demand_market,
                    wholesaler_env_test.I_mat[t, :].sum()
                )

                # load the retailer demand
                demand_retailer = retailer_env_test.order[t]

                # wholesaler replenishment
                # wholesaler syncnize the episode and time
                wholesaler.sync_time(episode, t)
                wholesaler_env_test.sync_time(episode, t)

                wholesaler_env_test.get_state(t)

                # order decision
                wholesaler_order = wholesaler.order(wholesaler_env_test)

                wholesaler_env_test.step(
                    wholesaler_order,
                    demand_retailer,
                    1e6
                )
            # store data during this episode
            test_data_store.record(
                episode,
                args.test_iteration,
                retailer,
                wholesaler,
                retailer_env_test,
                wholesaler_env_test,
            )
        args.test_iteration += 1

# save the data about cost, net inventory, lost sale, waste and loss
train_data_store.save()
test_data_store.save()

log.info(f"{'End':*^75}")
log.info_(f"Runing time: {(time.time()-args.start_time)/60:.2f} mins.")

# send message to WeChat by Autodl.com
send_msg()