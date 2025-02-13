import os
import time
import logging
import logging.config
import numpy as np
import argparse
from typing import Literal
from DataRecorder import Recorder
from tensorboardX import SummaryWriter

# Set the directory where the Python file is located as the working directory
# ===========================================================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the arguments
class Args:
    def __init__(self):
        self.default_args = {
            'model': {
                'abbr': 'm',
                'type': str,
                'default': "Rolling_BlE",
                'help': 'model name'
            },
            'seed': {
                'abbr': 's',
                'type': int,
                'default': 1,
                'help': 'seed'
            },
            'train_episodes': {
                'abbr': 'tra_epd',
                'type': int,
                'default': 10000,
                'help': 'the max episodes for training'
            },
            'test_episodes': {
                'abbr': 'tes_epd',
                'type': int,
                'default': 1000,
                'help': 'the max episodes for testing'
            },
            'test_interval': {
                'abbr': 'tes_itvl',
                'type': int,
                'default': 100,
                'help': 'the interval of testing. test the model every test_interval episodes'
            },
            'test_iteration': {
                'abbr': 'tes_iter',
                'type': int,
                'default': 1,
                'help': 'the number of test iteration'
            },
            'T': {
                'abbr': 'T',
                'type': int,
                'default': 28,
                'help': 'duration of replenishment period'
            },
            'dm_max': {
                'abbr': 'dm_max',
                'type': int,
                'default': 6,
                'help': 'the maximum demand of market'
            },
            'dm_min': {
                'abbr': 'dm_min',
                'type': int,
                'default': 1,
                'help': 'the minimum demand of market'
            },
            'lr_max': {
                'abbr': 'lr_max',
                'type': int,
                'default': 2,
                'help': 'the maximum lead time of retailer'
            },
            'lr_min': {
                'abbr': 'lr_min',
                'type': int,
                'default': 1,
                'help': 'the minimum lead time of retailer'
            },
            'lw_max': {
                'abbr': 'lw_max',
                'type': int,
                'default': 3,
                'help': 'the maximum lead time of wholesaler'
            },
            'lw_min': {
                'abbr': 'lw_min',
                'type': int,
                'default': 1,
                'help': 'the minimum lead time of wholesaler'
            },
            'Ir_init': {
                'abbr': 'Ir_init',
                'type': int,
                'default': None,
                'help': 'the initial inventory of retailer'
            },
            'Iw_init': {
                'abbr': 'Iw_init',
                'type': int,
                'default': None,
                'help': 'the initial inventory of wholesaler'
            },
            'k': {
                'abbr': 'k',
                'type': int,
                'default': 3,
                'help': 'the shelf life of fresh produce. if the fresh produce is not sold within k periods, it will be cleaned up'
            },
            'tau': {
                'abbr': 'tau',
                'type': int,
                'default': 10,
                'help': 'the length of the historical data in seasonal states'
            },
            'a': {
                'abbr': 'a',
                'type': float,
                'default': 0.02,
                'help': 'the shape parameter of the Weibull distribution'
            },
            'b': {
                'abbr': 'b',
                'type': float,
                'default': 1.50,
                'help': 'the scale parameter of the Weibull distribution'
            },
            'c': {
                'abbr': 'c',
                'type': float,
                'default': 1.00,
                'help': 'the location parameter of the Weibull distribution'
            },
            'cost_order': {
                'abbr': 'c_o',
                'type': int,
                'default': 3,
                'help': 'the unit cost of ordering'
            },
            'cost_holding': {
                'abbr': 'c_h',
                'type': int,
                'default': 1,
                'help': 'the unit cost of holding'
            },
            'cost_lost_sales': {
                'abbr': 'c_l',
                'type': int,
                'default': 5,
                'help': 'the unit cost of lost sales'
            },
            'cost_waste': {
                'abbr': 'c_w',
                'type': int,
                'default': 7,
                'help': 'the unit cost of waste'
            },
            'max_action': {
                'abbr': 'max_a',
                'type': int,
                'default': 10,
                'help': 'the maximum order quantity'
            },
            'min_action': {
                'abbr': 'min_a',
                'type': int,
                'default': 0,
                'help': 'the minimum order quantity'
            },
            'buffer_size': {
                'abbr': 'buf_sz',
                'type': int,
                'default': 50000,
                'help': 'the size of replay buffer'
            },
            'batch_size': {
                'abbr': 'bat_sz',
                'type': int,
                'default': 128,
                'help': 'the size of batch'
            },
            'gamma': {
                'abbr': 'gam',
                'type': float,
                'default': 0.9,
                'help': 'the discount factor'
            },
            'lr': {
                'abbr': 'lr',
                'type': float,
                'default': 0.0005,
                'help': 'the learning rate'
            },
            'learning_rate_decay': {
                'abbr': 'lr_decay',
                'type': float,
                'default': 1e-4,
                'help': 'the learning rate decay'
            },
            'target_network_update_frequency': {
                'abbr': 'C',
                'type': int,
                'default': None,
                'help': 'the frequency of updating the target network'
            },
            'epsilon_max': {
                'abbr': 'eps_max',
                'type': float,
                'default': 0.9,
                'help': 'the maximum epsilon'
            },
            'epsilon_min': {
                'abbr': 'eps_min',
                'type': float,
                'default': 0,
                'help': 'the minimum epsilon'
            },
            'epsilon_decay_rate': {
                'abbr': 'eps_decay_rate',
                'type': float,
                'default': None,
                'help': 'the steps of epsilon decay rate. this is the percentage of train_episodes'
            },
            'input_size': {
                'abbr': 'in_sz',
                'type': int,
                'default': None,
                'help': 'the size of input layer. automatically generate. DO NOT SET THIS PARAMETER!'
            },
            'hidden_size': {
                'abbr': 'hid_sz',
                'type': int,
                'default': 128,
                'help': 'the size of hidden layer'
            },
            'num_hidden_layers': {
                'abbr': 'num_hid_lyr',
                'type': int,
                'default': 3,
                'help': 'the number of hidden layers'
            },
            'output_size': {
                'abbr': 'out_sz',
                'type': int,
                'default': None,
                'help': 'the size of output layer. automatically generate. DO NOT SET THIS PARAMETER!'
            },
            'omega': {
                'abbr': 'omg',
                'type': float,
                'default': 0,
                'help': 'the forgiveness threshold. the omega parameter of reward shaping'
            },
            'rho': {
                'abbr': 'rho',
                'type': float,
                'default': 4.0,
                'help': 'for regularization. the rho parameter of reward shaping'
            },
            'e': {
                'abbr': 'e',
                'type': float,
                'default': 1e-3,
                'help': 'a small positive value to avoid division by zero'
            },
            'input_D_size': {
                'abbr': 'in_D_sz',
                'type': int,
                'default': None,
                'help': 'the size of input layer of demand forecasting module. automatically generate. DO NOT SET THIS PARAMETER!'
            },
            'hidden_D_size': {
                'abbr': 'hid_D_sz',
                'type': int,
                'default': 50,
                'help': 'the size of hidden layer of demand forecasting module'
            },
            'out_D_size': {
                'abbr': 'out_D_sz',
                'type': int,
                'default': None,
                'help': 'the size of output layer of demand forecasting module. automatically generate. DO NOT SET THIS PARAMETER!'
            },
            'input_L_size': {
                'abbr': 'in_L_sz',
                'type': int,
                'default': None,
                'help': 'the size of input layer of leadtime forecasting module. automatically generate. DO NOT SET THIS PARAMETER!'
            },
            'hidden_L_size': {
                'abbr': 'hid_L_sz',
                'type': int,
                'default': 50,
                'help': 'the size of hidden layer of leadtime forecasting module'
            },
            'out_L_size': {
                'abbr': 'out_L_sz',
                'type': int,
                'default': 1,
                'help': 'the size of output layer of leadtime forecasting module'
            },
            'tuning_param_middle_path': {
                'abbr': 'tuning_path',
                'type': str,
                'default': "",
                'help': 'the middle path represented by a string about tuning parameters'
            },
        }
        self.args = None
        self.parse_args()
        self.check_modified_args()
        self.add_calculated_args()
        self.make_directory()

    def parse_args(self):
        # Create the parser
        parser = argparse.ArgumentParser(
            description="IoT-driven Dynamic Replenishment of Fresh Produce in the Presence of Seasonal Variations: A Deep Reinforcement Learning Approach Using Reward Shaping"
            )

        # Add the arguments to the parser
        for key, value in self.default_args.items():
            if value["default"] is not None:
                parser.add_argument(f'-{value["abbr"]}', f'--{key}', type=value["type"], default=value["default"], help=value["help"])
            else:
                parser.add_argument(f'-{value["abbr"]}', f'--{key}', type=value["type"], help=value["help"])

        self.args = parser.parse_args()

    def check_modified_args(self):
        tuning_params = []
        for key, _ in self.default_args.items():
            if key != "tuning_param_middle_path":
                if getattr(self.args, key) != self.default_args[key]["default"]:
                    tuning_params.append(f"{self.default_args[key]['abbr']}={getattr(self.args, key)}")
        if tuning_params != []:
            self.args.tuning_param_middle_path = "_".join(tuning_params)
            self.args.tuning_param_middle_path += "/"

    def add_calculated_args(self):
        # Update Ir_init and Iw_init if not set
        self.args.Ir_init = 1 * round((self.args.dm_max + self.args.dm_min) / 2) if self.args.Ir_init is None else self.args.Ir_init
        self.args.Iw_init = 2 * round((self.args.dm_max + self.args.dm_min) / 2) if self.args.Iw_init is None else self.args.Iw_init
        self.args.action_space = range(self.args.min_action, self.args.max_action + 1)
        # The state size is defined as the length of state_I and state_Q
        # in the case of a DQN and none-DQN models. Conversely, if the
        # model is DQN-SV or DQN=SV-SR, the state size is determined by
        # the lengths of state_I, state_Q, state_Lr, state_Lw, state_Dm,
        # and state_Dr.
        self.args.state_size = [self.args.k, 1, self.args.tau, self.args.tau, self.args.tau, self.args.tau] if self.args.model in ["DQN-SV-SR", "DQN-SV"] else [self.args.k, 1]
        self.args.epsilon_decay_steps = 0.1 * self.args.train_episodes if self.args.epsilon_decay_rate is None else self.args.epsilon_decay_rate * self.args.train_episodes
        self.args.input_size = sum(self.args.state_size)
        self.args.output_size = len(self.args.action_space)
        self.args.target_network_update_frequency = 100 * self.args.T if self.args.target_network_update_frequency is None else self.args.target_network_update_frequency
        self.args.start_time = time.time()
        # Our period is 1-indexed, and the next state of the maximum period will occur at (T+1).
        # Therefore, we need to add 2 to the length of the time window.
        self.args.time_window_length = self.args.T + 2

        self.args.input_D_size = self.args.tau
        self.args.out_D_size = self.args.lw_max + 3

        self.args.input_L_size = self.args.tau
        self.args.out_L_size = self.args.lw_max + 3

    def make_directory(self):
        # Make the directory for the log, dataset, result, and tensorboard
        # =================================================================
        # make log directory
        if not os.path.exists(f"log/seed={self.args.seed}/{self.args.tuning_param_middle_path}"):
            os.makedirs(f"log/seed={self.args.seed}/{self.args.tuning_param_middle_path}")
        self.args.log_parnet_dir = f"log/seed={self.args.seed}/{self.args.tuning_param_middle_path}"

        # make dataset directory
        if not os.path.exists(f"dataset/seed={self.args.seed}"):
            os.makedirs(f"dataset/seed={self.args.seed}")
        self.args.dataset_parent_dir = f"dataset/seed={self.args.seed}"

        # make result directory
        if not os.path.exists(f"result/seed={self.args.seed}/{self.args.tuning_param_middle_path}"):
            os.makedirs(f"result/seed={self.args.seed}/{self.args.tuning_param_middle_path}")
        self.args.result_parent_dir = f"result/seed={self.args.seed}/{self.args.tuning_param_middle_path}"

        # make tensorboard directory
        if not os.path.exists(f"tensorboard/seed={self.args.seed}/{self.args.tuning_param_middle_path}"):
            os.makedirs(f"tensorboard/seed={self.args.seed}/{self.args.tuning_param_middle_path}")
        self.args.tensorboard_parent_dir = f"tensorboard/seed={self.args.seed}/{self.args.tuning_param_middle_path}"

        # make model directory
        if not os.path.exists(f"model/seed={self.args.seed}/{self.args.tuning_param_middle_path}/state_dict"):
            os.makedirs(f"model/seed={self.args.seed}/{self.args.tuning_param_middle_path}/state_dict")

        if not os.path.exists(f"model/seed={self.args.seed}/{self.args.tuning_param_middle_path}/complete_model"):
            os.makedirs(f"model/seed={self.args.seed}/{self.args.tuning_param_middle_path}/complete_model")

# Instantiate the Args class
args = Args().args

# Define color codes
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m',# Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m'# Magenta
    }
    RESET = '\033[0m'
    Magenta = '\033[95m'  # Yellow

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        # record.msg = f"{log_color}{record.msg}{self.RESET}"
        formatted_record = super().format(record)
        formatted_record = formatted_record.replace(record.asctime, f"{self.Magenta}{record.asctime}{self.RESET}")
        return formatted_record

# Condigure the logger
class Log:
    def __init__(self,
                 args):
        self.args = args
        self.log_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simpleFormatter': {
                    'format': '%(asctime)s|%(levelname)-5s|%(message)s',
                    'datefmt': '%Y-%m-%d|%H:%M:%S',
                },
                'colorFormatter': {
                    '()': ColorFormatter,
                    'format': '%(asctime)s|%(levelname)-5s|%(message)s',
                    'datefmt': '%Y-%m-%d|%H:%M:%S',
                },
            },
            'handlers': {
                'consoleHandler': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'colorFormatter',
                    'stream': 'ext://sys.stdout',
                },
                'fileHandler': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'simpleFormatter',
                    'filename': f'{self.args.log_parnet_dir}/replenishment.log',
                    'maxBytes': 5 * 1024 * 1024,
                    'backupCount': 1000,
                },
            },
            'loggers': {
                'replenishment_log': {
                    'level': 'DEBUG',
                    'handlers': ['fileHandler', 'consoleHandler'],
                    'propagate': False,
                },
            },
            'root': {
                'level': 'DEBUG',
                'handlers': ['fileHandler'],
            },
        }

        logging.config.dictConfig(self.log_config)
        self.logger = logging.getLogger('replenishment_log')

    def info(self, msg):
        self.logger.info(msg)

    def info_(self, msg):
        self.logger.info(msg)
        self.logger.info(f"{'='*75}")

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

# Instantiate the Log class
log = Log(args)
log.info_(f"Model: {args.model} | Seed: {args.seed}| Modified Parameters: {args.tuning_param_middle_path if args.tuning_param_middle_path != '' else 'None'}")


class Data_Store:
    def __init__(
        self,
        type_: Literal["train", "test"] = "train",
        whether_to_record_retailer_env_data: bool = True,
        whether_to_record_wholeshaler_env_data: bool = True,
        whether_to_record_retailer_loss: bool = False,
        whether_to_record_wholesaler_loss: bool = True
        ):
        self.type = type_
        self.whether_to_record_retailer_env_data = whether_to_record_retailer_env_data
        self.whether_to_record_wholeshaler_env_data = whether_to_record_wholeshaler_env_data
        self.whether_to_record_retailer_loss = whether_to_record_retailer_loss
        self.whether_to_record_wholesaler_loss = whether_to_record_wholesaler_loss

        # initialize the result recorder by DataRecorder
        self.header = [self.type] + ["episode"] + [f"t={t}" for t in range(1, args.T+1)]

        def initialize_recorder(member, recorder_suffix):
            path = f"{args.result_parent_dir}/{self.type}_{member}_{recorder_suffix}.csv"
            recorder = Recorder(path=path, cache_size=10)
            recorder.set.show_msg(False)
            recorder.add_data(self.header)
            return recorder

        self.recorders = {}

        if self.whether_to_record_retailer_env_data:
            self.recorders.update({
                'retailer_order': initialize_recorder('retailer', 'order'),
                'retailer_net_inventory': initialize_recorder('retailer', 'net_inventory'),
                'retailer_lost_sales': initialize_recorder('retailer', 'lost_sales'),
                'retailer_waste': initialize_recorder('retailer', 'waste'),
                'retailer_cost': initialize_recorder('retailer', 'cost')
            })

        if self.whether_to_record_wholeshaler_env_data:
            self.recorders.update({
                'wholesaler_order': initialize_recorder('wholesaler', 'order'),
                'wholesaler_net_inventory': initialize_recorder('wholesaler', 'net_inventory'),
                'wholesaler_lost_sales': initialize_recorder('wholesaler', 'lost_sales'),
                'wholesaler_waste': initialize_recorder('wholesaler', 'waste'),
                'wholesaler_cost': initialize_recorder('wholesaler', 'cost')
            })

        if self.whether_to_record_retailer_loss:
            self.recorders['retailer_loss'] = initialize_recorder('retailer', 'loss')

        if self.whether_to_record_wholesaler_loss:
            self.recorders['wholesaler_loss'] = initialize_recorder('wholesaler', 'loss')

        # initialize tensorboard
        self.writer = SummaryWriter(f"{args.tensorboard_parent_dir}/{self.type}")

    def record(self,
               episode: int,
               test_iteration: int = 0,
               retailer_agent=None,
               wholesaler_agent=None,
               retailer_env=None,
               wholesaler_env=None):
        if self.type == 'train':
            step = episode
        else:
            step = test_iteration

        def record_env_data(env, prefix):
            self.recorders[f'{prefix}_order'].add_data([test_iteration] + [episode] + list(env.order[1:-1]))
            self.recorders[f'{prefix}_net_inventory'].add_data([test_iteration] + [episode] + list(env.net_inventory[1:-1]))
            self.recorders[f'{prefix}_lost_sales'].add_data([test_iteration] + [episode] + list(env.lost_sales[1:-1]))
            self.recorders[f'{prefix}_waste'].add_data([test_iteration] + [episode] + list(env.waste[1:-1]))
            self.recorders[f'{prefix}_cost'].add_data([test_iteration] + [episode] + list(env.cost[1:-1]))

            self.writer.add_scalar(f'{prefix}/order', np.mean(env.order[1:-1]), step)
            self.writer.add_scalar(f'{prefix}/net_inventory', np.mean(env.net_inventory[1:-1]), step)
            self.writer.add_scalar(f'{prefix}/lost_sales', np.mean(env.lost_sales[1:-1]), step)
            self.writer.add_scalar(f'{prefix}/waste', np.mean(env.waste[1:-1]), step)
            self.writer.add_scalar(f'{prefix}/cost', np.mean(env.cost[1:-1]), step)

        def record_agent_data(agent, prefix):
            self.recorders[f'{prefix}_loss'].add_data([test_iteration] + [episode] + list(agent.losses))
            self.writer.add_scalar(f'{prefix}/loss', np.mean(agent.losses), step)

        if self.whether_to_record_retailer_env_data and retailer_env is not None:
            record_env_data(retailer_env, 'retailer')

        if self.whether_to_record_wholeshaler_env_data and wholesaler_env is not None:
            record_env_data(wholesaler_env, 'wholesaler')

        if self.whether_to_record_retailer_loss and retailer_agent is not None:
            record_agent_data(retailer_agent, 'retailer')

        if self.whether_to_record_wholesaler_loss and wholesaler_agent is not None:
            record_agent_data(wholesaler_agent, 'wholesaler')

    def save(self):
        for recorder in self.recorders.values():
            recorder.record()
        self.writer.close()