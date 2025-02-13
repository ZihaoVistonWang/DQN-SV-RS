import os
import numpy as np
from config import args, log

np.random.seed(args.seed)

# Generate the dataset by cosine function


class Cosine_Dataset_Generator:
    def __init__(self):
        log.info('CosineDatasetGenerator has been initialized.')

        self.time_window_length = args.T+2

    def generate_train_dataset(self):
        if not os.path.exists(f"{args.dataset_parent_dir}/dm_train.npy"):
            log.info('Generating the train dataset.')

            # generate period t
            t = np.arange(0, self.time_window_length)
            t = t.reshape(1, -1)

            # generate demand
            dm_eta_t = np.random.rand(args.train_episodes+1, self.time_window_length)
            dm = np.floor((args.dm_max + args.dm_min)/2 - (args.dm_max -
                        args.dm_min)/2 * np.cos((2*np.pi*t)/(args.T/2)) + dm_eta_t).astype(int)
            log.info(f'Train damend has been generated. dm_t shape: ({args.train_episodes}, {args.T}).')

            # generate lead time
            lr_eta_t = np.random.rand(args.train_episodes+1, self.time_window_length)
            lr = np.floor((args.lr_max + args.lr_min)/2 + (args.lr_max -
                        args.lr_min)/2 * np.cos((2*np.pi*t)/(args.T/2)) + lr_eta_t).astype(int)
            log.info(f'Train lead time of retailer has been generated. lr_t shape: ({args.train_episodes}, {args.T}).')

            lw_eta_t = np.random.rand(args.train_episodes+1, self.time_window_length)
            lw = np.floor((args.lw_max + args.lw_min)/2 + (args.lw_max -
                        args.lw_min)/2 * np.cos((2*np.pi*t)/(args.T/2)) + lw_eta_t).astype(int)
            log.info_(f'Train lead time of wholesaler has been generated. lw_t shape: ({args.train_episodes}, {args.T}).')

            # save the dataset
            np.save(f'{args.dataset_parent_dir}/dm_train.npy', dm)
            np.save(f'{args.dataset_parent_dir}/lr_train.npy', lr)
            np.save(f'{args.dataset_parent_dir}/lw_train.npy', lw)
        else:
            log.info('Train dataset already exists. Loading the dataset.')
            dm = np.load(f'{args.dataset_parent_dir}/dm_train.npy')
            lr = np.load(f'{args.dataset_parent_dir}/lr_train.npy')
            lw = np.load(f'{args.dataset_parent_dir}/lw_train.npy')
        return dm, lr, lw

    def generate_test_dataset(self):
        if not os.path.exists(f"{args.dataset_parent_dir}/dm_test.npy"):
            log.info('Generating the test dataset.')

            # generate period t
            t = np.arange(0, self.time_window_length)
            t = t.reshape(1, -1)

            # generate demand
            dm_eta_t = np.random.rand(args.test_episodes+1, self.time_window_length)
            dm = np.floor((args.dm_max + args.dm_min)/2 - (args.dm_max -
                        args.dm_min)/2 * np.cos((2*np.pi*t)/(args.T/2)) + dm_eta_t).astype(int)
            log.info(f'Test damend has been generated. dm_t shape: ({args.test_episodes}, {args.T}).')

            # generate lead time
            lr_eta_t = np.random.rand(args.test_episodes+1, self.time_window_length)
            lr = np.floor((args.lr_max + args.lr_min)/2 + (args.lr_max -
                        args.lr_min)/2 * np.cos((2*np.pi*t)/(args.T/2)) + lr_eta_t).astype(int)
            log.info(f'Test lead time of retailer has been generated. lr_t shape: ({args.test_episodes}, {args.T}).')

            lw_eta_t = np.random.rand(args.test_episodes+1, self.time_window_length)
            lw = np.floor((args.lw_max + args.lw_min)/2 + (args.lw_max -
                        args.lw_min)/2 * np.cos((2*np.pi*t)/(args.T/2)) + lw_eta_t).astype(int)
            log.info_(f'Test lead time of wholesaler has been generated. lw_t shape: ({args.test_episodes}, {args.T}).')

            # save the dataset
            np.save(f'{args.dataset_parent_dir}/dm_test.npy', dm)
            np.save(f'{args.dataset_parent_dir}/lr_test.npy', lr)
            np.save(f'{args.dataset_parent_dir}/lw_test.npy', lw)
        else:
            log.info('Test dataset already exists. Loading the dataset.')
            dm = np.load(f'{args.dataset_parent_dir}/dm_test.npy')
            lr = np.load(f'{args.dataset_parent_dir}/lr_test.npy')
            lw = np.load(f'{args.dataset_parent_dir}/lw_test.npy')
        return dm, lr, lw


if __name__ == '__main__':
    data_gen = Cosine_Dataset_Generator()

    dm_train, lr_train, lw_train = data_gen.generate_train_dataset()
    dm_test, lr_test, lw_test = data_gen.generate_test_dataset()
