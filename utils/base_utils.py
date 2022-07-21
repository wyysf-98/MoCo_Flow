import os
import time
import json
import torch
import random
import shutil
import logging
import numpy as np
from pathlib import Path
from itertools import repeat

class TrainClock(object):
    """ Clock object to track epoch and step during training
    """
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self, step=1):
        self.minibatch += 1
        self.step += step

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']

class WorklogLogger:
    def __init__(self, log_file):
        logging.basicConfig(filename=log_file,
                            level=logging.INFO,
                            format='%(asctime)s - %(threadName)s -  %(levelname)s - %(message)s')

        self.logger = logging.getLogger()

    def put_line(self, line):
        self.logger.info(line)


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if os.path.exists(path):
        response = input('Path {} already exists. overwrite, ignore or exit? (o/i/e) '.format(path))
        if response == 'o':
            shutil.rmtree(path)
            os.makedirs(path)
        elif response == 'i':
            pass
        else:
            exit()
    else:
        os.makedirs(path)


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def remkdir(path):
    """
    if dir exists, remove it and create a new one
    :param path:
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def merge_dict(dict1, dict2): 
    ''' merge two dicts. '''
    res = {**dict1, **dict2} 
    return res 

def set_seed(random_seed):
    ''' set random seed for random, numpy and torch. '''
    if random_seed is None:
        print('do not use specified seed.')
    else:
        print('use random seed {}.'.format(random_seed))
        # fix random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

    return random_seed

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                