#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import os
import random
from re import U

import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

def main(args, init_distributed=False):
    utils.import_user_module(args)

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    # Print args
    # print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Build model and criterion
    modelA = task.build_model(args)
    modelB = task.build_model(args)
    modelC = task.build_model(args)

    files = []

    stateA = checkpoint_utils.load_checkpoint_to_cpu(files[0])
    stateB = checkpoint_utils.load_checkpoint_to_cpu(files[1])
    stateC = checkpoint_utils.load_checkpoint_to_cpu(files[2])

    modelA.load_state_dict(stateA['model'], strict=True)
    modelB.load_state_dict(stateB['model'], strict=True)
    modelC.load_state_dict(stateC['model'], strict=True)
    
    for key in modelA:
        modelA[key] = (modelA[key] + modelB[key] + modelC[key]) / 3.

    model = task.build_model(args)
    model.load_state_dict(model)

def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == '__main__':
    cli_main()