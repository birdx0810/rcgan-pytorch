# -*- coding: UTF-8 -*-
# Local modules
import argparse
import logging
import os
import random
import shutil
import time

# 3rd-Party Modules
import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Self-Written Modules
from data.data_preprocess import data_preprocess
from metrics.metric_utils import (
    one_step_ahead_prediction, classification_task, reidentify_score
)

from models.rcgan import RCGAN
from models.utils import rcgan_trainer, rcgan_generator

def main(args):
    ##############################################
    # Initialize output directories
    ##############################################

    ## Runtime directory
    code_dir = os.path.abspath(".")
    if not os.path.exists(code_dir):
        raise ValueError(f"Code directory not found at {code_dir}.")

    ## Data directory
    data_path = os.path.abspath("./data")
    if not os.path.exists(data_path):
        raise ValueError(f"Data file not found at {data_path}.")
    data_dir = os.path.dirname(data_path)
    data_file_name = os.path.basename(data_path)

    ## Output directories
    args.model_path = f"./output/{args.exp}/"
    out_dir = os.path.abspath(args.model_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    ## TensorBoard directory
    tensorboard_path = os.path.abspath("./tensorboard")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)

    print(f"\nCode directory:\t\t\t{code_dir}")
    print(f"Data directory:\t\t\t{data_path}")
    print(f"Output directory:\t\t{out_dir}")
    print(f"TensorBoard directory:\t\t{tensorboard_path}\n")

    ##############################################
    # Initialize random seed and CUDA
    ##############################################

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        print("Using CUDA\n")
        args.device = torch.device("cuda:0")
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        args.device = torch.device("cpu")

    #########################
    # Load and preprocess data for model
    #########################

    data_path = "./data/stock.csv"
    X, T, Y, _ = data_preprocess(
        data_path, args
    )

    print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n")
    print(f"Original data preview:\n{X[:2, :10, :2]}\n")

    args.feature_dim = X.shape[-1]
    args.Z_dim = X.shape[-1]
    args.C_dim = Y.shape[-1]

    # Shuffle data sequence
    idx = np.random.permutation(X.shape[0])
    X, T, Y = X[idx], T[idx], Y[idx]

    # Train-Test Split data and time
    train_data, test_data = train_test_split(
        X, test_size=args.train_rate, shuffle=False
    )
    train_time, test_time = train_test_split(
        T, test_size=args.train_rate, shuffle=False
    )
    train_label, test_label = train_test_split(
        Y, test_size=args.train_rate, shuffle=False
    )
    
    print(f"Train data: {train_data.shape} (Idx x MaxSeqLen x Features)\n")
    print(f"Test data: {test_data.shape} (Idx x MaxSeqLen x Features)\n")

    if train_data.shape[0] < args.batch_size:
        raise ValueError("Batch size is larger than dataset")

    #########################
    # Initialize and Run model
    #########################

    # Log start time
    start = time.time()

    model = RCGAN(args)
    if args.is_train == True:
        rcgan_trainer(
            model=model, 
            data=train_data, 
            time=train_time, 
            label=train_label,
            args=args)
    generated_data, generated_label, generated_time = rcgan_generator(
        model=model, 
        time=train_time, 
        label=train_label, 
        args=args)
    
    # Log end time
    end = time.time()

    print(f"Generated data preview:\n{generated_data[:2, -10:, :2]}\n")
    print(f"Model Runtime: {(end - start)/60} mins\n")

    #########################
    # Save train and generated data for visualization
    #########################
    
    joblib.dump(train_data, f"{out_dir}/train_data.jlb")
    joblib.dump(train_time, f"{out_dir}/train_time.jlb")
    joblib.dump(train_label, f"{out_dir}/train_label.jlb")
    joblib.dump(generated_data, f"{out_dir}/generated_data.jlb")
    joblib.dump(generated_time, f"{out_dir}/generated_time.jlb")
    joblib.dump(generated_label, f"{out_dir}/generated_label.jlb")
    joblib.dump(test_data, f"{out_dir}/test_data.jlb")
    joblib.dump(test_time, f"{out_dir}/train_time.jlb")
    joblib.dump(test_label, f"{out_dir}/train_label.jlb")

    #########################
    # Preprocess data for seeker
    #########################

    # Define enlarge data and its labels
    enlarge_data = np.concatenate((train_data, test_data), axis=0)
    enlarge_time = np.concatenate((train_time, test_time), axis=0)
    enlarge_data_label = np.concatenate((np.ones([train_data.shape[0], 1]), np.zeros([test_data.shape[0], 1])), axis=0)

    # Mix the order
    idx = np.random.permutation(enlarge_data.shape[0])
    enlarge_data = enlarge_data[idx]
    enlarge_data_label = enlarge_data_label[idx]

    #########################
    # Evaluate the performance
    #########################

    # 1. Classification task
    print("\nRunning classification task using original data...")
    ori_classification_perf = classification_task(
        (train_data, train_time, train_label),
        (test_data, test_time, test_label)
    )
    print("Running classification task using generated data...")
    new_classification_perf = classification_task(
        (generated_data, generated_time, generated_label),
        (test_data, test_time, test_label)
    )

    classification = [ori_classification_perf, new_classification_perf]

    print('Classification Task results:\n' +
        f'(1) Ori: {str(np.round(ori_classification_perf, 4))}\n' +
        f'(2) New: {str(np.round(new_classification_perf, 4))}')

    # 2. One step ahead prediction
    print("Running one step ahead prediction using original data...")
    ori_step_ahead_pred_perf = one_step_ahead_prediction(
        (train_data, train_time), 
        (test_data, test_time)
    )
    print("Running one step ahead prediction using generated data...")
    new_step_ahead_pred_perf = one_step_ahead_prediction(
        (generated_data, generated_time),
        (test_data, test_time)
    )

    step_ahead_pred = [ori_step_ahead_pred_perf, new_step_ahead_pred_perf]

    print('One step ahead prediction results:\n' +
          f'(1) Ori: {str(np.round(ori_step_ahead_pred_perf, 4))}\n' +
          f'(2) New: {str(np.round(new_step_ahead_pred_perf, 4))}\n')

    print(f"Total Runtime: {(time.time() - start)/60} mins\n")

    return None

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--exp',
        default='test',
        type=str)
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--feat_pred_no',
        default=2,
        type=int)

    # Data Arguments
    parser.add_argument(
        '--data_name',
        choices=['phys12', 'bci'],
        default='phys12',
        type=str)
    parser.add_argument(
        '--max_seq_len',
        default=100,
        type=int)
    parser.add_argument(
        "--padding_value",
        default=-1.0,
        type=float)
    parser.add_argument(
        '--train_rate',
        default=0.5,
        type=float)

    # Model Arguments
    parser.add_argument(
        '--epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int)
    parser.add_argument(
        '--d_iters',
        default=1,
        type=int)
    parser.add_argument(
        '--g_iters',
        default=1,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=20,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--dis_thresh',
        default=0.15,
        type=float)
    parser.add_argument(
        '--optimizer',
        choices=['adam'],
        default='adam',
        type=str)
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float)

    args = parser.parse_args()

    # Call main function
    main(args)
