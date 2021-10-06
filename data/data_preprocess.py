"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar,
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data,"
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: Oct 17th 2020
Code author: Jinsung Yoon, Evgeny Saveliev
Contact: jsyoon0823@gmail.com, e.s.saveliev@gmail.com


-----------------------------

(1) data_preprocess: Load the data and preprocess into a 3d numpy array
(2) imputater: Impute missing data
"""
# Local packages
import os
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

# 3rd party modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def data_preprocess(
    file_name: str,
    args: Dict,
    impute_method: str="mode",
    scaling_method: str="minmax",
):
    """Load the data and preprocess into 3d numpy array.
    Preprocessing includes:
    1. Remove outliers
    2. Extract sequence length for each patient id
    3. Impute missing data
    4. Normalize data
    5. Sort dataset according to sequence length

    Args:
    - file_name (str): CSV file name
    - args (dict): parameters for preprocessing data
    - impute_method (str): The imputation method ("median" or "mode")
    - scaling_method (str): The scaler method ("standard" or "minmax")

    Returns:
    - data: preprocessed data
    - time: ndarray of ints indicating the length for each data
    - params: the parameters to rescale the data
    """
    padding_value = args.padding_value
    if not isinstance(padding_value, float):
        raise ValueError("Must provide padding value with type `float`")

    #########################
    # Load data
    #########################

    index = 'Idx'
    label = 'Label'

    # Load .csv file, columns are typically as follows:
    # | Index | Time | Feature_1 | ... | Feature_n | Label |
    print("Loading data...\n")
    ori_data = pd.read_csv(file_name)

    # Remove spurious column, so that column 0 is now 'Index'.
    if ori_data.columns[0] == "Unnamed: 0":
        ori_data = ori_data.drop(["Unnamed: 0"], axis=1)

    #########################
    # Remove outliers from dataset
    #########################

    no = ori_data.shape[0]
    z_scores = stats.zscore(ori_data, axis=0, nan_policy='omit')
    z_filter = np.nanmax(np.abs(z_scores), axis=1) < 3
    ori_data = ori_data[z_filter]
    print(f"Dropped {no - ori_data.shape[0]} rows (outliers)\n")

    # Parameters
    uniq_id = np.unique(ori_data[index])
    no = len(uniq_id)
    dim = len(ori_data.columns) - 1 # Ignore index

    #########################
    # Impute, scale and pad data
    #########################

    # Initialize scaler
    if scaling_method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(ori_data)
        params = [scaler.data_min_, scaler.data_max_]

    elif scaling_method == "standard":
        scaler = StandardScaler()
        scaler.fit(ori_data)
        params = [scaler.mean_, scaler.var_]

    # Imputation values
    if impute_method == "median":
        impute_vals = ori_data.median()
    elif impute_method == "mode":
        impute_vals = stats.mode(ori_data).mode[0]
    else:
        raise ValueError("Imputation method should be `median` or `mode`")

    # TODO: Sanity check for padding value
    # if np.any(ori_data == padding_value):
    #     print(f"Padding value `{padding_value}` found in data")
    #     padding_value = np.nanmin(ori_data.to_numpy()) - 1
    #     print(f"Changed padding value to: {padding_value}\n")
    args.padding_value = padding_value

    # Output initialization
    output = np.empty([no, args.max_seq_len, dim])  # Shape:[no, max_seq_len, dim]
    output.fill(args.padding_value)
    labels = np.empty([no, 1], dtype=np.int)
    time = np.empty([no], dtype=np.int)

    print("Preprocessing data...\n")
    # For each uniq id
    for i in tqdm(range(no)):
        # Extract the time-series data and label with a certain admissionid
        curr_data = ori_data[ori_data[index] == uniq_id[i]].to_numpy()
        curr_label = int(curr_data[0, -1])

        # Impute missing data
        curr_data = imputer(curr_data, impute_vals)

        # Normalize data
        curr_data = scaler.transform(curr_data)

        # Extract time and assign to the preprocessed data (Excluding ID)
        curr_no = len(curr_data)

        # Pad data to `max_seq_len`
        if curr_no >= args.max_seq_len:
            output[i, :, :] = curr_data[:args.max_seq_len, 1:]  # Shape: [1, max_seq_len, dim]
            time[i] = args.max_seq_len
        else:
            output[i, :curr_no, :] = curr_data[:, 1:]  # Shape: [1, max_seq_len, dim]
            time[i] = curr_no

        # Pad label for data
        labels[i] = curr_label

    return output, time, labels, params


def imputer(
    curr_data: np.ndarray,
    impute_vals: List,
    zero_fill: bool = True
) -> np.ndarray:
    """Impute missing data given values for each columns.

    Args:
        curr_data (np.ndarray): Data before imputation.
        impute_vals (list): Values to be filled for each column.
        zero_fill (bool, optional): Whather to Fill with zeros the cases where
            impute_val is nan. Defaults to True.

    Returns:
        np.ndarray: Imputed data.
    """

    curr_data = pd.DataFrame(data=curr_data)
    impute_vals = pd.Series(impute_vals)

    # Impute data
    imputed_data = curr_data.fillna(impute_vals)

    # Zero-fill, in case the `impute_vals` for a particular feature is `nan`.
    imputed_data = imputed_data.fillna(0.0)

    # Check for any N/A values
    if imputed_data.isnull().any().any():
        raise ValueError("NaN values remain after imputation")

    return imputed_data.to_numpy()
