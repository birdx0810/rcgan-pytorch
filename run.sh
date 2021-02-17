#!/bin/bash
train=false
export TZ="GMT-8"

# Experiment variables
exp="test"

# Iteration variables
epochs=50

python main.py \
--device            cuda \
--exp               $exp \
--is_train          $train \
--seed              42 \
--feat_pred_no      1 \
--max_seq_len       100 \
--train_rate        0.5 \
--epochs            $epochs \
--batch_size        32 \
--hidden_dim        20 \
--num_layers        3 \
--dis_thresh        0.15 \
--optimizer         adam \
--learning_rate     1e-3 \