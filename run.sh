#!/bin/bash
export TZ="GMT-8"
mkdir -p logs

# Data variables
train=false

# Experiment variables
data='ner2015'
seeker='knn-norm'
hider='timegan'
opt='adam'
epochs=10
layers=3
# exp="${hider}_${opt}_${data}_${epochs}"
exp="test"

# Iteration variables
emb_epochs=$epochs
sup_epochs=$epochs
gan_epochs=$epochs

python -i main.py \
--exp                   $exp \
--device                cuda \
--train                 $train \
--run_feat_pred         false \
--run_step_pred         false \
--run_seeker            true \
--max_seq_len           100 \
--train_rate            0.5 \
--hider_model           $hider \
--emb_epochs            $emb_epochs \
--sup_epochs            $sup_epochs \
--gan_epochs            $gan_epochs \
--batch_size            128 \
--hidden_dim            20 \
--num_layers            $layers \
--loss_fn               timegan \
--dis_thresh            0.15 \
--optimizer             adam \
--learning_rate         1e-3 \
--noise_multiplier      0.3 \
--seeker_model          $seeker \
--seeker_k              1 \
--seeker_r              5.0 \
