#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a LSTM network on FBANK features.
# The LSTM code comes from Yiayu DU, and Wei Li, thanks!

. ./cmd.sh
. ./path.sh

data=data
ali=exp/tri4b_ali
ali_dev=exp/tri4b_ali_cv

extra_dir=exp/lv_model

mlp_proto=nnet_extra_input.proto
#if [ $# != 3 ]; then
#   echo "Usage: $0 <temperature> <hard_scale> <soft_scale>";
#   exit;
#fi
extra_input_layer=6
dir=exp/train96_dnn_lv-out3_extra-input-cmvn-g-to${extra_input_layer}_lr0.008

stage=1
. utils/parse_options.sh || exit 1;

if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_extra_input.sh   --hid-dim 1200 --hid-layers 4 \
      --learn-rate 0.008  --mlp_proto $mlp_proto --cmvn-opts "--norm-means=true --norm-vars=false" \
       --feat-type lda --lda-dim 200 \
      --train-tool "nnet-train-frmshuff-extra-input-alone --extra-input-layer=${extra_input_layer}" \
    $data/train $data/cv $data/lang $ali $ali_dev $dir $extra_dir || exit 1;
fi

# TODO : sequence training,

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
