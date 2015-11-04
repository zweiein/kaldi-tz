#!/bin/bash


. ./cmd.sh
. ./path.sh

data=
ali=
ali_dev=
ref_dir=

if [ $# != 3 ]; then
   echo "Usage: $0 <temperature> <hard_scale> <soft_scale>";
   exit;
fi

temp=$1
hard_scale=$2
soft_scale=$3
dir=

. utils/parse_options.sh || exit 1;

  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_dual_targets_dnn.sh   --hid-dim 2048 --hid-layers 4  \
      --learn-rate 0.008 --temperature $temp \
      --cmvn-opts "--norm-means=true --norm-vars=false" --feat-type lda --lda-dim 200 \
      --hard-scale $hard_scale  --soft-scale $soft_scale \
      --train-tool "nnet-train-frmshuff-dual-targets" \
      --ref-dir $ref_dir \
    $data/train $data/cv data/lang $ali $ali_dev $dir || exit 1;

echo Success
exit 0

