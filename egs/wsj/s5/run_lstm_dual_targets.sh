#!/bin/bash

. ./cmd.sh
. ./path.sh

data=
ali=
ali_dev=
ref_dir=

if [ $# != 3 ]; then
  echo "Usage: $0 <temperature> <hard_scale> <soft_scale>";
  exit 1;
fi

temp=$1
hard_scale=$2
soft_scale=$3
dir=

. utils/parse_options.sh || exit 1;

  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_dual_targets.sh --network-type lstm --learn-rate 0.0001 --temperature $temp \
      --cmvn-opts "--norm-means=true --norm-vars=false" --feat-type plain --splice 0 \
      --train-opts "--momentum 0.9 --halving-factor 0.7" \
      --hard-scale $hard_scale  --soft-scale $soft_scale \
      --train-tool "nnet-train-lstm-streams-dual-targets --num-stream=4 --targets-delay=5" \
      --ref-dir $ref_dir \
    $data/train $data/dev data/lang $ali $ali_dev $dir || exit 1;

echo Success
exit 0

