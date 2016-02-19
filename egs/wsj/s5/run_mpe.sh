#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

. utils/parse_options.sh || exit 1;


graphdir=exp/chain/tdnn_2o/graph_huawei3gram
modeldir=exp/chain/tdnn_2o
dir=${modeldir}_mpe

data_dir=train_2w  #train
tri_latdir=exp/tri4_lats_nodup
ctc_trans_mdl=exp/chain/tdnn_2o/0.trans_mdl
chain_tree=exp/chain/tdnn_2o/tree
stage=3
mpe_stage=-3



if [ $stage -le 1 ]; then
 echo "make alimgents"
   steps/nnet3/align.sh --nj 8 --cmd "$decode_cmd" --frame-subsampling-factor 3 \
     data/$data_dir data/lang_chain_2o $modeldir ${modeldir}_ali || exit 1;
fi

#--frame-subsampling-factor 3 already in make_denlats.sh
if [ $stage -le 2 ]; then
 echo "make lattices"
   steps/nnet3/make_denlats.sh  --nj 8 --cmd "$decode_cmd" --acwt 1.0 --post-decode-acwt 10.0 --stage 1 \
      $graphdir data/$data_dir $modeldir ${modeldir}_denlats || exit 1;
fi

if [ $stage -le 3 ]; then
 echo "mpe training"
   steps/nnet3/chain/train_discriminative.sh --cmd "$cuda_cmd" --cpu-cmd "$cuda_cmd" --learning-rate 0.00008 \
     --frame_subsampling_factor 3 \
     --num-jobs-nnet 8 --stage $mpe_stage --num-pdfs 6975 --left-context 19 --right-context 14 \
     --tri-latdir $tri_latdir --ctc-trans-mdl $ctc_trans_mdl --chain_tree $chain_tree \
     data/$data_dir data/lang_chain_2o ${modeldir}_ali ${modeldir}_denlats $modeldir/final.mdl $dir || exit 1;
fi

exit 0;


