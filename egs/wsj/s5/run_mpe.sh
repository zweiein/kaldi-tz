#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

. utils/parse_options.sh || exit 1;


chunk_width=75         # for make_denlats.sh
chunk_left_context=30

stage=33  #0
mpe_stage=-8  #-8
cmd=run.pl

trans_model=exp/ctc/tri5b_tree_monophone/final.mdl
trans_dir=exp/ctc/tri5b_tree_monophone
tri_ali=$trans_dir
graphdir=exp/ctc/lstm_i/graph_bd_tgpr_0.15
modeldir=exp/ctc/lstm_i
#tri_ali=${modeldir}_ali
dir=${modeldir}_mpe

if [ $stage -le 0 ]; then
 echo "get nnet3-am from nnet3-ctc"
   nnet3-ctc-copy --raw=true $modeldir/final.mdl $modeldir/final.mdl.raw || exit 1;
   #nnet3-am-init $trans_model $modeldir/final.mdl.raw $modeldir/final.mdl.nnet3am || exit 1;
 echo "train the transitions, set the priors"
   $cmd $modeldir/log/get_nnet3am.log \
     nnet3-am-init $trans_model $modeldir/final.mdl.raw - \| \
     nnet3-am-train-transitions-cctc - "ark:gunzip -c $tri_ali/ali.*.gz|" $modeldir/final.mdl.nnet3am.linear || exit 1;
     #---TO DO---
     #replace objective "linear" with "mpe" to get final.mdl.nnet3am.mpe
     #final.mdl.nnet3am -> final.mdl.nnet3am.mpe
fi

if [ ! -f  $modeldir/final.mdl.nnet3am ];then
  echo "replace objective "linear" with "mpe" to get final.mdl.nnet3am.mpe, linked by final.mdl.nnet3am"
  exit 1;
fi

if [ $stage -le 11 ]; then
 echo "make alimgents"
 # refer to modeldir/final.mdl.nnet3am (or final.mdl)
   steps/nnet3/ctc/align.sh --nj 30 --cmd "$decode_cmd" \
     data/train data/lang_ctc $modeldir ${modeldir}_ali || exit 1;
fi

if [ $stage -le 22 ]; then
 echo "make lattices"
#   steps/nnet3/ctc/make_denlats.sh  --nj 8 --cmd "$decode_cmd" \
#     --frames-per-chunk $chunk_width --extra-left-context $chunk_left_context \
#      $graphdir data/train $modeldir ${modeldir}_denlats || exit 1;
   graphdir_tree=$trans_dir/graph_lang_test_bd_tgpr
   utils/mkgraph.sh data/lang_test_bd_tgpr $trans_dir $graphdir_tree 
   #modeldir_tree=exp/ctc/tri5b_tree
   steps/nnet3/make_denlats.sh  --nj 30 --cmd "$decode_cmd" \
      $graphdir_tree data/train $modeldir ${modeldir}_denlats || exit 1;
fi

if [ $stage -le 33 ]; then

 echo "mpe training"
   steps/nnet3/ctc/train_discriminative.sh --cmd "$cuda_cmd" --learning-rate 0.000005 \
     --num-jobs-nnet 4 --stage $mpe_stage --num-pdfs 193 \
     data/train data/lang ${modeldir}_ali ${modeldir}_denlats \
     $modeldir/final.mdl.nnet3am $dir || exit 1;

 echo "get nnet3-ctc from nnet3-am"
   nnet3-am-copy --raw=true $dir/final.mdl $dir/final.mdl.raw || exit 1;
   nnet3-ctc-copy --set-raw-nnet=$dir/final.mdl.raw $modeldir/0.ctc_trans_mdl $dir/final.mdl.ctc || exit 1;
   mv $dir/final.mdl $dir/final.mdl.nnet3am
   mv $dir/final.mdl.ctc $dir/final.mdl

fi

exit 0;


