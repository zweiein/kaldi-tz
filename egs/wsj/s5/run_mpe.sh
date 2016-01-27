#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

. utils/parse_options.sh || exit 1;


chunk_width=75         # for make_denlats.sh
chunk_left_context=30
stage=3
mpe_stage=-10  #-10,-4
cmd=run.pl

trans_model=exp/ctc/tri5b_tree/final.mdl
ctc_trans_mdl=exp/ctc/lstm_h/0.ctc_trans_mdl
graphdir=exp/ctc/lstm_h/graph_bd_tgpr_0.15
modeldir=exp/ctc/lstm_h
tri_ali=exp/ctc/tri5b_tree
tri_latdir=exp/tri4b_lats_si284
#tri_ali=${modeldir}_ali
dir=${modeldir}_mpe

hmm_mdl=exp/ctc/tri5b_tree/final.mdl

if [ $stage -le 1 ]; then
 echo "make alimgents"
   steps/nnet3/ctc/align.sh --nj 8 --cmd "$decode_cmd" --frame_subsampling_factor 3 \
     data/train data/lang_ctc $modeldir ${modeldir}_ali || exit 1;
fi

if [ $stage -le 2 ]; then
 echo "make lattices"
   steps/nnet3/ctc/make_denlats.sh  --nj 8 --cmd "$decode_cmd" --frame_subsampling_factor 3 \
    --frames-per-chunk $chunk_width --extra-left-context $chunk_left_context \
      $graphdir data/train $modeldir ${modeldir}_denlats_sub1 || exit 1;
fi

if [ $stage -le 3 ]; then

 if [ ! -f $modeldir/final.mdl.softmax.mpe ];then
   echo "$modeldir/final.mdl.softmax.mpe should have SoftmaxComponent instead of LogSoftmaxComponent 
     and mpe instead of linear for mpe training, and copy final.mdl.softmax.mpe to final.mdl"
   nnet3-ctc-copy --binary=false $modeldir/final.mdl $modeldir/final.mdl.softmax.mpe
   sed -i 's/LogSoftmaxComponent/SoftmaxComponent/g'  $modeldir/final.mdl.softmax.mpe
   sed -i 's/linear/mpe/g'  $modeldir/final.mdl.softmax.mpe
   nnet3-ctc-copy $modeldir/final.mdl.softmax.mpe $modeldir/final.mdl
 fi


 dir=${dir}_lr0.0002
 echo "mpe training"
   steps/nnet3/ctc/train_discriminative.sh --cmd "$cuda_cmd" --train_cmd "$train_cmd" --learning-rate 0.0002 \
     --num-jobs-nnet 4 --stage $mpe_stage --hmm_mdl $hmm_mdl --frame_subsampling_factor 3 \
     --tri_latdir $tri_latdir --ctc_trans_mdl $ctc_trans_mdl \
     data/train data/lang ${modeldir}_ali ${modeldir}_denlats \
     $modeldir/final.mdl $dir || exit 1;

 if [ -f $dir ]; then
   echo "SoftmaxComponent back to LogSoftmaxComponent for decoding"
   nnet3-ctc-copy --binary=false $dir/final.mdl $dir/final.mdl.logsoftmax
   sed -i 's/SoftmaxComponent/LogSoftmaxComponent/g'  $dir/final.mdl.logsoftmax
   nnet3-ctc-copy $dir/final.mdl.logsoftmax $dir/final.mdl
 fi

 echo "mpe done."

fi


exit 0;


