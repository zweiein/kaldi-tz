#!/bin/bash

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


expdir=exp/nnet3


  # this does offline decoding that should give the same results as the real
  # online decoding.
  #for lm_suffix in tgpr bd_tgpr; do
  for lm_suffix in bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    # use already-built graphs.
    for year in  eval92; do
      #for mdl in `ls $expdir | grep speech`; do
       mdl=self_start10_period5_T2
        echo $mdl
        [ -e $expdir/$mdl/final.mdl ] || continue;
        [ -e $expdir/$mdl/decode_${lm_suffix}_${year}/wer_11_0.0 ] || steps/nnet3/decode.sh --nj 8 \
         --cmd "$decode_cmd" --stage -2 \
        $graph_dir data_fbank/test_${year}  $expdir/$mdl/decode_${lm_suffix}_${year} || exit 1;
      #done
    done
  done


exit 0;

# results:
grep WER exp/nnet3/nnet_tdnn_a/decode_{tgpr,bd_tgpr}_{eval92,dev93}/scoring_kaldi/best_wer
exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/scoring_kaldi/best_wer:%WER 6.03 [ 340 / 5643, 74 ins, 20 del, 246 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/wer_13_1.0
exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/scoring_kaldi/best_wer:%WER 9.35 [ 770 / 8234, 162 ins, 84 del, 524 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/wer_11_0.5
exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/scoring_kaldi/best_wer:%WER 3.81 [ 215 / 5643, 30 ins, 18 del, 167 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/wer_10_1.0
exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/scoring_kaldi/best_wer:%WER 6.74 [ 555 / 8234, 69 ins, 72 del, 414 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/wer_11_0.0
b03:s5:
