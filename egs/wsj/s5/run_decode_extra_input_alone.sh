#!/bin/bash

#name=eval92
#data=data-fbank/test_$name
#name=dev_0330
#data=data-fbank/$name

. ./cmd.sh


extra_dir=exp/lv_model

for i in test_chinglish test_eval92; do
name=$i
data=$i
for dir in exp/*3*; do
    echo $dir
    [ ! -e $dir/decode_tgpr_5k_$name/wer_11 ] && [ -e $dir/final.nnet ] && steps/nnet/decode_extra_input_alone.sh --cmd "$decode_cmd" --nj 10 --acwt 0.10 --config conf/decode_dnn.config \
  graph  $data $extra_dir $dir/decode_tgpr_5k_$name &
done
done
wait
