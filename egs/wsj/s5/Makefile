TK_PATH = /nfs/raid88/thintern/tangzy_sino_1000h_16k_interspeech/tk


# running options
NCPUS = 100
#USE_SGE = FALSE
USE_SGE = TRUE


DATA_DIR = /nfs/raid88/thintern/tangzy_sino_1000h_16k_interspeech/data
DATA_SET = train

#CV_SET = cv
#FEAT_TYPE = mfcc
#include $(TK_PATH)/common.mk


ifeq ($(USE_SGE), TRUE)
        TRAIN_CMD = "queue.pl -q 110.q"
else
        TRAIN_CMD = "run.pl"
endif

ifeq ($(USE_SGE), TRUE)
        CUDA_CMD = "queue.pl -q gpu96.q"
else
       CUDA_CMD = "run.pl"
endif


gmm: quick_ali quick_ali_cv
dnn: nnet



########## soft pre-training #########

# default: ref_mdl was trained by the same data for the new model

ref_mdl = exp_nnet3/relu_6-2000_splice5_200h/final.mdl
nnet3_relu_4-400_splice5_200h_pretrain:
	steps/nnet3/train_tdnn_soft.sh --stage -6 \
        --egs-dir /nfs/raid88/zhangzhiyong/work/train_sino_1000h_16k/exp_nnet3/relu_6-2000_splice5_200h/egs \
        --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 4 \
        --splice-indexes "-5,-4,-3,-2,-1,0,1,2,3,4,5 0 0 0" \
        --feat-type raw \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --initial-effective-lrate 0.008 --final-effective-lrate 0.0005 \
        --cmd $(CUDA_CMD) \
        --relu-dim 400 \
        --remove-egs false --temperature 0.5 \
	$(DATA_DIR)/train_200h $(DATA_DIR)/lang exp/tri4b $(ref_mdl) exp_nnet3/relu_4-400_splice5_200h_pretrain_T0.5

ref_mdl = exp_nnet3/relu_6-2000_splice5/final.mdl
nnet3_relu_4-400_splice5_1000h_pretrain:
	steps/nnet3/train_tdnn_soft.sh --stage -6 \
        --egs-dir /nfs/raid88/zhangzhiyong/work/train_sino_1000h_16k/exp_nnet3/relu_6-2000_splice5/egs \
        --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 4 \
        --splice-indexes "-5,-4,-3,-2,-1,0,1,2,3,4,5 0 0 0" \
        --feat-type raw \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --initial-effective-lrate 0.008 --final-effective-lrate 0.0005 \
        --cmd $(CUDA_CMD) \
        --relu-dim 400 \
        --remove-egs false --temperature 2.0 \
	$(DATA_DIR)/train $(DATA_DIR)/lang exp/tri4b $(ref_mdl) exp_nnet3/relu_4-400_splice5_pretrain_T2.0

########## hard re-training #########

## retrain directly, stage > (layers-1)*add_layer_period  ##
nnet3_relu_4-400_splice5_200h_pretrain_refine:
	steps/nnet3/train_tdnn.sh --stage 7 \
        --egs-dir /nfs/raid88/zhangzhiyong/work/train_sino_1000h_16k/exp_nnet3/relu_6-2000_splice5_200h/egs \
        --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
        --splice-indexes "-5,-4,-3,-2,-1,0,1,2,3,4,5 0 0 0" \
        --feat-type raw \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --initial-effective-lrate 0.008 --final-effective-lrate 0.0005 \
        --cmd $(CUDA_CMD) \
        --relu-dim 400 \
        --remove-egs false \
	$(DATA_DIR)/train_200h $(DATA_DIR)/lang exp/tri4b exp_nnet3/relu_4-400_splice5_200h_pretrain_T2.0_refine

## rand last layer, stage = (layers-1)*add_layer_period  ##
nnet3_relu_4-400_splice5_200h_pretrain_T2.0_rand_retrain:
	steps/nnet3/train_tdnn.sh --stage 6 \
        --egs-dir /nfs/raid88/zhangzhiyong/work/train_sino_1000h_16k/exp_nnet3/relu_6-2000_splice5_200h/egs \
        --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
        --splice-indexes "-5,-4,-3,-2,-1,0,1,2,3,4,5 0 0 0" \
        --feat-type raw \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --initial-effective-lrate 0.008 --final-effective-lrate 0.0005 \
        --cmd $(CUDA_CMD) \
        --relu-dim 400 \
        --remove-egs false \
        $(DATA_DIR)/train_200h $(DATA_DIR)/lang exp/tri4b exp_nnet3/relu_4-400_splice5_200h_pretrain_T2.0_rand_retrain



########## baseline ##########

nnet3_relu_4-400_splice5:
	steps/nnet3/train_tdnn.sh --stage -6 \
        --egs-dir /nfs/raid88/zhangzhiyong/work/train_sino_1000h_16k/exp_nnet3/relu_6-2000_splice5/egs \
        --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
        --splice-indexes "-5,-4,-3,-2,-1,0,1,2,3,4,5 0 0 0" \
        --feat-type raw \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --initial-effective-lrate 0.008 --final-effective-lrate 0.0005 \
        --cmd $(CUDA_CMD) \
        --relu-dim 400 \
        --remove-egs false \
	$(DATA_DIR)/$(DATA_SET) $(DATA_DIR)/lang exp/tri4b exp_nnet3/relu_4-400_splice5

nnet3_relu_4-400_splice5_500h:
	steps/nnet3/train_tdnn.sh --stage -6 \
        --egs-dir /nfs/raid88/zhangzhiyong/work/train_sino_1000h_16k/exp_nnet3/relu_6-2000_splice5_500h/egs \
        --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
        --splice-indexes "-5,-4,-3,-2,-1,0,1,2,3,4,5 0 0 0" \
        --feat-type raw \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --initial-effective-lrate 0.008 --final-effective-lrate 0.0005 \
        --cmd $(CUDA_CMD) \
        --relu-dim 400 \
        --remove-egs false \
	$(DATA_DIR)/train_500h $(DATA_DIR)/lang exp/tri4b exp_nnet3/relu_4-400_splice5_500h

nnet3_relu_4-400_splice5_200h_final4:
	steps/nnet3/train_tdnn.sh --stage 24 \
        --egs-dir /nfs/raid88/zhangzhiyong/work/train_sino_1000h_16k/exp_nnet3/relu_6-2000_splice5_200h/egs \
        --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 4 \
        --splice-indexes "-5,-4,-3,-2,-1,0,1,2,3,4,5 0 0 0" \
        --feat-type raw \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --initial-effective-lrate 0.008 --final-effective-lrate 0.0005 \
        --cmd $(CUDA_CMD) \
        --relu-dim 400 \
        --remove-egs false \
	$(DATA_DIR)/train_200h $(DATA_DIR)/lang exp/tri4b exp_nnet3/relu_4-400_splice5_200h_final4

nnet3_relu_6-600_splice5:
	steps/nnet3/train_tdnn.sh --stage -6 \
        --egs-dir /nfs/raid88/zhangzhiyong/work/train_sino_1000h_16k/exp_nnet3/relu_6-2000_splice5/egs \
        --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
        --splice-indexes "-5,-4,-3,-2,-1,0,1,2,3,4,5 0 0 0 0 0" \
        --feat-type raw \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --initial-effective-lrate 0.008 --final-effective-lrate 0.0005 \
        --cmd $(CUDA_CMD) \
        --relu-dim 600 \
        --remove-egs false \
	$(DATA_DIR)/$(DATA_SET) $(DATA_DIR)/lang exp/tri4b exp_nnet3/relu_6-600_splice5

nnet3_relu_6-600_splice5_500h:
	steps/nnet3/train_tdnn.sh --stage -6 \
        --egs-dir /nfs/raid88/zhangzhiyong/work/train_sino_1000h_16k/exp_nnet3/relu_6-2000_splice5_500h/egs \
        --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
        --splice-indexes "-5,-4,-3,-2,-1,0,1,2,3,4,5 0 0 0 0 0" \
        --feat-type raw \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --initial-effective-lrate 0.008 --final-effective-lrate 0.0005 \
        --cmd $(CUDA_CMD) \
        --relu-dim 600 \
        --remove-egs false \
	$(DATA_DIR)/train_500h $(DATA_DIR)/lang exp/tri4b exp_nnet3/relu_6-600_splice5_500h

nnet3_relu_6-600_splice5_200h:
	steps/nnet3/train_tdnn.sh --stage -6 \
        --egs-dir /nfs/raid88/zhangzhiyong/work/train_sino_1000h_16k/exp_nnet3/relu_6-2000_splice5_200h/egs \
        --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
        --splice-indexes "-5,-4,-3,-2,-1,0,1,2,3,4,5 0 0 0 0 0" \
        --feat-type raw \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --initial-effective-lrate 0.008 --final-effective-lrate 0.0005 \
        --cmd $(CUDA_CMD) \
        --relu-dim 600 \
        --remove-egs false \
	$(DATA_DIR)/train_200h $(DATA_DIR)/lang exp/tri4b exp_nnet3/relu_6-600_splice5_200h

