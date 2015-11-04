// nnetbin/nnet-train-lstm-streams-dual-targets.cc

// Copyright 2015  Brno University of Technology (Author: Karel Vesely)
//           2014  Jiayu DU (Jerry), Wei Li

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.



#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of LSTM training by Stochastic Gradient Descent.\n"
        "The updates are done per-utterance, shuffling options are dummy for compatibility reason.\n"
        "\n"
        "Usage: nnet-train-lstm-streams [options] <feature-rspecifier> <hard-targets-rspecifier> <soft-targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-lstm-streams scp:feature.scp ark:posterior.ark ark:Matrix.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");
    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    /*
    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");
    */

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 
    
    //<jiayu>
    int32 targets_delay=5;
    po.Register("targets-delay", &targets_delay, "---LSTM--- BPTT targets delay"); 

    int32 batch_size=20;
    po.Register("batch-size", &batch_size, "---LSTM--- BPTT batch size"); 

    int32 num_stream=4;
    po.Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training"); 

    int32 dump_interval=0;
    po.Register("dump-interval", &dump_interval, "---LSTM--- num utts between model dumping [ 0 == disabled ]"); 
    //</jiayu>

    // Add dummy randomizer options, to make the tool compatible with standard scripts
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    bool randomize = false;
    po.Register("randomize", &randomize, "Dummy option, for compatibility...");
    //

    BaseFloat temperature = 1.0;
    po.Register("temperature", &temperature, "");

    BaseFloat hard_scale = 0.0;
    po.Register("hard-scale", &hard_scale, "");

    BaseFloat soft_scale = 1.0;
    po.Register("soft-scale", &soft_scale, "");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 5-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      hard_targets_rspecifier = po.GetArg(2),
      soft_targets_rspecifier = po.GetArg(3),
      model_filename = po.GetArg(4);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(5);
    }

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    KALDI_ASSERT(temperature > 0.0);
    

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader hard_target_reader(hard_targets_rspecifier);
    SequentialBaseFloatMatrixReader soft_target_reader(soft_targets_rspecifier);
    
    /*
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }
    */

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    Xent xent_hard, xent_soft;
    Mse mse;
    
    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;

    //  book-keeping for multi-streams
    std::vector<std::string> keys(num_stream);
    std::vector<Matrix<BaseFloat> > feats(num_stream);
    std::vector<Posterior> hard_targets(num_stream);
    std::vector<Matrix<BaseFloat> > soft_targets(num_stream);
    std::vector<int> curt(num_stream, 0);
    std::vector<int> lent(num_stream, 0);
    std::vector<int> new_utt_flags(num_stream, 0);

    // bptt batch buffer
    int32 feat_dim = nnet.InputDim();
    int32 out_dim = nnet.OutputDim();
    Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
    Matrix<BaseFloat> feat(batch_size * num_stream, feat_dim, kSetZero);
    Posterior hard_target(batch_size * num_stream);
    Matrix<BaseFloat> soft_target(batch_size * num_stream, out_dim, kSetZero);    
    CuMatrix<BaseFloat> feat_transf, nnet_out_hard, nnet_out_soft, obj_diff_hard, obj_diff_soft;

    while (1) {
        // loop over all streams, check if any stream reaches the end of its utterance,
        // if any, feed the exhausted stream with a new utterance, update book-keeping infos
        for (int s = 0; s < num_stream; s++) {
            // this stream still has valid frames
            if (curt[s] < lent[s]) {
                new_utt_flags[s] = 0;
                continue;
            }
            // else, this stream exhausted, need new utterance
            while (!feature_reader.Done()) {
                const std::string& key = feature_reader.Key();
                // get the feature matrix,
                const Matrix<BaseFloat> &mat = feature_reader.Value();
                // forward the features through a feature-transform,
                nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feat_transf);
                
                // get the labels,
                if (!hard_target_reader.HasKey(key) || soft_target_reader.Key() != key) {
                    KALDI_WARN << key << ", missing targets";
                    num_no_tgt_mat++;
                    feature_reader.Next();
                    soft_target_reader.Next();
                    continue;
                }
                const Posterior& hard_target_utt= hard_target_reader.Value(key);
                const Matrix<BaseFloat>& soft_target_utt = soft_target_reader.Value();

                // check that the length matches,
                if (feat_transf.NumRows() != hard_target_utt.size() || feat_transf.NumRows() != soft_target_utt.NumRows()) {
                    KALDI_WARN << key << ", length miss-match between feats and targets, skip";
                    num_other_error++;
                    feature_reader.Next();
                    soft_target_reader.Next();
                    continue;
                }

                // checks ok, put the data in the buffers,
                keys[s] = key;
                feats[s].Resize(feat_transf.NumRows(), feat_transf.NumCols());
                feat_transf.CopyToMat(&feats[s]); 
                hard_targets[s] = hard_target_utt;
                soft_targets[s] = soft_target_utt;
                curt[s] = 0;
                lent[s] = feats[s].NumRows();
                new_utt_flags[s] = 1;  // a new utterance feeded to this stream
                feature_reader.Next();
                soft_target_reader.Next();
                break;
            }
        }

        // we are done if all streams are exhausted
        int done = 1;
        for (int s = 0; s < num_stream; s++) {
            if (curt[s] < lent[s]) done = 0;  // this stream still contains valid data, not exhausted
        }
        if (done) break;

        // fill a multi-stream bptt batch
        // * frame_mask: 0 indicates padded frames, 1 indicates valid frames
        // * target: padded to batch_size
        // * feat: first shifted to achieve targets delay; then padded to batch_size
        for (int t = 0; t < batch_size; t++) {
            for (int s = 0; s < num_stream; s++) {
                // frame_mask & targets padding
                if (curt[s] < lent[s]) {
                    frame_mask(t * num_stream + s) = 1;
                    hard_target[t * num_stream + s] = hard_targets[s][curt[s]];
                    soft_target.Row(t * num_stream + s).CopyFromVec(soft_targets[s].Row(curt[s]));
                } else {
                    frame_mask(t * num_stream + s) = 0;
                    hard_target[t * num_stream + s] = hard_targets[s][lent[s]-1];
                    soft_target.Row(t * num_stream + s).CopyFromVec(soft_targets[s].Row(curt[s]-1));
                }
                // feat shifting & padding
                if (curt[s] + targets_delay < lent[s]) {
                    feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s]+targets_delay));
                } else {
                    feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(lent[s]-1));
                }

                curt[s]++;
            }
        }

        // for streams with new utterance, history states need to be reset
        nnet.ResetLstmStreams(new_utt_flags);

        // forward pass
        nnet.Propagate(CuMatrix<BaseFloat>(feat), temperature, &nnet_out_hard, &nnet_out_soft);
    
        // evaluate objective function we've chosen
        if (objective_function == "xent") {
            xent_hard.Eval(frame_mask, nnet_out_hard, hard_target, &obj_diff_hard);
            xent_soft.Eval(frame_mask, nnet_out_soft, CuMatrix<BaseFloat>(soft_target), &obj_diff_soft);
        //} else if (objective_function == "mse") {     // not supported yet
        //    mse.Eval(frame_mask, nnet_out, targets_batch, &obj_diff);
        } else {
            KALDI_ERR << "Unknown objective function code : " << objective_function;
        }
    
        // backward pass
        if (!crossvalidate) {
            obj_diff_hard.Scale(hard_scale);
            if(hard_scale != 0.0) { //when hard targets are used too
              obj_diff_soft.Scale(temperature);
            }
            obj_diff_soft.Scale(soft_scale);
            obj_diff_soft.AddMat(1.0, obj_diff_hard);
            nnet.Backpropagate(obj_diff_soft, NULL);
        }

        // 1st minibatch : show what happens in network 
        if (kaldi::g_kaldi_verbose_level >= 1 && total_frames == 0) { // vlog-1
            KALDI_VLOG(1) << "### After " << total_frames << " frames,";
            KALDI_VLOG(1) << nnet.InfoPropagate();
            if (!crossvalidate) {
                KALDI_VLOG(1) << nnet.InfoBackPropagate();
                KALDI_VLOG(1) << nnet.InfoGradient();
            }
        }

        int frame_progress = frame_mask.Sum();
        total_frames += frame_progress;

        int num_done_progress = 0;
        for (int i =0; i < new_utt_flags.size(); i++) {
            num_done_progress += new_utt_flags[i];
        }
        num_done += num_done_progress;
        
        // monitor the NN training
        if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
            if ((total_frames-frame_progress)/25000 != (total_frames/25000)) { // print every 25k frames
                KALDI_VLOG(2) << "### After " << total_frames << " frames,";
                KALDI_VLOG(2) << nnet.InfoPropagate();
                if (!crossvalidate) {
                    KALDI_VLOG(2) << nnet.InfoBackPropagate();
                    KALDI_VLOG(2) << nnet.InfoGradient();
                }
            }
        }

        // report the speed
        if ((num_done-num_done_progress)/1000 != (num_done/1000)) {
            double time_now = time.Elapsed();
            KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
            
#if HAVE_CUDA==1
            // check the GPU is not overheated
            CuDevice::Instantiate().CheckGpuHealth();
#endif
        }

        if (dump_interval > 0) { // disabled by 'dump_interval == 0',
          if ((num_done-num_done_progress)/dump_interval != (num_done/dump_interval)) {
              char nnet_name[512];
              if (!crossvalidate) {
                  sprintf(nnet_name, "%s_utt%d", target_model_filename.c_str(), num_done);
                  nnet.Write(nnet_name, binary);
              }
          }
        }
    }
      
    // after last minibatch : show what happens in network 
    if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
      KALDI_VLOG(1) << "### After " << total_frames << " frames,";
      KALDI_VLOG(1) << nnet.InfoPropagate();
      if (!crossvalidate) {
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  

    if (objective_function == "xent") {
      KALDI_LOG << xent_hard.Report();
    } else if (objective_function == "mse") {
      KALDI_LOG << mse.Report();
    } else {
      KALDI_ERR << "Unknown objective function code : " << objective_function;
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
