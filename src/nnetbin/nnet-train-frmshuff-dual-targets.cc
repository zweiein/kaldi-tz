// nnetbin/nnet-train-frmshuff-dual-targets.cc

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)

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
        "Perform one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "Usage:  nnet-train-frmshuff [options] <feature-rspecifier> <hard-targets-rspecifier> <soft-targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-frmshuff scp:feature.scp ark:posterior.ark ark:Matrix.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false,
         randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");
    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    std::string utt_weights;
    po.Register("utt-weights", &utt_weights, "Per-utterance weights (scalar applied to frame-weights).");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
    
    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value");

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

    if (dropout_retention > 0.0) {
      nnet_transf.SetDropoutRetention(dropout_retention);
      nnet.SetDropoutRetention(dropout_retention);
    }
    if (crossvalidate) {
      nnet_transf.SetDropoutRetention(1.0);
      nnet.SetDropoutRetention(1.0);
    }

    KALDI_ASSERT(temperature > 0.0);

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader hard_targets_reader(hard_targets_rspecifier);
    SequentialBaseFloatMatrixReader soft_targets_reader(soft_targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }
    RandomAccessBaseFloatReader utt_weights_reader;
    if (utt_weights != "") {
      utt_weights_reader.Open(utt_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer hard_targets_randomizer(rnd_opts);
    MatrixRandomizer soft_targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    Xent xent_hard, xent_soft;
    Mse mse_hard, mse_soft;
    
    MultiTaskLoss multitask_hard, multitask_soft;
    if (0 == objective_function.compare(0,9,"multitask")) {
      // objective_function contains something like : 
      // 'multitask,xent,2456,1.0,mse,440,0.001'
      //
      // the meaning is following:
      // 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
      multitask_hard.InitFromString(objective_function);
      multitask_soft.InitFromString(objective_function);
    }
    
    CuMatrix<BaseFloat> feats_transf, nnet_out_hard, nnet_out_soft, obj_diff_hard, obj_diff_soft;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    while (!feature_reader.Done()) {
#if HAVE_CUDA==1
      // check the GPU is not overheated
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the randomizer
      for ( ; !feature_reader.Done() && !soft_targets_reader.Done(); feature_reader.Next(), soft_targets_reader.Next()) {
        if (feature_randomizer.IsFull()) break; // suspend, keep utt for next loop
        std::string utt = feature_reader.Key();
        KALDI_VLOG(3) << "Reading " << utt;
        // check that we have targets
        if (!hard_targets_reader.HasKey(utt) || soft_targets_reader.Key() != utt) {
          KALDI_WARN << utt << ", missing targets";
          num_no_tgt_mat++;
          continue;
        }
        // check we have per-frame weights
        if (frame_weights != "" && !weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-frame weights";
          num_other_error++;
          continue;
        }
        // check we have per-utterance weights
        if (utt_weights != "" && !utt_weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-utterance weight";
          num_other_error++;
          continue;
        }
        // get feature / target pair
        Matrix<BaseFloat> mat = feature_reader.Value();
        Posterior hard_targets = hard_targets_reader.Value(utt);
        Matrix<BaseFloat> soft_targets = soft_targets_reader.Value();
        // get per-frame weights
        Vector<BaseFloat> weights;
        if (frame_weights != "") {
          weights = weights_reader.Value(utt);
        } else { // all per-frame weights are 1.0
          weights.Resize(mat.NumRows());
          weights.Set(1.0);
        }
        // multiply with per-utterance weight,
        if (utt_weights != "") {
          BaseFloat w = utt_weights_reader.Value(utt);
          KALDI_ASSERT(w >= 0.0);
          if (w == 0.0) continue; // remove sentence from training,
          weights.Scale(w);
        }

        // correct small length mismatch ... or drop sentence
        {
          // add lengths to vector
          std::vector<int32> lenght;
          lenght.push_back(mat.NumRows());
          lenght.push_back(hard_targets.size());
          lenght.push_back(soft_targets.NumRows());
          lenght.push_back(weights.Dim());
          // find min, max
          int32 min = *std::min_element(lenght.begin(),lenght.end());
          int32 max = *std::max_element(lenght.begin(),lenght.end());
          // fix or drop ?
          if (max - min < length_tolerance) {
            if(mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
            if(hard_targets.size() != min) hard_targets.resize(min);
            if(soft_targets.NumRows() != min) soft_targets.Resize(min, soft_targets.NumCols(), kCopyData);
            if(weights.Dim() != min) weights.Resize(min, kCopyData);
          } else {
            KALDI_WARN << utt << ", length mismatch of targets " << hard_targets.size()
                       << " and features " << mat.NumRows();
            num_other_error++;
            continue;
          }
        }
        // apply optional feature transform
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

        // pass data to randomizers
        KALDI_ASSERT(feats_transf.NumRows() == hard_targets.size());
        KALDI_ASSERT(feats_transf.NumRows() == soft_targets.NumRows());
        feature_randomizer.AddData(feats_transf);
        hard_targets_randomizer.AddData(hard_targets);
        soft_targets_randomizer.AddData(CuMatrix<BaseFloat>(soft_targets));
        weights_randomizer.AddData(weights);
        num_done++;
      
        // report the speed
        if (num_done % 5000 == 0) {
          double time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
        }
      }

      // randomize
      if (!crossvalidate && randomize) {
        const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());
        feature_randomizer.Randomize(mask);
        hard_targets_randomizer.Randomize(mask);
        soft_targets_randomizer.Randomize(mask);
        weights_randomizer.Randomize(mask);
      }

      // train with data from randomizers (using mini-batches)
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                          hard_targets_randomizer.Next(),
                                          soft_targets_randomizer.Next(),
                                          weights_randomizer.Next()) {
        // get block of feature/target pairs
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const Posterior& nnet_tgt_hard = hard_targets_randomizer.Value();
        const CuMatrixBase<BaseFloat>& nnet_tgt_soft = soft_targets_randomizer.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

        // forward pass
        nnet.SetTemperature(1.0);
        nnet.Propagate(nnet_in, &nnet_out_hard);
        nnet.SetTemperature(temperature);        
        nnet.Propagate(nnet_in, &nnet_out_soft);
        

        // evaluate objective function we've chosen
        if (objective_function == "xent") {
          // gradients re-scaled by weights in Eval,
          xent_hard.Eval(frm_weights, nnet_out_hard, nnet_tgt_hard, &obj_diff_hard);
          xent_soft.Eval(frm_weights, nnet_out_soft, nnet_tgt_soft, &obj_diff_soft);
        } else if (objective_function == "mse") {
          // gradients re-scaled by weights in Eval,
          mse_hard.Eval(frm_weights, nnet_out_hard, nnet_tgt_hard, &obj_diff_hard);
          mse_soft.Eval(frm_weights, nnet_out_soft, nnet_tgt_soft, &obj_diff_soft);
        } else if (0 == objective_function.compare(0,9,"multitask")) {
          // gradients re-scaled by weights in Eval,
          multitask_hard.Eval(frm_weights, nnet_out_hard, nnet_tgt_hard, &obj_diff_hard);
          multitask_soft.Eval(frm_weights, nnet_out_soft, nnet_tgt_soft, &obj_diff_soft);
        } else {
          KALDI_ERR << "Unknown objective function code : " << objective_function;
        }

        // backward pass
        if (!crossvalidate) {
          // backpropagate
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
        
        // monitor the NN training
        if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
          if ((total_frames/25000) != ((total_frames+nnet_in.NumRows())/25000)) { // print every 25k frames
            KALDI_VLOG(2) << "### After " << total_frames << " frames,";
            KALDI_VLOG(2) << nnet.InfoPropagate();
            if (!crossvalidate) {
              KALDI_VLOG(2) << nnet.InfoGradient();
            }
          }
        }
        
        total_frames += nnet_in.NumRows();
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
      KALDI_LOG << mse_hard.Report();
    } else if (0 == objective_function.compare(0,9,"multitask")) {
      KALDI_LOG << multitask_hard.Report();
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
