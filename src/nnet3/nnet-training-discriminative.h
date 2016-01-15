// nnet3/nnet-training-.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_TRAINING_DISCRIMINATIVE_H_
#define KALDI_NNET3_NNET_TRAINING_DISCRIMINATIVE_H_

#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-example-utils.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {
namespace nnet3 {

struct NnetTrainerOptions {
  bool zero_component_stats;
  bool store_component_stats;
  int32 print_interval;
  bool debug_computation;
  BaseFloat momentum;
  BaseFloat max_param_change;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  std::string criterion; // "mmi" or "mpe" or "smbr"
  BaseFloat acoustic_scale; // e.g. 0.1
  bool drop_frames; // for MMI, true if we ignore frames where alignment
                    // pdf-id is not in the lattice.
  bool one_silence_class;  // Affects MPE/SMBR>
  BaseFloat boost; // for MMI, boosting factor (would be Boosted MMI)... e.g. 0.1.

  std::string silence_phones_str; // colon-separated list of integer ids of silence phones,
                                  // for MPE/SMBR only.
  NnetTrainerOptions():
      zero_component_stats(true),
      store_component_stats(true),
      print_interval(100),
      debug_computation(false),
      momentum(0.0),
      max_param_change(1.0),
      criterion("smbr"), 
      acoustic_scale(0.1),
      drop_frames(false),
      one_silence_class(false),
      boost(0.0) { }
	  
	  
  void Register(OptionsItf *opts) {
    opts->Register("store-component-stats", &store_component_stats,
                   "If true, store activations and derivatives for nonlinear "
                   "components during training.");
    opts->Register("zero-component-stats", &zero_component_stats,
                   "If both this and --store-component-stats are true, then "
                   "the component stats are zeroed before training.");
    opts->Register("print-interval", &print_interval, "Interval (measured in "
                   "minibatches) after which we print out objective function "
                   "during training\n");
    opts->Register("max-param-change", &max_param_change, "The maximum change in"
                   "parameters allowed per minibatch, measured in Frobenius norm "
                   "over the entire model (change will be clipped to this value)");
    opts->Register("momentum", &momentum, "momentum constant to apply during "
                   "training (help stabilize update).  e.g. 0.9.  Note: we "
                   "automatically multiply the learning rate by (1-momenum) "
                   "so that the 'effective' learning rate is the same as "
                   "before (because momentum would normally increase the "
                   "effective learning rate by 1/(1-momentum))");
	opts->Register("criterion", &criterion, "Criterion, 'mmi'|'mpe'|'smbr', "
                   "determines the objective function to use.  Should match "
                   "option used when we created the examples.");
    opts->Register("acoustic-scale", &acoustic_scale, "Weighting factor to "
                   "apply to acoustic likelihoods.");
    opts->Register("drop-frames", &drop_frames, "For MMI, if true we drop frames "
                   "with no overlap of num and den frames");
    opts->Register("boost", &boost, "Boosting factor for boosted MMI (e.g. 0.1)");
    opts->Register("one-silence-class", &one_silence_class, "If true, newer "
                   "behavior which will tend to reduce insertions.");
    opts->Register("silence-phones", &silence_phones_str,
                   "For MPFE or SMBR, colon-separated list of integer ids of "
                   "silence phones, e.g. 1:2:3");


    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);


    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
  }
};

struct NnetDiscriminativeStats {
  double tot_t; // total number of frames
  double tot_t_weighted; // total number of frames times weight.
  double tot_num_count; // total count of numerator posterior (should be
                        // identical to denominator-posterior count, so we don't
                        // separately compute that).
  double tot_num_objf;  // for MMI, the (weighted) numerator likelihood; for
                        // SMBR/MPFE, 0.
  double tot_den_objf;  // for MMI, the (weighted) denominator likelihood; for
                        // SMBR/MPFE, the objective function.
  NnetDiscriminativeStats() { std::memset(this, 0, sizeof(*this)); }
  void Print(std::string criterion); // const NnetDiscriminativeUpdateOptions &opts);
  void Add(const NnetDiscriminativeStats &other);
};

class NnetTrainerDiscriminative {
 public:

  NnetTrainerDiscriminative (const NnetTrainerOptions &opts, 
                                        const ctc::CctcTransitionModel &tmodel, 
                                        Nnet *nnet, 
                                        NnetDiscriminativeStats *stats);

  void Train (const NnetExample &eg, const Lattice &clat, const std::vector<int32> &num_ali);

  ~NnetTrainerDiscriminative ();
  
 private:
  void ProcessOutputs(const NnetExample &eg, const Lattice &clat, 
                            const std::vector<int32> &num_ali, NnetComputer *computer);
  const NnetTrainerOptions &opts_;
  const ctc::CctcTransitionModel &tmodel_;
  
  Nnet *nnet_;
  Nnet *delta_nnet_;  // Only used if momentum != 0.0.  nnet representing
                      // accumulated parameter-change (we'd call this
                      // gradient_nnet_, but due to natural-gradient update,
                      // it's better to consider it as a delta-parameter nnet.
  NnetDiscriminativeStats *stats_; // the objective function, etc. 
  
  
  std::vector<int32> silence_phones_;
  CachingOptimizingCompiler compiler_;

};

/**
   This function computes the objective function, and if supply_deriv = true,
   supplies its derivative to the NnetComputation object.
   See also the function ComputeAccuracy(), declared in nnet-diagnostics.h.

  @param [in]  supervision   A GeneralMatrix, typically derived from a NnetExample,
                             containing the supervision posteriors or features.
  @param [in] objective_type The objective function type: kLinear = output *
                             supervision, or kQuadratic = -0.5 * (output -
                             supervision)^2.  kLinear is used for softmax
                             objectives; the network contains a LogSoftmax
                             layer which correctly normalizes its output.
  @param [in] output_name    The name of the output node (e.g. "output"), used to
                             look up the output in the NnetComputer object.

  @param [in] supply_deriv   If this is true, this function will compute the
                             derivative of the objective function and supply it
                             to the network using the function
                             NnetComputer::AcceptOutputDeriv
  @param [in,out] computer   The NnetComputer object, from which we get the
                             output using GetOutput and to which we may supply
                             the derivatives using AcceptOutputDeriv.
  @param [out] tot_weight    The total weight of the training examples.  In the
                             kLinear case, this is the sum of the supervision
                             matrix; in the kQuadratic case, it is the number of
                             rows of the supervision matrix.  In order to make
                             it possible to weight samples with quadratic
                             objective functions, we may at some point make it
                             possible for the supervision matrix to have an
                             extra column containing weights.  At the moment,
                             this is not supported.
  @param [out] tot_objf      The total objective function; divide this by the
                             tot_weight to get the normalized objective function.
*/
 void LatticeComputations(NnetDiscriminativeStats *stats, 
                                  ObjectiveType objective_type,
                                  const NnetTrainerOptions &opts,
                                  const ctc::CctcTransitionModel &tmodel,
                                  const Lattice &clat,
                                  const std::vector<int32> &num_ali,
                                  const GeneralMatrix &supervision,
                                  const std::string &output_name,
                                  bool supply_deriv,
                                  NnetComputer *computer,
                                  BaseFloat *tot_weight,
                                  BaseFloat *tot_objf);

 double GetDiscriminativePosteriors(const NnetTrainerOptions &opts,
                                                const ctc::CctcTransitionModel &tmodel,
                                                const GeneralMatrix &supervision, 
                                                const Lattice &clat,
                                                const std::vector<int32> &num_ali,
                                                Posterior *post);   

 static inline Int32Pair MakePair(int32 first, int32 second) {
    Int32Pair ans;
    ans.first = first;
    ans.second = second;
    return ans;
 }

 
} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_TRAINING_H_
