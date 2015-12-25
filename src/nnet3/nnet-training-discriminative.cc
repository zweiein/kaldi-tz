// nnet3/nnet-training-discriminative.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-training-discriminative.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/am-nnet-simple.h"
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "lat/kaldi-lattice.h"
#include "matrix/sparse-matrix.h"
#include "matrix/kaldi-matrix.h"
#include "lat/kaldi-lattice.h"
#include "matrix/matrix-functions.h"
#include <cmath>

namespace kaldi {
namespace nnet3 {
	

NnetTrainerDiscriminative::NnetTrainerDiscriminative (const NnetTrainerOptions &opts, 
                                        const AmNnetSimple &am_nnet, 
                                        const TransitionModel &tmodel, 
                                        Nnet *nnet, 
                                        NnetDiscriminativeStats *stats):
            opts_(opts), am_nnet_(am_nnet), tmodel_(tmodel),
			nnet_(nnet), stats_(stats), compiler_(*nnet, opts_.optimize_config) {
 
      if (opts_.zero_component_stats) 
        ZeroComponentStats(nnet_);
      if (opts_.momentum == 0.0 && opts_.max_param_change == 0.0) {
        delta_nnet_= NULL;
      } else {
        KALDI_ASSERT(opts_.momentum >= 0.0 &&
                 opts_.max_param_change >= 0.0);
        delta_nnet_ = nnet_->Copy();
        bool is_gradient = false;  //setting this to true would disable the
                                   // natural-gradient updates.
        SetZero(is_gradient, delta_nnet_);
      }

}
            
void NnetTrainerDiscriminative::Train(const NnetExample &eg, 
                                        const Lattice &clat, 
                                        const std::vector<int32> &num_ali) {
  bool need_model_derivative = true;
  ComputationRequest request;

  GetComputationRequest(*nnet_, eg, need_model_derivative,
                        opts_.store_component_stats,
                        &request);
  const NnetComputation *computation = compiler_.Compile(request);

  NnetComputer computer(opts_.compute_config, *computation,
                        *nnet_,
                        (delta_nnet_ == NULL ? nnet_ : delta_nnet_));
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.io);
  computer.Forward();
  this->ProcessOutputs(eg, clat, num_ali, &computer);
  computer.Backward();

  if (delta_nnet_ != NULL) {
    BaseFloat scale = (1.0 - opts_.momentum);
    if (opts_.max_param_change != 0.0) {
      BaseFloat param_delta =
          std::sqrt(DotProduct(*delta_nnet_, *delta_nnet_)) * scale;
      if (param_delta > opts_.max_param_change) {
        if (param_delta - param_delta != 0.0) {
          KALDI_WARN << "Infinite parameter change, will not apply.";
          SetZero(false, delta_nnet_);
        } else {
          scale *= opts_.max_param_change / param_delta;
          KALDI_LOG << "Parameter change too big: " << param_delta << " > "
                    << "--max-param-change=" << opts_.max_param_change
                    << ", scaling by " << opts_.max_param_change / param_delta;
        }
      }
    }
    AddNnet(*delta_nnet_, scale, nnet_);
    ScaleNnet(opts_.momentum, delta_nnet_);
  }
}

void NnetTrainerDiscriminative::ProcessOutputs(const NnetExample &eg, const Lattice &clat, 
                            const std::vector<int32> &num_ali, NnetComputer *computer) {

	if (!SplitStringToIntegers(opts_.silence_phones_str, ":", false,
				                             &silence_phones_)) {
		    KALDI_ERR << "Bad value for --silence-phones option: "
				              << opts_.silence_phones_str;
	}
	
  std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
      end = eg.io.end();
  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 node_index = nnet_->GetNodeIndex(io.name);
    KALDI_ASSERT(node_index >= 0);
    if (nnet_->IsOutputNode(node_index)) {
      ObjectiveType obj_type = nnet_->GetNode(node_index).u.objective_type;
      BaseFloat tot_weight, tot_objf;
      bool supply_deriv = true;
      LatticeComputations(stats_,obj_type,
                          opts_,am_nnet_,tmodel_,
                          clat,num_ali,io.features,
                          io.name,supply_deriv, computer,
                          &tot_weight, &tot_objf);
    }
  }
}

NnetTrainerDiscriminative::~NnetTrainerDiscriminative() {
  delete delta_nnet_;
}
void LatticeComputations (NnetDiscriminativeStats *stats, 
                                  ObjectiveType objective_type,
                                  const NnetTrainerOptions &opts,
                                  const AmNnetSimple &am_nnet,
                                  const TransitionModel &tmodel,
                                  const Lattice &lat,
                                  const std::vector<int32> &num_ali,
                                  const GeneralMatrix &supervision,
                                  const std::string &output_name,
                                  bool supply_deriv,
                                  NnetComputer *computer,
                                  BaseFloat *tot_weight,
                                  BaseFloat *tot_objf) {

 BaseFloat eg_weight = 1;
 Lattice clat = lat;
 std::vector<int32> silence_phones; // derived from opts.silence_phones_str
   if (!SplitStringToIntegers(opts.silence_phones_str, ":", false,
                             &silence_phones)) {
    KALDI_ERR << "Bad value for --silence-phones option: "
              << opts.silence_phones_str;
   }
 
 const CuMatrixBase<BaseFloat> &output = computer->GetOutput(output_name);
 if (output.NumCols() != supervision.NumCols())
    KALDI_ERR << "Nnet versus example output dimension (num-classes) "
              << "mismatch for '" << output_name << "': " << output.NumCols()
              << " (nnet) vs. " << supervision.NumCols() << " (egs)\n";
 
 switch (objective_type) {
	case kMpe: {
  if (opts.criterion == "mmi" && opts.boost != 0.0) {
    BaseFloat max_silence_error = 0.0;
    LatticeBoost(tmodel, num_ali, silence_phones,opts.boost, max_silence_error, &clat);
  }
  
  int32 num_frames = num_ali.size();
  stats->tot_t += num_frames;
  stats->tot_t_weighted += num_frames * eg_weight;
  std::vector<CuMatrix<BaseFloat> > forward_data_; 
  const VectorBase<BaseFloat> &priors = am_nnet.Priors();
  const CuMatrixBase<BaseFloat> &posteriors = output;
  KALDI_ASSERT(posteriors.NumRows() == num_frames);
  int32 num_pdfs = posteriors.NumCols();
  KALDI_ASSERT(num_pdfs == priors.Dim());
  
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  
  // We need to look up the posteriors of some pdf-ids in the matrix
  // "posteriors".  Rather than looking them all up using operator (), which is
  // very slow because each lookup involves a separate CUDA call with
  // communication over PciExpress, we look them up all at once using
  // CuMatrix::Lookup().
  // Note: regardless of the criterion, we evaluate the likelihoods in
  // the numerator alignment.  Even though they may be irrelevant to
  // the optimization, they will affect the value of the objective function.
  
  std::vector<Int32Pair> requested_indexes;
  BaseFloat wiggle_room = 1.3; // value not critical.. it's just 'reserve'
  requested_indexes.reserve(num_frames + wiggle_room * clat.NumStates());

  if (opts.criterion == "mmi") { // need numerator probabilities...
    for (int32 t = 0; t < num_frames; t++) {
      int32 tid = num_ali[t], pdf_id = tmodel.TransitionIdToPdf(tid);
      KALDI_ASSERT(pdf_id >= 0 && pdf_id < num_pdfs);
      requested_indexes.push_back(MakePair(t, pdf_id));
	}
  }
  std::vector<int32> state_times;
 int32 T = LatticeStateTimes(clat, &state_times);
 KALDI_ASSERT(T == num_frames);
 StateId num_states = clat.NumStates();
 for (StateId s = 0; s < num_states; s++) {
   StateId t = state_times[s];
   for (fst::ArcIterator<Lattice> aiter(clat, s); !aiter.Done(); aiter.Next()) {
	 const Arc &arc = aiter.Value();
	 if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
	   int32 tid = arc.ilabel, pdf_id = tmodel.TransitionIdToPdf(tid);
	   requested_indexes.push_back(MakePair(t, pdf_id));
	 }
   }
 }

 std::vector<BaseFloat> answers;
 CuArray<Int32Pair> cu_requested_indexes(requested_indexes);
 answers.resize(requested_indexes.size());
 posteriors.Lookup(cu_requested_indexes, &(answers[0]));

 int32 num_floored = 0;

 BaseFloat floor_val = 1.0e-20; // floor for posteriors.
 size_t index;
 // Replace "answers" with the vector of scaled log-probs.	If this step takes
   // too much time, we can look at other ways to do it, using the CUDA card.
   for (index = 0; index < answers.size(); index++) {
	 BaseFloat post  = answers[index];
	 if (post < floor_val) {
	   post = floor_val;
	   num_floored++;
	 }
	 int32 pdf_id = requested_indexes[index].second;
	 BaseFloat pseudo_loglike = Log(post / priors(pdf_id)) * opts.acoustic_scale;
	 answers[index] = pseudo_loglike;
   }
   if (num_floored > 0) {
	 KALDI_WARN << "Floored " << num_floored << " probabilities from nnet.";
   }
    
   
   index = 0;
   
   if (opts.criterion == "mmi") {
	 double tot_num_like = 0.0;
	 for (; index < num_ali.size(); index++)
	   tot_num_like += answers[index];
	 stats->tot_num_objf += eg_weight * tot_num_like;
   }
   
   // Now put the (scaled) acoustic log-likelihoods in the lattice.
	for (StateId s = 0; s < num_states; s++) {
	  for (fst::MutableArcIterator<Lattice> aiter(&clat, s);
		   !aiter.Done(); aiter.Next()) {
		Arc arc = aiter.Value();
		if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
		  arc.weight.SetValue2(-answers[index]);
		  index++;
		  aiter.SetValue(arc);
		}
	  }
	  LatticeWeight final = clat.Final(s);
	  if (final != LatticeWeight::Zero()) {
		final.SetValue2(0.0); // make sure no acoustic term in final-prob.
		clat.SetFinal(s, final);
	  }
	}

	KALDI_ASSERT(index == answers.size());

	// Get the MPE or MMI posteriors.
	Posterior post;
	stats->tot_den_objf += eg_weight * GetDiscriminativePosteriors(opts,am_nnet,tmodel,supervision,clat,num_ali,&post);
	ScalePosterior(eg_weight, &post);
	double tot_num_post = 0.0, tot_den_post = 0.0;
	  std::vector<MatrixElement<BaseFloat> > sv_labels;
	  sv_labels.reserve(answers.size());
	  for (int32 t = 0; t < post.size(); t++) {
		for (int32 i = 0; i < post[t].size(); i++) {
		  int32 pdf_id = post[t][i].first;
		  BaseFloat weight = post[t][i].second;
		  //KALDI_LOG<<"t="<<t<<" pdf_id="<<pdf_id<<" weight="<<weight;
		  if (weight > 0.0) { tot_num_post += weight; }
		  else { tot_den_post -= weight; }
		  MatrixElement<BaseFloat> elem = {t, pdf_id, weight};
		  sv_labels.push_back(elem);
		}
	  }
	  stats->tot_num_count += tot_num_post;
	    CuMatrix<BaseFloat> output_deriv;
	    output_deriv.Resize(output.NumRows(), output.NumCols()); // zeroes it.
	  {  
	  // We don't actually need tot_objf and tot_weight; we have already
		// computed the objective function.
		BaseFloat tot_objf, tot_weight;
		output_deriv.CompObjfAndDeriv(sv_labels, CuMatrix<BaseFloat>(output), &tot_objf, &tot_weight);
		KALDI_LOG<<"diff:"<<output_deriv;
		if (supply_deriv)
              computer->AcceptOutputDeriv(output_name, &output_deriv);
		break;
		// Now output_derivwill contan the derivative at the output.
	  // Our work here is done..
     
}
}
   default:
      KALDI_ERR << "Objective function type " << objective_type
	                  << " not handled.";
}
}





double GetDiscriminativePosteriors(const NnetTrainerOptions &opts,
                                                const AmNnetSimple &am_nnet,
                                                const TransitionModel &tmodel,
                                                const GeneralMatrix &supervision, 
                                                const Lattice &clat,
                                                const std::vector<int32> &num_ali,
                                                Posterior *post) {
	std::vector<int32> silence_phones;
    if (!SplitStringToIntegers(opts.silence_phones_str, ":", false,
                             &silence_phones)) {
    KALDI_ERR << "Bad value for --silence-phones option: "
              << opts.silence_phones_str;
    } 
  if (opts.criterion == "mpfe" || opts.criterion == "smbr") {
    Posterior tid_post;
    double ans;
    ans = LatticeForwardBackwardMpeVariants(tmodel, silence_phones, clat,
                                            num_ali, opts.criterion,
                                            opts.one_silence_class,
                                            &tid_post);
	KALDI_LOG<<"silence_phones:"<<opts.silence_phones_str;
	KALDI_LOG<<"opts.one_silence_class:"<<opts.one_silence_class;
	//for(int32 i=0; i<num_ali.size(); i++)
	//	KALDI_LOG<<"i="<<i<<" num_ali:"<<num_ali[i];
	//WriteLattice(std::cerr, false, clat);
	//KALDI_LOG<<"tid_post.size="<<tid_post.size();
    ConvertPosteriorToPdfs(tmodel, tid_post, post);
    return ans; // returns the objective function.
  } else {
    KALDI_ASSERT(opts.criterion == "mmi");
    bool convert_to_pdfs = true, cancel = true;
    // we'll return the denominator-lattice forward backward likelihood,
    // which is one term in the objective function.

     return LatticeForwardBackwardMmi(tmodel, clat, num_ali,
                                         opts.drop_frames, convert_to_pdfs,
                                         cancel, post);
      }
  }
void NnetDiscriminativeStats::Add(const NnetDiscriminativeStats &other) {
  tot_t += other.tot_t;
  tot_t_weighted += other.tot_t_weighted;
  tot_num_count += other.tot_num_count;
  tot_num_objf += other.tot_num_objf;
  tot_den_objf += other.tot_den_objf;
}
void NnetDiscriminativeStats::Print(std::string criterion) {
  KALDI_ASSERT(criterion == "mmi" || criterion == "smbr" ||
               criterion == "mpfe");

  double avg_post_per_frame = tot_num_count / tot_t_weighted;
  KALDI_LOG << "Number of frames is " << tot_t
            << " (weighted: " << tot_t_weighted
            << "), average (num or den) posterior per frame is "
            << avg_post_per_frame;
  
  if (criterion == "mmi") {
    double num_objf = tot_num_objf / tot_t_weighted,
        den_objf = tot_den_objf / tot_t_weighted,
        objf = num_objf - den_objf;
    KALDI_LOG << "MMI objective function is " << num_objf << " - "
 << den_objf << " = " << objf << " per frame, over "
              << tot_t_weighted << " frames.";
  } else if (criterion == "mpfe") {
    double objf = tot_den_objf / tot_t_weighted; // this contains the actual
                                                 // summed objf
    KALDI_LOG << "MPFE objective function is " << objf
              << " per frame, over " << tot_t_weighted << " frames.";
  } else {
    double objf = tot_den_objf / tot_t_weighted; // this contains the actual
                                                 // summed objf
    KALDI_LOG << "SMBR objective function is " << objf
              << " per frame, over " << tot_t_weighted << " frames.";
  }

}
  

} // namespace nnet3
} // namespace kaldi
