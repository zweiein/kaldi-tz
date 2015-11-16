// nnet/nnet-affine-transform-extra.h

// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_AFFINE_TRANSFORM_EXTRA_H_
#define KALDI_NNET_NNET_AFFINE_TRANSFORM_EXTRA_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class AffineTransformExtra : public UpdatableComponent {
 public:
  AffineTransformExtra(int32 dim_in, int32 dim_out) 
    : UpdatableComponent(dim_in, dim_out), 
      linearity_(dim_out, dim_in), bias_(dim_out),
      linearity_corr_(dim_out, dim_in), bias_corr_(dim_out),
      learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0), max_norm_(0.0)
  { }
  ~AffineTransformExtra()
  { }

  Component* Copy() const { return new AffineTransformExtra(*this); }
  ComponentType GetType() const { return kAffineTransformExtra; }
  
  void InitData(std::istream &is) {
    // define options
    float bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1;
    float learn_rate_coef = 1.0, bias_learn_rate_coef = 1.0;
    float max_norm = 0.0;
	float input_dim_extra;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<BiasMean>")    ReadBasicType(is, false, &bias_mean);
      else if (token == "<BiasRange>")   ReadBasicType(is, false, &bias_range);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef);
      else if (token == "<MaxNorm>") ReadBasicType(is, false, &max_norm);
      else if (token == "<InputDimExtra>") ReadBasicType(is, false, &input_dim_extra);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|LearnRateCoef|BiasLearnRateCoef)";
      is >> std::ws; // eat-up whitespace
    }

    //
    // initialize
    //
    Matrix<BaseFloat> mat(output_dim_, input_dim_);
    for (int32 r=0; r<output_dim_; r++) {
      for (int32 c=0; c<input_dim_; c++) {
        mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
      }
    }
    linearity_ = mat;

    Matrix<BaseFloat> mat_extra(output_dim_, input_dim_extra);
    for (int32 r_extra=0; r_extra<output_dim_; r_extra++) {
      for (int32 c_extra=0; c_extra<input_dim_extra; c_extra++) {
        mat_extra(r_extra,c_extra) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
      }
    }
    linearity_extra_ = mat_extra;
    linearity_extra_corr_.Resize(output_dim_, input_dim_extra);
    //
    Vector<BaseFloat> vec(output_dim_);
    for (int32 i=0; i<output_dim_; i++) {
      // +/- 1/2*bias_range from bias_mean:
      vec(i) = bias_mean + (RandUniform() - 0.5) * bias_range; 
    }
    bias_ = vec;
    //
    learn_rate_coef_ = learn_rate_coef;
    bias_learn_rate_coef_ = bias_learn_rate_coef;
    max_norm_ = max_norm;
    input_dim_extra_ = input_dim_extra;
    //
  }

  void ReadData(std::istream &is, bool binary) {
    // optional learning-rate coefs
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<InputDimExtra>");
      ReadBasicType(is, binary, &input_dim_extra_);
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
      ExpectToken(is, binary, "<BiasLearnRateCoef>");
      ReadBasicType(is, binary, &bias_learn_rate_coef_);
    }
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<MaxNorm>");
      ReadBasicType(is, binary, &max_norm_);
    }
    // weights
    linearity_.Read(is, binary);
    linearity_extra_.Read(is, binary);
    bias_.Read(is, binary);
    linearity_extra_corr_.Resize(output_dim_, input_dim_extra_);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_);
    KALDI_ASSERT(linearity_extra_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_extra_.NumCols() == input_dim_extra_);
    KALDI_ASSERT(bias_.Dim() == output_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<InputDimExtra>");
    WriteBasicType(os, binary, input_dim_extra_);
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    WriteToken(os, binary, "<MaxNorm>");
    WriteBasicType(os, binary, max_norm_);
    // weights
    linearity_.Write(os, binary);
    linearity_extra_.Write(os, binary);
    bias_.Write(os, binary);
  }

  int32 NumParams() const { return linearity_.NumRows()*linearity_.NumCols() + linearity_extra_.NumRows()*linearity_extra_.NumCols() + bias_.Dim(); }
  
  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());
    int32 linearity_num_elem = linearity_.NumRows() * linearity_.NumCols(); 
    int32 linearity_extra_num_elem = linearity_extra_.NumRows() * linearity_extra_.NumCols(); 
    wei_copy->Range(0,linearity_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(linearity_));
    wei_copy->Range(linearity_num_elem,linearity_extra_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(linearity_extra_));
    wei_copy->Range(linearity_num_elem+linearity_extra_num_elem, bias_.Dim()).CopyFromVec(Vector<BaseFloat>(bias_));
  }
  
  std::string Info() const {
    return std::string("\n  linearity") + MomentStatistics(linearity_) +
		   "\n  linearity_extra" + MomentStatistics(linearity_extra_) +
           "\n  bias" + MomentStatistics(bias_);
  }
  std::string InfoGradient() const {
    return std::string("\n  linearity_grad") + MomentStatistics(linearity_corr_) + 
		   ", linearity_extra_grad" + MomentStatistics(linearity_extra_corr_) + 
           ", lr-coef " + ToString(learn_rate_coef_) +
           ", max-norm " + ToString(max_norm_) +
           "\n  bias_grad" + MomentStatistics(bias_corr_) + 
           ", lr-coef " + ToString(bias_learn_rate_coef_);
           
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // precopy bias
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);
  }


  void PropagateFncExtra(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &in_extra, CuMatrix<BaseFloat> *out) {
    // precopy bias
    KALDI_ASSERT(in_extra.NumCols()==linearity_extra_.NumCols());
	KALDI_ASSERT(in.NumCols() == linearity_.NumCols());
	KALDI_ASSERT(in.NumRows() == in_extra.NumRows());
    out->Resize(in.NumRows(), linearity_.NumRows());
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);
	out->AddMatMat(1.0, in_extra, kNoTrans, linearity_extra_, kTrans, 1.0);
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    const BaseFloat l1 = opts_.l1_penalty;
    // we will also need the number of frames in the mini-batch
    const int32 num_frames = input.NumRows();
    // compute gradient (incl. momentum)
    linearity_corr_.AddMatMat(1.0, diff, kTrans, input, kNoTrans, mmt);
    bias_corr_.AddRowSumMat(1.0, diff, mmt);
    // l2 regularization
    if (l2 != 0.0) {
      linearity_.AddMat(-lr*l2*num_frames, linearity_);
    }
    // l1 regularization
    if (l1 != 0.0) {
      cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames, lr);
    }
    // update
    linearity_.AddMat(-lr, linearity_corr_);
    bias_.AddVec(-lr_bias, bias_corr_);
    // max-norm
    if (max_norm_ > 0.0) {
      CuMatrix<BaseFloat> lin_sqr(linearity_);
      lin_sqr.MulElements(linearity_);
      CuVector<BaseFloat> l2(OutputDim());
      l2.AddColSumMat(1.0, lin_sqr, 0.0);
      l2.ApplyPow(0.5); // we have per-neuron L2 norms
      CuVector<BaseFloat> scl(l2);
      scl.Scale(1.0/max_norm_);
      scl.ApplyFloor(1.0);
      scl.InvertElements();
      linearity_.MulRowsVec(scl); // shink to sphere!
    }
  }

  void UpdateExtra(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &input_extra, const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    const BaseFloat l1 = opts_.l1_penalty;
    // we will also need the number of frames in the mini-batch
    const int32 num_frames = input.NumRows();
    const int32 num_frames_extra= input_extra.NumRows();
	//linearity_extra_corr_.Resize(output_dim_, input_dim_extra_);
    // compute gradient (incl. momentum)
    linearity_corr_.AddMatMat(1.0, diff, kTrans, input, kNoTrans, mmt);
    linearity_extra_corr_.AddMatMat(1.0, diff, kTrans, input_extra, kNoTrans, mmt);
    bias_corr_.AddRowSumMat(1.0, diff, mmt);
    // l2 regularization
    if (l2 != 0.0) {
      linearity_.AddMat(-lr*l2*num_frames, linearity_);
	  linearity_extra_.AddMat(-lr*l2*num_frames_extra, linearity_extra_);
    }
    // l1 regularization
    if (l1 != 0.0) {
      cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames, lr);
	  cu::RegularizeL1(&linearity_extra_, &linearity_extra_corr_, lr*l1*num_frames_extra, lr);
    }
    // update
    linearity_.AddMat(-lr, linearity_corr_);
	linearity_extra_.AddMat(-lr, linearity_extra_corr_);
    bias_.AddVec(-lr_bias, bias_corr_);
    // max-norm
    if (max_norm_ > 0.0) {
      CuMatrix<BaseFloat> lin_sqr(linearity_);
      lin_sqr.MulElements(linearity_);
      CuVector<BaseFloat> l2(OutputDim());
      l2.AddColSumMat(1.0, lin_sqr, 0.0);
      l2.ApplyPow(0.5); // we have per-neuron L2 norms
      CuVector<BaseFloat> scl(l2);
      scl.Scale(1.0/max_norm_);
      scl.ApplyFloor(1.0);
      scl.InvertElements();
      linearity_.MulRowsVec(scl); // shink to sphere!

      CuMatrix<BaseFloat> lin_sqr_extra(linearity_extra_);
      lin_sqr_extra.MulElements(linearity_extra_);
      CuVector<BaseFloat> l2_extra(OutputDim());
      l2_extra.AddColSumMat(1.0, lin_sqr_extra, 0.0);
      l2_extra.ApplyPow(0.5); // we have per-neuron L2 norms
      CuVector<BaseFloat> scl_extra(l2_extra);
      scl_extra.Scale(1.0/max_norm_);
      scl_extra.ApplyFloor(1.0);
      scl_extra.InvertElements();
      linearity_extra_.MulRowsVec(scl_extra); // shink to sphere!
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // multiply error derivative by weights
    in_diff->AddMatMat(1.0, out_diff, kNoTrans, linearity_, kNoTrans, 0.0);
  }


  /// Accessors to the component parameters
  const CuVectorBase<BaseFloat>& GetBias() const {
    return bias_;
  }

  void SetBias(const CuVectorBase<BaseFloat>& bias) {
    KALDI_ASSERT(bias.Dim() == bias_.Dim());
    bias_.CopyFromVec(bias);
  }

  const CuMatrixBase<BaseFloat>& GetLinearity() const {
    return linearity_;
  }

    const CuMatrixBase<BaseFloat>& GetLinearityExtra() const {
    return linearity_extra_;
  }

  void SetLinearity(const CuMatrixBase<BaseFloat>& linearity) {
    KALDI_ASSERT(linearity.NumRows() == linearity_.NumRows());
    KALDI_ASSERT(linearity.NumCols() == linearity_.NumCols());
    linearity_.CopyFromMat(linearity);
  }

  void SetLinearityExtra(const CuMatrixBase<BaseFloat>& linearity_extra) {
    KALDI_ASSERT(linearity_extra.NumRows() == linearity_extra_.NumRows());
    KALDI_ASSERT(linearity_extra.NumCols() == linearity_extra_.NumCols());
    linearity_extra_.CopyFromMat(linearity_extra);
  }


  const CuVectorBase<BaseFloat>& GetBiasCorr() const {
    return bias_corr_;
  }

  const CuMatrixBase<BaseFloat>& GetLinearityCorr() const {
    return linearity_corr_;
  }

  const CuMatrixBase<BaseFloat>& GetLinearityExtraCorr() const {
    return linearity_extra_corr_;
  }


 private:
  CuMatrix<BaseFloat> linearity_;
  CuMatrix<BaseFloat> linearity_extra_;
  CuVector<BaseFloat> bias_;

  int32 input_dim_extra_;
  
  CuMatrix<BaseFloat> linearity_corr_;
  CuMatrix<BaseFloat> linearity_extra_corr_;
  CuVector<BaseFloat> bias_corr_;

  BaseFloat learn_rate_coef_;
  BaseFloat bias_learn_rate_coef_;
  BaseFloat max_norm_;
};

} // namespace nnet1
} // namespace kaldi

#endif
