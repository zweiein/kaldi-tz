// nnet/nnet-activation.h

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_ACTIVATION_H_
#define KALDI_NNET_NNET_ACTIVATION_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"
#include "util/text-utils.h"

namespace kaldi {
namespace nnet1 {

class Softmax : public Component {
 public:
  Softmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out), temperature_(1.0)
  { }
  ~Softmax()
  { }

  Component* Copy() const { return new Softmax(*this); }
  ComponentType GetType() const { return kSoftmax; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    const CuMatrixBase<BaseFloat> *in_tmp = &in;
    CuMatrix<BaseFloat> tmp;
    if (temperature_ != 1.0) {
      tmp.Resize(in.NumRows(), in.NumCols());
      tmp.CopyFromMat(in);
      tmp.Scale(1.0 / temperature_);
      in_tmp = &tmp;
    }
    // y = e^x_j/sum_j(e^x_j)
    out->ApplySoftMaxPerRow(*in_tmp);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // simply copy the error derivative
    // (ie. assume crossentropy error function, 
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    in_diff->CopyFromMat(out_diff);
  }

  void SetTemperature(BaseFloat t) {
    KALDI_ASSERT(t > 0.0);
    temperature_ = t;
  }


private:
  BaseFloat temperature_;
};



class BlockSoftmax : public Component {
 public:
  BlockSoftmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~BlockSoftmax()
  { }

  Component* Copy() const { return new BlockSoftmax(*this); }
  ComponentType GetType() const { return kBlockSoftmax; }
  
  void InitData(std::istream &is) {
    // parse config
    std::string token,
      dims_str;
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<BlockDims>") is >> dims_str;
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (BlockDims)";
      is >> std::ws; // eat-up whitespace
    }
    // parse dims,
    if (!kaldi::SplitStringToIntegers(dims_str, ",:", false, &block_dims))
      KALDI_ERR << "Invalid block-dims " << dims_str;
    // sanity check
    int32 sum = 0;
    for (int32 i=0; i<block_dims.size(); i++) {
      sum += block_dims[i];
    }
    KALDI_ASSERT(sum == OutputDim()); 
  }

  void ReadData(std::istream &is, bool binary) {
    ReadIntegerVector(is, binary, &block_dims);
    block_offset.resize(block_dims.size()+1, 0);
    for (int32 i = 0; i < block_dims.size(); i++) {
      block_offset[i+1] = block_offset[i] + block_dims[i];
    }
    // check
    KALDI_ASSERT(OutputDim() == block_offset[block_offset.size()-1]);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteIntegerVector(os, binary, block_dims);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // perform softmax per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> in_bl = in.ColRange(block_offset[bl], block_dims[bl]);
      CuSubMatrix<BaseFloat> out_bl = out->ColRange(block_offset[bl], block_dims[bl]);
      // y = e^x_j/sum_j(e^x_j)
      out_bl.ApplySoftMaxPerRow(in_bl);
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // copy the error derivative:
    // (assuming we already got softmax-cross-entropy derivative in out_diff)
    in_diff->CopyFromMat(out_diff);
    
    // zero-out line-in-block, where sum different from zero,
    // process per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> diff_bl = in_diff->ColRange(block_offset[bl], block_dims[bl]);
      CuVector<BaseFloat> row_sum(diff_bl.NumRows());
      row_sum.AddColSumMat(1.0, diff_bl, 0.0); // 0:keep, 1:zero-out
      // we'll scale rows by 0/1 masks
      CuVector<BaseFloat> row_diff_mask(row_sum);
      row_diff_mask.Scale(-1.0); // 0:keep, -1:zero-out
      row_diff_mask.Add(1.0); // 1:keep, 0:zero-out
      // here we should have only 0 and 1
      diff_bl.MulRowsVec(row_diff_mask);
    }
  }

  std::string Info() const {
    return "\n  softmax-dims " + ToString(block_dims);
  }

  std::vector<int32> block_dims;
  std::vector<int32> block_offset;
};




class Sigmoid : public Component {
 public:
  Sigmoid(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Sigmoid()
  { }

  Component* Copy() const { return new Sigmoid(*this); }
  ComponentType GetType() const { return kSigmoid; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = 1/(1+e^-x)
    out->Sigmoid(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->DiffSigmoid(out, out_diff);
  }
};



class Tanh : public Component {
 public:
  Tanh(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Tanh()
  { }

  Component* Copy() const { return new Tanh(*this); }
  ComponentType GetType() const { return kTanh; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = (e^x - e^(-x)) / (e^x + e^(-x))
    out->Tanh(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = (1 - y^2)ex
    in_diff->DiffTanh(out, out_diff);
  }
};



class Dropout : public Component {
 public:
  Dropout(int32 dim_in, int32 dim_out):
      Component(dim_in, dim_out), dropout_retention_(0.5)
  { }
  ~Dropout()
  { }

  Component* Copy() const { return new Dropout(*this); }
  ComponentType GetType() const { return kDropout; }

  void InitData(std::istream &is) {
    is >> std::ws; // eat-up whitespace
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<DropoutRetention>") ReadBasicType(is, false, &dropout_retention_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (DropoutRetention)";
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void ReadData(std::istream &is, bool binary) {
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<DropoutRetention>");
      ReadBasicType(is, binary, &dropout_retention_);
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<DropoutRetention>");
    WriteBasicType(os, binary, dropout_retention_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    out->CopyFromMat(in);
    // switch off 50% of the inputs...
    dropout_mask_.Resize(out->NumRows(),out->NumCols());
    dropout_mask_.Set(dropout_retention_);
    rand_.BinarizeProbs(dropout_mask_,&dropout_mask_);
    out->MulElements(dropout_mask_);
    // rescale to keep same dynamic range as w/o dropout
    out->Scale(1.0/dropout_retention_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    // use same mask on the error derivatives...
    in_diff->MulElements(dropout_mask_);
    // enlarge output to fit dynamic range w/o dropout
    in_diff->Scale(1.0/dropout_retention_);
  }
  
  BaseFloat GetDropoutRetention() {
    return dropout_retention_;
  }

  void SetDropoutRetention(BaseFloat dr) {
    dropout_retention_ = dr;
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

 private:
  CuRand<BaseFloat> rand_;
  CuMatrix<BaseFloat> dropout_mask_;
  BaseFloat dropout_retention_;
};



class Rectifier : public Component {
 public:
  Rectifier(int32 dim_in, int32 dim_out)
    : Component(dim_in, dim_out)
  { }
  ~Rectifier()
  { }

  Component* Copy() const { return new Rectifier(*this); }
  ComponentType GetType() const { return kRectifier; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    out->CopyFromMat(in);
    out->ApplyFloor(0.0);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                  const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(in);
    in_diff->ApplyHeaviside();
    in_diff->MulElements(out_diff);
  }
};



class LeakyRectifier : public Component {
 public:
  LeakyRectifier(int32 dim_in, int32 dim_out)
    : Component(dim_in, dim_out)
  { }
  ~LeakyRectifier()
  { }

  Component* Copy() const { return new LeakyRectifier(*this); }
  ComponentType GetType() const { return kLeakyRectifier; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    out->CopyFromMat(in);
    out->ApplyLeakyFloor(0.0, 0.01);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                  const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(in);
    in_diff->ApplyLeakyHeaviside(0.01);
    in_diff->MulElements(out_diff);

  }
};



class TemporalRectifier : public Component {
 public:
  TemporalRectifier(int32 dim_in, int32 dim_out)
    : Component(dim_in, dim_out), nstream_(0), momentum_(0.9), floor_coef(0.0), training_stat_(false)
  { }
  ~TemporalRectifier()
  { }

  Component* Copy() const { return new TemporalRectifier(*this); }
  ComponentType GetType() const { return kTemporalRectifier; }

  void ResetRectifierStreams(const std::vector<int32> &stream_reset_flag, int32 hid_dim) {
      // allocate prev_rectifier_state_ if not done yet,
      if (nstream_ == 0) {
        // Karel: we just got number of streams! (before the 1st batch comes)
        nstream_ = stream_reset_flag.size(); 
        prev_rectifier_state_.Resize(nstream_, hid_dim, kSetZero);
        KALDI_LOG << "Running training with " << nstream_ << " streams.";
      }
      // reset flag: 1 - reset stream network state
      KALDI_ASSERT(prev_rectifier_state_.NumRows() == stream_reset_flag.size());
      for (int s = 0; s < stream_reset_flag.size(); s++) {
          if (stream_reset_flag[s] == 1) {
              prev_rectifier_state_.Row(s).SetZero();
          }
      }
  }


  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        int32 hid_dim = in.NumCols();

        if (training_stat_ == false){
            nstream_ = 1;
            prev_rectifier_state_.Resize(nstream_, hid_dim, kSetZero);
            KALDI_LOG << "Running nnet-forward with per-utterance LSTM-state reset";
        }

        KALDI_ASSERT(nstream_ > 0);

        KALDI_ASSERT(in.NumRows() % nstream_ == 0);
        int32 T = in.NumRows() / nstream_;
        int32 S = nstream_;

        // 0:forward pass history, [1, T]:current sequence, T+1:dummy
        rectifier_states_.Resize((T+2)*S, hid_dim, kSetZero);  
        CuMatrix<BaseFloat> out_buf;
        out_buf.Resize(in.NumRows(), in.NumCols());
        out_buf.CopyFromMat(in);
        CuSubMatrix<BaseFloat> state_(rectifier_states_.RowRange(0*S,S));
        state_.CopyFromMat(prev_rectifier_state_);

        for (int t = 1; t <= T; t++) {
            CuSubMatrix<BaseFloat> out_t(out_buf.RowRange((t-1)*S,S));
            CuSubMatrix<BaseFloat> stata_t(rectifier_states_.RowRange((t-1)*S,S));
            CuSubMatrix<BaseFloat> stata_t_1(rectifier_states_.RowRange(t*S,S));
            out_t.ApplyTemporalFloor(stata_t, floor_coef);
            
            //more delicate method to calculate future threshhold waits to be analysed
            stata_t_1.AddMat(momentum_, stata_t, kNoTrans); 
            stata_t_1.AddMat(1.0-momentum_, out_t, kNoTrans); 
            
        }

        prev_rectifier_state_.CopyFromMat(rectifier_states_.RowRange(T*S,S));

        out->CopyFromMat(out_buf);
    
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
                  
        int32 T = in.NumRows() / nstream_;
        int32 S = nstream_;
        CuSubMatrix<BaseFloat> stata_t(rectifier_states_.RowRange(S, T*S)); 
        in_diff->CopyFromMat(in);
        in_diff->ApplyTemporalHeaviside( stata_t, floor_coef);
        in_diff->MulElements(out_diff);
  }

  bool GetRectifierTrainingStat(){
      return training_stat_;
  }

  void SetRectifierTrainingStat(bool training_stat){
      training_stat_ = training_stat;
  }

  void SetRectifierMomentum(BaseFloat momentum){
      momentum_ = momentum;

  }


 private:
    int32 nstream_;
    BaseFloat momentum_;
    BaseFloat floor_coef;
    bool training_stat_;

    CuMatrix<BaseFloat> prev_rectifier_state_;
    CuMatrix<BaseFloat> rectifier_states_;
 
};




} // namespace nnet1
} // namespace kaldi

#endif

