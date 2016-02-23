// nnet3bin/nnet3-train-soft.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-training.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3 neural network parameters with backprop and stochastic\n"
        "gradient descent.  Minibatches are to be created by nnet3-merge-egs in\n"
        "the input pipeline.  This training program is single-threaded (best to\n"
        "use it with a GPU); see nnet3-train-parallel for multi-threaded training\n"
        "that is better suited to CPUs.\n"
        "\n"
        "Usage:  nnet3-train-soft [options] <raw-model-in> <training-examples-in> <soft-targets> <raw-model-out>\n"
        "\n"
        "e.g.:\n"
        "nnet3-train-soft 1.raw 'ark:nnet3-merge-egs 1.egs ark:-|' 1.ark 2.raw\n";

    bool binary_write = true;
    std::string use_gpu = "yes";
    BaseFloat temperature = 1.0;
    NnetTrainerOptions train_config;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("temperature", &temperature, "temperature for soft_targets in knowledge distilling");

    train_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        soft_targets_rspecifier = po.GetArg(3),
        nnet_wxfilename = po.GetArg(4);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    NnetTrainerSoft trainer(train_config, &nnet);

    SequentialNnetExampleReader example_reader(examples_rspecifier);    
    SequentialBaseFloatMatrixReader soft_reader(soft_targets_rspecifier);

    for (; !example_reader.Done() && !soft_reader.Done(); example_reader.Next(), soft_reader.Next()) {
        if (example_reader.Key() != soft_reader.Key()) {
            KALDI_WARN << example_reader.Key() << ", missing soft targets";
            continue;
        }
        // knowledge distilling for soft targets
        CuMatrix<BaseFloat> soft_targets_cu= CuMatrix<BaseFloat>(soft_reader.Value());
        soft_targets_cu.Scale( 1.0 / temperature );
        CuMatrix<BaseFloat> soft_targets_tmp;
        soft_targets_tmp.ApplySoftMaxPerRow(soft_targets_cu);
        const Matrix<BaseFloat>& soft_targets = Matrix<BaseFloat>(soft_targets_tmp);
        
        trainer.Train(example_reader.Value(), GeneralMatrix(soft_targets));
    }

    bool ok = trainer.PrintTotalStats();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    KALDI_LOG << "Wrote model to " << nnet_wxfilename;
    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


