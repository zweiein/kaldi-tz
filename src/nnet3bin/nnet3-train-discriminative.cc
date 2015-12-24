// nnet3bin/nnet3-train.cc

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
#include "nnet3/nnet-training-discriminative.h"
#include "nnet3/nnet-example.h"
#include "lat/lattice-functions.h"

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
        "Usage:  nnet3-train-discriminative [options] <raw-model-in> <raw-model-mdl> <clat > <ali> <training-examples-in> <raw-model-out>\n"
        "\n"
        "e.g.:\n"
        "nnet3-train-discriminative 1.raw 0.mdl 'ark:nnet3-merge-egs 1.egs ark:-|' 2.raw\n";

    bool binary_write = true;
    std::string use_gpu = "yes";
	//NnetDiscriminativeUpdateOptions update_opts;
  //  NnetTrainerOptions train_config;
    NnetTrainerOptions opts;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }
   
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        mdl_rxfilename = po.GetArg(2),
		clat_rspecifier = po.GetArg(3),
		ref_ali_rspecifier = po.GetArg(4),
        examples_rspecifier = po.GetArg(5),
        nnet_wxfilename = po.GetArg(6);
    
	Nnet nnet;
	ReadKaldiObject(nnet_rxfilename, &nnet);

          
      TransitionModel trans_model;
      AmNnetSimple am_nnet;
      {
        bool binary_read;
        Input ki(mdl_rxfilename, &binary_read);
        trans_model.Read(ki.Stream(), binary_read);
        am_nnet.Read(ki.Stream(), binary_read);
      }
 //   KALDI_LOG << "trans_model.NumTransitionIds()"<< trans_model.NumTransitionIds();
//	KALDI_LOG << "am_nnet.NumPdfs()" <<am_nnet.NumPdfs();
    NnetDiscriminativeStats stats;

    bool ok;
    {
       NnetTrainer trainer(opts, am_nnet,trans_model, &nnet, &stats);
    //  NnetTrainer trainer;
     //RandomAccessLatticeReader clat_reader(clat_rspecifier);
     SequentialLatticeReader clat_reader(clat_rspecifier);
	 RandomAccessInt32VectorReader ref_ali_reader(ref_ali_rspecifier);
	 SequentialNnetExampleReader example_reader(examples_rspecifier);
	  //Lattice* den_lat_tmp = NULL;
	  //&den_lat_tmp = clat_reader.Value();
      for (; !example_reader.Done(); example_reader.Next()) {
          std::string utt = example_reader.Key();
		  KALDI_LOG << "utt" <<utt;
          //Lattice clat;
		  if (clat_reader.Key() != example_reader.Key()) {
			    continue;
		  }
		  Lattice clat = clat_reader.Value();
		  std::vector<int32> ref_ali = ref_ali_reader.Value(utt);
          kaldi::uint64 props = clat.Properties(fst::kFstProperties, false);
         if (!(props & fst::kTopSorted)) {
              if (fst::TopSort(&clat) == false)
                 KALDI_ERR << "Cycles detected in lattice.";
          }
	//	  KALDI_LOG << "clat.Start()"<< clat.Start();
		  //KALDI_LOG << "ref_ali.size()"<<ref_ali.size();
		  //WriteLattice(std::cerr, false, clat);
          trainer.Train(&stats,opts,am_nnet,trans_model,example_reader.Value(), clat, ref_ali);
		  clat_reader.Next();
//
            stats.Print(opts.criterion);
     //     ok = trainer.PrintTotalStats();
            ok = 1;
	  }
      // need trainer's destructor to be called before we write model.
    }
     //stats.Print(opts.criterion);
        
      /*{
        Output ko(nnet_wxfilename, binary_write);
        trans_model.Write(ko.Stream(), binary_write);
        am_nnet.Write(ko.Stream(), binary_write);
      } */

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


