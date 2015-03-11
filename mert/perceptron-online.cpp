// $Id$
// vim:tabstop=2
/***********************************************************************
K-best Batch MIRA for Moses
Copyright (C) 2012, National Research Council Canada / Conseil national
de recherches du Canada
***********************************************************************/

/**
  * k-best Batch Mira, as described in:
  *
  * Colin Cherry and George Foster
  * Batch Tuning Strategies for Statistical Machine Translation
  * NAACL 2012
  *
  * Implemented by colin.cherry@nrc-cnrc.gc.ca
  *
  * To license implementations of any of the other tuners in that paper,
  * please get in touch with any member of NRC Canada's Portage project
  *
  * Input is a set of n-best lists, encoded as feature and score files.
  *
  * Output is a weight file that results from running MIRA on these
  * n-btest lists for J iterations. Will return the set that maximizes
  * training BLEU.
 **/

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

#include "util/exception.hh"

#include "BleuScorer.h"
#include "PerceptronDecoder.h"
#include "MiraFeatureVector.h"
#include "MiraWeightVector.h"

#include "Scorer.h"
#include "ScorerFactory.h"

#include "moses/IOWrapper.h"
#include "moses/Hypothesis.h"
#include "moses/Manager.h"
#include "moses/StaticData.h"
#include "moses/TypeDef.h"
#include "moses/Util.h"
#include "moses/Timer.h"
#include "moses/TranslationModel/PhraseDictionary.h"
#include "moses/FF/StatefulFeatureFunction.h"
#include "moses/FF/StatelessFeatureFunction.h"
#include "moses/TranslationTask.h"

#include <map>

using namespace std;
using namespace MosesTuning;
using namespace Moses;

namespace po = boost::program_options;

char**convert(const vector<std::string> & svec)
{
  char ** arr = new char*[svec.size()];
  for (size_t i = 0; i < svec.size(); i++) {
    arr[i] = new char[svec[i].size() + 1];
    std::strcpy(arr[i], svec[i].c_str());
  }
   return arr;
}

void SparseVec2ScoreComp(const SparseVector& svec, ScoreComponentCollection& score, size_t denseSize) {

  // dense features
  map<string, pair<size_t, size_t> > nameIndexMap = score.GetCoreNameIndexes();
  map<string, bool > tuneMap = score.GetTunableMap();
  std::valarray<float> denseScore(StaticData::Instance().GetAllWeights().getCoreFeatures());
  for(map<string, pair<size_t, size_t> >::const_iterator iter = nameIndexMap.begin();
      iter != nameIndexMap.end(); ++iter) {
    string name = iter->first;
    pair<size_t, size_t> pos = iter->second;

    if (! tuneMap.find(name)->second)
      continue;

    if (pos.second - pos.first == 1) {
      denseScore[pos.first] = svec.get(name);
      tuneMap[name] = true;
      //cerr << name << endl;
    } else if (pos.second > pos.first + 1) {
      size_t feature_ctr = 1;
      for(size_t idx = pos.first; idx < pos.second; ++idx,++feature_ctr) {
        stringstream namestr;
        namestr << name << "_" << feature_ctr;
        denseScore[idx] = svec.get(namestr.str());
        tuneMap[namestr.str()] = true;
        //cerr << namestr.str() << endl;
      }
    }
  }
  for(size_t i = 0; i < denseScore.size(); i++)
    score.Assign(i, denseScore[i]);

  // sparse features
  vector<size_t> feats = svec.feats();
  for(size_t i = 0; i < feats.size(); i++) {
    string name = SparseVector::decode(feats[i]);
    if (tuneMap.find(name) == tuneMap.end()) {
      score.Assign(name, svec.get(name));
    } else {
      cerr << "not sparse: " << name << endl;
    }
  }

}

void UpdateDecoderWeights(const MiraWeightVector& wv, size_t denseSize) {
  StaticData& staticData = StaticData::InstanceNonConst();
  const ScoreComponentCollection& weights = staticData.GetAllWeights();
  //cerr << weights << endl;
  SparseVector svec;
  wv.ToSparse(&svec, denseSize);
  //svec.write(cerr, " "); cerr << endl;
  ScoreComponentCollection update;
  SparseVec2ScoreComp(svec, update, denseSize);
  //cerr << update << endl;
  staticData.SetAllWeights(update);
  //cerr << staticData.GetAllWeights() << endl;
}

int main(int argc, char** argv)
{
  bool help;
  string denseInitFile;
  string sparseInitFile;
  string type = "maxvio";
  string sctype = "BLEU";
  string scconfig = "";
  vector<string> scoreFiles;
  vector<string> featureFiles;
  vector<string> referenceFiles; //for hg mira
  string hgDir;
  string hgDirRef;
  int seed;
  string outputFile;
  float c = 1.0;      // Step-size cap C
  float decay = 0.999; // Pseudo-corpus decay \gamma
  int n_iters = 1;    // Max epochs J
  bool streaming = false; // Stream all k-best lists?
  bool streaming_out = false; // Stream output after each sentence?
  bool no_shuffle = true; // Don't shuffle, even for in memory version
  bool model_bg = false; // Use model for background corpus
  bool verbose = false; // Verbose updates
  bool safe_hope = false; // Model score cannot have more than BLEU_RATIO times more influence than BLEU
  size_t hgPruning = 0; //prune hypergraphs to have this many edges per reference word

  string mosesargs;
  string inputFile;
  string decoderCmd = "";
  bool readRef = false;
  bool readHyp = false;
  bool noavg = false;
  int batch = 1;

  // Command-line processing follows pro.cpp
  po::options_description desc("Allowed options");
  desc.add_options()
  ("help,h", po::value(&help)->zero_tokens()->default_value(false), "Print this help message and exit")
  ("type,t", po::value<string>(&type), "maxvio")
  ("sctype", po::value<string>(&sctype), "the scorer type (default BLEU)")
  ("scconfig,c", po::value<string>(&scconfig), "configuration string passed to scorer")
  ("scfile,S", po::value<vector<string> >(&scoreFiles), "Scorer data files")
  ("ffile,F", po::value<vector<string> > (&featureFiles), "Feature data files")
  ("hgdir,H", po::value<string> (&hgDir), "Directory containing hypergraphs")
  ("hgdirref", po::value<string> (&hgDirRef), "Directory containing hypergraphs")
  ("reference,R", po::value<vector<string> > (&referenceFiles), "Reference files, only required for hypergraph mira")
  ("random-seed,r", po::value<int>(&seed), "Seed for random number generation")
  ("output-file,o", po::value<string>(&outputFile), "Output file")
  ("cparam,C", po::value<float>(&c), "MIRA C-parameter, lower for more regularization (default 0.01)")
  ("decay,D", po::value<float>(&decay), "BLEU background corpus decay rate (default 0.999)")
  ("iters,J", po::value<int>(&n_iters), "Number of MIRA iterations to run (default 60)")
  ("dense-init,d", po::value<string>(&denseInitFile), "Weight file for dense features. This should have 'name= value' on each line, or (legacy) should be the Moses mert 'init.opt' format.")
  ("sparse-init,s", po::value<string>(&sparseInitFile), "Weight file for sparse features")
  ("streaming", po::value(&streaming)->zero_tokens()->default_value(false), "Stream n-best lists to save memory, implies --no-shuffle")
  ("streaming-out", po::value(&streaming_out)->zero_tokens()->default_value(false), "Stream weights to stdout after each sentence")
  ("model-bg", po::value(&model_bg)->zero_tokens()->default_value(false), "Use model instead of hope for BLEU background")
  ("verbose", po::value(&verbose)->zero_tokens()->default_value(false), "Verbose updates")
  ("safe-hope", po::value(&safe_hope)->zero_tokens()->default_value(false), "Mode score's influence on hope decoding is limited")
  ("hg-prune", po::value<size_t>(&hgPruning), "Prune hypergraphs to have this many edges per reference word")
  ("mosesargs", po::value<string>(&mosesargs), "decoder args")
  ("read-ref", po::value(&readRef)->zero_tokens()->default_value(false), "read ref hypergraph into memory")
  ("noavg", po::value(&noavg)->zero_tokens()->default_value(false), "output averaged perceptron")
  ("batch", po::value<int>(&batch), "batch parallel, batch size")
  ;

  po::options_description cmdline_options;
  cmdline_options.add(desc);
  po::variables_map vm;
  po::store(po::command_line_parser(argc,argv).
            options(cmdline_options).run(), vm);
  po::notify(vm);
  if (help) {
    cout << "Usage: " + string(argv[0]) +  " [options]" << endl;
    cout << desc << endl;
    exit(0);
  }

  cerr << "kbmira with c=" << c << " decay=" << decay << " no_shuffle=" << no_shuffle << endl;

  if (vm.count("random-seed")) {
    cerr << "Initialising random seed to " << seed << endl;
    srand(seed);
  } else {
    cerr << "Initialising random seed from system clock" << endl;
    srand(time(NULL));
  }

  // Initialize weights
  ///
  // Dense
  vector<parameter_t> initParams;
  if(!denseInitFile.empty()) {
    ifstream opt(denseInitFile.c_str());
    string buffer;
    if (opt.fail()) {
      cerr << "could not open dense initfile: " << denseInitFile << endl;
      exit(3);
    }
    if (verbose) cerr << "Reading dense features:" << endl;
    parameter_t val;
    getline(opt,buffer);
    if (buffer.find_first_of("=") == buffer.npos) {
      UTIL_THROW_IF(type == "hypergraph", util::Exception, "For hypergraph version, require dense features in 'name= value' format");
      cerr << "WARN: dense features in deprecated Moses mert format. Prefer 'name= value' format." << endl;
      istringstream strstrm(buffer);
      while(strstrm >> val) {
        initParams.push_back(val);
        if(verbose) cerr << val << endl;
      }
    } else {
      vector<string> names;
      string last_name = "";
      size_t feature_ctr = 1;
      do {
        size_t equals = buffer.find_last_of("=");
        UTIL_THROW_IF(equals == buffer.npos, util::Exception, "Incorrect format in dense feature file: '"
                      << buffer << "'");
        string name = buffer.substr(0,equals);
        names.push_back(name);
        initParams.push_back(boost::lexical_cast<ValType>(buffer.substr(equals+2)));

        //Names for features with several values need to have their id added
        if (name != last_name) feature_ctr = 1;
        last_name = name;
        if (feature_ctr>1) {
          stringstream namestr;
          namestr << names.back() << "_" << feature_ctr;
          names[names.size()-1] = namestr.str();
          if (feature_ctr == 2) {
            stringstream namestr;
            namestr << names[names.size()-2] << "_" << (feature_ctr-1);
            names[names.size()-2] = namestr.str();
          }
        }
        ++feature_ctr;

      } while(getline(opt,buffer));


      //Make sure that SparseVector encodes dense feature names as 0..n-1.
      for (size_t i = 0; i < names.size(); ++i) {
        size_t id = SparseVector::encode(names[i]);
        assert(id == i);
        if (verbose) cerr << names[i] << " " << initParams[i] << endl;
      }

    }

    opt.close();
  }
  size_t initDenseSize = initParams.size();
  // Sparse
  if(!sparseInitFile.empty()) {
    if(initDenseSize==0) {
      cerr << "sparse initialization requires dense initialization" << endl;
      exit(3);
    }
    ifstream opt(sparseInitFile.c_str());
    if(opt.fail()) {
      cerr << "could not open sparse initfile: " << sparseInitFile << endl;
      exit(3);
    }
    int sparseCount=0;
    parameter_t val;
    std::string name;
    while(opt >> name >> val) {
      size_t id = SparseVector::encode(name) + initDenseSize;
      while(initParams.size()<=id) initParams.push_back(0.0);
      initParams[id] = val;
      sparseCount++;
    }
    cerr << "Found " << sparseCount << " initial sparse features" << endl;
    opt.close();
  }

  MiraWeightVector wv(initParams);
  //SparseVector sv;
  //wv.ToSparse(&sv);
  //MiraWeightVector wv2(vector<parameter_t>(initParams.size(), 0.0));

  // Initialize scorer
  if(sctype != "BLEU" && type == "hypergraph") {
  //if(sctype != "BLEU") {
    UTIL_THROW(util::Exception, "hypergraph mira only supports BLEU");
  }
  boost::scoped_ptr<Scorer> scorer(ScorerFactory::getScorer(sctype, scconfig));

  // Initialize background corpus
  vector<ValType> bg(scorer->NumberOfScores(), 1);

  boost::scoped_ptr<PerceptronDecoder> decoder;
  if (type == "nbest") {
    decoder.reset(new NbestPerceptronDecoder(featureFiles, scoreFiles, streaming, no_shuffle, safe_hope, scorer.get()));
  } else if (type == "hypergraph") {
    decoder.reset(new HypergraphPerceptronDecoder(hgDir, referenceFiles, initDenseSize, streaming, no_shuffle, safe_hope, hgPruning, wv, scorer.get()));
  } else if (type == "maxvio") {
    decoder.reset(new MaxvioPerceptronDecoder(hgDir, hgDirRef, referenceFiles, initDenseSize, streaming, no_shuffle, safe_hope, hgPruning, wv, scorer.get(), readRef, readHyp));
  } else {
    UTIL_THROW(util::Exception, "Unknown batch mira type: '" << type << "'");
  }

  // Training loop
  //if (!streaming_out)
  //  cerr << "Initial BLEU = " << decoder->Evaluate(wv.avg()) << endl;
  ValType bestBleu = 0;
  int totalCount = 1;


    ////
    ////   init moses for online decoding
    ////

    vector<string> vecargs = Moses::Tokenize(mosesargs);
    vecargs.insert(vecargs.begin(),"executable");
    char** argv2 = convert(vecargs);
    Parameter params;
    if (!params.LoadParam(int(vecargs.size()), (char**) argv2)) {
      exit(1);
    }
    //

    if (!StaticData::LoadDataStatic(&params, argv[0])) {
      exit(1);
    }
    const StaticData& staticData = StaticData::Instance();

 for(int j=0; j<n_iters; j++) {
    IOWrapper* ioWrapper = new IOWrapper();
    if (ioWrapper == NULL) {
      cerr << "Error; Failed to create IO object" << endl;
      exit(1);
    }

    InputType* source = NULL;
    size_t lineCount = staticData.GetStartTranslationId();

    // MIRA train for one epoch
    int iNumExamples = 0;
    int iNumUpdates = 0;
    ValType totalLoss = 0.0;
    size_t sentenceIndex = 0;
    for(decoder->reset(); !decoder->finished(); decoder->next()) {

#ifdef WITH_THREADS
      ThreadPool pool(staticData.ThreadCount());
#endif

      int b = 0;
        // decode a sentence
      while (ioWrapper->ReadInput(staticData.GetInputType(),source)) {
        source->SetTranslationId(lineCount);
        // set up task of translating one sentence
        TranslationTask* task = new TranslationTask(source, *ioWrapper);

#ifdef WITH_THREADS
        pool.Submit(task);
#else
      task->Run();
      delete task;
#endif

        source = NULL;
        ++lineCount;

        b++;
        if (b == batch) {
          break;
        }
      }

#ifdef WITH_THREADS
      pool.Stop(true); //flush remaining jobs
#endif

      //cerr << wv << endl;
      // compute violation
      PerceptronData hfd;
      decoder->Perceptron(bg,wv,&hfd, b);

      // Update weights
      if (!hfd.hopeModelEqual && hfd.hopeBleu  > hfd.modelBleu) {
        // Vector difference
        MiraFeatureVector diff = hfd.hopeFeatures - hfd.modelFeatures;
        // Bleu difference
        //assert(hfd.hopeBleu + 1e-8 >= hfd.fearBleu);
        ValType delta = hfd.hopeBleu - hfd.modelBleu;
        // Loss and update
        ValType diff_score = wv.score(diff);
        //ValType loss = delta - diff_score;
        if(verbose) {
          cerr << "Updating sent " << sentenceIndex << endl;
          cerr << "Wght: " << wv << endl;
          cerr << "Hope: " << hfd.hopeFeatures << " BLEU:" << hfd.hopeBleu << " Score:" << wv.score(hfd.hopeFeatures) << endl;
          cerr << "Model: " << hfd.modelFeatures << " BLEU:" << hfd.modelBleu << " Score:" << wv.score(hfd.modelFeatures) << endl;
          cerr << "Diff: " << diff << " BLEU:" << delta << " Score:" << diff_score << endl;
          //cerr << "Loss: " << loss <<  " Scale: " << 1 << endl;
          cerr << endl;
        }
        /*if(loss > 0) {
          ValType eta = min(c, loss / diff.sqrNorm());
          wv.update(diff,eta);
          totalLoss+=loss;
          iNumUpdates++;
        }*/

        if (diff_score < 0) {
          //cerr << wv << endl;
          //cerr << diff << endl;

          ValType delta = c / hfd.updateCount;

          wv.update(diff, delta);
          //cerr << wv << endl;
          UpdateDecoderWeights(wv, initDenseSize);
          //wv2.update(diff,1.0*totalCount);
          totalLoss+=diff_score;
          iNumUpdates++;
        }

        // Update BLEU statistics
        /*for(size_t k=0; k<bg.size(); k++) {
          bg[k]*=decay;
          if(model_bg)
            bg[k]+=hfd.modelStats[k];
          else
            bg[k]+=hfd.hopeStats[k];
        }*/
      }
      iNumExamples++;
      ++sentenceIndex;
      ++totalCount;
      if (streaming_out)
        cout << wv << endl;

      // jump
      for(int bi = 0; bi < b-1; bi++)
        decoder->next();

    }
    // Training Epoch summary
    cerr << iNumUpdates << "/" << iNumExamples << " updates"
         << ", avg loss = " << (totalLoss / iNumExamples);

    // Evaluate current average weights

    /*if (avgPerceptron) {
      SparseVector svec;
      wv2.ToSparse(&svec, initDenseSize);
      wv.update(MiraFeatureVector(svec,initDenseSize), 1.0/totalCount);
    }*/

    AvgWeightVector avg = wv.avg();
    avg.noavg = noavg;
    ValType bleu = decoder->Evaluate(avg);
    //cerr << ", BLEU = " << bleu << endl;

    if (bleu > bestBleu) {
      bestBleu = bleu;

      ostream* out;
      ofstream outFile;
      if (!outputFile.empty() ) {
        outFile.open(outputFile.c_str());
        if (!(outFile)) {
          cerr << "Error: Failed to open " << outputFile << endl;
          exit(1);
        }
        out = &outFile;
      } else {
        out = &cout;
      }
      for(size_t i=0; i<avg.size(); i++) {
        if(i<initDenseSize)
          *out << "F" << i << " " << avg.weight(i) << endl;
        else {
          if(abs(avg.weight(i))>1e-8)
            *out << SparseVector::decode(i-initDenseSize) << " " << avg.weight(i) << endl;
        }
      }
      outFile.close();
   }
  }
      cerr << ", Best BLEU = " << bestBleu << endl;
  }


  // averaged perceptron
  /*SparseVector svec;
  wv2.ToSparse(&svec);
  wv.update(MiraFeatureVector(svec,initDenseSize), 1.0/totalCount);

  // output
  AvgWeightVector avg = wv.avg();
  avg.noavg = true;

  ostream* out;
  ofstream outFile;
  if (!outputFile.empty() ) {
    outFile.open(outputFile.c_str());
    if (!(outFile)) {
      cerr << "Error: Failed to open " << outputFile << endl;
      exit(1);
    }
    out = &outFile;
  } else {
    out = &cout;
  }
  for(size_t i=0; i<avg.size(); i++) {
    if(i<initDenseSize)
      *out << "F" << i << " " << avg.weight(i) << endl;
    else {
      if(abs(avg.weight(i))>1e-8)
        *out << SparseVector::decode(i-initDenseSize) << " " << avg.weight(i) << endl;
    }
  }
  outFile.close();*/

// cerr << "Best BLEU = " << bestBleu << endl;
//}
// --Emacs trickery--
// Local Variables:
// mode:c++
// c-basic-offset:2
// End:
