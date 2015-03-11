/***********************************************************************
Moses - factored phrase-based language decoder
Copyright (C) 2014- University of Edinburgh

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
***********************************************************************/

#include <algorithm>
#include <cmath>
#include <iterator>

#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "util/exception.hh"
#include "util/file_piece.hh"

#include "Scorer.h"
#include "PerceptronDecoder.h"
#include "PerceptronForestRescore.h"

#include <boost/shared_ptr.hpp>

using namespace std;
namespace fs = boost::filesystem;

namespace MosesTuning
{

static const ValType BLEU_RATIO = 5;

ValType PerceptronDecoder::Evaluate(const AvgWeightVector& wv)
{
  vector<ValType> stats(scorer_->NumberOfScores(),0);
  for(reset(); !finished(); next()) {
    vector<ValType> sent;
    MaxModel(wv,&sent);
    for(size_t i=0; i<sent.size(); i++) {
      stats[i]+=sent[i];
    }
  }
  return scorer_->calculateScore(stats);
}

NbestPerceptronDecoder::NbestPerceptronDecoder(
  const vector<string>& featureFiles,
  const vector<string>&  scoreFiles,
  bool streaming,
  bool  no_shuffle,
  bool safe_hope,
  Scorer* scorer
) : safe_hope_(safe_hope)
{
  scorer_ = scorer;
  if (streaming) {
    train_.reset(new StreamingHypPackEnumerator(featureFiles, scoreFiles));
  } else {
    train_.reset(new RandomAccessHypPackEnumerator(featureFiles, scoreFiles, no_shuffle));
  }
}


void NbestPerceptronDecoder::next()
{
  train_->next();
}

bool NbestPerceptronDecoder::finished()
{
  return train_->finished();
}

void NbestPerceptronDecoder::reset()
{
  train_->reset();
}

void NbestPerceptronDecoder::Perceptron(
  const std::vector<ValType>& backgroundBleu,
  const MiraWeightVector& wv,
  PerceptronData* Perceptron, int batch=1
)
{


  // Hope / fear decode
  ValType hope_scale = 1.0;
  size_t hope_index=0, fear_index=0, model_index=0;
  ValType hope_score=0, fear_score=0, model_score=0;
  for(size_t safe_loop=0; safe_loop<2; safe_loop++) {
    ValType hope_bleu, hope_model;
    for(size_t i=0; i< train_->cur_size(); i++) {
      const MiraFeatureVector& vec=train_->featuresAt(i);
      ValType score = wv.score(vec);
      ValType bleu = scorer_->calculateSentenceLevelBackgroundScore(train_->scoresAt(i),backgroundBleu);
      // Hope
      if(i==0 || (bleu) > hope_score) {
        hope_score = bleu;
        hope_index = i;
        hope_bleu = bleu;
        hope_model = score;
      }
      // Fear
      //if(i==0 || (bleu) > fear_score) {
      //  fear_score = bleu;
      //  fear_index = i;
      //}
      // Model
      if(i==0 || score > model_score) {
        model_score = score;
        model_index = i;
      }
    }
    // Outer loop rescales the contribution of model score to 'hope' in antagonistic cases
    // where model score is having far more influence than BLEU
    //hope_bleu *= BLEU_RATIO; // We only care about cases where model has MUCH more influence than BLEU
    //if(safe_hope_ && safe_loop==0 && abs(hope_model)>1e-8 && abs(hope_bleu)/abs(hope_model)<hope_scale)
    //  hope_scale = abs(hope_bleu) / abs(hope_model);
    //else break;
    break;
  }
  Perceptron->modelFeatures = train_->featuresAt(model_index);
  Perceptron->hopeFeatures = train_->featuresAt(hope_index);
  Perceptron->fearFeatures = train_->featuresAt(fear_index);

  Perceptron->hopeStats = train_->scoresAt(hope_index);
  Perceptron->hopeBleu = scorer_->calculateSentenceLevelBackgroundScore(Perceptron->hopeStats, backgroundBleu);
  const vector<float>& fear_stats = train_->scoresAt(fear_index);
  Perceptron->fearBleu = scorer_->calculateSentenceLevelBackgroundScore(fear_stats, backgroundBleu);

  Perceptron->modelStats = train_->scoresAt(model_index);
  Perceptron->modelBleu = scorer_->calculateSentenceLevelBackgroundScore(Perceptron->modelStats, backgroundBleu);
  Perceptron->PerceptronEqual = (hope_index == fear_index);
  Perceptron->hopeModelEqual = (hope_index == model_index);
}

void NbestPerceptronDecoder::MaxModel(const AvgWeightVector& wv, std::vector<ValType>* stats)
{
  // Find max model
  size_t max_index=0;
  ValType max_score=0;
  for(size_t i=0; i<train_->cur_size(); i++) {
    MiraFeatureVector vec(train_->featuresAt(i));
    ValType score = wv.score(vec);
    if(i==0 || score > max_score) {
      max_index = i;
      max_score = score;
    }
  }
  *stats = train_->scoresAt(max_index);
}



HypergraphPerceptronDecoder::HypergraphPerceptronDecoder
(
  const string& hypergraphDir,
  const vector<string>& referenceFiles,
  size_t num_dense,
  bool streaming,
  bool no_shuffle,
  bool safe_hope,
  size_t hg_pruning,
  const MiraWeightVector& wv,
  Scorer* scorer
) :
  num_dense_(num_dense)
{

  UTIL_THROW_IF(streaming, util::Exception, "Streaming not currently supported for hypergraphs");
  UTIL_THROW_IF(!fs::exists(hypergraphDir), HypergraphException, "Directory '" << hypergraphDir << "' does not exist");
  UTIL_THROW_IF(!referenceFiles.size(), util::Exception, "No reference files supplied");
  references_.Load(referenceFiles, vocab_);

  SparseVector weights;
  wv.ToSparse(&weights, num_dense);
  scorer_ = scorer;

  static const string kWeights = "weights";
  fs::directory_iterator dend;
  size_t fileCount = 0;

  cerr << "Reading  hypergraphs" << endl;
  for (fs::directory_iterator di(hypergraphDir); di != dend; ++di) {
    const fs::path& hgpath = di->path();
    if (hgpath.filename() == kWeights) continue;
    //  cerr << "Reading " << hgpath.filename() << endl;
    Graph graph(vocab_);
    size_t id = boost::lexical_cast<size_t>(hgpath.stem().string());
    util::scoped_fd fd(util::OpenReadOrThrow(hgpath.string().c_str()));
    //util::FilePiece file(di->path().string().c_str());
    util::FilePiece file(fd.release());
    ReadGraph(file,graph);

    //cerr << "ref length " << references_.Length(id) << endl;
    size_t edgeCount = hg_pruning * references_.Length(id);
    boost::shared_ptr<Graph> prunedGraph;
    prunedGraph.reset(new Graph(vocab_));
    graph.Prune(prunedGraph.get(), weights, edgeCount);
    graphs_[id] = prunedGraph;
    // cerr << "Pruning to v=" << graphs_[id]->VertexSize() << " e=" << graphs_[id]->EdgeSize()  << endl;
    ++fileCount;
    if (fileCount % 10 == 0) cerr << ".";
    if (fileCount % 400 ==  0) cerr << " [count=" << fileCount << "]\n";
  }
  cerr << endl << "Done" << endl;

  sentenceIds_.resize(graphs_.size());
  for (size_t i = 0; i < graphs_.size(); ++i) sentenceIds_[i] = i;
  if (!no_shuffle) {
    random_shuffle(sentenceIds_.begin(), sentenceIds_.end());
  }

}

void HypergraphPerceptronDecoder::reset()
{
  sentenceIdIter_ = sentenceIds_.begin();
}

void HypergraphPerceptronDecoder::next()
{
  sentenceIdIter_++;
}

bool HypergraphPerceptronDecoder::finished()
{
  return sentenceIdIter_ == sentenceIds_.end();
}

void HypergraphPerceptronDecoder::Perceptron(
  const vector<ValType>& backgroundBleu,
  const MiraWeightVector& wv,
  PerceptronData* Perceptron, int batch=1
)
{
  size_t sentenceId = *sentenceIdIter_;
  SparseVector weights;
  wv.ToSparse(&weights, num_dense_);
  const Graph& graph = *(graphs_[sentenceId]);

  ValType hope_scale = 1.0;
  HgHypothesis hopeHypo, fearHypo, modelHypo;
  for(size_t safe_loop=0; safe_loop<2; safe_loop++) {

    //hope decode
    Viterbi(graph, weights, 1, references_, sentenceId, backgroundBleu, &hopeHypo);

    //fear decode
    Viterbi(graph, weights, -1, references_, sentenceId, backgroundBleu, &fearHypo);

    //Model decode
    Viterbi(graph, weights, 0, references_, sentenceId, backgroundBleu, &modelHypo);


    // Outer loop rescales the contribution of model score to 'hope' in antagonistic cases
    // where model score is having far more influence than BLEU
    //  hope_bleu *= BLEU_RATIO; // We only care about cases where model has MUCH more influence than BLEU
    //  if(safe_hope_ && safe_loop==0 && abs(hope_model)>1e-8 && abs(hope_bleu)/abs(hope_model)<hope_scale)
    //    hope_scale = abs(hope_bleu) / abs(hope_model);
    //  else break;
    //TODO: Don't currently get model and bleu so commented this out for now.
    break;
  }

  //modelFeatures, hopeFeatures and fearFeatures
  Perceptron->modelFeatures = MiraFeatureVector(modelHypo.featureVector, num_dense_);
  Perceptron->hopeFeatures = MiraFeatureVector(hopeHypo.featureVector, num_dense_);
  Perceptron->fearFeatures = MiraFeatureVector(fearHypo.featureVector, num_dense_);

  //Need to know which are to be mapped to dense features!

  //Only C++11
  //Perceptron->modelStats.assign(std::begin(modelHypo.bleuStats), std::end(modelHypo.bleuStats));
  vector<ValType> fearStats(scorer_->NumberOfScores());
  Perceptron->hopeStats.reserve(scorer_->NumberOfScores());
  Perceptron->modelStats.reserve(scorer_->NumberOfScores());
  for (size_t i = 0; i < fearStats.size(); ++i) {
    Perceptron->modelStats.push_back(modelHypo.bleuStats[i]);
    Perceptron->hopeStats.push_back(hopeHypo.bleuStats[i]);

    fearStats[i] = fearHypo.bleuStats[i];
  }
  /*
  cerr << "hope" << endl;;
  for (size_t i = 0; i < hopeHypo.text.size(); ++i) {
    cerr << hopeHypo.text[i]->first << " ";
  }
  cerr << endl;
  for (size_t i = 0; i < fearStats.size(); ++i) {
    cerr << hopeHypo.bleuStats[i] << " ";
  }
  cerr << endl;
  cerr << "fear";
  for (size_t i = 0; i < fearHypo.text.size(); ++i) {
    cerr << fearHypo.text[i]->first << " ";
  }
  cerr << endl;
  for (size_t i = 0; i < fearStats.size(); ++i) {
    cerr  << fearHypo.bleuStats[i] << " ";
  }
  cerr << endl;
  cerr << "model";
  for (size_t i = 0; i < modelHypo.text.size(); ++i) {
    cerr << modelHypo.text[i]->first << " ";
  }
  cerr << endl;
  for (size_t i = 0; i < fearStats.size(); ++i) {
    cerr << modelHypo.bleuStats[i] << " ";
  }
  cerr << endl;
  */
  Perceptron->hopeBleu = sentenceLevelBackgroundBleu(Perceptron->hopeStats, backgroundBleu);
  Perceptron->fearBleu = sentenceLevelBackgroundBleu(fearStats, backgroundBleu);
  Perceptron->modelBleu = sentenceLevelBackgroundBleu(Perceptron->modelStats, backgroundBleu);

  //If fv and bleu stats are equal, then assume equal
  Perceptron->PerceptronEqual = true; //(Perceptron->hopeBleu - Perceptron->fearBleu) >= 1e-8;
  if (Perceptron->PerceptronEqual) {
    for (size_t i = 0; i < fearStats.size(); ++i) {
      if (fearStats[i] != Perceptron->hopeStats[i]) {
        Perceptron->PerceptronEqual = false;
        break;
      }
    }
  }
  Perceptron->PerceptronEqual = Perceptron->PerceptronEqual && (Perceptron->fearFeatures == Perceptron->hopeFeatures);

  //If fv and bleu stats are equal, then assume equal
  Perceptron->hopeModelEqual = true; //(Perceptron->hopeBleu - Perceptron->fearBleu) >= 1e-8;
  if (Perceptron->hopeModelEqual) {
    for (size_t i = 0; i < Perceptron->modelStats.size(); ++i) {
      if (Perceptron->modelStats[i] != Perceptron->hopeStats[i]) {
        Perceptron->hopeModelEqual = false;
        break;
      }
    }
  }
  Perceptron->hopeModelEqual = Perceptron->hopeModelEqual && (Perceptron->modelFeatures == Perceptron->hopeFeatures);
}

void HypergraphPerceptronDecoder::MaxModel(const AvgWeightVector& wv, vector<ValType>* stats)
{
  assert(!finished());
  HgHypothesis bestHypo;
  size_t sentenceId = *sentenceIdIter_;
  SparseVector weights;
  wv.ToSparse(&weights, num_dense_);
  vector<ValType> bg(scorer_->NumberOfScores());
  //cerr << "Calculating bleu on " << sentenceId << endl;
  Viterbi(*(graphs_[sentenceId]), weights, 0, references_, sentenceId, bg, &bestHypo);
  stats->resize(bestHypo.bleuStats.size());
  /*
  for (size_t i = 0; i < bestHypo.text.size(); ++i) {
    cerr << bestHypo.text[i]->first << " ";
  }
  cerr << endl;
  */
  for (size_t i = 0; i < bestHypo.bleuStats.size(); ++i) {
    (*stats)[i] = bestHypo.bleuStats[i];
  }
}

// maxvio

MaxvioPerceptronDecoder::MaxvioPerceptronDecoder
(
  const string& hypergraphDir,
  const string& hypergraphDirRef,
  const vector<string>& referenceFiles,
  size_t num_dense,
  bool streaming,
  bool no_shuffle,
  bool safe_hope,
  size_t hg_pruning,
  const MiraWeightVector& wv,
  Scorer* scorer, bool readRef, bool readHyp
) :
  num_dense_(num_dense),
  hypergraphDirHyp (hypergraphDir),
  hypergraphDirRef (hypergraphDirRef),
  hg_pruning (hg_pruning), readRef_ (readRef), readHyp_(readHyp)
{

  UTIL_THROW_IF(streaming, util::Exception, "Streaming not currently supported for maxvio");
  //UTIL_THROW_IF(!fs::exists(hypergraphDir), HypergraphException, "Directory '" << hypergraphDir << "' does not exist");
  UTIL_THROW_IF(!fs::exists(hypergraphDirRef), HypergraphException, "Directory '" << hypergraphDirRef << "' does not exist");
  UTIL_THROW_IF(!referenceFiles.size(), util::Exception, "No reference files supplied");
  references_.Load(referenceFiles, vocab_);

  SparseVector weights;
  wv.ToSparse(&weights, num_dense);
  scorer_ = scorer;

  static const string kWeights = "weights";
  fs::directory_iterator dend;
  size_t fileCount = 0;

  cerr << "counting  ref hypergraphs" << endl;
  for (fs::directory_iterator di(hypergraphDirRef); di != dend; ++di) {
    const fs::path& hgpath = di->path();
    if (hgpath.filename() == kWeights) continue;

    //  cerr << "Reading " << hgpath.filename() << endl;
    if (readRef) {
      Graph graph(vocab_);
      //boost::shared_ptr<Graph> prunedGraph;
      //prunedGraph.reset(new Graph(vocab_));

      size_t id = boost::lexical_cast<size_t>(hgpath.stem().string());
      util::scoped_fd fd(util::OpenReadOrThrow(hgpath.string().c_str()));
      //util::FilePiece file(di->path().string().c_str());
      util::FilePiece file(fd.release());
      ReadGraph(file,graph);

      //cerr << "ref length " << references_.Length(id) << endl;
      size_t edgeCount = hg_pruning * references_.Length(id);

      if (edgeCount == 0) {
        edgeCount = std::numeric_limits<size_t>::max();
      }

      boost::shared_ptr<Graph> prunedGraph;
      prunedGraph.reset(new Graph(vocab_));
      graph.Prune(prunedGraph.get(), weights, edgeCount);
      graphs_ref[id] = prunedGraph;
      // cerr << "Pruning to v=" << graphs_[id]->VertexSize() << " e=" << graphs_[id]->EdgeSize()  << endl;

      /*for(size_t i = 0; i < (*prunedGraph).VertexSize(); i++) {
        const Vertex& vi = (*prunedGraph).GetVertex(i);
        cerr << vi.startPos << " " << vi.endPos << endl;
      }
      exit(1);*/

    }
    ++fileCount;
    if (fileCount % 10 == 0) cerr << ".";
    if (fileCount % 400 ==  0) cerr << " [count=" << fileCount << "]\n";

    //++fileCount;
  }
  cerr << fileCount << endl << "Done" << endl;


  if (readHyp) {
    size_t fileCount2 = 0;
    cerr << "counting  hyp hypergraphs" << endl;
    for (fs::directory_iterator di(hypergraphDir); di != dend; ++di) {
      const fs::path& hgpath = di->path();
      if (hgpath.filename() == kWeights) continue;

      //  cerr << "Reading " << hgpath.filename() << endl;
      Graph graph(vocab_);

      //boost::shared_ptr<Graph> prunedGraph;
      //prunedGraph.reset(new Graph(vocab_));

      size_t id = boost::lexical_cast<size_t>(hgpath.stem().string());
      util::scoped_fd fd(util::OpenReadOrThrow(hgpath.string().c_str()));
      //util::FilePiece file(di->path().string().c_str());
      util::FilePiece file(fd.release());
      ReadGraph(file,graph);

      //cerr << "ref length " << references_.Length(id) << endl;
      size_t edgeCount = hg_pruning * references_.Length(id);

      if (edgeCount == 0) {
        edgeCount = std::numeric_limits<size_t>::max();
      }

      boost::shared_ptr<Graph> prunedGraph;
      prunedGraph.reset(new Graph(vocab_));
      graph.Prune(prunedGraph.get(), weights, edgeCount);
      graphs_hyp[id] = prunedGraph;
      // cerr << "Pruning to v=" << graphs_[id]->VertexSize() << " e=" << graphs_[id]->EdgeSize()  << endl;
      ++fileCount2;
      if (fileCount2 % 10 == 0) cerr << ".";
      if (fileCount2 % 400 ==  0) cerr << " [count=" << fileCount2 << "]\n";

      //++fileCount;
    }
    cerr << fileCount2 << endl << "Done" << endl;

    assert(fileCount==fileCount2);
  }

  sentenceIds_.resize(fileCount);
  for (size_t i = 0; i < fileCount; ++i) sentenceIds_[i] = i;
  if (!no_shuffle) {
    random_shuffle(sentenceIds_.begin(), sentenceIds_.end());
  }

}

void MaxvioPerceptronDecoder::reset()
{
  sentenceIdIter_ = sentenceIds_.begin();
}

void MaxvioPerceptronDecoder::next()
{
  sentenceIdIter_++;
}

bool MaxvioPerceptronDecoder::finished()
{
  return sentenceIdIter_ == sentenceIds_.end();
}

void MaxvioPerceptronDecoder::ReadAGraph(size_t sentenceId, const string& hypergraphDir, Graph* graph, const SparseVector& weights) {
  // read hypergraph of hypoes.
  //fs::directory_iterator di(hypergraphDir);
  //di = di + sentenceId;
  //for (size_t i = 0; i < sentenceId; i++)
  //  ++di;
  //const fs::path& hgpath = di->path();
  fs::path hgpath(hypergraphDir+"/"+boost::lexical_cast<std::string>(sentenceId)+".gz");
  //Graph graph(vocab_);
  //size_t id = boost::lexical_cast<size_t>(hgpath.stem().string());
  //fs::path newpath = hgpath.parent_path() +fs::path(Moses::SPrint<size_t>(sentenceId)) + hgpath.stem()
  util::scoped_fd fd(util::OpenReadOrThrow(hgpath.string().c_str()));
  util::FilePiece file(fd.release());
  ReadGraph(file,*graph);
  //return &graph;

  size_t edgeCount = hg_pruning * references_.Length(sentenceId);
  if (edgeCount == 0) {
    edgeCount = std::numeric_limits<size_t>::max();
  }
  boost::shared_ptr<Graph> prunedGraph;
  prunedGraph.reset(new Graph(vocab_));
  graph->Prune(prunedGraph.get(), weights, edgeCount);
  graph = prunedGraph.get();
}

void MaxvioPerceptronDecoder::Perceptron(
  const vector<ValType>& backgroundBleu,
  const MiraWeightVector& wv,
  PerceptronData* Perceptron, int batch=1
)
{
  size_t sentenceId = *sentenceIdIter_;

  SparseVector weights;
  wv.ToSparse(&weights, num_dense_);

  Perceptron->hopeModelEqual = true;

  int updateCount = 0;

  for(int index = 0; index < batch; index++) {
    sentenceId = *sentenceIdIter_ + index;

    //Graph* graphHyp = new Graph(vocab_);
    //Graph* graphRef = new Graph(vocab_);
    boost::shared_ptr<Graph> graphHyp;
    boost::shared_ptr<Graph> graphRef;

    if (!readHyp_) {
      graphHyp.reset(new Graph(vocab_));
      ReadAGraph(sentenceId, hypergraphDirHyp, graphHyp.get(), weights);
    } else {
      //graphHyp.reset(graphs_hyp[sentenceId], new Graph(vocab_));
      graphHyp = graphs_hyp[sentenceId];
    }
    if (!readRef_) {
      graphRef.reset(new Graph(vocab_));
      ReadAGraph(sentenceId, hypergraphDirRef, graphRef.get(), weights);
    } else {
      //graphRef.reset(graphs_ref[sentenceId], new Graph(vocab_));
      graphRef = graphs_ref[sentenceId];
    }

    ValType hope_scale = 1.0;

    VioColl hypVio;
    VioColl refVio;
    for(size_t safe_loop=0; safe_loop<2; safe_loop++) {

      //Model decode
      Viterbi(*(graphHyp), weights, 0, references_, sentenceId, backgroundBleu, hypVio);
      Viterbi(*(graphRef), weights, 0, references_, sentenceId, backgroundBleu, refVio);

      break;
    }

    // find the max vio
    Range bestr(0,0);
    float maxvio = 0.0;

    for(VioColl::const_iterator ri = refVio.begin(); ri != refVio.end(); ++ri) {
      VioColl::const_iterator hi = hypVio.find(ri->first);
      if (hi == hypVio.end())
        continue;

      float scoreHyp = inner_product((*hi->second).featureVector, weights);
      float scoreRef = inner_product((*ri->second).featureVector, weights);


      if (scoreRef < scoreHyp && scoreHyp-scoreRef > maxvio && ri->first.second - ri->first.first > 1) {
        bestr = ri->first;
        maxvio = scoreHyp-scoreRef;
      }
    }

    if (bestr.second - bestr.first <= 1) {
      continue;
    }
      //Perceptron->hopeModelEqual = true;
      //return;
    Perceptron->updateCount++;

  //modelFeatures, hopeFeatures and fearFeatures
    if (Perceptron->hopeModelEqual) {
      Perceptron->modelFeatures = MiraFeatureVector((*(hypVio.find(bestr)->second)).featureVector, num_dense_);
      Perceptron->hopeFeatures = MiraFeatureVector((*(refVio.find(bestr)->second)).featureVector, num_dense_);
    } else {
      Perceptron->modelFeatures = Perceptron->modelFeatures+MiraFeatureVector((*(hypVio.find(bestr)->second)).featureVector, num_dense_);
      Perceptron->hopeFeatures = Perceptron->hopeFeatures+MiraFeatureVector((*(refVio.find(bestr)->second)).featureVector, num_dense_);
    }

    //Need to know which are to be mapped to dense features!

    //Only C++11
    //Perceptron->modelStats.assign(std::begin(modelHypo.bleuStats), std::end(modelHypo.bleuStats));
    //vector<ValType> fearStats(scorer_->NumberOfScores());
    size_t size = (*graphHyp).GetVertex((*graphHyp).VertexSize()-1).SourceCovered();
    Range fullRange(0, size-1);
    const HgHypothesis& modelHypo = *(hypVio.find(bestr)->second);
    const HgHypothesis& hopeHypo = *(refVio.find(bestr)->second);

    if (Perceptron->hopeModelEqual) {
      Perceptron->hopeStats.reserve(scorer_->NumberOfScores());
      Perceptron->modelStats.reserve(scorer_->NumberOfScores());
      for (size_t i = 0; i < scorer_->NumberOfScores(); ++i) {
        Perceptron->modelStats.push_back(modelHypo.bleuStats[i]);
        Perceptron->hopeStats.push_back(hopeHypo.bleuStats[i]);
      }
    } else {
      for (size_t i = 0; i < scorer_->NumberOfScores(); ++i) {
        Perceptron->modelStats[i] += modelHypo.bleuStats[i];
        Perceptron->hopeStats[i] += hopeHypo.bleuStats[i];
      }
    }

    Perceptron->hopeModelEqual = false;
  }

  Perceptron->hopeBleu = 1.0;//sentenceLevelBackgroundBleu(Perceptron->hopeStats, backgroundBleu);
  Perceptron->modelBleu = 0.0;//sentenceLevelBackgroundBleu(Perceptron->modelStats, backgroundBleu);

}

void MaxvioPerceptronDecoder::MaxModelCurrSent(const MiraWeightVector& wv, PerceptronData* Perceptron)
{
  assert(!finished());
  HgHypothesis bestHypo;
  size_t sentenceId = *sentenceIdIter_;
  SparseVector weights;
  wv.ToSparse(&weights, num_dense_);
  vector<ValType> bg(scorer_->NumberOfScores());
  //cerr << "Calculating bleu on " << sentenceId << endl;
  if (!readHyp_) {
    Graph graphHyp(vocab_);
    ReadAGraph(sentenceId, hypergraphDirHyp, &graphHyp, weights);
    Viterbi(graphHyp, weights, 0, references_, sentenceId, bg, &bestHypo);
  } else {
    Viterbi(*(graphs_hyp[sentenceId]), weights, 0, references_, sentenceId, bg, &bestHypo);
  }

  Perceptron->modelStats.reserve(scorer_->NumberOfScores());
  for (size_t i = 0; i < scorer_->NumberOfScores(); ++i) {
    Perceptron->modelStats.push_back(bestHypo.bleuStats[i]);
  }
}

void MaxvioPerceptronDecoder::MaxModel(const AvgWeightVector& wv, vector<ValType>* stats)
{
  assert(!finished());
  HgHypothesis bestHypo;
  size_t sentenceId = *sentenceIdIter_;
  SparseVector weights;
  wv.ToSparse(&weights, num_dense_);
  vector<ValType> bg(scorer_->NumberOfScores());
  //cerr << "Calculating bleu on " << sentenceId << endl;
  if (!readHyp_) {
    Graph graphHyp(vocab_);
    ReadAGraph(sentenceId, hypergraphDirHyp, &graphHyp, weights);
    Viterbi(graphHyp, weights, 0, references_, sentenceId, bg, &bestHypo);
  } else {
    Viterbi(*(graphs_hyp[sentenceId]), weights, 0, references_, sentenceId, bg, &bestHypo);
  }
  stats->resize(bestHypo.bleuStats.size());
  /*
  for (size_t i = 0; i < bestHypo.text.size(); ++i) {
    cerr << bestHypo.text[i]->first << " ";
  }
  cerr << endl;
  */
  for (size_t i = 0; i < bestHypo.bleuStats.size(); ++i) {
    (*stats)[i] = bestHypo.bleuStats[i];
  }
}

};
