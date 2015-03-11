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
#ifndef MERT_PERCEPTRONDECODER_H
#define MERT_PERCEPTRONDECODER_H

#include <vector>

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include "ForestRescore.h"
#include "Hypergraph.h"
#include "HypPackEnumerator.h"
#include "MiraFeatureVector.h"
#include "MiraWeightVector.h"

//
// Used by batch mira to get the hope, fear and model hypothesis. This wraps
// the n-best list and lattice/hypergraph implementations
//

namespace MosesTuning
{

class Scorer;

/** To be filled in by the decoder */
struct PerceptronData {
  MiraFeatureVector modelFeatures;
  MiraFeatureVector hopeFeatures;
  MiraFeatureVector fearFeatures;

  std::vector<float> modelStats;
  std::vector<float> hopeStats;

  ValType hopeBleu;
  ValType fearBleu;
  ValType modelBleu;

  bool PerceptronEqual;
  bool hopeModelEqual;

  int updateCount = 0;
};

//Abstract base class
class PerceptronDecoder
{
public:
  //iterator methods
  virtual void reset() = 0;
  virtual void next() = 0;
  virtual bool finished() = 0;

  virtual ~PerceptronDecoder() {};

  /**
    * Calculate hope, fear and model hypotheses
    **/
  virtual void Perceptron(
    const std::vector<ValType>& backgroundBleu,
    const MiraWeightVector& wv,
    PerceptronData* Perceptron, int batch=1
  ) = 0;

  /** Max score decoding */
  virtual void MaxModel(const AvgWeightVector& wv, std::vector<ValType>* stats)
  = 0;

  virtual void MaxModelCurrSent(const MiraWeightVector& wv, PerceptronData* Perceptron) {}

  /** Calculate bleu on training set */
  ValType Evaluate(const AvgWeightVector& wv);

protected:
  Scorer* scorer_;
};


/** Gets hope-fear from nbest lists */
class NbestPerceptronDecoder : public virtual PerceptronDecoder
{
public:
  NbestPerceptronDecoder(const std::vector<std::string>& featureFiles,
                       const std::vector<std::string>&  scoreFiles,
                       bool streaming,
                       bool  no_shuffle,
                       bool safe_hope,
                       Scorer* scorer
                      );

  virtual void reset();
  virtual void next();
  virtual bool finished();

  virtual void Perceptron(
    const std::vector<ValType>& backgroundBleu,
    const MiraWeightVector& wv,
    PerceptronData* Perceptron, int batch
  );

  virtual void MaxModel(const AvgWeightVector& wv, std::vector<ValType>* stats);

private:
  boost::scoped_ptr<HypPackEnumerator> train_;
  bool safe_hope_;

};



/** Gets hope-fear from hypergraphs */
class HypergraphPerceptronDecoder : public virtual PerceptronDecoder
{
public:
  HypergraphPerceptronDecoder(
    const std::string& hypergraphDir,
    const std::vector<std::string>& referenceFiles,
    size_t num_dense,
    bool streaming,
    bool no_shuffle,
    bool safe_hope,
    size_t hg_pruning,
    const MiraWeightVector& wv,
    Scorer* scorer_
  );

  virtual void reset();
  virtual void next();
  virtual bool finished();

  virtual void Perceptron(
    const std::vector<ValType>& backgroundBleu,
    const MiraWeightVector& wv,
    PerceptronData* Perceptron, int batch
  );

  virtual void MaxModel(const AvgWeightVector& wv, std::vector<ValType>* stats);

private:
  size_t num_dense_;
  //maps sentence Id to graph ptr
  typedef std::map<size_t, boost::shared_ptr<Graph> > GraphColl;
  GraphColl graphs_;
  std::vector<size_t> sentenceIds_;
  std::vector<size_t>::const_iterator sentenceIdIter_;
  ReferenceSet references_;
  Vocab vocab_;
};

/** Gets hope-fear from hypergraphs */
class MaxvioPerceptronDecoder : public virtual PerceptronDecoder
{
public:
  MaxvioPerceptronDecoder(
    const std::string& hypergraphDir,
    const std::string& hypergraphDirRef,
    const std::vector<std::string>& referenceFiles,
    size_t num_dense,
    bool streaming,
    bool no_shuffle,
    bool safe_hope,
    size_t hg_pruning,
    const MiraWeightVector& wv,
    Scorer* scorer_,
    bool readRef, bool readHyp
  );

  virtual void reset();
  virtual void next();
  virtual bool finished();

  virtual void Perceptron(
    const std::vector<ValType>& backgroundBleu,
    const MiraWeightVector& wv,
    PerceptronData* Perceptron, int batch
  );

  virtual void MaxModel(const AvgWeightVector& wv, std::vector<ValType>* stats);
  void MaxModelCurrSent(const MiraWeightVector& wv, PerceptronData* Perceptron);
  void ReadAGraph(size_t sentenceId, const std::string& hypergraphDir, Graph* graph, const SparseVector& weights);

private:
  size_t num_dense_;
  //maps sentence Id to graph ptr
  typedef std::map<size_t, boost::shared_ptr<Graph> > GraphColl;
  GraphColl graphs_ref;
  GraphColl graphs_hyp;
  std::vector<size_t> sentenceIds_;
  std::vector<size_t>::const_iterator sentenceIdIter_;
  ReferenceSet references_;
  Vocab vocab_;
  std::string hypergraphDirHyp;
  std::string hypergraphDirRef;
  size_t hg_pruning;
  bool readRef_;
  bool readHyp_;
  int batch;
};

};

#endif

