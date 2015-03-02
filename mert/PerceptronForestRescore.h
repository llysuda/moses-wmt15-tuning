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

#include <cmath>
#include <limits>
#include <list>

#include <boost/unordered_set.hpp>

#include "util/file_piece.hh"
#include "util/tokenize_piece.hh"

#include "BleuScorer.h"
#include "ForestRescore.h"

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

using namespace std;

namespace MosesTuning
{

typedef pair<const Edge*,FeatureStatsType> BackPointer;

static bool GetBestHypothesis(size_t vertexId, const Graph& graph, const vector<BackPointer>& bps,
                              HgHypothesis* bestHypo)
{
  //cerr << "Expanding " << vertexId << " Score: " << bps[vertexId].second << endl;
  //UTIL_THROW_IF(bps[vertexId].second == kMinScore+1, HypergraphException, "Landed at vertex " << vertexId << " which is a dead end");
  if (!bps[vertexId].first) return false;
  const Edge* prevEdge = bps[vertexId].first;
  bestHypo->featureVector += *(prevEdge->Features().get());
  size_t childId = 0;
  for (size_t i = 0; i < prevEdge->Words().size(); ++i) {
    if (prevEdge->Words()[i] != NULL) {
      bestHypo->text.push_back(prevEdge->Words()[i]);
    } else {
      size_t childVertexId = prevEdge->Children()[childId++];
      HgHypothesis childHypo;
      GetBestHypothesis(childVertexId,graph,bps,&childHypo);
      bestHypo->text.insert(bestHypo->text.end(), childHypo.text.begin(), childHypo.text.end());
      bestHypo->featureVector += childHypo.featureVector;
    }
  }
  return true;
}

typedef pair<size_t, size_t> Range ;
typedef map<Range, boost::shared_ptr<HgHypothesis> > VioColl;

void Viterbi(const Graph& graph, const SparseVector& weights, float bleuWeight, const ReferenceSet& references , size_t sentenceId, const std::vector<FeatureStatsType>& backgroundBleu,  VioColl& bestHypos)
{
  size_t size = graph.GetVertex(graph.VertexSize()-1).SourceCovered();
  //bestHypos.resize(size);
  //for(size_t i = 0; i < size; i++) {
  //  bestHypos[i].resize(size-i, HgHypothesis());
  //}

  //std::set<pair<size_t, size_t> > exists;

  BackPointer init(NULL,kMinScore);
  vector<BackPointer> backPointers(graph.VertexSize(),init);
  HgBleuScorer bleuScorer(references, graph, sentenceId, backgroundBleu);
  vector<FeatureStatsType> winnerStats(kBleuNgramOrder*2+1);
  for (size_t vi = 0; vi < graph.VertexSize(); ++vi) {
//    cerr << "vertex id " << vi <<  endl;
    FeatureStatsType winnerScore = kMinScore;
    const Vertex& vertex = graph.GetVertex(vi);
    const vector<const Edge*>& incoming = vertex.GetIncoming();
    if (!incoming.size()) {
      //UTIL_THROW(HypergraphException, "Vertex " << vi << " has no incoming edges");
      //If no incoming edges, vertex is a dead end
      backPointers[vi].first = NULL;
      backPointers[vi].second = kMinScore;
    } else {
      //cerr << "\nVertex: " << vi << endl;
      for (size_t ei = 0; ei < incoming.size(); ++ei) {
        //cerr << "edge id " << ei << endl;
        FeatureStatsType incomingScore = incoming[ei]->GetScore(weights);
        for (size_t i = 0; i < incoming[ei]->Children().size(); ++i) {
          size_t childId = incoming[ei]->Children()[i];
          //UTIL_THROW_IF(backPointers[childId].second == kMinScore,
          //  HypergraphException, "Graph was not topologically sorted. curr=" << vi << " prev=" << childId);
          incomingScore = max(incomingScore + backPointers[childId].second, kMinScore);
        }
        vector<FeatureStatsType> bleuStats(kBleuNgramOrder*2+1);
        // cerr << "Score: " << incomingScore << " Bleu: ";
        // if (incomingScore > nonbleuscore) {nonbleuscore = incomingScore; nonbleuid = ei;}
        FeatureStatsType totalScore = incomingScore;
        if (bleuWeight) {
          FeatureStatsType bleuScore = bleuScorer.Score(*(incoming[ei]), vertex, bleuStats);
          if (isnan(bleuScore)) {
            cerr << "WARN: bleu score undefined" << endl;
            cerr << "\tVertex id : " << vi << endl;
            cerr << "\tBleu stats : ";
            for (size_t i = 0; i < bleuStats.size(); ++i) {
              cerr << bleuStats[i] << ",";
            }
            cerr << endl;
            bleuScore = 0;
          }
          //UTIL_THROW_IF(isnan(bleuScore), util::Exception, "Bleu score undefined, smoothing problem?");
          totalScore += bleuWeight * bleuScore;
          //  cerr << bleuScore << " Total: " << incomingScore << endl << endl;
          //cerr << "is " << incomingScore << " bs " << bleuScore << endl;
        }
        if (totalScore >= winnerScore) {
          //We only store the feature score (not the bleu score) with the vertex,
          //since the bleu score is always cumulative, ie from counts for the whole span.
          winnerScore = totalScore;
          backPointers[vi].first = incoming[ei];
          backPointers[vi].second = incomingScore;
          winnerStats = bleuStats;
        }
      }
      //update with winner
      //if (bleuWeight) {
      //TODO: Not sure if we need this when computing max-model solution
      if (backPointers[vi].first) {
        bleuScorer.UpdateState(*(backPointers[vi].first), vi, winnerStats);
      }

    }
//    cerr  << "backpointer[" << vi << "] = (" << backPointers[vi].first << "," << backPointers[vi].second << ")" << endl;
    boost::shared_ptr<HgHypothesis> bestHypo (new HgHypothesis());
    bool flag = GetBestHypothesis(vi, graph, backPointers, bestHypo.get());
    if (!flag)
      continue;

    /*vector<size_t> feats = bestHypo.featureVector.feats();
    for(size_t i = 0; i < feats.size(); i++) {
      //assert(feats[i] < SparseVector::m_id_to_name.size());
      SparseVector::decode(feats[i]);
    }*/

    // update BLEU
    (*bestHypo).bleuStats.resize(kBleuNgramOrder*2+1);
    NgramCounter counts;
    list<WordVec> openNgrams;
    for (size_t i = 0; i < (*bestHypo).text.size(); ++i) {
      const Vocab::Entry* entry = (*bestHypo).text[i];
      if (graph.IsBoundary(entry)) continue;
      openNgrams.push_front(WordVec());
      for (list<WordVec>::iterator k = openNgrams.begin(); k != openNgrams.end();  ++k) {
        k->push_back(entry);
        ++counts[*k];
      }
      if (openNgrams.size() >=  kBleuNgramOrder) openNgrams.pop_back();
    }
    for (NgramCounter::const_iterator ngi = counts.begin(); ngi != counts.end(); ++ngi) {
      size_t order = ngi->first.size();
      size_t count = ngi->second;
      (*bestHypo).bleuStats[(order-1)*2 + 1] += count;
      (*bestHypo).bleuStats[(order-1) * 2] += min(count, references.NgramMatches(sentenceId,ngi->first,true));
    }
    (*bestHypo).bleuStats[kBleuNgramOrder*2] = references.Length(sentenceId);

    //
    size_t s = vertex.startPos;
    size_t e = vertex.endPos;

    Range r(s,e);

    VioColl::iterator iter = bestHypos.find(r);
    if (iter == bestHypos.end()  || inner_product((*iter->second).featureVector,weights) < inner_product((*bestHypo).featureVector,weights)) {
      bestHypos[r] = bestHypo;
      //bestHypo.featureVector.write(cerr, " "); cerr << endl;
    }
  }

  /*cerr << "BEGIN" << endl;
  map<Range, HgHypothesis >::iterator iter = bestHypos.begin();
  for(; iter != bestHypos.end(); iter++) {
    vector<size_t> feats = iter->second.featureVector.feats();
    for(size_t i = 0; i < feats.size(); i++) {
      //assert(feats[i] < SparseVector::m_id_to_name.size());
      SparseVector::decode(feats[i]);
    }
  }
  cerr << "END" << endl;*/
  //exit(1);

  //expand back pointers


  //bleu stats and fv

  //Need the actual (clipped) stats
  //TODO: This repeats code in bleu scorer - factor out
  /*bestHypo->bleuStats.resize(kBleuNgramOrder*2+1);
  NgramCounter counts;
  list<WordVec> openNgrams;
  for (size_t i = 0; i < bestHypo->text.size(); ++i) {
    const Vocab::Entry* entry = bestHypo->text[i];
    if (graph.IsBoundary(entry)) continue;
    openNgrams.push_front(WordVec());
    for (list<WordVec>::iterator k = openNgrams.begin(); k != openNgrams.end();  ++k) {
      k->push_back(entry);
      ++counts[*k];
    }
    if (openNgrams.size() >=  kBleuNgramOrder) openNgrams.pop_back();
  }
  for (NgramCounter::const_iterator ngi = counts.begin(); ngi != counts.end(); ++ngi) {
    size_t order = ngi->first.size();
    size_t count = ngi->second;
    bestHypo->bleuStats[(order-1)*2 + 1] += count;
    bestHypo->bleuStats[(order-1) * 2] += min(count, references.NgramMatches(sentenceId,ngi->first,true));
  }
  bestHypo->bleuStats[kBleuNgramOrder*2] = references.Length(sentenceId);*/
}


};
