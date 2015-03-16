#include "RedScorer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <vector>

#include <boost/thread/mutex.hpp>

#include "ScoreStats.h"
#include "Util.h"

#include "util/file_piece.hh"
#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"
#include "FeatureDataIterator.h"

using namespace std;

namespace MosesTuning
{

RedScorer::RedScorer(const string& config)
  : StatisticsBasedScorer("RED",config)
{
  m_currIndex = 0;
  m_prevSid = 0;

  stat_file = getConfig("stat", "");
  if (stat_file == "") {
    //throw runtime_error("stat file is required: --scconfig stat:stat_file");
    cerr << "no stat file specified" << endl;
    return;
  }

  TRACE_ERR("loading nbest stats from " << stat_file << endl);
  util::FilePiece in(stat_file.c_str());

  string sentence;
  int sentence_index;
  int prev_index = -1;

  while (true) {
    try {
      StringPiece line = in.ReadLine();
      if (line.empty()) continue;

      util::TokenIter<util::MultiCharacter> it(line, util::MultiCharacter("|||"));

      sentence_index = ParseInt(*it);
      ++it;
      sentence = it->as_string();
      ++it;

      if (sentence_index != prev_index) {
        m_stats.push_back(vector<string>());
        prev_index = sentence_index;
      }

      m_stats[sentence_index].push_back(sentence);

    } catch (util::EndOfFileException &e) {
      PrintUserTime("Loaded N-best stats");
      break;
    }
  }
}

RedScorer::~RedScorer()
{
  // Cleanup IO
}

void RedScorer::setReferenceFiles(const vector<string>& referenceFiles)
{
  // Just store strings since we're sending lines to an external process
  for (int incRefs = 0; incRefs < (int)referenceFiles.size(); incRefs++) {
    m_references.clear();
    ifstream in(referenceFiles.at(incRefs).c_str());
    if (!in) {
      throw runtime_error("Unable to open " + referenceFiles.at(incRefs));
    }
    string line;
    while (getline(in, line)) {
      line = this->preprocessSentence(line);
      m_references.push_back(line);
    }
    m_multi_references.push_back(m_references);
  }
  m_references=m_multi_references.at(0);
}

void RedScorer::prepareStats(size_t sid, const string& text, ScoreStats& entry)
{
  if (sid != m_prevSid) {
    m_currIndex = 0;
    m_prevSid = sid;
  }

  string sentence = this->preprocessSentence(text);
  string stats_str = m_stats[sid][m_currIndex];
  entry.set(stats_str);
  m_currIndex++;
}

float RedScorer::calculateScore(const vector<ScoreStatsType>& comps) const
{
  if (comps.size() != NumberOfScores()) {
    throw runtime_error("RED stats num mis-match !");
  }
  ScoreStatsType score = comps[2]/comps[1];
  return score;
}


}
