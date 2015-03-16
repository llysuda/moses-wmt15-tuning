#ifndef MERT_RED_SCORER_H_
#define MERT_RED_SCORER_H_

#include <set>
#include <string>
#include <vector>

#ifdef WITH_THREADS
#include <boost/thread/mutex.hpp>
#endif

#include "Types.h"
#include "StatisticsBasedScorer.h"

namespace MosesTuning
{

class ofdstream;
class ifdstream;
class ScoreStats;

/**
 * Meteor scoring
 *
 * https://github.com/mjdenkowski/meteor
 * http://statmt.org/wmt11/pdf/WMT07.pdf
 *
 * Config:
 * jar - location of meteor-*.jar (meteor-1.4.jar at time of writing)
 * lang - optional language code (default: en)
 * task - optional task (default: tune)
 * m - optional quoted, space delimited module string "exact stem synonym paraphrase" (default varies by language)
 * p - optional quoted, space delimited parameter string "alpha beta gamma delta" (default for tune: "0.5 1.0 0.5 0.5")
 * w - optional quoted, space delimited weight string "w_exact w_stem w_synonym w_paraphrase" (default for tune: "1.0 0.5 0.5 0.5")
 *
 * Usage with mert-moses.pl:
 * --mertargs="--sctype METEOR --scconfig jar:/path/to/meteor-1.4.jar"
 */
class RedScorer: public StatisticsBasedScorer
{
public:
  explicit RedScorer(const std::string& config = "");
  ~RedScorer();

  virtual void setReferenceFiles(const std::vector<std::string>& referenceFiles);
  virtual void prepareStats(std::size_t sid, const std::string& text, ScoreStats& entry);

  virtual std::size_t NumberOfScores() const {
    // reflen count totalScore
    return 3;
  }

  virtual float getReferenceLength(const std::vector<ScoreStatsType>& totals) const {
     return totals[0];
  }

  virtual float calculateScore(const std::vector<ScoreStatsType>& comps) const;

private:
  // Meteor and process IO
  std::string stat_file;
  std::vector<std::vector<std::string> > m_stats;
  int m_currIndex;
  int m_prevSid;

  // data extracted from reference files
  std::vector<std::string> m_references;
  std::vector<std::vector<std::string> > m_multi_references;

  // no copying allowed
  RedScorer(const RedScorer&);
  RedScorer& operator=(const RedScorer&);

};

}

#endif // MERT_METEOR_SCORER_H_
