#include "dart/utils/CSVParser.hpp"

#include "dart/common/Uri.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/SimmSpline.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

namespace dart {
namespace utils {

/// SkelParser
namespace CSVParser {

//==============================================================================
common::ResourceRetrieverPtr ensureRetriever(
    const common::ResourceRetrieverPtr& _retriever)
{
  if (_retriever)
  {
    return _retriever;
  }
  else
  {
    auto newRetriever = std::make_shared<utils::CompositeResourceRetriever>();
    newRetriever->addSchemaRetriever(
        "file", std::make_shared<common::LocalResourceRetriever>());
    newRetriever->addSchemaRetriever("dart", DartResourceRetriever::create());
    return newRetriever;
  }
}

/// Read World from skel file
std::vector<std::map<std::string, std::string>> parseFile(
    const common::Uri& uri, const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  std::string content = retriever->readAll(uri);

  std::vector<std::string> columnNames;
  std::vector<std::map<std::string, std::string>> values;

  int lineNumber = 0;
  auto start = 0U;
  auto end = content.find("\n");
  while (end != std::string::npos)
  {
    std::string line = content.substr(start, end - start);

    std::map<std::string, std::string> rowValues;

    int tokenNumber = 0;
    auto tokenStart = 0U;
    auto tokenEnd = line.find(",");
    while (tokenEnd != std::string::npos)
    {
      assert(tokenStart < line.size());
      std::string token = line.substr(tokenStart, tokenEnd - tokenStart);

      /////////////////////////////////////////////////////////
      // Process the token, given tokenNumber and lineNumber

      if (lineNumber == 0)
      {
        columnNames.push_back(token);
      }
      else
      {
        rowValues[columnNames[tokenNumber]] = token;
      }

      /////////////////////////////////////////////////////////

      tokenStart = tokenEnd + 1; // ",".length()
      tokenEnd = line.find(",", tokenStart);
      tokenNumber++;
    }

    if (lineNumber > 0)
    {
      values.push_back(rowValues);
    }

    start = end + 1; // "\n".length()
    end = content.find("\n", start);
    lineNumber++;
  }

  return values;
};

} // namespace CSVParser

} // namespace utils
} // namespace dart
