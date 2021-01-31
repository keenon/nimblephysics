#ifndef AMC_READER_HPP
#define AMC_READER_HPP

#include <deque>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <assert.h>

using std::cerr;
using std::deque;
using std::endl;
using std::istream;
using std::istringstream;
using std::make_pair;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace Reader {

class BaseReader
{
public:
  BaseReader();

  // for manipulating seperator list
  void set_seperators(set<char> const&);
  void add_seperator(char);
  void clear_seperators();

  // manipulate stream state
  void set_stream(istream& stream);
  void clear_stream();

  // get tokens
  bool get_token(string& into);
  bool get_token_noeol(string& into);

  // save and restore token list state
  void push_token_list();
  void restore_token_list();
  void ignore_token_list();

private:
  set<char> seperators;
  istream* stream;
  deque<string> token_list;
  vector<deque<string> > token_list_stack;
};

template <typename OBJ>
class Reader;

class BaseMatchData
{
public:
  virtual ~BaseMatchData()
  {
  }
  // nothing.
private:
  // to make class polymorphic.
  virtual void _dummy()
  {
  }
};

class BasePattern
{
public:
  virtual ~BasePattern()
  {
  }
  // attempt to read the pattern from the MotionReader
  virtual BaseMatchData* operator()(BaseReader& from) = 0;
};

class NullPattern : public BasePattern
{
public:
  virtual BaseMatchData* operator()(BaseReader& /* from */)
  {
    static NullPattern::MatchData data;
    return &data;
  }
  class MatchData : public BaseMatchData
  {
  public:
    // empty
  };
  static BasePattern* get_instance()
  {
    static NullPattern pat;
    return &pat;
  }
};

template <typename PATA, typename PATB>
class PairPattern : public BasePattern
{
public:
  virtual BaseMatchData* operator()(BaseReader& from)
  {
    static typename PairPattern<PATA, PATB>::MatchData data;
    from.push_token_list();
    typename PATA::MatchData* mata = dynamic_cast<typename PATA::MatchData*>(
        (*PATA::get_instance())(from));
    typename PATB::MatchData* matb = dynamic_cast<typename PATB::MatchData*>(
        (*PATB::get_instance())(from));
    if (mata && matb)
    {
      from.ignore_token_list();
      data.first = *mata;
      data.second = *matb;
      return &data;
    }
    from.restore_token_list();
    return NULL;
  }
  class MatchData : public BaseMatchData
  {
  public:
    typename PATA::MatchData first;
    typename PATB::MatchData second;
  };
  static BasePattern* get_instance()
  {
    static PairPattern<PATA, PATB> pat;
    return &pat;
  }
};

template <typename TYPE>
class TypePattern : public BasePattern
{
public:
  virtual BaseMatchData* operator()(BaseReader& from)
  {
    static typename TypePattern<TYPE>::MatchData data;
    string token;
    from.push_token_list();
    if (from.get_token(token) && token != "E O L")
    {
      istringstream in(token);
      if (in >> data.mat)
      {
        from.ignore_token_list();
        return &data;
      }
      else
      {
        cerr << "Token '" << token << "' failed to match type." << endl;
      }
    }
    from.restore_token_list();
    return NULL;
  }
  class MatchData : public BaseMatchData
  {
  public:
    TYPE mat;
  };
  static BasePattern* get_instance()
  {
    static TypePattern<TYPE> pat;
    return &pat;
  }
};

template <class PAT, int MIN, int MAX>
class WildPattern : public BasePattern
{
public:
  virtual BaseMatchData* operator()(BaseReader& from)
  {
    static typename WildPattern<PAT, MIN, MAX>::MatchData data;
    from.push_token_list();

    vector<typename PAT::MatchData> vec;

    typename PAT::MatchData* match;
    while ((MAX < 0 || (signed)vec.size() < MAX)
           && (match = dynamic_cast<typename PAT::MatchData*>(
                   (*PAT::get_instance())(from))))
    {
      vec.push_back(*match);
    }

    if (MIN < 0 || (signed)vec.size() >= MIN)
    {
      data.data = vec;
      from.ignore_token_list();
      return &data;
    }

    from.restore_token_list();
    return NULL;
  }
  class MatchData : public BaseMatchData
  {
  public:
    vector<typename PAT::MatchData> data;
  };
  static BasePattern* get_instance()
  {
    static WildPattern<PAT, MIN, MAX> pat;
    return &pat;
  };
};

template <class PAT>
class StarPattern : public WildPattern<PAT, -1, -1>
{
  /* nothing to add */
};

template <class PAT, int NUM>
class VectorPattern : public WildPattern<PAT, NUM, NUM>
{
  /* nothing to add */
};

template <class OBJ>
class BaseHandler
{
public:
  virtual ~BaseHandler()
  {
  }
  virtual bool operator()(
      BaseMatchData* data, OBJ& out, Reader<OBJ> const& reader) const = 0;
};

template <class PAT, class OBJ>
class BasePatternHandler : public BaseHandler<OBJ>
{
public:
  // a PatternHandler takes match data and sticks it into an object
  virtual bool operator()(
      BaseMatchData* data, OBJ& obj, Reader<OBJ> const& reader) const
  {
    typename PAT::MatchData* tdata
        = dynamic_cast<typename PAT::MatchData*>(data);
    if (tdata)
    {
      return use_data(*tdata, obj, reader);
    }
    else
    {
      return false;
    }
  }
  virtual bool use_data(
      typename PAT::MatchData& data,
      OBJ& obj,
      Reader<OBJ> const& reader) const = 0;
};

template <class PAT, class OBJ>
class IgnorePatternHandler : public BasePatternHandler<PAT, OBJ>
{
public:
  virtual bool use_data(
      typename PAT::MatchData& /* data */,
      OBJ& /* obj */,
      Reader<OBJ> const& /* reader */) const
  {
    /* nothing, just ignore */
    return true;
  }
};

template <typename OBJ>
class Reader : public BaseReader
{
public:
  typedef pair<BasePattern*, BaseHandler<OBJ>*> HandlerInfo;

  template <class PAT>
  void set_section_handler(
      string const& section, BasePatternHandler<PAT, OBJ>* handler)
  {
    set_section_handler(section, handler, PAT::get_instance());
  }

  void set_section_handler(
      string const& section, BaseHandler<OBJ>* handler, BasePattern* pattern)
  {
    assert(handler);
    assert(pattern);
    if (section_handlers.count(section))
    {
      section_handlers[section] = make_pair(pattern, handler);
    }
    else
    {
      section_handlers.insert(make_pair(section, make_pair(pattern, handler)));
    }
  }

  void clear_section_handler(string const& section)
  {
    if (section_handlers.count(section))
    {
      section_handlers.erase(section_handlers.find(section));
    }
  }

  HandlerInfo get_section_handler(string const& section)
  {
    if (section_handlers.count(section))
    {
      return section_handlers[section];
    }
    else if (section_handlers.count(""))
    {
      return section_handlers[""];
    }
    else
    {
      return make_pair((BasePattern*)NULL, (BaseHandler<OBJ>*)NULL);
    }
  }

  template <class PAT>
  void set_handler(
      string const& section,
      string const& keyword,
      BasePatternHandler<PAT, OBJ>* handler)
  {
    set_handler(section, keyword, handler, PAT::get_instance());
  }

  void set_handler(
      string const& section,
      string const& keyword,
      BaseHandler<OBJ>* handler,
      BasePattern* pattern)
  {
    assert(handler);
    assert(pattern);
    if (!data_handlers.count(section))
    {
      data_handlers.insert(make_pair(section, map<string, HandlerInfo>()));
    }
    if (!data_handlers[section].count(keyword))
    {
      data_handlers[section].insert(
          make_pair(keyword, make_pair(pattern, handler)));
    }
    else
    {
      data_handlers[section][keyword] = make_pair(pattern, handler);
    }
  }

  void clear_handler(string const& section, string const& keyword)
  {
    if (data_handlers.count(section) && data_handlers[section].count(keyword))
    {
      data_handlers[section].erase(data_handlers[section].find(keyword));
    }
  }

  void clear_handlers(string const& section)
  {
    if (data_handlers.count(section))
    {
      data_handlers.erase(data_handlers.find(section));
    }
  }

  HandlerInfo get_handler(string const& section, string const& keyword)
  {
    if (data_handlers.count(section) && data_handlers[section].count(keyword))
    {
      return data_handlers[section][keyword];
    }
    else if (data_handlers.count(section) && data_handlers[section].count(""))
    {
      return data_handlers[section][""];
    }
    else if (data_handlers.count("") && data_handlers[""].count(keyword))
    {
      return data_handlers[""][keyword];
    }
    else if (data_handlers.count("") && data_handlers[""].count(""))
    {
      return data_handlers[""][""];
    }
    else
    {
      return make_pair((BasePattern*)NULL, (BaseHandler<OBJ>*)NULL);
    }
  }

  bool parse(istream& str, OBJ& out)
  {
    // establish context
    set_stream(str);

    current_section = "";
    current_keyword = "";

    bool fine = true;

    string token;
    while (get_token_noeol(token))
    {
      assert(token.size());
      if (token[0] == ':')
      {
        // think that this is a section marker
        current_section = token;
        HandlerInfo i = get_section_handler(current_section);
        if (i.first)
        {
          BaseMatchData* match = (*i.first)(*this);
          if (match)
          {
            assert(i.second);
            if (!(*i.second)(match, out, *this))
            {
              fine = false;
            }
          }
          else
          {
            cerr << "Section " << current_section << " doesn't match pattern."
                 << endl;
            fine = false;
          }
        }
        else
        {
          cerr << "No handler for section marker '" << current_section << "'."
               << endl;
        }
      }
      else
      {
        // think that this is probably a keyword
        current_keyword = token;
        HandlerInfo i = get_handler(current_section, current_keyword);
        if (i.first)
        {
          BaseMatchData* match = (*i.first)(*this);
          if (match)
          {
            assert(i.second);
            if (!(*i.second)(match, out, *this))
            {
              fine = false;
            }
          }
          else
          {
            cerr << "Section " << current_section << ", keyword "
                 << current_keyword << " doesn't match pattern." << endl;
            fine = false;
          }
        }
        else
        {
          cerr << "No handler for section '" << current_section
               << "', keyword '" << current_keyword << "'." << endl;
        }
      }
    }

    // bye, bye context!
    clear_stream();

    return fine;
  }

  map<string, HandlerInfo> section_handlers;
  map<string, map<string, HandlerInfo> > data_handlers;

  string current_section;
  string current_keyword;
};

} // namespace Reader

#endif // MOTIONREADER_HPP
