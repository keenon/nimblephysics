#include "dart/utils/amc/Reader.hpp"

namespace Reader {

BaseReader::BaseReader() : stream(NULL)
{
}

// for manipulating seperator list
void BaseReader::set_seperators(set<char> const& sep)
{
  assert(!stream);
  seperators = sep;
}

void BaseReader::add_seperator(char s)
{
  assert(!stream);
  if (seperators.find(s) == seperators.end())
  {
    seperators.insert(s);
  }
}

void BaseReader::clear_seperators()
{
  seperators.clear();
}

// manipulate stream state
void BaseReader::set_stream(istream& str)
{
  stream = &str;
  token_list.clear();
  token_list_stack.clear();
}

void BaseReader::clear_stream()
{
  stream = NULL;
}

// get tokens
bool BaseReader::get_token(string& into)
{
  char c;
  string acc = "";
  bool eol = false;
  while (token_list.empty() && stream && stream->get(c))
  {
    if (c == '#')
    {
      while (stream && stream->get(c) && c != '\n')
      {
        /* keep grabbing */
      }
      eol = true;
      break;
    }
    else if (seperators.find(c) == seperators.end())
    {
      acc += c;
    }
    else if (c == '\n')
    {
      eol = true;
      break;
    }
    else
    {
      if (acc.size())
      {
        stream->putback((char)c);
        break;
      }
    }
  }
  if (acc.size())
  {
    token_list.push_back(acc);
    for (unsigned int i = 0; i < token_list_stack.size(); ++i)
    {
      token_list_stack[i].push_back(acc);
    }
  }
  if (eol)
  {
    token_list.push_back("E O L");
    for (unsigned int i = 0; i < token_list_stack.size(); ++i)
    {
      token_list_stack[i].push_back("E O L");
    }
  }
  if (!token_list.empty())
  {
    into = token_list.front();
    token_list.pop_front();
    return true;
  }
  return false;
}

bool BaseReader::get_token_noeol(string& into)
{
  bool got_token = false;
  while ((got_token = get_token(into)) && into == "E O L")
  {
    // keep pulling
  }
  if (got_token)
  {
    return true;
  }
  into = "";
  return false;
}

// save and restore token list state
void BaseReader::push_token_list()
{
  token_list_stack.push_back(token_list);
}

void BaseReader::restore_token_list()
{
  assert(!token_list_stack.empty());
  token_list = token_list_stack.back();
  token_list_stack.pop_back();
}

void BaseReader::ignore_token_list()
{
  assert(!token_list_stack.empty());
  token_list_stack.pop_back();
}

}; // namespace Reader
