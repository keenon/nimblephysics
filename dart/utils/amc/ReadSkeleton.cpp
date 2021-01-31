#include "dart/utils/amc/ReadSkeleton.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <set>

#include <arpa/inet.h>
#include <assert.h>

#include "dart/utils/amc/Reader.hpp"

using namespace Library;
using std::cerr;
using std::endl;
using std::getline;
using std::ifstream;
using std::set;
using std::transform;

Reader::Reader<Skeleton>& get_reader();

bool ReadSkeleton(string filename, Skeleton& into)
{
  // quick hack to load .vsk's instead
  if (filename.substr(filename.size() - 4, 4) == ".vsk"
      || filename.substr(filename.size() - 4, 4) == ".VSK")
  {
#ifdef LIBRARY_USE_VFILE
    return ReadSkeletonV(filename, into);
#else
    cerr << "In this compile, vsk files are not supported, sorry." << endl;
    return false;
#endif
  }

  Reader::Reader<Skeleton>& reader = get_reader();
  ifstream file(filename.c_str());
  if (!file)
  {
    return false;
  }
  into.filename = filename;
  into.init_parse();
  bool ret = reader.parse(file, into);
  return ret && into.check_parse();
}

bool ReadAnimationBin(
    string filename, Skeleton const& skeleton, vector<double>& positions)
{
  ifstream file(filename.c_str());
  if (!file)
    return false;
  { // check magic number.
    char magic[4];
    if (!file.read(magic, 4)
        || !(
            magic[0] == 'b' && magic[1] == 'm' && magic[2] == 'c'
            && magic[3] == 'D'))
    {
      cerr << "Magic number doesn't match for file '" << filename << "'."
           << endl;
      return false;
    }
  }
  unsigned int frames = 0;
  if (!file.read((char*)(&frames), 4))
  {
    cerr << "Couldn't read frames count." << endl;
    return false;
  }
  frames = ntohl(frames);
  { // check 'skel' chunk header.
    char head[4];
    if (!file.read(head, 4)
        || !(
            head[0] == 's' && head[1] == 'k' && head[2] == 'e'
            && head[3] == 'l'))
    {
      cerr << "No skeleton chunk header." << endl;
      return false;
    }
  }
  { // check names in skeleton chunk versus skeleton.
    unsigned int size = 0;
    if (!file.read((char*)(&size), 4))
    {
      cerr << "Couldn't read skeleton chunk size." << endl;
      return false;
    }
    size = ntohl(size);
    if (size < 8)
    {
      cerr << "chunk size too small" << endl;
      return false;
    }
    vector<char> expected;
    expected.push_back('r');
    expected.push_back('o');
    expected.push_back('o');
    expected.push_back('t');
    expected.push_back('\0');
    unsigned int net_six = htonl(6);
    expected.push_back(((char*)(&net_six))[0]);
    expected.push_back(((char*)(&net_six))[1]);
    expected.push_back(((char*)(&net_six))[2]);
    expected.push_back(((char*)(&net_six))[3]);
    int ofs = 6;
    for (unsigned int b = 0; b < skeleton.bones.size(); ++b)
    {
      for (unsigned int i = 0; i < skeleton.bones[b].name.size(); ++i)
      {
        expected.push_back(skeleton.bones[b].name[i]);
      }
      expected.push_back('\0');
      unsigned int net_dof = htonl(skeleton.bones[b].dof.size());
      expected.push_back(((char*)(&net_dof))[0]);
      expected.push_back(((char*)(&net_dof))[1]);
      expected.push_back(((char*)(&net_dof))[2]);
      expected.push_back(((char*)(&net_dof))[3]);
      if (skeleton.bones[b].frame_offset != ofs)
      {
        cerr << "Frame offsets not in order. This reader will be incompatible!"
             << endl;
        return false;
      }
      ofs += skeleton.bones[b].dof.size();
    }
    if (expected.size() + 8 != size)
    {
      cerr << "Expected skeleton chunk (" << expected.size() + 8
           << ") not the same size as skeleton chunk (" << size << ")." << endl;
      return false;
    }
    vector<char> test(expected.size());
    if (!file.read(&(test[0]), test.size()))
    {
      cerr << "Cannot read skeleton chunk." << endl;
      return false;
    }
    for (unsigned int i = 0; i < expected.size(); ++i)
    {
      if (test[i] != expected[i])
      {
        cerr << "Expected skeleton chunk does not match." << endl;
        for (unsigned int j = 0; j < expected.size(); ++j)
        {
          if (test[j] >= 'a' && test[j] <= 'z')
          {
            cerr << test[j];
          }
          else
          {
            cerr << '.';
          }
          if (expected[j] >= 'a' && expected[j] <= 'z')
          {
            cerr << expected[j];
          }
          else
          {
            cerr << '.';
          }
          cerr << " " << (int)test[j] << " " << (int)test[j];
          if (test[j] != expected[j])
          {
            cerr << " XXX";
          }
          cerr << endl;
        }
        return false;
      }
    }
  }
  positions.resize(frames * skeleton.frame_size);
  // finally, let's read some frames.
  for (unsigned int f = 0; f < frames; ++f)
  {
    { // frame header.
      char head[4];
      if (!file.read(head, 4))
      {
        cerr << "No frame chunk header (frame " << f + 1 << " of " << frames
             << ")." << endl;
        return false;
      }
      else if (!(head[0] == 'f' && head[1] == 'r' && head[2] == 'a'
                 && head[3] == 'm'))
      {
        cerr << "Improper frame chunk header:'" << head[0] << head[1] << head[2]
             << head[3] << "'." << endl;
        return false;
      }
    }
    { // frame number
      unsigned int number = 0;
      if (!file.read((char*)&number, 4) || ntohl(number) != f)
      {
        cerr << "Couldn't read frame number or wrong frame number." << endl;
        return false;
      }
    }
    { // frame itself.
      if (!file.read(
              (char*)&positions[f * skeleton.frame_size],
              skeleton.frame_size * sizeof(double)))
      {
        cerr << "Couldn't read frame." << endl;
        return false;
      }
      // This is kind of ugly; maybe I should go length-agnostic or something.
      for (unsigned int i = 0; i < skeleton.order.size(); ++i)
      {
        if (skeleton.order[i] != tolower(skeleton.order[i]))
        {
          positions[f * skeleton.frame_size + i] *= skeleton.length;
        }
      }
    }
  }
  if (file.get() && file)
  {
    cerr << "WARNING: trailing data in bmc. Maybe frame count was wrong?"
         << endl;
  }
  return true;
}

bool ReadAnimation(
    string filename, Skeleton const& on, vector<double>& positions)
{
  // quick hack to load .v's
  if (filename.substr(filename.size() - 2, 2) == ".v"
      || filename.substr(filename.size() - 2, 2) == ".V")
  {
#ifdef LIBRARY_USE_VFILE
    return ReadAnimationV(filename, on, positions);
#else
    cerr << "In this compile, .v files are not supported, sorry." << endl;
    return false;
#endif
  }
  if (filename.size() >= 4
      && (filename.substr(filename.size() - 4, 4) == ".amc"))
  {
    string temp = filename;
    temp[temp.size() - 3] = 'b';
    if (ReadAnimationBin(temp, on, positions))
    {
      cerr << "Using .bmc version of '" << filename << "'" << endl;
      return true;
    }
  }
  if (filename.size() >= 4
      && (filename.substr(filename.size() - 4, 4) == ".bmc"))
  {
    return ReadAnimationBin(filename, on, positions);
  }

  map<string, int> bone_map;
  for (unsigned int b = 0; b < on.bones.size(); ++b)
  {
    bone_map.insert(make_pair(on.bones[b].name, b));
  }

  // clear this. woo.
  positions.clear();

  // suddenly, Jim gets bored of using 'Reader'.
  ifstream file(filename.c_str());
  if (!file)
  {
    return false;
  }
  string line_in;
  int current_frame = -1;
  int dof_read = 0;
  while (getline(file, line_in))
  {
    string tline = "";
    for (unsigned int i = 0; i < line_in.size(); ++i)
    {
      if (line_in[i] == '#')
        break;
      tline += line_in[i];
    }
    istringstream line(tline);
    string tok;
    if (line >> tok)
    {
      if (tok[0] != ':')
      {
        if (tok == "root")
        {
          if (current_frame < 0)
          {
            cerr << "We started getting bone data outside a frame." << endl;
            return false;
          }
          int p = current_frame * on.frame_size;
          int read = 0;
          double info;
          while (line >> info)
          {
            if (p + read > (signed)positions.size())
            {
              cerr << "Overflow while reading." << endl;
              return false;
            }
            positions[p + read] = info;
            if (on.order[read] != tolower(on.order[read]))
            {
              positions[p + read] *= on.length;
            }
            else if (!on.ang_is_deg)
            {
              positions[p + read] *= 180.0 / M_PI;
            }
            ++read;
          }
          if ((unsigned)read != 6)
          {
            cerr << "We read " << read << " things but were expecting " << 6
                 << " things for root." << endl;
            return false;
          }
          dof_read += read;
        }
        else if (bone_map.count(tok))
        {
          if (current_frame < 0)
          {
            cerr << "We started getting bone data outside a frame." << endl;
            return false;
          }
          int b = bone_map[tok];
          int p = current_frame * on.frame_size + on.bones[b].frame_offset;
          int read = 0;
          double info;
          while (line >> info)
          {
            if (p + read > (signed)positions.size())
            {
              cerr << "Overflow while reading." << endl;
              return false;
            }
            positions[p + read] = info;
            ++read;
          }
          if ((unsigned)read != on.bones[b].dof.size())
          {
            cerr << "We read " << read << " things but were expecting "
                 << on.bones[b].dof.size() << " things for bone "
                 << on.bones[b].name << "." << endl;
            return false;
          }
          dof_read += read;
        }
        else
        {
          int num;
          istringstream nums(tok);
          if (nums >> num)
          {
            current_frame += 1;
            if (current_frame != 0 && dof_read != on.frame_size)
            {
              cerr << "We read only " << dof_read << " of the total "
                   << on.frame_size << " things we wanted." << endl;
              return false;
            }
            dof_read = 0;
            for (int i = 0; i < on.frame_size; ++i)
            {
              positions.push_back(0.0);
            }
          }
          else
          {
            cerr << "We got '" << tok
                 << "' which doesn't appear to be a frame number or bone name."
                 << endl;
            return false;
          }
        }
      }
    }
  }
  return true;
}

class OrderTokenPattern : public Reader::BasePattern
{
public:
  virtual Reader::BaseMatchData* operator()(Reader::BaseReader& from)
  {
    static MatchData data;
    from.push_token_list();
    string tok;
    if (!from.get_token(tok))
    {
      from.restore_token_list();
      return nullptr;
    }
    if (tok.size() == 2)
    {
      int ind = 0;
      if (tok[0] == 't' || tok[0] == 'T')
      {
        ind = 0;
      }
      else if (tok[0] == 'r' || tok[0] == 'R')
      {
        ind = 3;
      }
      else
      {
        ind = 6;
      }
      if (tok[1] == 'x' || tok[1] == 'X')
      {
        ind += 0;
      }
      else if (tok[1] == 'y' || tok[1] == 'Y')
      {
        ind += 1;
      }
      else if (tok[1] == 'z' || tok[1] == 'Z')
      {
        ind += 2;
      }
      else
      {
        ind += 6;
      }
      if (ind < 6 && ind >= 0)
      {
        static char ident[] = "XYZxyz";
        data.ident = ident[ind];
        from.ignore_token_list();
        return &data;
      }
    }
    cerr << "Order token '" << tok << "' unrecognized." << endl;
    from.restore_token_list();
    return NULL;
  }
  class MatchData : public Reader::BaseMatchData
  {
  public:
    char ident;
  };
  static BasePattern* get_instance()
  {
    static OrderTokenPattern pat;
    return &pat;
  }
};

class OrderHandler : public Reader::BasePatternHandler<
                         Reader::VectorPattern<OrderTokenPattern, 6>,
                         Skeleton>
{
public:
  virtual bool use_data(
      Reader::VectorPattern<OrderTokenPattern, 6>::MatchData& data,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    obj.order = "";
    set<char> seen;
    for (unsigned int i = 0; i < data.data.size(); ++i)
    {
      char const& c = data.data[i].ident;
      if (seen.count(c))
      {
        cerr << "Order identifier " << c << " appears more than once." << endl;
        return false;
      }
      obj.order += c;
      seen.insert(c);
    }
    return true;
  }
};

class RootAxisHandler
  : public Reader::BasePatternHandler<Reader::TypePattern<string>, Skeleton>
{
public:
  virtual bool use_data(
      Reader::TypePattern<string>::MatchData& data,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    obj.offset_order = data.mat;
    transform(
        obj.offset_order.begin(),
        obj.offset_order.end(),
        obj.offset_order.begin(),
        tolower);
    return true;
  }
};

template <typename TYPE>
class TypeHandler
  : public Reader::BasePatternHandler<Reader::TypePattern<TYPE>, Skeleton>
{
  virtual bool use_data(
      typename Reader::TypePattern<TYPE>::MatchData& data,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    TYPE& t = get_storage(obj);
    t = data.mat;
    postprocess(obj, t);
    return true;
  }
  virtual TYPE& get_storage(Skeleton& obj) const = 0;
  virtual void postprocess(Skeleton& /* obj */, TYPE& /* t */) const
  {
  }
};

class UnitsMassHandler : public TypeHandler<double>
{
  virtual double& get_storage(Skeleton& obj) const
  {
    return obj.mass;
  }
};

class UnitsLengthHandler : public TypeHandler<double>
{
  virtual double& get_storage(Skeleton& obj) const
  {
    return obj.length;
  }
};

class UnitsTimestepHandler : public TypeHandler<double>
{
  virtual double& get_storage(Skeleton& obj) const
  {
    return obj.timestep;
  }
};

class BoneDataIdHandler : public TypeHandler<int>
{
  virtual int& get_storage(Skeleton& obj) const
  {
    if (!obj.in_bone)
    {
      static int temp;
      cerr << "Got an id outside of begin/end pair." << endl;
      return temp;
    }
    else
    {
      static int temp;
      // we actually ignore the Id anyway.
      return temp;
    }
  }
};

class BoneDataNameHandler : public TypeHandler<string>
{
  virtual string& get_storage(Skeleton& obj) const
  {
    if (!obj.in_bone)
    {
      static string temp;
      cerr << "Got a name outside of begin/end pair." << endl;
      return temp;
    }
    else
    {
      return obj.bones.back().name;
    }
  }
};

class BoneDataRadiusHandler : public TypeHandler<double>
{
  virtual double& get_storage(Skeleton& obj) const
  {
    if (!obj.in_bone)
    {
      static double temp;
      cerr << "Got a radius outside of begin/end pair." << endl;
      return temp;
    }
    else
    {
      return obj.bones.back().radius;
    }
  }
};

class BoneDataLengthHandler : public TypeHandler<double>
{
  virtual double& get_storage(Skeleton& obj) const
  {
    if (!obj.in_bone)
    {
      static double temp;
      cerr << "Got a length outside of begin/end pair." << endl;
      return temp;
    }
    else
    {
      return obj.bones.back().length;
    }
  }
  virtual void postprocess(Skeleton& obj, double& t) const
  {
    t *= obj.length;
  }
};

class BoneDataDensityHandler : public TypeHandler<double>
{
  virtual double& get_storage(Skeleton& obj) const
  {
    if (!obj.in_bone)
    {
      static double temp;
      cerr << "Got a density outside of begin/end pair." << endl;
      return temp;
    }
    else
    {
      return obj.bones.back().density;
    }
  }
};

class UnitsAngleHandler
  : public Reader::BasePatternHandler<Reader::TypePattern<string>, Skeleton>
{
  virtual bool use_data(
      Reader::TypePattern<string>::MatchData& data,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    if (data.mat == "deg")
    {
      obj.ang_is_deg = true;
    }
    else if (data.mat == "rad")
    {
      obj.ang_is_deg = false;
    }
    else
    {
      cerr << "Expecting 'rad' or 'deg' for angle. Got '" << data.mat << "'"
           << endl;
    }
    return true;
  }
};

template <typename NUM, int size>
class VectorHandler : public Reader::BasePatternHandler<
                          Reader::VectorPattern<Reader::TypePattern<NUM>, size>,
                          Skeleton>
{
  virtual bool use_data(
      typename Reader::VectorPattern<Reader::TypePattern<NUM>, size>::MatchData&
          data,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    assert(data.data.size() == size);
    Vector<NUM, size>& store = get_storage(obj);
    for (int i = 0; i < size; ++i)
    {
      store[i] = data.data[i].mat;
    }
    postprocess(obj, store);
    return true;
  }
  virtual Vector<NUM, size>& get_storage(Skeleton& obj) const = 0;
  virtual void postprocess(
      Skeleton& /* obj */, Vector<NUM, size>& /* storage */) const
  {
  }
};

class RootOrientationHandler : public VectorHandler<double, 3>
{
  virtual Vector<double, 3>& get_storage(Skeleton& obj) const
  {
    return obj.axis_offset;
  }
  virtual void postprocess(Skeleton& obj, Vector<double, 3>& storage) const
  {
    if (!obj.ang_is_deg)
    {
      for (int i = 0; i < 3; ++i)
      {
        storage[i] *= 180.0 / M_PI;
      }
    }
  }
};

class RootPositionHandler : public VectorHandler<double, 3>
{
  virtual Vector<double, 3>& get_storage(Skeleton& obj) const
  {
    return obj.position;
  }
};

class BoneDataDirectionHandler : public VectorHandler<double, 3>
{
  virtual Vector<double, 3>& get_storage(Skeleton& obj) const
  {
    if (!obj.in_bone)
    {
      static Vector3d temp;
      cerr << "Got a direction while not in a begin/end pair" << endl;
      return temp;
    }
    else
    {
      return obj.bones.back().direction;
    }
  }
};

class BoneDataAxisHandler
  : public Reader::BasePatternHandler<
        Reader::PairPattern<
            Reader::VectorPattern<Reader::TypePattern<double>, 3>,
            Reader::TypePattern<string> >,
        Skeleton>
{
public:
  virtual bool use_data(
      Reader::PairPattern<
          Reader::VectorPattern<Reader::TypePattern<double>, 3>,
          Reader::TypePattern<string> >::MatchData& data,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    if (!obj.in_bone)
    {
      cerr << "Looks like we have an axis not in a bone. Ignoring." << endl;
      return false;
    }
    else
    {
      assert(data.first.data.size() == 3);
      obj.bones.back().axis_offset[0] = data.first.data[0].mat;
      obj.bones.back().axis_offset[1] = data.first.data[1].mat;
      obj.bones.back().axis_offset[2] = data.first.data[2].mat;
      if (!obj.ang_is_deg)
      {
        for (int i = 0; i < 3; ++i)
        {
          obj.bones.back().axis_offset[i] *= M_PI / 180.0;
        }
      }
      obj.bones.back().offset_order = "";
      bool good = true;
      if (data.second.mat.size() != 3)
      {
        good = false;
      }
      else
      {
        for (int i = 0; i < 3; ++i)
        {
          char c = '_';
          if (data.second.mat[i] == 'x' || data.second.mat[i] == 'X')
          {
            c = 'x';
          }
          else if (data.second.mat[i] == 'y' || data.second.mat[i] == 'Y')
          {
            c = 'y';
          }
          else if (data.second.mat[i] == 'z' || data.second.mat[i] == 'Z')
          {
            c = 'z';
          }
          else
          {
            good = false;
          }
          obj.bones.back().offset_order += c;
        }
      }
      if (good)
      {
        return true;
      }
      obj.bones.back().offset_order = "xyz";
      cerr << "I got '" << data.second.mat
           << "' when I was looking for an axis-order value." << endl;
      return false;
    }
  }
};

class DofTokenPattern : public Reader::BasePattern
{
public:
  virtual Reader::BaseMatchData* operator()(Reader::BaseReader& from)
  {
    static MatchData data;
    from.push_token_list();
    string tok;
    if (!from.get_token(tok))
    {
      from.restore_token_list();
      return nullptr;
    }
    if (tok.size() == 2)
    {
      int ind = 0;
      if (tok[0] == 't' || tok[0] == 'T')
      {
        ind = 0;
      }
      else if (tok[0] == 'r' || tok[0] == 'R')
      {
        ind = 3;
      }
      else
      {
        ind = 6;
      }
      if (tok[1] == 'x' || tok[1] == 'X')
      {
        ind += 0;
      }
      else if (tok[1] == 'y' || tok[1] == 'Y')
      {
        ind += 1;
      }
      else if (tok[1] == 'z' || tok[1] == 'Z')
      {
        ind += 2;
      }
      else
      {
        ind += 6;
      }
      if (ind < 6 && ind >= 0)
      {
        static char ident[] = "XYZxyz";
        data.ident = ident[ind];
        from.ignore_token_list();
        return &data;
      }
    }
    else if (tok == "l" || tok == "L")
    {
      data.ident = 'l';
      return &data;
    }
    else if (tok == "E O L")
    {
      from.restore_token_list();
      // don't complain on EOL just drop it.
      return NULL;
    }
    cerr << "Dof token '" << tok << "' unrecognized." << endl;
    from.restore_token_list();
    return NULL;
  }
  class MatchData : public Reader::BaseMatchData
  {
  public:
    char ident;
  };
  static BasePattern* get_instance()
  {
    static DofTokenPattern pat;
    return &pat;
  }
};

class BoneDataDofHandler
  : public Reader::
        BasePatternHandler<Reader::StarPattern<DofTokenPattern>, Skeleton>
{
public:
  virtual bool use_data(
      Reader::StarPattern<DofTokenPattern>::MatchData& data,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    if (!obj.in_bone)
    {
      cerr << "Looks like we got a dof while not in a bone." << endl;
      return false;
    }
    else
    {
      obj.bones.back().dof = "";
      set<char> seen;
      for (unsigned int i = 0; i < data.data.size(); ++i)
      {
        if (seen.count(data.data[i].ident))
        {
          cerr << "We saw token '" << data.data[i].ident
               << "' again in dof string." << endl;
          return false;
        }
        else
        {
          obj.bones.back().dof += data.data[i].ident;
          seen.insert(data.data[i].ident);
        }
      }
      return true;
    }
  }
};

class LimitsPattern : public Reader::BasePattern
{
public:
  virtual Reader::BaseMatchData* operator()(Reader::BaseReader& from)
  {
    static MatchData data;
    from.push_token_list();
    string t1, t2, t3;
    if (from.get_token(t1) && from.get_token(t2) && from.get_token(t3))
    {
      istringstream i1(t1);
      istringstream i2(t2);
      double low, high;
      if (i1 >> low && i2 >> high && t3 == "E O L")
      {
        data.low = low;
        data.high = high;
        from.ignore_token_list();
        return &data;
      }
    }
    from.restore_token_list();
    return NULL;
  }
  class MatchData : public Reader::BaseMatchData
  {
  public:
    double low;
    double high;
  };
  static BasePattern* get_instance()
  {
    static LimitsPattern pat;
    return &pat;
  }
};

class BoneDataLimitsHandler
  : public Reader::
        BasePatternHandler<Reader::StarPattern<LimitsPattern>, Skeleton>
{
public:
  virtual bool use_data(
      Reader::StarPattern<LimitsPattern>::MatchData& /* data */,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    if (!obj.in_bone)
    {
      cerr << "Looks like we got a limits while not in a bone." << endl;
      return false;
    }
    else
    {
      /* ignore limits */
      return true;
    }
  }
};

class BoneDataTorqueLimitsHandler
  : public Reader::
        BasePatternHandler<Reader::StarPattern<LimitsPattern>, Skeleton>
{
public:
  virtual bool use_data(
      Reader::StarPattern<LimitsPattern>::MatchData& data,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    if (!obj.in_bone)
    {
      cerr << "Looks like we got a torque_limits while not in a bone." << endl;
      return false;
    }
    else
    {
      obj.bones.back().torque_limits.clear();
      for (unsigned int i = 0; i < data.data.size(); ++i)
      {
        obj.bones.back().torque_limits.push_back(
            make_vector(data.data[i].low, data.data[i].high));
      }
      return true;
    }
  }
};

class BoneDataBeginHandler
  : public Reader::BasePatternHandler<Reader::NullPattern, Skeleton>
{
  virtual bool use_data(
      Reader::NullPattern::MatchData& /* data */,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    if (obj.in_bone)
    {
      cerr << "Looks like the last bone hasn't finished yet." << endl;
      return false;
    }
    obj.in_bone = true;
    obj.bones.push_back(Bone());
    obj.bones.back().pre_parse();
    return true;
  }
};

class BoneDataEndHandler
  : public Reader::BasePatternHandler<Reader::NullPattern, Skeleton>
{
public:
  virtual bool use_data(
      Reader::NullPattern::MatchData& /* data */,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& /* reader */) const
  {
    if (!obj.in_bone)
    {
      cerr << "Looks like we're trying to end without begining." << endl;
      return false;
    }
    obj.in_bone = false;
    return obj.bones.back().post_parse();
  }
};

class HeirarchyHandler : public Reader::BasePatternHandler<
                             Reader::StarPattern<Reader::TypePattern<string> >,
                             Skeleton>
{
public:
  virtual bool use_data(
      Reader::StarPattern<Reader::TypePattern<string> >::MatchData& data,
      Skeleton& obj,
      Reader::Reader<Skeleton> const& reader) const
  {
    int par = -2;
    if (reader.current_keyword == "root")
    {
      par = -1;
    }
    for (unsigned int i = 0; i < obj.bones.size(); ++i)
    {
      if (obj.bones[i].name == reader.current_keyword)
      {
        if (par != -2)
        {
          cerr << "More than one bone has name '" << reader.current_keyword
               << "'." << endl;
        }
        par = i;
      }
    }
    if (par == -2)
    {
      cerr << "Could not find bone named '" << reader.current_keyword << "'."
           << endl;
      return false;
    }
    for (unsigned int c = 0; c < data.data.size(); ++c)
    {
      int chi = -1;
      for (unsigned int i = 0; i < obj.bones.size(); ++i)
      {
        if (obj.bones[i].name == data.data[c].mat)
        {
          if (chi != -1)
          {
            cerr << "More than one bone has name '" << data.data[c].mat << "'."
                 << endl;
          }
          chi = i;
        }
      }
      if (chi == -1)
      {
        cerr << "Missing bone with name '" << data.data[c].mat << "'." << endl;
      }
      else
      {
        obj.bones[chi].parent = par;
      }
    }
    return true;
  }
};

Reader::Reader<Skeleton>& get_reader()
{
  static Reader::Reader<Skeleton> reader;
  static bool inited = false;
  if (!inited)
  {
    reader.clear_seperators();
    reader.add_seperator(' ');
    reader.add_seperator('\r');
    reader.add_seperator('\n');
    reader.add_seperator('\t');
    reader.add_seperator(',');
    reader.add_seperator('(');
    reader.add_seperator(')');
    // insert section handler objects here soon
    reader.set_section_handler(
        ":version",
        new Reader::
            IgnorePatternHandler<Reader::TypePattern<double>, Skeleton>());
    reader.set_section_handler(
        ":name",
        new Reader::
            IgnorePatternHandler<Reader::TypePattern<string>, Skeleton>());
    reader.set_section_handler(
        ":documentation",
        new Reader::IgnorePatternHandler<Reader::NullPattern, Skeleton>());
    reader.set_handler(
        ":documentation",
        "",
        new Reader::IgnorePatternHandler<Reader::NullPattern, Skeleton>());

    reader.set_section_handler(
        ":units",
        new Reader::IgnorePatternHandler<Reader::NullPattern, Skeleton>());
    reader.set_handler(":units", "mass", new UnitsMassHandler());
    reader.set_handler(":units", "length", new UnitsLengthHandler());
    reader.set_handler(":units", "timestep", new UnitsTimestepHandler());
    reader.set_handler(":units", "angle", new UnitsAngleHandler());

    reader.set_section_handler(
        ":root",
        new Reader::IgnorePatternHandler<Reader::NullPattern, Skeleton>());
    reader.set_handler(":root", "order", new OrderHandler());
    reader.set_handler(":root", "axis", new RootAxisHandler());
    reader.set_handler(":root", "position", new RootPositionHandler());
    reader.set_handler(":root", "orientation", new RootOrientationHandler());

    reader.set_section_handler(
        ":bonedata",
        new Reader::IgnorePatternHandler<Reader::NullPattern, Skeleton>());
    reader.set_handler(":bonedata", "begin", new BoneDataBeginHandler());
    reader.set_handler(":bonedata", "id", new BoneDataIdHandler());
    reader.set_handler(":bonedata", "name", new BoneDataNameHandler());
    reader.set_handler(
        ":bonedata", "direction", new BoneDataDirectionHandler());
    reader.set_handler(":bonedata", "length", new BoneDataLengthHandler());
    reader.set_handler(":bonedata", "axis", new BoneDataAxisHandler());
    reader.set_handler(":bonedata", "dof", new BoneDataDofHandler());
    reader.set_handler(
        ":bonedata", "torque_limits", new BoneDataTorqueLimitsHandler());
    reader.set_handler(":bonedata", "limits", new BoneDataLimitsHandler());
    reader.set_handler(":bonedata", "radius", new BoneDataRadiusHandler());
    reader.set_handler(":bonedata", "density", new BoneDataDensityHandler());
    reader.set_handler(":bonedata", "end", new BoneDataEndHandler());

    reader.set_section_handler(
        ":hierarchy",
        new Reader::IgnorePatternHandler<Reader::NullPattern, Skeleton>());
    reader.set_handler(
        ":hierarchy",
        "begin",
        new Reader::IgnorePatternHandler<Reader::NullPattern, Skeleton>());
    reader.set_handler(
        ":hierarchy",
        "end",
        new Reader::IgnorePatternHandler<Reader::NullPattern, Skeleton>());
    reader.set_handler(":hierarchy", "", new HeirarchyHandler());

    inited = true;
  }
  return reader;
}
