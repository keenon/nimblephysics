/**
 * Original version:
 * https://github.com/pyomeca/ezc3d/blob/dev/src/modules/ForcePlatforms.cpp
 *
 * Copied here to allow easier bug fixes
 */

#include "dart/biomechanics/C3DForcePlatforms.hpp"

#include <ezc3d_all.h>

namespace dart {

namespace biomechanics {

const extern int FORCE_PLATFORM_NUM_CONVENTIONS = 2;

ForcePlatform::ForcePlatform()
{
}

ForcePlatform::ForcePlatform(size_t idx, const ezc3d::c3d& c3d, int convention)
{
  _meanCorners.setZero();
  _origin.setZero();
  _calMatrix.setZero();
  _refFrame.setZero();
  // Extract the required values from the C3D
  extractUnits(c3d);
  extractType(idx, c3d);
  extractCorners(idx, c3d);
  extractOrigin(idx, c3d);
  extractCalMatrix(idx, c3d);
  computePfReferenceFrame();
  extractDataWithConvention(idx, c3d, convention);
}

const std::string& ForcePlatform::forceUnit() const
{
  return _unitsForce;
}

const std::string& ForcePlatform::momentUnit() const
{
  return _unitsMoment;
}

const std::string& ForcePlatform::positionUnit() const
{
  return _unitsPosition;
}

void ForcePlatform::extractUnits(const ezc3d::c3d& c3d)
{
  const ezc3d::ParametersNS::GroupNS::Group& groupPoint(
      c3d.parameters().group("POINT"));
  const ezc3d::ParametersNS::GroupNS::Group& groupFP(
      c3d.parameters().group("FORCE_PLATFORM"));

  // Position units
  try
  {
    if (groupPoint.isParameter("UNITS")
        && groupPoint.parameter("UNITS").dimension()[0] > 0)
    {
      _unitsPosition = groupPoint.parameter("UNITS").valuesAsString()[0];
    }
    else
    {
      // Assume meter if not provided
      _unitsPosition = "m";
    }
  }
  catch (const std::exception& e)
  {
    // Assume meter if not provided
    _unitsPosition = "m";
  }

  // Force units
  try
  {
    if (groupFP.isParameter("UNITS")
        && groupFP.parameter("UNITS").dimension()[0] > 0)
    {
      _unitsForce = groupFP.parameter("UNITS").valuesAsString()[0];
    }
    else
    {
      // Assume Newton if not provided
      _unitsForce = "N";
    }
  }
  catch (const std::exception& e)
  {
    // Assume Newton if not provided
    _unitsForce = "N";
  }

  // Moments units
  _unitsMoment = _unitsForce + _unitsPosition;
}

size_t ForcePlatform::nbFrames() const
{
  return _F.size();
}

size_t ForcePlatform::type() const
{
  return _type;
}

const Eigen::Matrix6s& ForcePlatform::calMatrix() const
{
  return _calMatrix;
}

const std::vector<Eigen::Vector3s>& ForcePlatform::corners() const
{
  return _corners;
}

const Eigen::Vector3s& ForcePlatform::meanCorners() const
{
  return _meanCorners;
}

const Eigen::Vector3s& ForcePlatform::origin() const
{
  return _origin;
}

const std::vector<Eigen::Vector3s>& ForcePlatform::forces() const
{
  return _F;
}

const std::vector<Eigen::Vector3s>& ForcePlatform::moments() const
{
  return _M;
}

const std::vector<Eigen::Vector3s>& ForcePlatform::CoP() const
{
  return _CoP;
}

const std::vector<Eigen::Vector3s>& ForcePlatform::Tz() const
{
  return _Tz;
}

void ForcePlatform::extractType(size_t idx, const ezc3d::c3d& c3d)
{
  const ezc3d::ParametersNS::GroupNS::Group& groupFP(
      c3d.parameters().group("FORCE_PLATFORM"));

  if (groupFP.parameter("TYPE").valuesAsInt().size() < idx + 1)
  {
    throw std::runtime_error(
        "FORCE_PLATFORM:IDX is not fill properly "
        "to extract Force platform informations");
  }
  _type = static_cast<size_t>(groupFP.parameter("TYPE").valuesAsInt()[idx]);

  // Make sure that particular type is supported
  if (_type == 1)
  {
  }
  else if (_type == 2 || _type == 4)
  {
  }
  else if (_type == 3 || _type == 7)
  {
    if (_type == 7)
    {
      throw std::runtime_error(
          "Type 7 is not supported yet, "
          "please open an Issue on github for "
          "support");
    }
  }
  else if (_type == 5)
  {
    throw std::runtime_error(
        "Type 5 is not supported yet, please "
        "open an Issue on github for support");
  }
  else if (_type == 6)
  {
    throw std::runtime_error(
        "Type 6 is not supported yet, please "
        "open an Issue on github for support");
  }
  else if (_type == 11 || _type == 12)
  {
    throw std::runtime_error(
        "Kistler Split Belt Treadmill is not "
        "supported for ForcePlatform analysis");
  }
  else if (_type == 21)
  {
    throw std::runtime_error(
        "AMTI-stairs is not supported "
        "for ForcePlatform analysis");
  }
  else
  {
    throw std::runtime_error(
        "Force platform type is non existant "
        "or not supported yet");
  }
}

ForcePlatforms::ForcePlatforms(const ezc3d::c3d& c3d, int convention)
{
  size_t nbForcePF(c3d.parameters()
                       .group("FORCE_PLATFORM")
                       .parameter("USED")
                       .valuesAsInt()[0]);
  for (size_t i = 0; i < nbForcePF; ++i)
  {
    _platforms.push_back(ForcePlatform(i, c3d, convention));
  }
}

void ForcePlatform::extractCorners(size_t idx, const ezc3d::c3d& c3d)
{
  const ezc3d::ParametersNS::GroupNS::Group& groupFP(
      c3d.parameters().group("FORCE_PLATFORM"));

  const std::vector<double>& all_corners(
      groupFP.parameter("CORNERS").valuesAsDouble());
  if (all_corners.size() < 12 * (idx + 1))
  {
    throw std::runtime_error(
        "FORCE_PLATFORM:CORNER is not fill properly "
        "to extract Force platform informations");
  }

  for (size_t i = 0; i < 4; ++i)
  {
    Eigen::Vector3s corner;
    for (size_t j = 0; j < 3; ++j)
    {
      corner(j) = all_corners[idx * 12 + i * 3 + j];
    }
    _corners.push_back(corner);
    _meanCorners += corner;
  }
  _meanCorners /= 4;
}

void ForcePlatform::extractOrigin(size_t idx, const ezc3d::c3d& c3d)
{
  const ezc3d::ParametersNS::GroupNS::Group& groupFP(
      c3d.parameters().group("FORCE_PLATFORM"));

  const std::vector<double>& all_origins(
      groupFP.parameter("ORIGIN").valuesAsDouble());
  if (all_origins.size() < 3 * (idx + 1))
  {
    throw std::runtime_error(
        "FORCE_PLATFORM:ORIGIN is not fill properly "
        "to extract Force platform informations");
  }
  for (size_t i = 0; i < 3; ++i)
  {
    if (_type == 1 && i < 2)
    {
      _origin(i) = 0;
    }
    else
    {
      _origin(i) = all_origins[idx * 3 + i];
    }
  }

  if ((_type >= 1 && _type <= 4) && _origin(2) > 0.0)
  {
    _origin = -1 * _origin;
  }
}

void ForcePlatform::extractCalMatrix(size_t idx, const ezc3d::c3d& c3d)
{
  const ezc3d::ParametersNS::GroupNS::Group& groupFP(
      c3d.parameters().group("FORCE_PLATFORM"));

  size_t nChannels(-1);
  if (_type >= 1 && _type <= 4)
  {
    nChannels = 6;
  }

  if (!groupFP.isParameter("CAL_MATRIX"))
  {
    if (_type == 2)
    {
      // CAL_MATRIX is ignore for type 2
      // If none is found, returns all zeros
      return;
    }
    else
    {
      throw std::runtime_error(
          "FORCE_PLATFORM:CAL_MATRIX was not found, but is "
          "required for the type of force platform");
    }
  }

  // Check dimensions
  const auto& calMatrixParam(groupFP.parameter("CAL_MATRIX"));
  if (calMatrixParam.dimension().size() < 3
      || calMatrixParam.dimension()[2] <= idx)
  {
    if (_type == 1 || _type == 2 || _type == 3)
    {
      // CAL_MATRIX is ignore for type 2
      // If none is found, returns all zeros
      return;
    }
    else
    {
      throw std::runtime_error(
          "FORCE_PLATFORM:CAL_MATRIX is not fill properly "
          "to extract Force platform informations");
    }
  }

  const std::vector<double>& val(calMatrixParam.valuesAsDouble());
  if (val.size() == 0)
  {
    // This is for Motion Analysis not providing a calibration matrix
    _calMatrix.setIdentity();
  }
  else
  {
    size_t skip(calMatrixParam.dimension()[0] * calMatrixParam.dimension()[1]);
    for (size_t i = 0; i < nChannels; ++i)
    {
      for (size_t j = 0; j < nChannels; ++j)
      {
        _calMatrix(i, j) = val[skip * idx + j * nChannels + i];
      }
    }
  }
}

void ForcePlatform::computePfReferenceFrame()
{
  Eigen::Vector3s axisX(_corners[0] - _corners[1]);
  Eigen::Vector3s axisY(_corners[0] - _corners[3]);
  Eigen::Vector3s axisZ(axisX.cross(axisY));
  axisY = axisZ.cross(axisX);

  axisX.normalize();
  axisY.normalize();
  axisZ.normalize();

  for (size_t i = 0; i < 3; ++i)
  {
    _refFrame(i, 0) = axisX(i);
    _refFrame(i, 1) = axisY(i);
    _refFrame(i, 2) = axisZ(i);
  }
}

void ForcePlatform::extractDataWithConvention(
    size_t idx, const ezc3d::c3d& c3d, int convention)
{
  assert(convention >= 0 && convention <= 1);

  const ezc3d::ParametersNS::GroupNS::Group& groupFP(
      c3d.parameters().group("FORCE_PLATFORM"));

  // Get elements from the force platform's type
  size_t nChannels(-1);
  if (_type == 1)
  {
    nChannels = 6;
  }
  else if (_type == 2 || _type == 4)
  {
    nChannels = 6;
  }
  else if (_type == 3)
  {
    nChannels = 8;
  }

  // Check the dimensions of FORCE_PLATFORM:CHANNEL are consistent
  const std::vector<size_t>& dimensions(
      groupFP.parameter("CHANNEL").dimension());
  if (dimensions[0] < nChannels)
  {
    throw std::runtime_error(
        "FORCE_PLATFORM:CHANNEL was not filled properly "
        "to extract Force platform informations");
  }
  if (dimensions[1] < idx + 1)
  {
    throw std::runtime_error(
        "FORCE_PLATFORM:CHANNEL was not filled properly "
        "to extract Force platform informations");
  }

  // Get the channels where the force platform are stored in the data
  std::vector<size_t> channel_idx(nChannels);
  const std::vector<int>& all_channel_idx(
      groupFP.parameter("CHANNEL").valuesAsInt());
  for (size_t i = 0; i < nChannels; ++i)
  {
    channel_idx[i] = all_channel_idx[idx * dimensions[0] + i] - 1; // 1-based
  }

  // Get the force and moment from these channel in global reference frame
  size_t nFramesTotal(c3d.header().nbFrames() * c3d.header().nbAnalogByFrame());
  _F.resize(nFramesTotal);
  _M.resize(nFramesTotal);
  _CoP.resize(nFramesTotal);
  _Tz.resize(nFramesTotal);
  size_t cmp(0);
  double* ch = new double[8];
  for (const auto& frame : c3d.data().frames())
  {
    for (size_t i = 0; i < frame.analogs().nbSubframes(); ++i)
    {
      const auto& subframe(frame.analogs().subframe(i));
      if (_type == 1)
      {
        Eigen::Vector3s force_raw = Eigen::Vector3s::Zero();
        Eigen::Vector3s cop_raw = Eigen::Vector3s::Zero();
        Eigen::Vector3s tz_raw = Eigen::Vector3s::Zero();
        // CalMatrix (the example I have does not have any)

        for (size_t j = 0; j < 3; ++j)
        {
          force_raw(j) = subframe.channel(channel_idx[j]).data();
          if (j < 2)
          {
            cop_raw(j) = subframe.channel(channel_idx[j + 3]).data();
          }
          else
          {
            tz_raw(j) = subframe.channel(channel_idx[j + 3]).data();
          }
        }

        assert(!force_raw.hasNaN());
        assert(!cop_raw.hasNaN());
        assert(!tz_raw.hasNaN());
        assert(!_refFrame.hasNaN());

        _F[cmp] = _refFrame * force_raw;
        _CoP[cmp] = _refFrame * cop_raw;
        _Tz[cmp] = _refFrame * tz_raw;
        assert(!_Tz[cmp].hasNaN());
        _M[cmp] = _F[cmp].cross(_CoP[cmp]) - _Tz[cmp];
        assert(!_M[cmp].hasNaN());
        _CoP[cmp] += _meanCorners;

        ++cmp;
      }
      else if (_type == 2 || _type == 3 || _type == 4)
      {
        Eigen::Vector3s force_raw = Eigen::Vector3s::Zero();
        Eigen::Vector3s moment_raw = Eigen::Vector3s::Zero();
        if (_type == 3)
        {
          for (size_t j = 0; j < 8; ++j)
          {
            ch[j] = subframe.channel(channel_idx[j]).data();
          }
          // CalMatrix (the example I have does not have any)

          force_raw(0) = ch[0] + ch[1];
          force_raw(1) = ch[2] + ch[3];
          force_raw(2) = ch[4] + ch[5] + ch[6] + ch[7];

          moment_raw(0) = _origin(1) * (ch[4] + ch[5] - ch[6] - ch[7]);
          moment_raw(1) = _origin(0) * (ch[5] + ch[6] - ch[4] - ch[7]);
          moment_raw(2)
              = _origin(1) * (ch[1] - ch[0]) + _origin(0) * (ch[2] - ch[3]);
          moment_raw += force_raw.cross(Eigen::Vector3s(0, 0, _origin(2)));
          assert(!moment_raw.hasNaN());
        }
        else
        {
          Eigen::Vector6s data_raw = Eigen::Vector6s::Zero();
          for (size_t j = 0; j < 3; ++j)
          {
            data_raw(j) = subframe.channel(channel_idx[j]).data();
            data_raw(j + 3) = subframe.channel(channel_idx[j + 3]).data();
          }
          if (_type == 4)
          {
            data_raw = _calMatrix * data_raw;
          }
          for (size_t j = 0; j < 3; ++j)
          {
            force_raw(j) = data_raw(j);
            moment_raw(j) = data_raw(j + 3);
          }
          if (convention == 1)
          {
            moment_raw += force_raw.cross(_origin);
          }
        }
        assert(!force_raw.hasNaN());
        assert(!moment_raw.hasNaN());
        assert(!_refFrame.hasNaN());
        _F[cmp] = _refFrame * force_raw;
        _M[cmp] = _refFrame * moment_raw;
        assert(!_M[cmp].hasNaN());

        Eigen::Vector3s CoP_raw = Eigen::Vector3s(
            moment_raw(1) == 0 ? 0 : -moment_raw(1) / force_raw(2),
            moment_raw(0) == 0 ? 0 : moment_raw(0) / force_raw(2),
            0);
        // Avoid NaNs in the CoP data
        if (moment_raw(1) != 0 && force_raw(2) == 0)
        {
          CoP_raw(0) = 0;
        }
        if (moment_raw(0) != 0 && force_raw(2) == 0)
        {
          CoP_raw(1) = 0;
        }
        Eigen::Vector3s originNoVertical = _origin;
        originNoVertical(2) = 0.0;
        if (convention == 0)
        {
          _CoP[cmp] = _refFrame * (CoP_raw + originNoVertical) + _meanCorners;
        }
        if (convention == 1)
        {
          _CoP[cmp] = _refFrame * CoP_raw + _meanCorners;
        }
        _Tz[cmp] = _refFrame * (moment_raw - force_raw.cross(-1 * CoP_raw));
        assert(!_Tz[cmp].hasNaN());
        ++cmp;
      }
    }
  }
  delete[] ch;
}

const std::vector<ForcePlatform>& ForcePlatforms::forcePlatforms() const
{
  return _platforms;
}

const ForcePlatform& ForcePlatforms::forcePlatform(size_t idx) const
{
  return _platforms.at(idx);
}

} // namespace biomechanics
} // namespace dart