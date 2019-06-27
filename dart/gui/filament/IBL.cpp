/*
 * Copyright (c) 2011-2018, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dart/gui/filament/IBL.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <fstream>
#include <sstream>
#include <string>

#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/Skybox.h>
#include <filament/Texture.h>

#include <image/KtxBundle.h>
#include <image/KtxUtility.h>

#include <filament/IndirectLight.h>
#include <utils/Path.h>

namespace dart {
namespace gui {
namespace flmt {

static constexpr float IBL_INTENSITY = 30000.0f;

IBL::IBL(filament::Engine& engine) : mEngine(engine)
{
}

IBL::~IBL()
{
  mEngine.destroy(mIndirectLight);
  mEngine.destroy(mTexture);
  mEngine.destroy(mSkybox);
  mEngine.destroy(mSkyboxTexture);
}

bool IBL::loadFromKtx(const std::string& prefix)
{
  // First check for compressed variants of the environment.
  utils::Path iblPath(prefix + "_ibl_s3tc.ktx");
  if (!iblPath.exists())
  {
    iblPath = utils::Path(prefix + "_ibl.ktx");
    if (!iblPath.exists())
    {
      return false;
    }
  }
  utils::Path skyPath(prefix + "_skybox_s3tc.ktx");
  if (!skyPath.exists())
  {
    skyPath = utils::Path(prefix + "_skybox.ktx");
    if (!skyPath.exists())
    {
      return false;
    }
  }

  auto createKtx = [](utils::Path path) {
    using namespace std;
    ifstream file(path.getPath(), ios::binary);
    vector<uint8_t> contents((istreambuf_iterator<char>(file)), {});
    return new image::KtxBundle(contents.data(), contents.size());
  };

  image::KtxBundle* iblKtx = createKtx(iblPath);
  image::KtxBundle* skyKtx = createKtx(skyPath);

  mSkyboxTexture = image::KtxUtility::createTexture(&mEngine, skyKtx, false);
  mTexture = image::KtxUtility::createTexture(&mEngine, iblKtx, false);

  std::istringstream shstring(iblKtx->getMetadata("sh"));
  for (filament::math::float3& band : mBands)
  {
    shstring >> band.x >> band.y >> band.z;
  }

  mIndirectLight = filament::IndirectLight::Builder()
                       .reflections(mTexture)
                       .irradiance(3, mBands)
                       .intensity(IBL_INTENSITY)
                       .build(mEngine);

  mSkybox = filament::Skybox::Builder()
                .environment(mSkyboxTexture)
                .showSun(true)
                .build(mEngine);

  return true;
}

bool IBL::loadFromDirectory(const utils::Path& path)
{
  // First check if KTX files are available.
  if (loadFromKtx(utils::Path::concat(path, path.getName())))
  {
    return true;
  }
  // Read spherical harmonics
  utils::Path sh(utils::Path::concat(path, "sh.txt"));
  if (sh.exists())
  {
    std::ifstream shReader(sh);
    shReader >> std::skipws;

    std::string line;
    for (filament::math::float3& band : mBands)
    {
      std::getline(shReader, line);
      int n = sscanf(
          line.c_str(),
          "(%f,%f,%f)",
          &band.r,
          &band.g,
          &band.b); // NOLINT(cert-err34-c)
      if (n != 3)
        return false;
    }
  }
  else
  {
    return false;
  }

  // Read mip-mapped cubemap
  const std::string prefix = "m";
  if (!loadCubemapLevel(&mTexture, path, 0, prefix + "0_"))
    return false;

  size_t numLevels = mTexture->getLevels();
  for (size_t i = 1; i < numLevels; i++)
  {
    const std::string levelPrefix = prefix + std::to_string(i) + "_";
    if (!loadCubemapLevel(&mTexture, path, i, levelPrefix))
      return false;
  }

  if (!loadCubemapLevel(&mSkyboxTexture, path))
    return false;

  mIndirectLight = filament::IndirectLight::Builder()
                       .reflections(mTexture)
                       .irradiance(3, mBands)
                       .intensity(IBL_INTENSITY)
                       .build(mEngine);

  mSkybox = filament::Skybox::Builder()
                .environment(mSkyboxTexture)
                .showSun(true)
                .build(mEngine);

  return true;
}

bool IBL::loadCubemapLevel(
    filament::Texture** texture,
    const utils::Path& path,
    size_t level,
    std::string const& levelPrefix) const
{
  filament::Texture::FaceOffsets offsets;
  filament::Texture::PixelBufferDescriptor buffer;
  bool success
      = loadCubemapLevel(texture, &buffer, &offsets, path, level, levelPrefix);
  if (!success)
    return false;
  (*texture)->setImage(mEngine, level, std::move(buffer), offsets);
  return true;
}

bool IBL::loadCubemapLevel(
    filament::Texture** texture,
    filament::Texture::PixelBufferDescriptor* outBuffer,
    filament::Texture::FaceOffsets* outOffsets,
    const utils::Path& path,
    size_t level,
    std::string const& levelPrefix) const
{
  static const char* faceSuffix[6] = {"px", "nx", "py", "ny", "pz", "nz"};

  size_t size = 0;
  size_t numLevels = 1;

  { // this is just a scope to avoid variable name hidding below
    int w, h;
    std::string faceName = levelPrefix + faceSuffix[0] + ".rgb32f";
    utils::Path facePath(utils::Path::concat(path, faceName));
    if (!facePath.exists())
    {
      std::cerr << "The face " << faceName << " does not exist" << std::endl;
      return false;
    }
    stbi_info(facePath.getAbsolutePath().c_str(), &w, &h, nullptr);
    if (w != h)
    {
      std::cerr << "width != height" << std::endl;
      return false;
    }

    size = (size_t)w;

    if (!levelPrefix.empty())
    {
      numLevels = (size_t)std::log2(size) + 1;
    }

    if (level == 0)
    {
      *texture = filament::Texture::Builder()
                     .width((uint32_t)size)
                     .height((uint32_t)size)
                     .levels((uint8_t)numLevels)
                     .format(filament::Texture::InternalFormat::R11F_G11F_B10F)
                     .sampler(filament::Texture::Sampler::SAMPLER_CUBEMAP)
                     .build(mEngine);
    }
  }

  // RGB_10_11_11_REV encoding: 4 bytes per pixel
  const size_t faceSize = size * size * sizeof(uint32_t);

  filament::Texture::FaceOffsets offsets;
  filament::Texture::PixelBufferDescriptor buffer(
      malloc(faceSize * 6),
      faceSize * 6,
      filament::Texture::Format::RGB,
      filament::Texture::Type::UINT_10F_11F_11F_REV,
      (filament::Texture::PixelBufferDescriptor::Callback)&free);

  bool success = true;
  uint8_t* p = static_cast<uint8_t*>(buffer.buffer);

  for (size_t j = 0; j < 6; j++)
  {
    offsets[j] = faceSize * j;

    std::string faceName = levelPrefix + faceSuffix[j] + ".rgb32f";
    utils::Path facePath(utils::Path::concat(path, faceName));
    if (!facePath.exists())
    {
      std::cerr << "The face " << faceName << " does not exist" << std::endl;
      success = false;
      break;
    }

    int w, h, n;
    unsigned char* data
        = stbi_load(facePath.getAbsolutePath().c_str(), &w, &h, &n, 4);
    if (w != h || w != size)
    {
      std::cerr << "Face " << faceName << "has a wrong size " << w << " x " << h
                << ", instead of " << size << " x " << size << std::endl;
      success = false;
      break;
    }

    if (data == nullptr || n != 4)
    {
      std::cerr << "Could not decode face " << faceName << std::endl;
      success = false;
      break;
    }

    memcpy(p + offsets[j], data, w * h * sizeof(uint32_t));

    stbi_image_free(data);
  }

  if (!success)
    return false;

  *outBuffer = std::move(buffer);
  *outOffsets = offsets;

  return true;
}

} // namespace flmt
} // namespace gui
} // namespace dart
