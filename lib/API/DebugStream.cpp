//===- DebugStream.cpp - Shader debug printing API ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "API/API.h"
#include "Support/Pipeline.h"
#include "llvm/Object/DXContainer.h"

using namespace offloadtest;

llvm::Error offloadtest::PrintDebugStream(const Buffer &Buffer,
                                          const Shader &Shader,
                                          llvm::raw_ostream &Out) {
  auto ExContainer =
      llvm::object::DXContainer::create(Shader.Shader->getMemBufferRef());
  // Currently this is only supported for DXContainer files where we have an
  // embedded string table.
  if (!ExContainer)
    return ExContainer.takeError();
  llvm::StringRef StringTable = "";
  for (const auto &Part : *ExContainer)
    if (memcmp(Part.Part.Name, "STAB", 4) == 0)
      StringTable = Part.Data;
  if (StringTable.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "No string table part found in DXContainer");
  const uint32_t *const Start =
      reinterpret_cast<const uint32_t *>(Buffer.Data[0].get());
  size_t const NumEntries = Buffer.Size / sizeof(uint32_t);
  const uint32_t *Data = Start;
  const uint32_t *const End = Start + NumEntries;

  // Must be able to read at least the wave count, string offset, and arg size!
  while (Data + 3 < End) {
    uint32_t const WaveCount = *Data++;
    if (WaveCount == 0)
      break;
    uint32_t const StrOffset = *Data++;
    uint32_t const ArgSize = *Data++;

    llvm::StringRef const Str =
        StringTable.slice(StrOffset, StringTable.find('\0', StrOffset));
    if (Data + (WaveCount * (ArgSize / sizeof(uint32_t))) > End)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Debug stream data goes out of bounds of the buffer");
    llvm::SmallString<512> WorkingStr;
    for (uint32_t I = 0; I < WaveCount; ++I) {
      llvm::raw_svector_ostream OS(WorkingStr);
      const uint32_t *const ArgEnd = Data + (ArgSize / sizeof(uint32_t));
      for (size_t S = 0; S < Str.size(); ++S) {
        if (Data > ArgEnd)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "Debug stream arguments exceed expected size");

        if (Str[S] == '%') {
          ++S;
          if (S >= Str.size()) {
            OS << "%";
            break;
          }
          switch (Str[S]) {
          case 'd': {
            int32_t const Val = *reinterpret_cast<const int32_t *>(Data++);
            OS << Val;
            break;
          }
          case 'u': {
            uint32_t const Val = *Data++;
            OS << Val;
            break;
          }
          case 'f': {
            float const Val = *reinterpret_cast<const float *>(Data++);
            OS << Val;
            break;
          }
          case 'x': {
            uint32_t const Val = *Data++;
            OS << llvm::format_hex(Val, 8, /*Upper=*/false);
            break;
          }
          case '%': {
            OS << '%';
            break;
          }
          default:
            // Unknown format specifier, just print it literally.
            OS << '%' << Str[S];
            break;
          }
        } else {
          OS << Str[S];
        }
      }
      OS << "\0";
      Out << WorkingStr;
      WorkingStr.clear();
    }
  }
  return llvm::Error::success();
}
