//===- API.h - Offload GPU API --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOADTEST_API_API_H
#define OFFLOADTEST_API_API_H

#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
}
namespace offloadtest {
struct Shader;
struct Buffer;

enum class GPUAPI { Unknown, DirectX, Vulkan, Metal };

llvm::Error PrintDebugStream(const Buffer &Buffer,
                      const Shader &Shader,
                      llvm::raw_ostream &Out);

}

#endif // OFFLOADTEST_API_API_H
