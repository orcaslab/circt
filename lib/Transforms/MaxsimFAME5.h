//===- MaxsimFAME5.h - FAME-5 multithreading of a flat model ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_LIB_TRANSFORMS_MAXSIMFAME5_H
#define CIRCT_LIB_TRANSFORMS_MAXSIMFAME5_H

#include "circt/Dialect/HW/HWOps.h"

namespace circt {
namespace maxsim {

/// Replaces `model` with its FAME-5 threaded module `name` of `threads`
/// instances, as `maxsim-fame5` with `clock-enable` and `thread-index` builds
/// it: each target port a channel, the current thread on `threadIdx`. With
/// `readySchedule`, a `thread_ready` input of one bit per thread says whose
/// inputs are all there, and the thread after the current one among those is
/// next.
FailureOr<hw::HWModuleOp> threadModel(hw::HWModuleOp model, unsigned threads,
                                      StringAttr name, bool readySchedule);

} // namespace maxsim
} // namespace circt

#endif // CIRCT_LIB_TRANSFORMS_MAXSIMFAME5_H
