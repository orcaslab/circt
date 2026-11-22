//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-inline-calls"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_INLINECALLSPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::SmallSetVector;

namespace {
/// Whether a coroutine never suspends: its body has no `llhd.wait` and every
/// coroutine it calls never suspends either. Such a coroutine is a
/// SystemVerilog task without timing controls; it runs to completion within
/// the caller's time step, exactly like a function, so its calls are inlined
/// like function calls. A coroutine that calls itself (directly or through
/// others) is not taken. `memo` caches the answer per coroutine; a coroutine
/// on the current path counts as suspending.
static bool neverSuspends(CoroutineOp coroutine, const SymbolTable &symbolTable,
                          DenseMap<Operation *, bool> &memo) {
  auto [it, inserted] = memo.try_emplace(coroutine, false);
  if (!inserted)
    return it->second;
  bool result = true;
  coroutine.walk([&](Operation *op) {
    if (isa<WaitOp>(op)) {
      result = false;
      return WalkResult::interrupt();
    }
    if (auto call = dyn_cast<CallCoroutineOp>(op)) {
      auto callee = dyn_cast_or_null<CoroutineOp>(
          symbolTable.lookup(call.getCalleeAttr().getAttr()));
      if (!callee || !neverSuspends(callee, symbolTable, memo)) {
        result = false;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  memo[coroutine] = result;
  return result;
}

/// Implementation of the `InlinerInterface` that allows calls in SSACFG regions
/// nested within `llhd.process`, `llhd.final`, and `llhd.combinational` ops to
/// be inlined. Coroutines are inlined only where the pass found that they never
/// suspend (`neverSuspends`).
struct FunctionInliner : public InlinerInterface {
  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const override {
    // Only inline `func.func` and (non-suspending) `llhd.coroutine` ops.
    if (!isa<func::FuncOp, CoroutineOp>(callable))
      return false;

    // Only inline into SSACFG regions embedded within LLHD processes.
    if (!mayHaveSSADominance(*call->getParentRegion()))
      return false;
    return call->getParentWithTrait<ProceduralRegion>();
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  bool shouldAnalyzeRecursively(Operation *op) const override { return false; }

  /// The UB dialect does not implement this inliner hook for its
  /// `ub.unreachable` terminator, which causes the default implementation to
  /// abort. Handle the op here instead: like a branch op, it needs no rewrite
  /// when inlined, since it never transfers control to the continuation block.
  void handleTerminator(Operation *op, Block *newDest) const override {
    if (isa<mlir::ub::UnreachableOp>(op))
      return;
    // A coroutine's `llhd.return` continues after the call, like a function's
    // return: a branch to the continuation block with the returned values.
    if (auto ret = dyn_cast<ReturnOp>(op)) {
      OpBuilder builder(op);
      cf::BranchOp::create(builder, op->getLoc(), newDest, ret.getOperands());
      op->erase();
      return;
    }
    InlinerInterface::handleTerminator(op, newDest);
  }

  void handleTerminator(Operation *op,
                        ValueRange valuesToReplace) const override {
    if (auto ret = dyn_cast<ReturnOp>(op)) {
      for (auto [from, to] : llvm::zip(valuesToReplace, ret.getOperands()))
        from.replaceAllUsesWith(to);
      return;
    }
    InlinerInterface::handleTerminator(op, valuesToReplace);
  }
};

/// Pass implementation.
struct InlineCallsPass
    : public llhd::impl::InlineCallsPassBase<InlineCallsPass> {
  using CallStack = SmallSetVector<Operation *, 8>;
  /// Per coroutine, whether it never suspends (`neverSuspends`), filled
  /// before the modules are walked in parallel.
  DenseMap<Operation *, bool> nonSuspending;
  void runOnOperation() override;
  LogicalResult runOnRegion(Region &region, const SymbolTable &symbolTable,
                            CallStack &callStack);
};
} // namespace

void InlineCallsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  nonSuspending.clear();
  for (auto coroutine : getOperation().getOps<CoroutineOp>())
    neverSuspends(coroutine, symbolTable, nonSuspending);
  if (failed(failableParallelForEach(
          &getContext(), getOperation().getOps<hw::HWModuleOp>(),
          [&](auto module) {
            CallStack callStack;
            return runOnRegion(module.getBody(), symbolTable, callStack);
          })))
    signalPassFailure();
}

LogicalResult InlineCallsPass::runOnRegion(Region &region,
                                           const SymbolTable &symbolTable,
                                           CallStack &callStack) {
  FunctionInliner inliner(&getContext());
  InlinerConfig config;
  SmallVector<Operation *> callsToErase;
  SmallVector<std::pair<Operation *, Operation *>> inlineEndMarkers;

  // Walk all calls in the HW module and inline each. Emit a diagnostic if a
  // call does not target a `func.func` op or the inliner fails for some reason.
  // We use a custom version of `Operation::walk` here to ensure that we visit
  // the inlined operations immediately after visiting the call.
  for (auto &block : region) {
    for (auto &op : block) {
      // Pop all calls that are followed by this op off the call stack.
      while (!inlineEndMarkers.empty() &&
             inlineEndMarkers.back().first == &op) {
        assert(inlineEndMarkers.back().second == callStack.back());
        LLVM_DEBUG(llvm::dbgs()
                   << "- Finished @"
                   << SymbolTable::getSymbolName(inlineEndMarkers.back().second)
                   << "\n");
        inlineEndMarkers.pop_back();
        callStack.pop_back();
      }

      // Handle nested regions.
      for (auto &nestedRegion : op.getRegions())
        if (failed(runOnRegion(nestedRegion, symbolTable, callStack)))
          return failure();

      // A call of a coroutine that never suspends is inlined like a function
      // call; one that may suspend stays a call.
      if (auto coCall = dyn_cast<CallCoroutineOp>(op)) {
        auto coroutine = dyn_cast_or_null<CoroutineOp>(
            symbolTable.lookup(coCall.getCalleeAttr().getAttr()));
        auto known = coroutine ? nonSuspending.find(coroutine)
                               : nonSuspending.end();
        if (known == nonSuspending.end() || !known->second)
          continue;
        if (!callStack.insert(coroutine))
          continue;
        inlineEndMarkers.push_back({op.getNextNode(), coroutine});
        LLVM_DEBUG(llvm::dbgs() << "- Inlining " << coCall << "\n");
        if (failed(inlineCall(inliner, config.getCloneCallback(),
                              cast<CallOpInterface>(op),
                              cast<CallableOpInterface>(coroutine.getOperation()),
                              &coroutine.getBody())))
          return coCall.emitError("coroutine call cannot be inlined");
        callsToErase.push_back(coCall);
        ++numInlined;
        continue;
      }

      // We only care about calls.
      auto callOp = dyn_cast<func::CallOp>(op);
      if (!callOp)
        continue;

      // Make sure we're calling a `func.func`.
      auto symbol = callOp.getCalleeAttr();
      auto *calledOp = symbolTable.lookup(symbol.getAttr());
      auto funcOp = dyn_cast<func::FuncOp>(calledOp);
      if (!funcOp) {
        auto d = callOp.emitError("function call cannot be inlined: call "
                                  "target is not a regular function");
        d.attachNote(calledOp->getLoc()) << "call target defined here";
        return failure();
      }

      // Skip extern declarations (e.g. DPI-C imports) — nothing to inline.
      if (funcOp.isDeclaration())
        continue;

      // Ensure that we are not recursively inlining a function, which would
      // just expand infinitely in the IR. Recursive functions can't be fully
      // inlined, so leave this particular call as-is and keep inlining the
      // rest of the module.
      if (!callStack.insert(funcOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Skipping recursive call " << callOp << "\n");
        continue;
      }
      inlineEndMarkers.push_back({op.getNextNode(), funcOp});

      // Inline the function body and remember the call for later removal. The
      // `inlineCall` function will inline the function body *after* the call
      // op, which allows the loop to immediately visit the inlined ops and
      // handling nested calls.
      LLVM_DEBUG(llvm::dbgs() << "- Inlining " << callOp << "\n");
      if (failed(inlineCall(inliner, config.getCloneCallback(), callOp, funcOp,
                            funcOp.getCallableRegion())))
        return callOp.emitError("function call cannot be inlined");
      callsToErase.push_back(callOp);
      ++numInlined;
    }
  }

  // Erase all call ops that were successfully inlined.
  for (auto *callOp : callsToErase)
    callOp->erase();

  return success();
}
