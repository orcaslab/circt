//===- MaxsimFAME5.cpp - FAME-5 multithreading of a flat model ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The FAME-1 channelization, with eager outputs, and the FAME-5 multithreading
// FireSim's Golden Gate applies to a model (midas/passes/fame/FAMETransform
// and MuxingMultithreader, with the `mem1` state layout), on HW/Comb/Seq IR.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"

namespace circt {
#define GEN_PASS_DEF_MAXSIMFAME5
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

struct MaxsimFAME5Pass : public circt::impl::MaxsimFAME5Base<MaxsimFAME5Pass> {
  using MaxsimFAME5Base::MaxsimFAME5Base;
  void runOnOperation() override;
};

/// The inputs each bit of a value depends on.
using BitDeps = SmallVector<llvm::BitVector>;

/// A target port of the model and the channel it becomes.
struct Channel {
  StringAttr name;
  Type type;
  /// For an input, its block argument in the model; for an output, its index.
  unsigned argNum = 0;
  /// The bits of the port the channel carries, if only some.
  bool sliced = false;
  unsigned low = 0;
  /// For an output, the value the model drives it with.
  Value current;
  /// The inputs an output depends on combinationally.
  llvm::BitVector deps;
  bool eager = false;
  Value next;
};

/// Builds the threaded module of one model.
class Threader {
public:
  using Slices = llvm::StringMap<SmallVector<std::pair<unsigned, unsigned>>>;

  Threader(hw::HWModuleOp model, unsigned threads, StringAttr name,
           hw::HWModuleExternOp gate, const Slices &slices)
      : model(model), threads(threads), name(name), gate(gate), slices(slices) {
  }

  LogicalResult run();

private:
  LogicalResult checkModel();
  void computeDeps();
  const BitDeps &getDeps(Value value);
  Value getNext(Value value, ImplicitLocOpBuilder &b);
  bool dependsOnState(Value value);
  /// A memory of one entry per thread, read at the current thread.
  Value createThreadState(ImplicitLocOpBuilder &b, StringRef stateName,
                          Type type, Value &memory);
  void writeThreadState(ImplicitLocOpBuilder &b, Value memory, Value clock,
                        Value data, Value enable = {});
  /// How a register is written: never (it only holds its value), or with
  /// `data` when `enable` (always, if null).
  struct Write {
    bool holds;
    Value enable, data;
  };
  /// The write of a register whose next value is `next`: the branches of its
  /// muxes that hold the register are the cycles it is not written.
  Write peelHolds(ImplicitLocOpBuilder &b, Value next, Value reg);

  hw::HWModuleOp model;
  unsigned threads;
  StringAttr name;
  hw::HWModuleExternOp gate;
  const Slices &slices;

  unsigned clockArg = 0;
  SmallVector<Channel> inputs, outputs;
  SmallVector<seq::FirRegOp> registers;
  SmallVector<seq::FirMemOp> memories;
  DenseMap<Value, bool> memoryDeps;
  bool dependsOnMemory(Value value);
  /// For each bit of an input port, the channel carrying it.
  DenseMap<Value, SmallVector<unsigned>> inputOfArg;
  DenseMap<Value, std::unique_ptr<BitDeps>> deps;
  DenseMap<Value, Value> nexts;
  DenseMap<Value, bool> stateDeps;
  Value threadIdx, hostClock;
};

} // namespace

LogicalResult Threader::checkModel() {
  auto *body = model.getBodyBlock();
  bool hasClock = false;
  for (auto [i, port] : llvm::enumerate(model.getPortList())) {
    if (isa<hw::InOutType>(port.type))
      return model.emitError("inout port '") << port.name << "' in a model";
    if (isa<seq::ClockType>(port.type)) {
      if (port.dir != hw::ModulePort::Direction::Input || hasClock)
        return model.emitError("a model has a single clock input");
      hasClock = true;
      clockArg = port.argNum;
      continue;
    }
    bool isInput = port.dir == hw::ModulePort::Direction::Input;
    // An input the model does not read would only hold its threads back.
    if (isInput && body->getArgument(port.argNum).use_empty())
      continue;
    // An output of no bits is a channel of tokens only.
    auto width = hw::getBitWidth(port.type);
    if (width < 0 || (width == 0 && isInput))
      return model.emitError("port '") << port.name << "' has no bit width";
    auto &channels = isInput ? inputs : outputs;
    SmallVector<unsigned> bitChannels(width, channels.size());

    // A port may be split into channels of some of its bits each. They must
    // cover an input exactly; an output's other bits are not exported.
    auto it = slices.find(port.name.getValue());
    if (it == slices.end()) {
      Channel channel;
      channel.name = port.name;
      channel.type = port.type;
      channel.argNum = port.argNum;
      channels.push_back(channel);
    } else {
      auto portSlices = it->second;
      llvm::sort(portSlices,
                 [](auto a, auto b) { return a.second < b.second; });
      unsigned next = 0;
      for (auto [hi, lo] : portSlices) {
        if (lo < next || (isInput && lo != next) || hi < lo ||
            hi >= unsigned(width))
          return model.emitError("slices of port '")
                 << port.name << "' do not cover it exactly";
        next = hi + 1;
        for (unsigned bit = lo; bit <= hi; ++bit)
          bitChannels[bit] = channels.size();
        Channel channel;
        channel.name = StringAttr::get(
            model.getContext(),
            port.name.getValue() + "_" + Twine(hi) +
                (hi == lo ? Twine("_") : "_" + Twine(lo) + "_"));
        channel.type = IntegerType::get(model.getContext(), hi - lo + 1);
        channel.argNum = port.argNum;
        channel.sliced = true;
        channel.low = lo;
        channels.push_back(channel);
      }
      if (isInput && next != unsigned(width))
        return model.emitError("slices of port '")
               << port.name << "' do not cover it exactly";
    }
    if (isInput)
      inputOfArg[body->getArgument(port.argNum)] = bitChannels;
  }
  if (!hasClock)
    return model.emitError("a model has a clock input");

  for (auto &op : *body) {
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      if (reg.getIsAsync())
        return reg.emitError("asynchronous reset in a threaded model");
      if (reg.getClk() != body->getArgument(clockArg))
        return reg.emitError("register not on the model's clock");
      if (hw::getBitWidth(reg.getType()) <= 0)
        return reg.emitError("register has no bit width");
      registers.push_back(reg);
      continue;
    }
    if (auto memory = dyn_cast<seq::FirMemOp>(op)) {
      if (memory.getInit())
        return memory.emitError("initialized memory in a threaded model");
      if (memory.getReadLatency() > 1 || memory.getWriteLatency() != 1)
        return memory.emitError(
            "a threaded memory has a read latency of 0 or 1 and a write "
            "latency of 1");
      if (!llvm::isPowerOf2_32(threads))
        return memory.emitError("threading a memory needs a power of two of "
                                "threads");
      memories.push_back(memory);
      continue;
    }
    if (isa<hw::OutputOp, seq::FirMemReadOp, seq::FirMemWriteOp>(op))
      continue;
    if (isa<hw::HWInstanceLike>(op) || op.getDialect()->getNamespace() == "seq")
      return op.emitError("a threaded model must be flat and keep its state "
                          "in registers and memories");
  }
  for (auto &use : body->getArgument(clockArg).getUses())
    if (!isa<seq::FirRegOp, seq::FirMemReadOp, seq::FirMemWriteOp>(
            use.getOwner()))
      return use.getOwner()->emitError(
          "clock used other than by a register or memory");
  return success();
}

/// The inputs each bit of `value` depends on combinationally. Bits are followed
/// through the ops that only move bits around, so that a signal packed with
/// data into an aggregate does not take on the data's dependences; any other
/// op makes each result bit depend on every operand bit.
const BitDeps &Threader::getDeps(Value value) {
  if (auto it = deps.find(value); it != deps.end())
    return *it->second;
  auto width = std::max<int64_t>(hw::getBitWidth(value.getType()), 0);
  auto result =
      std::make_unique<BitDeps>(width, llvm::BitVector(inputs.size()));
  // Placeholder against combinational cycles.
  deps[value] = std::make_unique<BitDeps>(*result);
  auto &bits = *result;
  auto all = [&](Value operand) {
    llvm::BitVector any(inputs.size());
    for (auto &bit : getDeps(operand))
      any |= bit;
    return any;
  };
  auto copyBits = [&](Value operand, unsigned from, unsigned to,
                      unsigned count) {
    auto operandBits = getDeps(operand);
    for (unsigned i = 0; i < count; ++i)
      bits[to + i] |= operandBits[from + i];
  };
  auto *op = value.getDefiningOp();
  APInt index;
  if (auto it = inputOfArg.find(value); it != inputOfArg.end()) {
    for (auto [bit, channel] : llvm::zip(bits, it->second))
      bit.set(channel);
  } else if (auto read = dyn_cast_or_null<seq::FirMemReadOp>(op)) {
    // A registered read is state; a combinational one follows its address.
    auto memory = read.getMemory().getDefiningOp<seq::FirMemOp>();
    if (!memory || memory.getReadLatency() == 0) {
      llvm::BitVector any = all(read.getAddress());
      if (read.getEnable())
        any |= all(read.getEnable());
      for (auto &bit : bits)
        bit = any;
    }
  } else if (!op ||
             isa<seq::FirRegOp, hw::ConstantOp, hw::AggregateConstantOp>(op)) {
    // Inputs of the threaded module's state only.
  } else if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
    unsigned at = 0;
    for (auto operand : llvm::reverse(concat.getInputs())) {
      auto w = hw::getBitWidth(operand.getType());
      copyBits(operand, 0, at, w);
      at += w;
    }
  } else if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
    copyBits(extract.getInput(), extract.getLowBit(), 0, width);
  } else if (auto replicate = dyn_cast<comb::ReplicateOp>(op)) {
    auto w = hw::getBitWidth(replicate.getInput().getType());
    for (unsigned at = 0; at + w <= unsigned(width); at += w)
      copyBits(replicate.getInput(), 0, at, w);
  } else if (isa<hw::BitcastOp>(op)) {
    copyBits(op->getOperand(0), 0, 0, width);
  } else if (isa<hw::ArrayCreateOp, hw::ArrayConcatOp>(op)) {
    // Operands are listed from the most significant one.
    unsigned at = 0;
    for (auto operand : llvm::reverse(op->getOperands())) {
      auto w = hw::getBitWidth(operand.getType());
      copyBits(operand, 0, at, w);
      at += w;
    }
  } else if (auto get = dyn_cast<hw::ArrayGetOp>(op)) {
    auto arrayType = hw::type_cast<hw::ArrayType>(get.getInput().getType());
    auto size = arrayType.getNumElements();
    if (matchPattern(get.getIndex(), m_ConstantInt(&index))) {
      if (index.getZExtValue() < size)
        copyBits(get.getInput(), index.getZExtValue() * width, 0, width);
    } else {
      for (unsigned e = 0; e < size; ++e)
        copyBits(get.getInput(), e * width, 0, width);
      auto indexDeps = all(get.getIndex());
      for (auto &bit : bits)
        bit |= indexDeps;
    }
  } else if (auto slice = dyn_cast<hw::ArraySliceOp>(op);
             slice &&
             matchPattern(slice.getLowIndex(), m_ConstantInt(&index))) {
    auto elementWidth = hw::getBitWidth(
        hw::type_cast<hw::ArrayType>(slice.getType()).getElementType());
    copyBits(slice.getInput(), index.getZExtValue() * elementWidth, 0, width);
  } else if (auto create = dyn_cast<hw::StructCreateOp>(op)) {
    // The first field is the most significant.
    unsigned at = 0;
    for (auto operand : llvm::reverse(create.getInput())) {
      auto w = hw::getBitWidth(operand.getType());
      copyBits(operand, 0, at, w);
      at += w;
    }
  } else if (auto extract = dyn_cast<hw::StructExtractOp>(op)) {
    auto structType =
        hw::type_cast<hw::StructType>(extract.getInput().getType());
    unsigned at = 0;
    auto elements = structType.getElements();
    for (auto i = elements.size(); i-- > extract.getFieldIndex() + 1;)
      at += hw::getBitWidth(elements[i].type);
    copyBits(extract.getInput(), at, 0, width);
  } else if (isa<comb::AndOp, comb::OrOp, comb::XorOp>(op)) {
    for (auto operand : op->getOperands())
      copyBits(operand, 0, 0, width);
  } else if (auto mux = dyn_cast<comb::MuxOp>(op)) {
    copyBits(mux.getTrueValue(), 0, 0, width);
    copyBits(mux.getFalseValue(), 0, 0, width);
    auto condDeps = all(mux.getCond());
    for (auto &bit : bits)
      bit |= condDeps;
  } else {
    llvm::BitVector any(inputs.size());
    for (auto operand : op->getOperands())
      any |= all(operand);
    for (auto &bit : bits)
      bit = any;
  }
  deps[value] = std::move(result);
  return *deps[value];
}

void Threader::computeDeps() {
  auto output = cast<hw::OutputOp>(model.getBodyBlock()->getTerminator());
  for (auto &channel : outputs) {
    auto value = output.getOperand(channel.argNum);
    channel.current = value;
    channel.deps = llvm::BitVector(inputs.size());
    auto width = hw::getBitWidth(channel.type);
    for (auto &bit : ArrayRef(getDeps(value)).slice(channel.low, width))
      channel.deps |= bit;
  }
}

/// Whether `value` is computed from a memory's read data. Golden Gate does not
/// compute the next value of such an output, so it does not fire it eagerly.
bool Threader::dependsOnMemory(Value value) {
  if (auto it = memoryDeps.find(value); it != memoryDeps.end())
    return it->second;
  auto *op = value.getDefiningOp();
  bool result = false;
  if (isa_and_nonnull<seq::FirMemReadOp>(op)) {
    result = true;
  } else if (op && !isa<seq::FirRegOp>(op)) {
    memoryDeps[value] = false;
    result = llvm::any_of(op->getOperands(), [&](Value operand) {
      return dependsOnMemory(operand);
    });
  }
  return memoryDeps[value] = result;
}

bool Threader::dependsOnState(Value value) {
  if (auto it = stateDeps.find(value); it != stateDeps.end())
    return it->second;
  auto *op = value.getDefiningOp();
  bool result = false;
  if (isa_and_nonnull<seq::FirRegOp>(op)) {
    result = true;
  } else if (op) {
    stateDeps[value] = false;
    result = llvm::any_of(op->getOperands(), [&](Value operand) {
      return dependsOnState(operand);
    });
  }
  return stateDeps[value] = result;
}

/// The value `value` takes in the next target cycle: its logic with every
/// register replaced by the value it is about to take. Only values that do
/// not depend on an input combinationally have one.
Value Threader::getNext(Value value, ImplicitLocOpBuilder &b) {
  if (auto it = nexts.find(value); it != nexts.end())
    return it->second;
  Value result;
  if (auto reg = value.getDefiningOp<seq::FirRegOp>()) {
    result = reg.getNext();
    if (reg.getReset())
      result =
          comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(), result);
  } else if (!dependsOnState(value)) {
    result = value;
  } else {
    auto *op = value.getDefiningOp();
    IRMapping mapping;
    for (auto operand : op->getOperands())
      mapping.map(operand, getNext(operand, b));
    auto *clone = b.clone(*op, mapping);
    for (auto [old, cloned] : llvm::zip(op->getResults(), clone->getResults()))
      nexts[old] = cloned;
    return nexts[value];
  }
  return nexts[value] = result;
}

Value Threader::createThreadState(ImplicitLocOpBuilder &b, StringRef stateName,
                                  Type type, Value &memory) {
  auto width = hw::getBitWidth(type);
  auto memType = seq::FirMemType::get(b.getContext(), threads, width, 1);
  memory = seq::FirMemOp::create(
      b, memType, /*readLatency=*/0, /*writeLatency=*/1, seq::RUW::Undefined,
      seq::WUW::Undefined, b.getStringAttr(stateName), hw::InnerSymAttr{},
      seq::FirMemInitAttr{}, StringAttr{}, Attribute{});
  Value data =
      seq::FirMemReadOp::create(b, memory, threadIdx, hostClock,
                                hw::ConstantOp::create(b, b.getI1Type(), 1));
  return b.createOrFold<hw::BitcastOp>(type, data);
}

void Threader::writeThreadState(ImplicitLocOpBuilder &b, Value memory,
                                Value clock, Value data, Value enable) {
  auto width = cast<seq::FirMemType>(memory.getType()).getWidth();
  data = b.createOrFold<hw::BitcastOp>(b.getIntegerType(width), data);
  if (!enable)
    enable = hw::ConstantOp::create(b, b.getI1Type(), 1);
  seq::FirMemWriteOp::create(b, memory, threadIdx, clock, enable, data,
                             Value{});
}

/// Bits `[low, low + width)` of the integer `value`, looking through wiring.
static Value sliceBits(ImplicitLocOpBuilder &b, Value value, unsigned low,
                       unsigned width) {
  if (low == 0 && hw::getBitWidth(value.getType()) == width)
    return value;
  if (auto extract = value.getDefiningOp<comb::ExtractOp>())
    return sliceBits(b, extract.getInput(), extract.getLowBit() + low, width);
  if (auto concat = value.getDefiningOp<comb::ConcatOp>()) {
    // The operands are most significant first.
    SmallVector<Value> pieces;
    unsigned operandLow = hw::getBitWidth(concat.getType());
    for (auto operand : concat.getInputs()) {
      unsigned operandWidth = hw::getBitWidth(operand.getType());
      operandLow -= operandWidth;
      unsigned lo = std::max(low, operandLow);
      unsigned hi = std::min(low + width, operandLow + operandWidth);
      if (lo < hi)
        pieces.push_back(sliceBits(b, operand, lo - operandLow, hi - lo));
    }
    return pieces.size() == 1 ? pieces[0]
                              : b.createOrFold<comb::ConcatOp>(pieces);
  }
  return b.createOrFold<comb::ExtractOp>(value, low, width);
}

/// The concatenation of `pieces`, adjacent slices of one value joined.
static Value joinBits(ImplicitLocOpBuilder &b, ArrayRef<Value> pieces) {
  SmallVector<Value> joined;
  for (auto piece : pieces) {
    auto extract = piece.getDefiningOp<comb::ExtractOp>();
    auto last = joined.empty() ? comb::ExtractOp()
                               : joined.back().getDefiningOp<comb::ExtractOp>();
    if (extract && last && last.getInput() == extract.getInput() &&
        extract.getLowBit() + hw::getBitWidth(extract.getType()) ==
            last.getLowBit()) {
      joined.back() = sliceBits(b, extract.getInput(), extract.getLowBit(),
                                hw::getBitWidth(extract.getType()) +
                                    hw::getBitWidth(last.getType()));
      continue;
    }
    joined.push_back(piece);
  }
  return joined.size() == 1 ? joined[0]
                            : b.createOrFold<comb::ConcatOp>(joined);
}

/// `value` as `type`, looking through bitcasts.
static Value castBits(ImplicitLocOpBuilder &b, Value value, Type type) {
  if (value.getType() == type)
    return value;
  if (auto cast = value.getDefiningOp<hw::BitcastOp>())
    return castBits(b, cast.getInput(), type);
  return b.createOrFold<hw::BitcastOp>(type, value);
}

/// `value` with the mux below its wiring (bitcasts, extracts and concats)
/// moved to the top, or `value` if there is none: a mux of the slices of the
/// register is the register.
static Value hoistMux(ImplicitLocOpBuilder &b, Value value) {
  auto *op = value.getDefiningOp();
  if (!op)
    return value;
  if (auto cast = dyn_cast<hw::BitcastOp>(op)) {
    auto mux = hoistMux(b, cast.getInput()).getDefiningOp<comb::MuxOp>();
    if (!mux)
      return value;
    return comb::MuxOp::create(
        b, mux.getCond(), castBits(b, mux.getTrueValue(), value.getType()),
        castBits(b, mux.getFalseValue(), value.getType()));
  }
  if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
    auto mux = hoistMux(b, extract.getInput()).getDefiningOp<comb::MuxOp>();
    if (!mux)
      return value;
    unsigned low = extract.getLowBit(),
             width = hw::getBitWidth(extract.getType());
    return comb::MuxOp::create(b, mux.getCond(),
                               sliceBits(b, mux.getTrueValue(), low, width),
                               sliceBits(b, mux.getFalseValue(), low, width));
  }
  if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
    SmallVector<Value> pieces;
    Value cond;
    for (auto operand : concat.getInputs()) {
      pieces.push_back(hoistMux(b, operand));
      if (auto mux = pieces.back().getDefiningOp<comb::MuxOp>(); mux && !cond)
        cond = mux.getCond();
    }
    if (!cond)
      return value;
    // The pieces under another condition are the same on both sides.
    SmallVector<Value> onTrue, onFalse;
    for (auto piece : pieces) {
      auto mux = piece.getDefiningOp<comb::MuxOp>();
      bool split = mux && mux.getCond() == cond;
      onTrue.push_back(split ? mux.getTrueValue() : piece);
      onFalse.push_back(split ? mux.getFalseValue() : piece);
    }
    return comb::MuxOp::create(b, cond, joinBits(b, onTrue),
                               joinBits(b, onFalse));
  }
  return value;
}

Threader::Write Threader::peelHolds(ImplicitLocOpBuilder &b, Value next,
                                    Value reg) {
  if (next == reg)
    return {/*holds=*/true, {}, {}};
  next = hoistMux(b, next);
  if (next == reg)
    return {/*holds=*/true, {}, {}};
  auto mux = next.getDefiningOp<comb::MuxOp>();
  if (!mux)
    return {false, {}, next};
  auto onTrue = peelHolds(b, mux.getTrueValue(), reg);
  auto onFalse = peelHolds(b, mux.getFalseValue(), reg);
  if (onTrue.holds && onFalse.holds)
    return {true, {}, {}};
  auto one = [&]() { return hw::ConstantOp::create(b, b.getI1Type(), 1); };
  auto cond = mux.getCond();
  if (onTrue.holds)
    return {false,
            onFalse.enable
                ? b.createOrFold<comb::AndOp>(
                      b.createOrFold<comb::XorOp>(cond, one()), onFalse.enable)
                : b.createOrFold<comb::XorOp>(cond, one()),
            onFalse.data};
  if (onFalse.holds)
    return {false,
            onTrue.enable ? b.createOrFold<comb::AndOp>(cond, onTrue.enable)
                          : cond,
            onTrue.data};
  Value enable;
  if (onTrue.enable || onFalse.enable)
    enable =
        b.createOrFold<comb::MuxOp>(cond, onTrue.enable ? onTrue.enable : one(),
                                    onFalse.enable ? onFalse.enable : one());
  Value data =
      onTrue.data == mux.getTrueValue() && onFalse.data == mux.getFalseValue()
          ? next
          : b.createOrFold<comb::MuxOp>(cond, onTrue.data, onFalse.data);
  return {false, enable, data};
}

LogicalResult Threader::run() {
  if (failed(checkModel()))
    return failure();
  computeDeps();

  // The threaded module's ports: the host clock and reset, then a decoupled
  // channel per target port, inputs first, its bits flat as Golden Gate's.
  auto *context = model.getContext();
  auto i1 = IntegerType::get(context, 1);
  SmallVector<hw::PortInfo> ports;
  size_t numIn = 0, numOut = 0;
  auto addPort = [&](const Twine &portName, Type type, bool isInput) {
    hw::PortInfo port;
    port.name = StringAttr::get(context, portName);
    port.type = type;
    port.dir = isInput ? hw::ModulePort::Direction::Input
                       : hw::ModulePort::Direction::Output;
    port.argNum = isInput ? numIn++ : numOut++;
    ports.push_back(port);
  };
  addPort("hostClock", seq::ClockType::get(context), true);
  addPort("hostReset", i1, true);
  for (auto &channel : inputs) {
    addPort(channel.name.getValue() + "_sink_ready", i1, false);
    addPort(channel.name.getValue() + "_sink_valid", i1, true);
    addPort(channel.name.getValue() + "_sink_bits",
            IntegerType::get(context, hw::getBitWidth(channel.type)), true);
  }
  for (auto &channel : outputs) {
    addPort(channel.name.getValue() + "_source_ready", i1, true);
    addPort(channel.name.getValue() + "_source_valid", i1, false);
    if (hw::getBitWidth(channel.type) > 0)
      addPort(channel.name.getValue() + "_source_bits",
              IntegerType::get(context, hw::getBitWidth(channel.type)), false);
  }

  OpBuilder top(model);
  auto threaded = hw::HWModuleOp::create(top, model.getLoc(), name,
                                         hw::ModulePortInfo(ports));
  auto *body = threaded.getBodyBlock();
  body->getTerminator()->erase();
  hostClock = body->getArgument(0);
  Value hostReset = body->getArgument(1);
  SmallVector<Value> sinkValid, sinkBits, sourceReady;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    sinkValid.push_back(body->getArgument(2 + 2 * i));
    sinkBits.push_back(body->getArgument(3 + 2 * i));
  }
  for (unsigned i = 0; i < outputs.size(); ++i)
    sourceReady.push_back(body->getArgument(2 + 2 * inputs.size() + i));

  // The target logic moves into the threaded module as is, reading the
  // input channels' bits.
  auto *modelBody = model.getBodyBlock();
  auto modelOutput = modelBody->getTerminator();
  body->getOperations().splice(body->end(), modelBody->getOperations(),
                               modelBody->begin(),
                               Block::iterator(modelOutput));
  ImplicitLocOpBuilder b(model.getLoc(), body, body->end());
  BackedgeBuilder backedges(b, model.getLoc());
  auto targetClock = backedges.get(seq::ClockType::get(context));
  modelBody->getArgument(clockArg).replaceAllUsesWith(targetClock);
  // Values kept aside for the outputs may be replaced below.
  DenseMap<Value, Value> replaced;
  // An input split into channels is their bits put back together.
  for (auto [i, channel] : llvm::enumerate(inputs)) {
    auto arg = modelBody->getArgument(channel.argNum);
    if (replaced.count(arg))
      continue;
    Value value = sinkBits[i];
    if (channel.sliced) {
      SmallVector<Value> parts;
      for (auto [j, other] : llvm::enumerate(inputs))
        if (other.argNum == channel.argNum)
          parts.push_back(sinkBits[j]);
      // Channels of a port are in ascending bit order; concat wants the most
      // significant first.
      std::reverse(parts.begin(), parts.end());
      value = b.createOrFold<comb::ConcatOp>(parts);
    }
    value = b.createOrFold<hw::BitcastOp>(arg.getType(), value);
    arg.replaceAllUsesWith(value);
    replaced[arg] = value;
  }

  // The current thread, advancing every host cycle.
  auto idxType = b.getIntegerType(std::max(1u, llvm::Log2_32_Ceil(threads)));
  auto idxNext = backedges.get(idxType);
  threadIdx = seq::FirRegOp::create(b, idxNext, hostClock,
                                    b.getStringAttr("threadIdx"));
  auto isLast =
      comb::ICmpOp::create(b, comb::ICmpPredicate::eq, threadIdx,
                           hw::ConstantOp::create(b, idxType, threads - 1));
  idxNext.setValue(comb::MuxOp::create(
      b, isLast, hw::ConstantOp::create(b, idxType, 0),
      comb::AddOp::create(b, threadIdx,
                          hw::ConstantOp::create(b, idxType, 1))));

  // An output that depends on no input fires eagerly, with the next cycle's
  // value once its thread has finished.
  for (auto &channel : outputs) {
    channel.eager = channel.deps.none() && !dependsOnMemory(channel.current);
    if (channel.eager)
      channel.next = getNext(channel.current, b);
  }

  // Each register keeps one value per thread. The branches of its next value
  // that hold it become the write enable of its memory rather than logic that
  // writes it back.
  for (auto reg : registers) {
    Value next = reg.getNext();
    if (reg.getReset())
      next = comb::MuxOp::create(b, reg.getReset(), reg.getResetValue(), next);
    auto write = peelHolds(b, next, reg.getResult());
    // Written every cycle: the next value as it is.
    if (!write.holds && !write.enable)
      write.data = next;
    Value memory;
    auto value = createThreadState(b, reg.getName(), reg.getType(), memory);
    reg.getResult().replaceAllUsesWith(value);
    replaced[reg.getResult()] = value;
    if (!write.holds)
      writeThreadState(b, memory, targetClock, write.data, write.enable);
    reg.erase();
  }

  // Each memory holds every thread's contents, the thread in the low address
  // bits, and is written on the target clock. A registered read reads on the
  // target clock too, and its data is kept per thread in a buffer filled on
  // the next host cycle, as FireSim's FameRam does, so that the memory stays
  // a block RAM.
  Value edgeSeen;
  auto getEdgeSeen = [&]() {
    if (edgeSeen)
      return edgeSeen;
    auto zero = b.getIntegerAttr(i1, 0);
    auto toggleNext = backedges.get(i1);
    auto toggle = seq::FirRegOp::create(b, toggleNext, targetClock,
                                        b.getStringAttr("target_toggle"),
                                        hw::InnerSymAttr{}, zero);
    toggleNext.setValue(
        comb::XorOp::create(b, toggle, hw::ConstantOp::create(b, i1, 1)));
    auto seen = seq::FirRegOp::create(b, toggle, hostClock,
                                      b.getStringAttr("target_toggle_seen"),
                                      hw::InnerSymAttr{}, zero);
    edgeSeen = comb::XorOp::create(b, toggle, seen);
    return edgeSeen;
  };
  auto idxWidth = threadIdx.getType().getIntOrFloatBitWidth();
  for (auto memory : memories) {
    auto memType = memory.getType();
    auto depth = memType.getDepth() * threads;
    auto threadedType = seq::FirMemType::get(
        b.getContext(), depth, memType.getWidth(), memType.getMaskWidth());
    auto threadedMemory = seq::FirMemOp::create(
        b, threadedType, memory.getReadLatency(), memory.getWriteLatency(),
        memory.getRuw(), memory.getWuw(), memory.getNameAttr(),
        hw::InnerSymAttr{}, seq::FirMemInitAttr{}, memory.getPrefixAttr(),
        memory.getOutputFileAttr());
    threadedMemory->setDialectAttrs(memory->getDialectAttrs());
    unsigned addrWidth = llvm::Log2_64_Ceil(depth) - idxWidth;
    auto addressOf = [&](Value addr) -> Value {
      if (!addrWidth)
        return threadIdx;
      Value low =
          b.createOrFold<comb::ExtractOp>(b.getIntegerType(addrWidth), addr, 0);
      return b.createOrFold<comb::ConcatOp>(low, threadIdx);
    };
    for (auto *user : llvm::make_early_inc_range(memory->getUsers())) {
      if (auto write = dyn_cast<seq::FirMemWriteOp>(user)) {
        seq::FirMemWriteOp::create(
            b, threadedMemory, addressOf(write.getAddress()), targetClock,
            write.getEnable(), write.getData(), write.getMask());
        write.erase();
        continue;
      }
      auto read = cast<seq::FirMemReadOp>(user);
      Value data;
      if (memory.getReadLatency() == 0) {
        data = seq::FirMemReadOp::create(b, threadedMemory,
                                         addressOf(read.getAddress()),
                                         hostClock, read.getEnable());
      } else {
        Value dout = seq::FirMemReadOp::create(b, threadedMemory,
                                               addressOf(read.getAddress()),
                                               targetClock, read.getEnable());
        Value enable = read.getEnable()
                           ? read.getEnable()
                           : Value(hw::ConstantOp::create(b, i1, 1));
        auto readEdge = seq::FirRegOp::create(b, enable, targetClock,
                                              b.getStringAttr("read_edge"));
        auto readThread = seq::FirRegOp::create(b, threadIdx, targetClock,
                                                b.getStringAttr("read_thread"));
        Value buffer;
        data = createThreadState(
            b, (memory.getName().value_or("mem") + "_rdata").str(),
            dout.getType(), buffer);
        seq::FirMemWriteOp::create(
            b, buffer, readThread, hostClock,
            comb::AndOp::create(b, getEdgeSeen(), readEdge), dout, Value{});
      }
      read.getData().replaceAllUsesWith(data);
      replaced[read.getData()] = data;
      read.erase();
    }
    memory.erase();
  }

  auto resolve = [&](Value value) {
    auto it = replaced.find(value);
    return it == replaced.end() ? value : it->second;
  };
  // A channel of some bits of an output carries just those.
  auto sliceOf = [&](Channel &channel, Value value) -> Value {
    if (!value || !channel.sliced)
      return value;
    auto width = hw::getBitWidth(value.getType());
    Value bits = b.createOrFold<hw::BitcastOp>(b.getIntegerType(width), value);
    return b.createOrFold<comb::ExtractOp>(channel.type, bits, channel.low);
  };
  for (auto &channel : outputs) {
    channel.current = sliceOf(channel, resolve(channel.current));
    channel.next = sliceOf(channel, resolve(channel.next));
  }

  // The channel flags of the current thread.
  auto one = hw::ConstantOp::create(b, i1, 1);
  auto zero = hw::ConstantOp::create(b, i1, 0);
  auto notOf = [&](Value v) { return comb::XorOp::create(b, v, one); };
  SmallVector<Value> inFired, inMem, outFired, outMem;
  for (auto &channel : inputs) {
    inMem.emplace_back();
    inFired.push_back(createThreadState(
        b, (channel.name.getValue() + "_fired").str(), i1, inMem.back()));
  }
  for (auto &channel : outputs) {
    outMem.emplace_back();
    outFired.push_back(createThreadState(
        b, (channel.name.getValue() + "_fired").str(), i1, outMem.back()));
  }
  Value clkEnabledMem;
  auto clkEnabled = createThreadState(b, "clk_enabled", i1, clkEnabledMem);

  // The thread finishes its target cycle when every input is valid and every
  // output has fired or fires now.
  auto finishing = backedges.get(i1);
  SmallVector<Value> outValid, firedOrFiring, finishTerms;
  for (auto [i, channel] : llvm::enumerate(outputs)) {
    Value fired = outFired[i], ready = sourceReady[i];
    Value valid, fof;
    if (channel.eager) {
      valid = comb::OrOp::create(b, notOf(fired),
                                 comb::AndOp::create(b, finishing, fired));
      fof = comb::OrOp::create(b, fired,
                               comb::AndOp::create(b, ready, notOf(fired)));
    } else {
      SmallVector<Value> terms;
      for (auto dep : channel.deps.set_bits())
        terms.push_back(sinkValid[dep]);
      terms.push_back(notOf(fired));
      valid = comb::AndOp::create(b, ValueRange(terms), false);
      fof = comb::OrOp::create(b, fired, comb::AndOp::create(b, ready, valid));
    }
    outValid.push_back(valid);
    firedOrFiring.push_back(fof);
    finishTerms.push_back(fof);
  }
  finishTerms.append(sinkValid.begin(), sinkValid.end());
  finishing.setValue(
      finishTerms.empty()
          ? Value(one)
          : comb::AndOp::create(b, ValueRange(finishTerms), false));

  auto unlessReset = [&](Value v) {
    return comb::MuxOp::create(b, hostReset, zero, v);
  };
  for (auto [i, channel] : llvm::enumerate(outputs)) {
    Value fired = outFired[i];
    Value onFinish = channel.eager
                         ? Value(comb::AndOp::create(b, fired, sourceReady[i]))
                         : Value(zero);
    writeThreadState(b, outMem[i], hostClock,
                     unlessReset(comb::MuxOp::create(b, finishing, onFinish,
                                                     firedOrFiring[i])));
  }
  SmallVector<Value> sinkReady;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    Value ready = comb::AndOp::create(b, finishing, notOf(inFired[i]));
    sinkReady.push_back(ready);
    Value accepted = comb::OrOp::create(
        b, inFired[i], comb::AndOp::create(b, ready, sinkValid[i]));
    writeThreadState(
        b, inMem[i], hostClock,
        unlessReset(comb::MuxOp::create(b, finishing, zero, accepted)));
  }

  // The target clock ticks for the thread when it finishes, from its second
  // finish after reset on.
  writeThreadState(b, clkEnabledMem, hostClock,
                   unlessReset(comb::OrOp::create(b, finishing, clkEnabled)));
  Value enable = comb::AndOp::create(
      b, ValueRange{clkEnabled, finishing, notOf(hostReset)}, false);
  auto clockGate =
      hw::InstanceOp::create(b, gate, b.getStringAttr("clk_buffer"),
                             ArrayRef<Value>{hostClock, enable});
  targetClock.setValue(clockGate.getResult(0));

  SmallVector<Value> results(sinkReady);
  for (auto [i, channel] : llvm::enumerate(outputs)) {
    results.push_back(outValid[i]);
    if (hw::getBitWidth(channel.type) == 0)
      continue;
    Value bits = channel.eager
                     ? Value(comb::MuxOp::create(b, outFired[i], channel.next,
                                                 channel.current))
                     : channel.current;
    results.push_back(b.createOrFold<hw::BitcastOp>(
        b.getIntegerType(hw::getBitWidth(channel.type)), bits));
  }
  hw::OutputOp::create(b, results);

  // Which output channels fire eagerly is part of the model's timing, which
  // the simulator around it relies on.
  SmallVector<Attribute> eager;
  for (auto &channel : outputs)
    if (channel.eager)
      eager.push_back(channel.name);
  threaded->setAttr("maxsim.eager_channels", b.getArrayAttr(eager));
  model.erase();
  return success();
}

void MaxsimFAME5Pass::runOnOperation() {
  auto top = getOperation();
  auto model = dyn_cast_or_null<hw::HWModuleOp>(
      SymbolTable::lookupSymbolIn(top, moduleName));
  if (!model) {
    top.emitError("no module '") << moduleName << "' to thread";
    return signalPassFailure();
  }
  if (threads < 2) {
    model.emitError("threading needs at least two threads");
    return signalPassFailure();
  }

  // The clock gate the FireSim platform provides.
  auto gate = dyn_cast_or_null<hw::HWModuleExternOp>(
      SymbolTable::lookupSymbolIn(top, "FireSimFabricClockGate"));
  if (!gate) {
    OpBuilder builder(top.getBody(), top.getBody()->begin());
    auto clock = seq::ClockType::get(&getContext());
    SmallVector<hw::PortInfo> ports(3);
    ports[0] = {
        {builder.getStringAttr("I"), clock, hw::ModulePort::Direction::Input},
        0};
    ports[1] = {{builder.getStringAttr("CE"), builder.getI1Type(),
                 hw::ModulePort::Direction::Input},
                1};
    ports[2] = {
        {builder.getStringAttr("O"), clock, hw::ModulePort::Direction::Output},
        0};
    gate = hw::HWModuleExternOp::create(
        builder, top.getLoc(), builder.getStringAttr("FireSimFabricClockGate"),
        hw::ModulePortInfo(ports));
  }

  auto name = threadedName.empty()
                  ? StringAttr::get(&getContext(),
                                    moduleName + "_threaded_" + moduleName)
                  : StringAttr::get(&getContext(), threadedName);
  Threader::Slices slices;
  for (auto &slice : portSlices) {
    SmallVector<StringRef, 3> fields;
    StringRef(slice).split(fields, ':');
    unsigned hi, lo;
    if (fields.size() != 3 || fields[1].getAsInteger(10, hi) ||
        fields[2].getAsInteger(10, lo)) {
      model.emitError("port slice '") << slice << "' is not <port>:<hi>:<lo>";
      return signalPassFailure();
    }
    slices[fields[0]].push_back({hi, lo});
  }
  if (failed(Threader(model, threads, name, gate, slices).run()))
    return signalPassFailure();
}
