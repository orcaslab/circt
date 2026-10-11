//===- MaxsimExtractModels.cpp - Cut child models out of a parent ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FireSim's Golden Gate simulates each partitioned class as its own model,
// connected to the others by channels. This pass cuts the instances of those
// classes out of a flat parent, leaving the parent's own logic as a model whose
// ports are what it exchanges with them.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace circt {
#define GEN_PASS_DEF_MAXSIMEXTRACTMODELS
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

struct MaxsimExtractModelsPass
    : public circt::impl::MaxsimExtractModelsBase<MaxsimExtractModelsPass> {
  using MaxsimExtractModelsBase::MaxsimExtractModelsBase;
  void runOnOperation() override;
};

/// Where one bit of a value comes from: logic, a constant, or a bit of a port
/// it is wired from.
struct BitSource {
  enum Kind { Logic, Constant, Port } kind = Logic;
  Value port;
  unsigned bit = 0;
  bool value = false;
};

using BitSources = SmallVector<BitSource>;

/// Traces the bits of values through the ops that only move bits around and,
/// optionally, through register bits that only ever take one constant value.
class BitTracer {
public:
  BitTracer(const DenseSet<Value> &ports, bool throughRegisters)
      : ports(ports), throughRegisters(throughRegisters) {}

  /// Whether `op` only moves bits around, and so is traced through.
  static bool isWiring(Operation *op);
  const BitSources &get(Value value);

private:
  BitSources compute(Value value);
  BitSources computeRegister(seq::FirRegOp reg);

  const DenseSet<Value> &ports;
  bool throughRegisters;
  DenseMap<Value, std::unique_ptr<BitSources>> cache;
};

} // namespace

static unsigned getElementWidth(Type arrayType) {
  return hw::getBitWidth(
      hw::type_cast<hw::ArrayType>(arrayType).getElementType());
}

bool BitTracer::isWiring(Operation *op) {
  if (isa<hw::ConstantOp, hw::AggregateConstantOp, comb::ConcatOp,
          comb::ExtractOp, hw::BitcastOp, hw::ArrayCreateOp, hw::ArrayConcatOp,
          hw::StructCreateOp, hw::StructExtractOp>(op))
    return true;
  if (auto get = dyn_cast<hw::ArrayGetOp>(op))
    return matchPattern(get.getIndex(), m_Constant());
  if (auto slice = dyn_cast<hw::ArraySliceOp>(op))
    return matchPattern(slice.getLowIndex(), m_Constant());
  return false;
}

/// The bits of an aggregate constant: the first element of an array or field
/// of a struct is in the high bits.
static std::optional<APInt> getAggregateValue(Attribute attr, Type type) {
  auto width = hw::getBitWidth(type);
  if (width < 0)
    return std::nullopt;
  if (auto integer = dyn_cast<IntegerAttr>(attr))
    return integer.getValue().zextOrTrunc(width);
  auto elements = dyn_cast<ArrayAttr>(attr);
  if (!elements)
    return std::nullopt;
  SmallVector<Type> types;
  if (auto array = hw::type_dyn_cast<hw::ArrayType>(type))
    types.assign(array.getNumElements(), array.getElementType());
  else if (auto structType = hw::type_dyn_cast<hw::StructType>(type))
    for (auto &field : structType.getElements())
      types.push_back(field.type);
  if (types.size() != elements.size())
    return std::nullopt;
  std::optional<APInt> result;
  for (auto [element, elementType] : llvm::zip(elements, types)) {
    auto value = getAggregateValue(element, elementType);
    if (!value)
      return std::nullopt;
    result = result ? result->concat(*value) : *value;
  }
  return result;
}

const BitSources &BitTracer::get(Value value) {
  auto it = cache.find(value);
  if (it != cache.end())
    return *it->second;
  // A value reached again while it is traced, through a register, is logic.
  auto width = std::max<int64_t>(hw::getBitWidth(value.getType()), 0);
  cache[value] = std::make_unique<BitSources>(width);
  auto sources = compute(value);
  return *(cache[value] = std::make_unique<BitSources>(std::move(sources)));
}

BitSources BitTracer::compute(Value value) {
  auto width = hw::getBitWidth(value.getType());
  BitSources sources(std::max<int64_t>(width, 0));
  if (width <= 0)
    return sources;
  if (ports.contains(value)) {
    for (unsigned bit = 0; bit < sources.size(); ++bit)
      sources[bit] = {BitSource::Port, value, bit};
    return sources;
  }
  auto *op = value.getDefiningOp();
  if (auto reg = dyn_cast_or_null<seq::FirRegOp>(op); reg && throughRegisters)
    return computeRegister(reg);
  if (!op || !isWiring(op))
    return sources;

  // The bits of `operands`, the last one in the low bits.
  auto appendReversed = [&](ValueRange operands) {
    BitSources result;
    for (auto operand : llvm::reverse(operands)) {
      auto bits = get(operand);
      result.append(bits.begin(), bits.end());
    }
    return result;
  };
  auto slice = [&](Value input, unsigned low) {
    auto bits = get(input);
    return BitSources(bits.begin() + low, bits.begin() + low + width);
  };

  if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
    for (auto [bit, source] : llvm::enumerate(sources)) {
      source.kind = BitSource::Constant;
      source.value = constant.getValue()[bit];
    }
    return sources;
  }
  if (auto constant = dyn_cast<hw::AggregateConstantOp>(op)) {
    auto value = getAggregateValue(constant.getFields(), constant.getType());
    if (!value)
      return sources;
    for (auto [bit, source] : llvm::enumerate(sources)) {
      source.kind = BitSource::Constant;
      source.value = (*value)[bit];
    }
    return sources;
  }
  if (auto concat = dyn_cast<comb::ConcatOp>(op))
    return appendReversed(concat.getInputs());
  if (auto extract = dyn_cast<comb::ExtractOp>(op))
    return slice(extract.getInput(), extract.getLowBit());
  if (auto bitcast = dyn_cast<hw::BitcastOp>(op))
    return get(bitcast.getInput());
  // Array element 0 is in the low bits, as the first operand of a create or a
  // concatenation is in the high ones.
  if (auto create = dyn_cast<hw::ArrayCreateOp>(op))
    return appendReversed(create.getInputs());
  if (auto concat = dyn_cast<hw::ArrayConcatOp>(op))
    return appendReversed(concat.getInputs());
  // A struct's first field is in the high bits.
  if (auto create = dyn_cast<hw::StructCreateOp>(op))
    return appendReversed(create.getInput());
  if (auto extract = dyn_cast<hw::StructExtractOp>(op)) {
    auto elements = hw::type_cast<hw::StructType>(extract.getInput().getType())
                        .getElements();
    unsigned low = 0;
    for (auto &element : elements.drop_front(extract.getFieldIndex() + 1))
      low += hw::getBitWidth(element.type);
    return slice(extract.getInput(), low);
  }
  APInt index;
  if (auto get = dyn_cast<hw::ArrayGetOp>(op);
      get && matchPattern(get.getIndex(), m_ConstantInt(&index)))
    return slice(get.getInput(), index.getZExtValue() *
                                     getElementWidth(get.getInput().getType()));
  if (auto arraySlice = dyn_cast<hw::ArraySliceOp>(op);
      arraySlice &&
      matchPattern(arraySlice.getLowIndex(), m_ConstantInt(&index)))
    return slice(arraySlice.getInput(),
                 index.getZExtValue() *
                     getElementWidth(arraySlice.getInput().getType()));
  return sources;
}

/// A register bit is that constant if its next value and its reset value, if
/// any, are always the same constant: its value is undefined only before the
/// first clock edge.
BitSources BitTracer::computeRegister(seq::FirRegOp reg) {
  BitSources sources(hw::getBitWidth(reg.getType()));
  auto next = get(reg.getNext());
  BitSources reset;
  if (reg.hasReset())
    reset = get(reg.getResetValue());
  for (auto [bit, source] : llvm::enumerate(sources)) {
    if (next[bit].kind != BitSource::Constant)
      continue;
    if (!reset.empty() && (reset[bit].kind != BitSource::Constant ||
                           reset[bit].value != next[bit].value))
      continue;
    source = next[bit];
  }
  return sources;
}

/// The name of the port carrying bits `lo` to `hi` of a port `width` wide.
static std::string sliceName(const Twine &base, unsigned hi, unsigned lo,
                             unsigned width) {
  if (lo == 0 && hi + 1 == width)
    return base.str();
  if (hi == lo)
    return (base + "_" + Twine(hi) + "_").str();
  return (base + "_" + Twine(hi) + "_" + Twine(lo) + "_").str();
}

/// A name usable as a Verilog identifier.
static std::string sanitize(StringRef name) {
  std::string result = name.str();
  for (auto &c : result)
    if (!llvm::isAlnum(c) && c != '_')
      c = '_';
  return result;
}

/// For each output of a model, the bits that are always the same constant.
static SmallVector<BitSources> getConstantOutputs(hw::HWModuleOp model) {
  auto *body = model.getBodyBlock();
  DenseSet<Value> ports(body->args_begin(), body->args_end());
  BitTracer tracer(ports, /*throughRegisters=*/true);
  SmallVector<BitSources> outputs;
  for (auto output : body->getTerminator()->getOperands())
    outputs.push_back(tracer.get(output));
  return outputs;
}

void MaxsimExtractModelsPass::runOnOperation() {
  auto top = getOperation();
  auto parent = dyn_cast_or_null<hw::HWModuleOp>(
      SymbolTable::lookupSymbolIn(top, moduleName));
  if (!parent) {
    top.emitError("no module '") << moduleName << "' to extract models from";
    return signalPassFailure();
  }
  llvm::StringSet<> modelNames;
  for (auto &name : models)
    modelNames.insert(name);
  llvm::StringMap<SmallVector<std::pair<unsigned, unsigned>>> givenInputs;
  for (auto &entry : inputBits) {
    SmallVector<StringRef, 3> fields;
    StringRef(entry).split(fields, ':');
    unsigned hi, lo;
    if (fields.size() != 3 || fields[1].getAsInteger(10, hi) ||
        fields[2].getAsInteger(10, lo) || hi < lo) {
      parent.emitError("input bits '") << entry << "' are not <port>:<hi>:<lo>";
      return signalPassFailure();
    }
    givenInputs[fields[0]].push_back({hi, lo});
  }

  auto *body = parent.getBodyBlock();
  SmallVector<hw::InstanceOp> instances;
  DenseSet<Value> ports(body->args_begin(), body->args_end());
  DenseMap<StringAttr, SmallVector<BitSources>> constantOutputs;
  for (auto instance : body->getOps<hw::InstanceOp>()) {
    if (!modelNames.contains(instance.getModuleName()))
      continue;
    for (auto type : instance->getOperandTypes())
      if (isa<hw::InOutType>(type)) {
        instance.emitError("inout port on a model");
        return signalPassFailure();
      }
    instances.push_back(instance);
    ports.insert(instance.getResults().begin(), instance.getResults().end());
    auto name = instance.getModuleNameAttr().getAttr();
    if (!constantOutputs.count(name))
      if (auto model = dyn_cast_or_null<hw::HWModuleOp>(
              SymbolTable::lookupSymbolIn(top, name)))
        constantOutputs[name] = getConstantOutputs(model);
  }

  // Each model input becomes an output of the parent, unless inputs of the
  // parent drive it, with constants: the model is connected to those directly.
  // One that only constants drive is a channel of tokens, with no bits.
  BitTracer tracer(ports, /*throughRegisters=*/false);
  auto builder = ImplicitLocOpBuilder::atBlockTerminator(parent.getLoc(), body);
  SmallVector<std::pair<StringAttr, Value>> newOutputs;
  for (auto instance : instances) {
    auto base = sanitize(instance.getInstanceName());
    for (auto [operand, name] :
         llvm::zip(instance.getInputs(), instance.getArgNames())) {
      auto width = hw::getBitWidth(operand.getType());
      if (isa<seq::ClockType>(operand.getType()) || width <= 0)
        continue;
      auto &sources = tracer.get(operand);
      auto isConstant = [](auto &source) {
        return source.kind == BitSource::Constant;
      };
      bool direct = llvm::all_of(sources, [&](auto &source) {
        return isConstant(source) || (source.kind == BitSource::Port &&
                                      isa<BlockArgument>(source.port));
      });
      if (direct && !llvm::all_of(sources, isConstant))
        continue;
      Value value = operand;
      if (direct)
        value = hw::ConstantOp::create(builder, builder.getIntegerType(0), 0);
      else if (!value.getType().isInteger(width))
        value = hw::BitcastOp::create(builder, builder.getIntegerType(width),
                                      value);
      newOutputs.push_back({builder.getStringAttr(
                                base + "_" + cast<StringAttr>(name).getValue()),
                            value});
    }
  }
  if (!newOutputs.empty())
    parent.insertOutputs(parent.getNumOutputPorts(), newOutputs);

  // Each model output becomes inputs of its bits, but for those that are
  // always the same constant.
  builder.setInsertionPoint(body, body->begin());
  for (auto instance : instances) {
    auto base = sanitize(instance.getInstanceName());
    auto constants =
        constantOutputs.lookup(instance.getModuleNameAttr().getAttr());
    for (auto [index, result, name] :
         llvm::enumerate(instance.getResults(), instance.getResultNames())) {
      auto width = hw::getBitWidth(result.getType());
      if (isa<seq::ClockType>(result.getType()) || width <= 0) {
        if (!result.use_empty()) {
          instance.emitError("model output '")
              << name << "' has no bits but is used";
          return signalPassFailure();
        }
        continue;
      }
      // Runs of bits from the low one: each a constant or an input.
      SmallVector<Value> parts;
      unsigned lo = 0;
      const BitSources *bits = constants.empty() ? nullptr : &constants[index];
      auto isConstant = [&](unsigned bit) {
        return bits && (*bits)[bit].kind == BitSource::Constant;
      };
      auto portBase = base + "_" + cast<StringAttr>(name).getValue().str();
      auto constantPart = [&](unsigned lo, unsigned hi) {
        APInt value(hi - lo + 1, 0);
        for (unsigned bit = lo; bit <= hi; ++bit)
          value.setBitVal(bit - lo, (*bits)[bit].value);
        parts.push_back(hw::ConstantOp::create(builder, value));
      };
      if (auto it = givenInputs.find(portBase); it != givenInputs.end()) {
        // The inputs are given; the bits they leave out must be constants.
        auto ranges = it->second;
        llvm::sort(ranges, [](auto a, auto b) { return a.second < b.second; });
        for (auto [hi, lo2] : ranges) {
          if (lo2 < lo || hi >= unsigned(width)) {
            instance.emitError("input bits of '")
                << portBase << "' overlap or exceed it";
            return signalPassFailure();
          }
          for (unsigned bit = lo; bit < lo2; ++bit)
            if (!isConstant(bit)) {
              instance.emitError("bit ")
                  << bit << " of '" << portBase
                  << "' is not a constant and is not in an input";
              return signalPassFailure();
            }
          if (lo2 > lo)
            constantPart(lo, lo2 - 1);
          parts.push_back(parent
                              .appendInput(sliceName(portBase, hi, lo2, width),
                                           builder.getIntegerType(hi - lo2 + 1))
                              .second);
          lo = hi + 1;
        }
        for (unsigned bit = lo; bit < unsigned(width); ++bit)
          if (!isConstant(bit)) {
            instance.emitError("bit ")
                << bit << " of '" << portBase
                << "' is not a constant and is not in an input";
            return signalPassFailure();
          }
        if (lo < unsigned(width))
          constantPart(lo, width - 1);
        lo = width;
      }
      while (lo < unsigned(width)) {
        unsigned hi = lo;
        while (hi + 1 < unsigned(width) && isConstant(hi + 1) == isConstant(lo))
          ++hi;
        if (isConstant(lo)) {
          constantPart(lo, hi);
        } else {
          auto portName = sliceName(portBase, hi, lo, width);
          parts.push_back(
              parent.appendInput(portName, builder.getIntegerType(hi - lo + 1))
                  .second);
        }
        lo = hi + 1;
      }
      std::reverse(parts.begin(), parts.end());
      Value value = parts.size() == 1
                        ? parts[0]
                        : comb::ConcatOp::create(builder, parts).getResult();
      if (value.getType() != result.getType())
        value = hw::BitcastOp::create(builder, result.getType(), value);
      result.replaceAllUsesWith(value);
    }
  }
  for (auto instance : instances)
    instance.erase();
}
