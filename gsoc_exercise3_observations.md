# GSoC 2026 - Exercise 3: Exploring SOFIE Keras and PyTorch Parsers

## Overview
This document summarizes my exploration of the SOFIE Keras and PyTorch parsers
as part of Exercise 3 for the GSoC 2026 SOFIE project. The goal was to
understand how each parser works, what layers are currently supported, and
what gaps exist that Exercise 4 addresses.

---

## How the Keras Parser Works

The Keras parser is implemented as a Python interface under the ROOT
pythonizations module. The entry point is PyKeras.Parse() which accepts a
saved .keras model file. Internally it reads the Keras model layer by layer
using a dispatch map that routes each layer type to a dedicated parsing
function. Each function extracts the layer weights and configuration and
adds them to an RModel object which is then used to generate the C++
inference header.

The parser lives in:
bindings/pyroot/pythonizations/python/ROOT/_pythonization/_tmva/_sofie/_parser/_keras/

Individual layer parsers are in separate files such as elu.py, batchnorm.py,
pooling.py and rnn.py. This modular structure makes it straightforward to
add support for new layer types.

## How the PyTorch Parser Works

The PyTorch parser is implemented in C++ in
tmva/sofie_parsers/src/RModelParser_PyTorch.cxx. It accepts a TorchScript
.pt file saved with torch.jit.script and parses the graph nodes directly.
Each node type is routed through a dispatch map called mapPyTorchNode which
maps operator strings like aten::linear and aten::relu to C++ parsing
functions. The parser extracts weight tensors and operator parameters and
builds an RModel for code generation.

A key difference from the Keras parser is that the PyTorch parser currently
only has a C++ interface. There is no Python-level entry point for PyTorch
models, which is exactly what Exercise 4 addresses by implementing a Python
parser using ONNX as an intermediate representation.

---

## Currently Supported Layers

### Keras Parser
The following layers are fully supported: Dense, Conv2D, BatchNormalization,
MaxPooling2D, AveragePooling2D, ReLU, ELU, SELU, Sigmoid, Softmax, Flatten
and Reshape. The RNN, LSTM and GRU layers have partial implementations in
rnn.py but are commented out in the main parser dispatch map and not yet
connected to the code generation pipeline.

### PyTorch Parser (C++)
The following operators are fully supported: Linear (Gemm), ReLU, SELU,
Sigmoid, Conv2d, Transpose, Reshape, Add, Flatten and Identity. Recurrent
operators like RNN, LSTM and GRU are not supported in the C++ parser.

---

## Key Observations from Source Code Study

Looking at the Keras parser source files, the batchnorm layer handles the
axis parameter carefully since Keras and ONNX use different conventions for
channel ordering. The pooling parser extracts kernel size, strides and
padding and maps them to ONNX-compatible attribute names. The rnn.py file
has implementations for RNN, LSTM and GRU but these are not yet wired into
the main parser, suggesting they were work in progress.

The most important insight from studying both parsers is that SOFIE uses
ONNX as its internal representation. The Keras parser converts Keras layers
to ONNX-compatible operator descriptions before adding them to the RModel.
This means a Python PyTorch parser can follow the same approach by first
exporting the PyTorch model to ONNX using torch.onnx.export and then
parsing the resulting ONNX graph, which is exactly the design chosen for
Exercise 4.

---

## Issues Discovered

During exploration it was found that PyTorch's ONNX export with
do_constant_folding=True causes BatchNorm2D to be optimised away entirely
in eval mode since the normalisation parameters get folded into the
preceding Conv layer. Setting do_constant_folding=False preserves the
BatchNormalization operator in the ONNX graph and allows it to be parsed
correctly. This fix was applied in Exercise 4.

Recurrent models require opset version 11 or lower for ONNX export since
higher opset versions change the RNN operator interface in ways that are
not compatible with the SOFIE operator definitions.

---

## What Exercise 4 Adds

Based on this exploration Exercise 4 implements a Python interface for the
PyTorch parser that supports the six operators not yet available in either
the C++ PyTorch parser or the Keras parser: ELU, MaxPool2D, BatchNorm2D,
RNN, LSTM and GRU. The implementation exports PyTorch models to ONNX
in memory and parses the resulting graph, producing SOFIELayerInfo objects
that mirror the fNode dictionary structure used in the C++ parser.
