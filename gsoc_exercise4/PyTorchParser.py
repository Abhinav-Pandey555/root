"""
TMVA::SOFIE - PyTorch Parser (Python Interface)
================================================
GSoC 2026 Exercise 4: Implement parsing functionality in Python to translate
PyTorch layers into a format suitable for SOFIE inference.

Supported layers:
    - ELU
    - MaxPool2D
    - BatchNorm2D
    - RNN
    - LSTM
    - GRU

Author: Abhinav Pandey (GSoC 2026 Candidate)
Based on: TMVA/sofie_parsers/src/RModelParser_PyTorch.cxx
"""

import torch
import torch.nn as nn
import torch.onnx
import io
import onnx
from onnx import numpy_helper
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


# =============================================================================
# Data Structures
# =============================================================================

class SOFIELayerInfo:
    """
    Holds parsed information about a single layer/operator node,
    mirroring the fNode dictionary used in the C++ PyTorch parser.
    """
    def __init__(self, node_type, attributes, inputs, outputs, dtypes, params):
        self.node_type  = node_type
        self.attributes = attributes
        self.inputs     = inputs
        self.outputs    = outputs
        self.dtypes     = dtypes
        self.params     = params

    def to_dict(self):
        """Return SOFIE C++-compatible dict format."""
        return {
            'nodeType'       : self.node_type,
            'nodeAttributes' : self.attributes,
            'nodeInputs'     : self.inputs,
            'nodeOutputs'    : self.outputs,
            'nodeDType'      : self.dtypes,
            'nodeParams'     : self.params,
        }

    def __repr__(self):
        return (f"SOFIELayerInfo(\n"
                f"  nodeType    = {self.node_type}\n"
                f"  nodeInputs  = {self.inputs}\n"
                f"  nodeOutputs = {self.outputs}\n"
                f"  nodeDType   = {self.dtypes}\n"
                f"  nodeParams  = {self.params}\n"
                f")")


class SOFIEModelInfo:
    """
    Container for all parsed model information.
    """
    def __init__(self):
        self.layers        = []
        self.weights       = {}
        self.weight_dtypes = {}
        self.input_names   = []
        self.input_shapes  = []
        self.output_names  = []
        self.model_name    = "PyTorchModel"

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  SOFIE Model Summary: {self.model_name}")
        print(f"{'='*60}")
        print(f"\n  Input Tensors:")
        for name, shape in zip(self.input_names, self.input_shapes):
            print(f"    {name}: {shape}")
        print(f"\n  Output Tensors: {self.output_names}")
        print(f"\n  Operators ({len(self.layers)} total):")
        for i, layer in enumerate(self.layers):
            print(f"    [{i}] {layer.node_type}")
            for k, v in layer.params.items():
                print(f"         {k}: {v}")
        print(f"\n  Initialized Tensors (weights): {len(self.weights)}")
        for name, arr in self.weights.items():
            print(f"    {name}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"{'='*60}\n")


# =============================================================================
# ONNX dtype mapping
# =============================================================================

ONNX_DTYPE_MAP = {
    1:  "float",
    2:  "uint8",
    3:  "int8",
    4:  "uint16",
    5:  "int16",
    6:  "int32",
    7:  "int64",
    10: "float16",
    11: "double",
}

def _onnx_dtype_str(dtype_int):
    return ONNX_DTYPE_MAP.get(dtype_int, "float")


# =============================================================================
# Layer Parsers
# =============================================================================

def parse_elu(node, value_info_map):
    """
    Parse ELU activation layer.
    ELU(x) = x if x>=0, else alpha*(exp(x)-1)
    SOFIE: ROperator_Elu<float>
    """
    attrs  = {attr.name: attr for attr in node.attribute}
    alpha  = attrs["alpha"].f if "alpha" in attrs else 1.0

    inputs  = list(node.input)
    outputs = list(node.output)
    dtypes  = [_onnx_dtype_str(value_info_map.get(o, 1)) for o in outputs]

    params = {
        "alpha"  : alpha,
        "input"  : inputs[0]  if inputs  else "",
        "output" : outputs[0] if outputs else "",
    }

    return SOFIELayerInfo(
        node_type  = "onnx::Elu",
        attributes = {"alpha": alpha},
        inputs     = inputs,
        outputs    = outputs,
        dtypes     = dtypes,
        params     = params,
    )


def parse_maxpool2d(node, value_info_map):
    """
    Parse MaxPool2D layer.
    SOFIE: ROperator_Pool<float>
    """
    attrs = {attr.name: attr for attr in node.attribute}

    def get_ints(name, default):
        if name in attrs:
            return list(attrs[name].ints)
        return default

    kernel_shape = get_ints("kernel_shape", [1, 1])
    pads         = get_ints("pads",         [0, 0, 0, 0])
    strides      = get_ints("strides",      [1, 1])
    dilations    = get_ints("dilations",    [1, 1])
    ceil_mode    = attrs["ceil_mode"].i if "ceil_mode" in attrs else 0

    inputs  = list(node.input)
    outputs = list(node.output)
    dtypes  = [_onnx_dtype_str(value_info_map.get(o, 1)) for o in outputs]

    params = {
        "kernel_shape" : kernel_shape,
        "pads"         : pads,
        "strides"      : strides,
        "dilations"    : dilations,
        "ceil_mode"    : ceil_mode,
        "input"        : inputs[0]  if inputs  else "",
        "output"       : outputs[0] if outputs else "",
    }

    return SOFIELayerInfo(
        node_type  = "onnx::MaxPool",
        attributes = {
            "kernel_shape" : kernel_shape,
            "pads"         : pads,
            "strides"      : strides,
            "dilations"    : dilations,
            "ceil_mode"    : ceil_mode,
        },
        inputs  = inputs,
        outputs = outputs,
        dtypes  = dtypes,
        params  = params,
    )


def parse_batchnorm2d(node, value_info_map):
    """
    Parse BatchNorm2D layer.
    ONNX inputs: X, scale(gamma), B(beta), mean, var
    SOFIE: ROperator_BatchNormalization<float>
    """
    attrs    = {attr.name: attr for attr in node.attribute}
    epsilon  = attrs["epsilon"].f  if "epsilon"  in attrs else 1e-5
    momentum = attrs["momentum"].f if "momentum" in attrs else 0.9

    inputs  = list(node.input)   # [X, scale, B, mean, var]
    outputs = list(node.output)
    dtypes  = [_onnx_dtype_str(value_info_map.get(o, 1)) for o in outputs]

    params = {
        "epsilon"    : epsilon,
        "momentum"   : momentum,
        "input_X"    : inputs[0] if len(inputs) > 0 else "",
        "input_scale": inputs[1] if len(inputs) > 1 else "",
        "input_bias" : inputs[2] if len(inputs) > 2 else "",
        "input_mean" : inputs[3] if len(inputs) > 3 else "",
        "input_var"  : inputs[4] if len(inputs) > 4 else "",
        "output"     : outputs[0] if outputs else "",
    }

    return SOFIELayerInfo(
        node_type  = "onnx::BatchNormalization",
        attributes = {"epsilon": epsilon, "momentum": momentum},
        inputs     = inputs,
        outputs    = outputs,
        dtypes     = dtypes,
        params     = params,
    )


def parse_rnn(node, value_info_map):
    """
    Parse RNN layer.
    h_t = tanh(W*x_t + b_x + U*h_{t-1} + b_h)
    SOFIE: ROperator_RNN<float>
    """
    attrs = {attr.name: attr for attr in node.attribute}

    hidden_size      = attrs["hidden_size"].i if "hidden_size" in attrs else 1
    direction        = attrs["direction"].s.decode() if "direction" in attrs else "forward"
    activations      = ([s.decode() for s in attrs["activations"].strings]
                        if "activations" in attrs else ["Tanh"])
    activation_alpha = list(attrs["activation_alpha"].floats) if "activation_alpha" in attrs else []
    activation_beta  = list(attrs["activation_beta"].floats)  if "activation_beta"  in attrs else []

    inputs  = list(node.input)
    outputs = list(node.output)
    dtypes  = [_onnx_dtype_str(value_info_map.get(o, 1)) for o in outputs]

    params = {
        "hidden_size"      : hidden_size,
        "direction"        : direction,
        "activations"      : activations,
        "activation_alpha" : activation_alpha,
        "activation_beta"  : activation_beta,
        "input_X"          : inputs[0] if len(inputs) > 0 else "",
        "input_W"          : inputs[1] if len(inputs) > 1 else "",
        "input_R"          : inputs[2] if len(inputs) > 2 else "",
        "input_B"          : inputs[3] if len(inputs) > 3 else "",
        "output_Y"         : outputs[0] if len(outputs) > 0 else "",
        "output_Y_h"       : outputs[1] if len(outputs) > 1 else "",
    }

    return SOFIELayerInfo(
        node_type  = "onnx::RNN",
        attributes = {
            "hidden_size"      : hidden_size,
            "direction"        : direction,
            "activations"      : activations,
            "activation_alpha" : activation_alpha,
            "activation_beta"  : activation_beta,
        },
        inputs  = inputs,
        outputs = outputs,
        dtypes  = dtypes,
        params  = params,
    )


def parse_lstm(node, value_info_map):
    """
    Parse LSTM layer.
    4 gates: input(i), forget(f), cell(g), output(o)
    SOFIE: ROperator_LSTM<float>
    """
    attrs = {attr.name: attr for attr in node.attribute}

    hidden_size      = attrs["hidden_size"].i  if "hidden_size"  in attrs else 1
    direction        = attrs["direction"].s.decode() if "direction" in attrs else "forward"
    input_forget     = attrs["input_forget"].i if "input_forget" in attrs else 0
    activations      = ([s.decode() for s in attrs["activations"].strings]
                        if "activations" in attrs else ["Sigmoid", "Tanh", "Tanh"])
    activation_alpha = list(attrs["activation_alpha"].floats) if "activation_alpha" in attrs else []
    activation_beta  = list(attrs["activation_beta"].floats)  if "activation_beta"  in attrs else []

    inputs  = list(node.input)
    outputs = list(node.output)
    dtypes  = [_onnx_dtype_str(value_info_map.get(o, 1)) for o in outputs]

    params = {
        "hidden_size"      : hidden_size,
        "direction"        : direction,
        "input_forget"     : input_forget,
        "activations"      : activations,
        "activation_alpha" : activation_alpha,
        "activation_beta"  : activation_beta,
        "input_X"          : inputs[0] if len(inputs) > 0 else "",
        "input_W"          : inputs[1] if len(inputs) > 1 else "",
        "input_R"          : inputs[2] if len(inputs) > 2 else "",
        "input_B"          : inputs[3] if len(inputs) > 3 else "",
        "output_Y"         : outputs[0] if len(outputs) > 0 else "",
        "output_Y_h"       : outputs[1] if len(outputs) > 1 else "",
        "output_Y_c"       : outputs[2] if len(outputs) > 2 else "",
    }

    return SOFIELayerInfo(
        node_type  = "onnx::LSTM",
        attributes = {
            "hidden_size"      : hidden_size,
            "direction"        : direction,
            "input_forget"     : input_forget,
            "activations"      : activations,
            "activation_alpha" : activation_alpha,
            "activation_beta"  : activation_beta,
        },
        inputs  = inputs,
        outputs = outputs,
        dtypes  = dtypes,
        params  = params,
    )


def parse_gru(node, value_info_map):
    """
    Parse GRU layer.
    3 gates: reset(r), update(z), hidden(h)
    SOFIE: ROperator_GRU<float>
    """
    attrs = {attr.name: attr for attr in node.attribute}

    hidden_size         = attrs["hidden_size"].i if "hidden_size" in attrs else 1
    direction           = attrs["direction"].s.decode() if "direction" in attrs else "forward"
    linear_before_reset = attrs["linear_before_reset"].i if "linear_before_reset" in attrs else 0
    activations         = ([s.decode() for s in attrs["activations"].strings]
                           if "activations" in attrs else ["Sigmoid", "Tanh"])
    activation_alpha    = list(attrs["activation_alpha"].floats) if "activation_alpha" in attrs else []
    activation_beta     = list(attrs["activation_beta"].floats)  if "activation_beta"  in attrs else []

    inputs  = list(node.input)
    outputs = list(node.output)
    dtypes  = [_onnx_dtype_str(value_info_map.get(o, 1)) for o in outputs]

    params = {
        "hidden_size"         : hidden_size,
        "direction"           : direction,
        "linear_before_reset" : linear_before_reset,
        "activations"         : activations,
        "activation_alpha"    : activation_alpha,
        "activation_beta"     : activation_beta,
        "input_X"             : inputs[0] if len(inputs) > 0 else "",
        "input_W"             : inputs[1] if len(inputs) > 1 else "",
        "input_R"             : inputs[2] if len(inputs) > 2 else "",
        "input_B"             : inputs[3] if len(inputs) > 3 else "",
        "output_Y"            : outputs[0] if len(outputs) > 0 else "",
        "output_Y_h"          : outputs[1] if len(outputs) > 1 else "",
    }

    return SOFIELayerInfo(
        node_type  = "onnx::GRU",
        attributes = {
            "hidden_size"         : hidden_size,
            "direction"           : direction,
            "linear_before_reset" : linear_before_reset,
            "activations"         : activations,
            "activation_alpha"    : activation_alpha,
            "activation_beta"     : activation_beta,
        },
        inputs  = inputs,
        outputs = outputs,
        dtypes  = dtypes,
        params  = params,
    )


# =============================================================================
# Operator dispatch map
# =============================================================================

LAYER_PARSER_MAP = {
    "Elu"                : parse_elu,
    "MaxPool"            : parse_maxpool2d,
    "BatchNormalization" : parse_batchnorm2d,
    "RNN"                : parse_rnn,
    "LSTM"               : parse_lstm,
    "GRU"                : parse_gru,
}

# Already supported by existing C++ parser
EXISTING_OPS = {
    "Gemm", "Conv", "Relu", "Selu", "Sigmoid",
    "Transpose", "Reshape", "Add", "Sub", "Mul",
    "Flatten", "Identity", "Gather", "MatMul"
}


# =============================================================================
# Core functions
# =============================================================================

def _export_to_onnx(model, input_shapes, opset=12):
    """Export PyTorch model to ONNX format in memory."""
    model.eval()
    dummy_inputs = tuple(torch.randn(*shape) for shape in input_shapes)

    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0],
        buffer,
        export_params       = True,
        opset_version       = opset,
        do_constant_folding = False,
        input_names         = [f"input_{i}" for i in range(len(input_shapes))],
        output_names        = ["output"],
        dynamic_axes        = None,
    )
    buffer.seek(0)
    return onnx.load_from_string(buffer.read())


def parse_pytorch_model(model_or_path, input_shapes, model_name="PyTorchModel",
                        opset=12, verbose=True):
    """
    Parse a PyTorch model and extract SOFIE-compatible layer information.

    Parameters
    ----------
    model_or_path : nn.Module or str
        PyTorch model object or path to .pt file
    input_shapes : list of tuples
        Input tensor shapes e.g. [(2, 16)]
    model_name : str
        Name for the model
    opset : int
        ONNX opset version (use 11 for recurrent models)
    verbose : bool
        Print progress

    Returns
    -------
    SOFIEModelInfo
    """
    # Load model if path given
    if isinstance(model_or_path, str):
        if verbose:
            print(f"[SOFIE] Loading model from: {model_or_path}")
        model = torch.jit.load(model_or_path)
    else:
        model = model_or_path

    model.eval()

    # Export to ONNX
    if verbose:
        print(f"[SOFIE] Exporting to ONNX (opset {opset})...")
    onnx_model = _export_to_onnx(model, input_shapes, opset=opset)

    # Build value info map (tensor name -> dtype)
    value_info_map = {}
    for vi in onnx_model.graph.value_info:
        value_info_map[vi.name] = vi.type.tensor_type.elem_type
    for inp in onnx_model.graph.input:
        value_info_map[inp.name] = inp.type.tensor_type.elem_type
    for out in onnx_model.graph.output:
        value_info_map[out.name] = out.type.tensor_type.elem_type

    # Build SOFIEModelInfo
    info             = SOFIEModelInfo()
    info.model_name  = model_name
    info.input_names = [inp.name for inp in onnx_model.graph.input]
    info.input_shapes= input_shapes
    info.output_names= [out.name for out in onnx_model.graph.output]

    # Extract weights
    for init in onnx_model.graph.initializer:
        arr = numpy_helper.to_array(init)
        info.weights[init.name]       = arr
        info.weight_dtypes[init.name] = str(arr.dtype)

    # Parse each node
    supported_new   = 0
    unsupported_ops = []

    for node in onnx_model.graph.node:
        op_type = node.op_type

        if op_type in LAYER_PARSER_MAP:
            parser     = LAYER_PARSER_MAP[op_type]
            layer_info = parser(node, value_info_map)
            info.layers.append(layer_info)
            supported_new += 1
            if verbose:
                print(f"[SOFIE]   + Parsed new operator: onnx::{op_type}")

        elif op_type in EXISTING_OPS:
            inputs  = list(node.input)
            outputs = list(node.output)
            dtypes  = [_onnx_dtype_str(value_info_map.get(o, 1)) for o in outputs]
            layer_info = SOFIELayerInfo(
                node_type  = f"onnx::{op_type}",
                attributes = {attr.name: attr for attr in node.attribute},
                inputs     = inputs,
                outputs    = outputs,
                dtypes     = dtypes,
                params     = {"note": "handled by existing C++ parser"},
            )
            info.layers.append(layer_info)
            if verbose:
                print(f"[SOFIE]   i Existing operator:   onnx::{op_type}")

        else:
            unsupported_ops.append(op_type)
            inputs  = list(node.input)
            outputs = list(node.output)
            dtypes  = [_onnx_dtype_str(value_info_map.get(o, 1)) for o in outputs]
            layer_info = SOFIELayerInfo(
                node_type  = f"onnx::{op_type}",
                attributes = {},
                inputs     = inputs,
                outputs    = outputs,
                dtypes     = dtypes,
                params     = {"note": "not yet supported"},
            )
            info.layers.append(layer_info)
            if verbose:
                print(f"[SOFIE]   ! Unsupported operator: onnx::{op_type}")

    if verbose:
        print(f"\n[SOFIE] Parsing complete.")
        print(f"        New operators parsed : {supported_new}")
        print(f"        Total operators      : {len(info.layers)}")
        if unsupported_ops:
            print(f"        Unsupported          : {set(unsupported_ops)}")

    return info


def get_layer_info(model_info, layer_type):
    """
    Filter layers by type from parsed model info.

    Parameters
    ----------
    model_info : SOFIEModelInfo
    layer_type : str e.g. 'onnx::LSTM'

    Returns
    -------
    list of matching SOFIELayerInfo objects
    """
    return [l for l in model_info.layers if l.node_type == layer_type]


def parse_from_file(pt_path, input_shapes, verbose=True):
    """Parse a saved .pt model file."""
    model = torch.jit.load(pt_path)
    name  = pt_path.split("/")[-1].replace(".pt", "")
    return parse_pytorch_model(model, input_shapes, model_name=name, verbose=verbose)
