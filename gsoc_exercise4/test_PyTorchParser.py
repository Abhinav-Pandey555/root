"""
Test suite for Exercise 4: PyTorch Parser (Python Interface)
============================================================
Tests parsing of: ELU, MaxPool2D, BatchNorm2D, RNN, LSTM, GRU

Run with:
    python3 test_PyTorchParser.py

Author: Abhinav Pandey (GSoC 2026 Candidate)
"""

import torch
import torch.nn as nn
import sys
import traceback

from PyTorchParser import parse_pytorch_model, get_layer_info, SOFIEModelInfo

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    print(f"  [{status}]  {name}")
    if not condition and detail:
        print(f"          Detail: {detail}")


def run_test(test_name, fn):
    print(f"\n{'─'*55}")
    print(f"  TEST: {test_name}")
    print(f"{'─'*55}")
    try:
        fn()
    except Exception as e:
        print(f"  [FAIL]  Exception: {e}")
        traceback.print_exc()
        results.append((test_name + " [EXCEPTION]", False))


# ─────────────────────────────────────────────────────────
# TEST 1: ELU
# ─────────────────────────────────────────────────────────
def test_elu():
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ELU(alpha=1.0),
        nn.Linear(32, 8),
    )
    info = parse_pytorch_model(model, input_shapes=[(2, 16)],
                               model_name="ELUModel", verbose=False)
    elu_layers = get_layer_info(info, "onnx::Elu")
    check("ELU layer detected", len(elu_layers) >= 1,
          f"Found {len(elu_layers)} ELU layers")
    if elu_layers:
        params = elu_layers[0].params
        check("ELU alpha extracted",      "alpha" in params)
        check("ELU alpha value correct",  abs(params.get("alpha", 0) - 1.0) < 1e-5,
              f"alpha={params.get('alpha')}")
        check("ELU has input name",       bool(params.get("input")))
        check("ELU has output name",      bool(params.get("output")))
        check("ELU dtype is float",       elu_layers[0].dtypes[0] == "float")
    check("Model has weights", len(info.weights) > 0)


# ─────────────────────────────────────────────────────────
# TEST 2: MaxPool2D
# ─────────────────────────────────────────────────────────
def test_maxpool2d():
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    info = parse_pytorch_model(model, input_shapes=[(1, 1, 8, 8)],
                               model_name="MaxPool2DModel", verbose=False)
    pool_layers = get_layer_info(info, "onnx::MaxPool")
    check("MaxPool2D layer detected", len(pool_layers) >= 1,
          f"Found {len(pool_layers)} MaxPool layers")
    if pool_layers:
        params = pool_layers[0].params
        check("kernel_shape extracted",   "kernel_shape" in params)
        check("kernel_shape is [2,2]",    params.get("kernel_shape") == [2, 2],
              f"kernel_shape={params.get('kernel_shape')}")
        check("strides extracted",        "strides" in params)
        check("strides is [2,2]",         params.get("strides") == [2, 2],
              f"strides={params.get('strides')}")
        check("pads extracted",           "pads" in params)
        check("MaxPool has input/output", bool(params.get("input")) and bool(params.get("output")))


# ─────────────────────────────────────────────────────────
# TEST 3: BatchNorm2D
# ─────────────────────────────────────────────────────────
def test_batchnorm2d():
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
    )
    info = parse_pytorch_model(model, input_shapes=[(2, 3, 8, 8)],
                               model_name="BatchNorm2DModel", verbose=False)
    bn_layers = get_layer_info(info, "onnx::BatchNormalization")
    check("BatchNorm2D layer detected", len(bn_layers) >= 1,
          f"Found {len(bn_layers)} BN layers")
    if bn_layers:
        params = bn_layers[0].params
        check("epsilon extracted",        "epsilon" in params)
        check("epsilon is small positive", 0 < params.get("epsilon", -1) < 0.01,
              f"epsilon={params.get('epsilon')}")
        check("momentum extracted",       "momentum" in params)
        check("BN has 5 inputs",          len(bn_layers[0].inputs) == 5,
              f"num inputs={len(bn_layers[0].inputs)}")
        check("BN has output name",       bool(params.get("output")))
    check("BN has weights", len(info.weights) >= 4)


# ─────────────────────────────────────────────────────────
# TEST 4: RNN
# ─────────────────────────────────────────────────────────
def test_rnn():
    class RNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(input_size=8, hidden_size=16,
                              num_layers=1, batch_first=True)
            self.fc  = nn.Linear(16, 4)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])

    model = RNNModel()
    info = parse_pytorch_model(model, input_shapes=[(1, 5, 8)],
                               model_name="RNNModel", opset=11, verbose=False)
    rnn_layers = get_layer_info(info, "onnx::RNN")
    check("RNN layer detected", len(rnn_layers) >= 1,
          f"Found {len(rnn_layers)} RNN layers")
    if rnn_layers:
        params = rnn_layers[0].params
        check("hidden_size extracted",  "hidden_size" in params)
        check("hidden_size is 16",      params.get("hidden_size") == 16,
              f"hidden_size={params.get('hidden_size')}")
        check("direction extracted",    "direction" in params)
        check("activations extracted",  "activations" in params)
        check("RNN has W and R inputs", bool(params.get("input_W")) and bool(params.get("input_R")))
        check("RNN has output",         bool(params.get("output_Y")))
    check("RNN weights present", len(info.weights) > 0)


# ─────────────────────────────────────────────────────────
# TEST 5: LSTM
# ─────────────────────────────────────────────────────────
def test_lstm():
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=10, hidden_size=20,
                                num_layers=1, batch_first=True)
            self.fc   = nn.Linear(20, 5)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = LSTMModel()
    info = parse_pytorch_model(model, input_shapes=[(1, 6, 10)],
                               model_name="LSTMModel", opset=11, verbose=False)
    lstm_layers = get_layer_info(info, "onnx::LSTM")
    check("LSTM layer detected", len(lstm_layers) >= 1,
          f"Found {len(lstm_layers)} LSTM layers")
    if lstm_layers:
        params = lstm_layers[0].params
        check("hidden_size extracted",   "hidden_size" in params)
        check("hidden_size is 20",       params.get("hidden_size") == 20,
              f"hidden_size={params.get('hidden_size')}")
        check("direction extracted",     "direction" in params)
        check("activations extracted",   "activations" in params)
        acts = params.get("activations", [])
        check("LSTM has 3 activations",  len(acts) == 3, f"activations={acts}")
        check("LSTM has W and R inputs", bool(params.get("input_W")) and bool(params.get("input_R")))
        check("input_forget extracted",  "input_forget" in params)
    check("LSTM weights present", len(info.weights) > 0)


# ─────────────────────────────────────────────────────────
# TEST 6: GRU
# ─────────────────────────────────────────────────────────
def test_gru():
    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=12, hidden_size=24,
                              num_layers=1, batch_first=True)
            self.fc  = nn.Linear(24, 6)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    model = GRUModel()
    info = parse_pytorch_model(model, input_shapes=[(1, 4, 12)],
                               model_name="GRUModel", opset=11, verbose=False)
    gru_layers = get_layer_info(info, "onnx::GRU")
    check("GRU layer detected", len(gru_layers) >= 1,
          f"Found {len(gru_layers)} GRU layers")
    if gru_layers:
        params = gru_layers[0].params
        check("hidden_size extracted",        "hidden_size" in params)
        check("hidden_size is 24",            params.get("hidden_size") == 24,
              f"hidden_size={params.get('hidden_size')}")
        check("direction extracted",          "direction" in params)
        check("activations extracted",        "activations" in params)
        acts = params.get("activations", [])
        check("GRU has 2 activations",        len(acts) == 2, f"activations={acts}")
        check("linear_before_reset extracted","linear_before_reset" in params)
        check("GRU has W and R inputs",       bool(params.get("input_W")) and bool(params.get("input_R")))
    check("GRU weights present", len(info.weights) > 0)


# ─────────────────────────────────────────────────────────
# TEST 7: Mixed model
# ─────────────────────────────────────────────────────────
def test_mixed_cnn_model():
    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 16),
    )
    info = parse_pytorch_model(model, input_shapes=[(1, 1, 8, 8)],
                               model_name="MixedCNNModel", verbose=False)
    check("ELU in mixed model",       len(get_layer_info(info, "onnx::Elu"))                >= 1)
    check("MaxPool in mixed model",   len(get_layer_info(info, "onnx::MaxPool"))            >= 1)
    check("BatchNorm in mixed model", len(get_layer_info(info, "onnx::BatchNormalization")) >= 1)
    check("Total operators parsed",   len(info.layers) >= 4)


# ─────────────────────────────────────────────────────────
# TEST 8: to_dict format
# ─────────────────────────────────────────────────────────
def test_sofie_dict_format():
    model = nn.Sequential(nn.Linear(8, 4), nn.ELU())
    info  = parse_pytorch_model(model, input_shapes=[(1, 8)],
                                model_name="DictTest", verbose=False)
    elu_layers = get_layer_info(info, "onnx::Elu")
    if elu_layers:
        d = elu_layers[0].to_dict()
        check("to_dict has nodeType",       "nodeType"       in d)
        check("to_dict has nodeAttributes", "nodeAttributes" in d)
        check("to_dict has nodeInputs",     "nodeInputs"     in d)
        check("to_dict has nodeOutputs",    "nodeOutputs"    in d)
        check("to_dict has nodeDType",      "nodeDType"      in d)
        check("nodeInputs is a list",       isinstance(d["nodeInputs"],  list))
        check("nodeOutputs is a list",      isinstance(d["nodeOutputs"], list))
        check("nodeDType is a list",        isinstance(d["nodeDType"],   list))
    else:
        check("ELU present for dict test", False)


# ─────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SOFIE PyTorch Parser - Exercise 4 Test Suite")
    print("="*55)

    run_test("ELU Layer",         test_elu)
    run_test("MaxPool2D Layer",   test_maxpool2d)
    run_test("BatchNorm2D Layer", test_batchnorm2d)
    run_test("RNN Layer",         test_rnn)
    run_test("LSTM Layer",        test_lstm)
    run_test("GRU Layer",         test_gru)
    run_test("Mixed CNN Model",   test_mixed_cnn_model)
    run_test("SOFIE Dict Format", test_sofie_dict_format)

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed

    print(f"\n{'='*55}")
    print(f"  Results: {passed}/{total} tests passed")
    if failed:
        print(f"  Failed tests:")
        for name, ok in results:
            if not ok:
                print(f"    FAIL: {name}")
    print("="*55 + "\n")

    sys.exit(0 if failed == 0 else 1)
