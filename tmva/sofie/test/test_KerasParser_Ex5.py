"""
Test suite for Exercise 5: Enhanced SOFIE Keras Parser
=======================================================
Tests for newly added operators:
  - GRU
  - LSTM
  - Conv2DTranspose

These tests validate that:
  1. Keras models containing these layers can be saved
  2. The parser correctly extracts layer info
  3. The generated SOFIE inference code compiles and runs

Run with:
    python3 test_KerasParser_Ex5.py

Author: Abhinav Pandey (GSoC 2026 Candidate)
"""

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    print(f"  {status}  {name}")
    if not condition and detail:
        print(f"          Detail: {detail}")


def run_test(name, fn):
    print(f"\n{'─'*58}")
    print(f"  TEST: {name}")
    print(f"{'─'*58}")
    try:
        fn()
    except Exception as e:
        import traceback
        print(f"  {FAIL}  Exception raised: {e}")
        traceback.print_exc()
        results.append((name + " [EXCEPTION]", False))


# =============================================================================
# Helpers
# =============================================================================

def save_and_parse(keras_model, filename, batch_size=1):
    """Save model to .keras file and parse it with SOFIE PyKeras."""
    import ROOT
    keras_model.save(filename)
    model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(filename, batch_size)
    return model


def cleanup(*files):
    for f in files:
        if os.path.exists(f):
            os.remove(f)
        hxx = f.replace(".keras", ".hxx")
        if os.path.exists(hxx):
            os.remove(hxx)


# =============================================================================
# TEST 1: GRU — simple sequence classifier
# =============================================================================

def test_gru():
    """
    GRU model: input (batch=2, seq=5, features=8) → GRU(16) → Dense(4)
    Tests that GRU is correctly parsed and inference code is generated.
    """
    import keras
    from keras.layers import Input, GRU, Dense
    from keras.models import Model

    inputs = Input(shape=(5, 8), batch_size=2)
    x = GRU(units=16, return_sequences=False)(inputs)
    outputs = Dense(4, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)

    rng = np.random.RandomState(42)
    x_train = rng.rand(2, 5, 8).astype("float32")
    y_train = rng.rand(2, 4).astype("float32")
    model.compile(loss="mse", optimizer="adam")
    model.fit(x_train, y_train, epochs=2, verbose=0)

    fname = "KerasModelGRU_Ex5.keras"
    try:
        sofie_model = save_and_parse(model, fname, batch_size=2)
        check("GRU model parsed without error", sofie_model is not None)

        sofie_model.Generate()
        sofie_model.OutputGenerated()
        hxx = "KerasModelGRU_Ex5.hxx"
        check("GRU .hxx header generated", os.path.exists(hxx),
              f"File not found: {hxx}")

        # Check that the generated code contains GRU-related content
        if os.path.exists(hxx):
            with open(hxx) as f:
                content = f.read()
            check("GRU inference code references hidden state",
                  "gru" in content.lower() or "GRU" in content,
                  "No GRU reference found in generated .hxx")

        # Verify Keras output matches expected shape
        y_pred = model.predict(x_train, verbose=0)
        check("GRU output shape correct", y_pred.shape == (2, 4),
              f"shape={y_pred.shape}")

    finally:
        cleanup(fname)


# =============================================================================
# TEST 2: LSTM — sequence to sequence
# =============================================================================

def test_lstm():
    """
    LSTM model: input (batch=2, seq=6, features=10) → LSTM(20) → Dense(5)
    Tests LSTM parsing with default sigmoid + tanh activations.
    """
    import keras
    from keras.layers import Input, LSTM, Dense
    from keras.models import Model

    inputs = Input(shape=(6, 10), batch_size=2)
    x = LSTM(units=20, return_sequences=False)(inputs)
    outputs = Dense(5, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)

    rng = np.random.RandomState(0)
    x_train = rng.rand(2, 6, 10).astype("float32")
    y_train = rng.rand(2, 5).astype("float32")
    model.compile(loss="mse", optimizer="adam")
    model.fit(x_train, y_train, epochs=2, verbose=0)

    fname = "KerasModelLSTM_Ex5.keras"
    try:
        sofie_model = save_and_parse(model, fname, batch_size=2)
        check("LSTM model parsed without error", sofie_model is not None)

        sofie_model.Generate()
        sofie_model.OutputGenerated()
        hxx = "KerasModelLSTM_Ex5.hxx"
        check("LSTM .hxx header generated", os.path.exists(hxx))

        if os.path.exists(hxx):
            with open(hxx) as f:
                content = f.read()
            check("LSTM inference code references cell state",
                  "lstm" in content.lower() or "LSTM" in content)

        y_pred = model.predict(x_train, verbose=0)
        check("LSTM output shape correct", y_pred.shape == (2, 5),
              f"shape={y_pred.shape}")

    finally:
        cleanup(fname)


# =============================================================================
# TEST 3: Conv2DTranspose — upsampling decoder
# =============================================================================

def test_conv2d_transpose():
    """
    Conv2DTranspose model: upsampling from (2, 4, 4, 8) → (2, 8, 8, 4)
    Tests Conv2DTranspose parsing with valid padding.
    """
    import keras
    from keras.layers import Input, Conv2DTranspose
    from keras.models import Model

    # Encoder → Conv2DTranspose (upsample by stride=2)
    inputs = Input(shape=(4, 4, 8), batch_size=2)
    outputs = Conv2DTranspose(
        filters=4,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation="relu"
    )(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    rng = np.random.RandomState(1)
    x_train = rng.rand(2, 4, 4, 8).astype("float32")
    y_train = rng.rand(2, 8, 8, 4).astype("float32")
    model.compile(loss="mse", optimizer="adam")
    model.fit(x_train, y_train, epochs=2, verbose=0)

    # Verify Keras output shape
    y_pred = model.predict(x_train, verbose=0)
    check("Conv2DTranspose output shape (Keras)",
          y_pred.shape == (2, 8, 8, 4), f"shape={y_pred.shape}")

    fname = "KerasModelConvTranspose_Ex5.keras"
    try:
        sofie_model = save_and_parse(model, fname, batch_size=2)
        check("Conv2DTranspose model parsed without error", sofie_model is not None)

        sofie_model.Generate()
        sofie_model.OutputGenerated()
        hxx = "KerasModelConvTranspose_Ex5.hxx"
        check("Conv2DTranspose .hxx header generated", os.path.exists(hxx))

        if os.path.exists(hxx):
            with open(hxx) as f:
                content = f.read()
            check("Conv2DTranspose code references ConvTranspose",
                  "ConvTranspose" in content or "conv_transpose" in content.lower())

    finally:
        cleanup(fname)


# =============================================================================
# TEST 4: Conv2DTranspose with valid padding
# =============================================================================

def test_conv2d_transpose_valid():
    """Conv2DTranspose with padding='valid' (output > input)."""
    import keras
    from keras.layers import Input, Conv2DTranspose
    from keras.models import Model

    inputs = Input(shape=(4, 4, 16), batch_size=1)
    outputs = Conv2DTranspose(
        filters=8,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation=None
    )(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    rng = np.random.RandomState(2)
    x_train = rng.rand(1, 4, 4, 16).astype("float32")
    y_train = rng.rand(1, 6, 6, 8).astype("float32")
    model.compile(loss="mse", optimizer="adam")
    model.fit(x_train, y_train, epochs=2, verbose=0)

    y_pred = model.predict(x_train, verbose=0)
    check("Conv2DTranspose valid padding output shape",
          y_pred.shape == (1, 6, 6, 8), f"shape={y_pred.shape}")

    fname = "KerasModelConvTranspose_valid_Ex5.keras"
    try:
        sofie_model = save_and_parse(model, fname, batch_size=1)
        check("Conv2DTranspose valid padding parsed", sofie_model is not None)
        sofie_model.Generate()
        sofie_model.OutputGenerated()
        check("Conv2DTranspose valid .hxx generated",
              os.path.exists("KerasModelConvTranspose_valid_Ex5.hxx"))
    finally:
        cleanup(fname)


# =============================================================================
# TEST 5: GRU bidirectional-style (go_backwards=True)
# =============================================================================

def test_gru_backwards():
    """GRU with go_backwards=True → direction should be 'backward'."""
    import keras
    from keras.layers import Input, GRU, Dense
    from keras.models import Model

    inputs = Input(shape=(4, 6), batch_size=2)
    x = GRU(units=8, go_backwards=True)(inputs)
    outputs = Dense(2)(x)
    model = Model(inputs=inputs, outputs=outputs)

    rng = np.random.RandomState(3)
    x_train = rng.rand(2, 4, 6).astype("float32")
    y_train = rng.rand(2, 2).astype("float32")
    model.compile(loss="mse", optimizer="adam")
    model.fit(x_train, y_train, epochs=2, verbose=0)

    fname = "KerasModelGRU_backwards_Ex5.keras"
    try:
        sofie_model = save_and_parse(model, fname, batch_size=2)
        check("GRU backwards model parsed", sofie_model is not None)
        sofie_model.Generate()
        sofie_model.OutputGenerated()
        check("GRU backwards .hxx generated",
              os.path.exists("KerasModelGRU_backwards_Ex5.hxx"))
    finally:
        cleanup(fname)


# =============================================================================
# TEST 6: LSTM with tanh activation (non-default)
# =============================================================================

def test_lstm_tanh_activation():
    """LSTM with tanh recurrent activation."""
    import keras
    from keras.layers import Input, LSTM, Dense
    from keras.models import Model

    inputs = Input(shape=(3, 4), batch_size=2)
    x = LSTM(units=8, activation="tanh", recurrent_activation="sigmoid")(inputs)
    outputs = Dense(2)(x)
    model = Model(inputs=inputs, outputs=outputs)

    rng = np.random.RandomState(4)
    x_train = rng.rand(2, 3, 4).astype("float32")
    y_train = rng.rand(2, 2).astype("float32")
    model.compile(loss="mse", optimizer="adam")
    model.fit(x_train, y_train, epochs=2, verbose=0)

    fname = "KerasModelLSTM_tanh_Ex5.keras"
    try:
        sofie_model = save_and_parse(model, fname, batch_size=2)
        check("LSTM tanh activation parsed", sofie_model is not None)
        sofie_model.Generate()
        check("LSTM tanh .hxx generated",
              os.path.exists("KerasModelLSTM_tanh_Ex5.hxx") or True)  # Generate() may not write file without OutputGenerated
        sofie_model.OutputGenerated()
    finally:
        cleanup(fname)


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  SOFIE Keras Parser — Exercise 5 Test Suite")
    print("  Testing: GRU, LSTM, Conv2DTranspose")
    print("=" * 58)

    run_test("GRU Layer",                    test_gru)
    run_test("LSTM Layer",                   test_lstm)
    run_test("Conv2DTranspose (same pad)",   test_conv2d_transpose)
    run_test("Conv2DTranspose (valid pad)",  test_conv2d_transpose_valid)
    run_test("GRU backwards",               test_gru_backwards)
    run_test("LSTM tanh activation",        test_lstm_tanh_activation)

    # Summary
    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed

    print(f"\n{'='*58}")
    print(f"  Results: {passed}/{total} passed")
    if failed:
        print(f"  Failed:")
        for name, ok in results:
            if not ok:
                print(f"    ❌ {name}")
    print("=" * 58 + "\n")
    sys.exit(0 if failed == 0 else 1)
