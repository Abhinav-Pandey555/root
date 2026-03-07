import os
import time

from . import get_keras_version
from .layers.batchnorm import MakeKerasBatchNorm
from .layers.binary import MakeKerasBinary
from .layers.concat import MakeKerasConcat
from .layers.conv import MakeKerasConv
from .layers.conv_transpose import MakeKerasConvTranspose   # GSoC Ex5: Conv2DTranspose
from .layers.dense import MakeKerasDense
from .layers.elu import MakeKerasELU
from .layers.flatten import MakeKerasFlatten
from .layers.layernorm import MakeKerasLayerNorm
from .layers.leaky_relu import MakeKerasLeakyRelu
from .layers.permute import MakeKerasPermute
from .layers.pooling import MakeKerasPooling
from .layers.relu import MakeKerasReLU
from .layers.reshape import MakeKerasReshape
from .layers.rnn import MakeKerasRNN                        # GSoC Ex5: GRU + LSTM
from .layers.selu import MakeKerasSeLU
from .layers.sigmoid import MakeKerasSigmoid
from .layers.softmax import MakeKerasSoftmax
from .layers.swish import MakeKerasSwish
from .layers.tanh import MakeKerasTanh


def MakeKerasActivation(layer):
    attributes = layer["layerAttributes"]
    activation = attributes["activation"]
    fLayerActivation = str(activation.__name__)

    if fLayerActivation in mapKerasLayer.keys():
        return mapKerasLayer[fLayerActivation](layer)
    else:
        raise Exception("TMVA.SOFIE - parsing keras activation layer " + fLayerActivation + " is not yet supported")


# Set global dictionaries, mapping layers to corresponding functions that create their ROperator instances
mapKerasLayer = {
    "Activation": MakeKerasActivation,
    "Permute": MakeKerasPermute,
    "BatchNormalization": MakeKerasBatchNorm,
    "LayerNormalization": MakeKerasLayerNorm,
    "Reshape": MakeKerasReshape,
    "Flatten": MakeKerasFlatten,
    "Concatenate": MakeKerasConcat,
    "swish": MakeKerasSwish,
    "silu": MakeKerasSwish,
    "Add": MakeKerasBinary,
    "Subtract": MakeKerasBinary,
    "Multiply": MakeKerasBinary,
    "Softmax": MakeKerasSoftmax,
    "tanh": MakeKerasTanh,
    #  "Identity": MakeKerasIdentity,
    #  "Dropout": MakeKerasIdentity,
    "ReLU": MakeKerasReLU,
    "relu": MakeKerasReLU,
    "ELU": MakeKerasELU,
    "elu": MakeKerasELU,
    "selu": MakeKerasSeLU,
    "sigmoid": MakeKerasSigmoid,
    "LeakyReLU": MakeKerasLeakyRelu,
    "leaky_relu": MakeKerasLeakyRelu,
    "softmax": MakeKerasSoftmax,
    "MaxPooling2D": MakeKerasPooling,
    "AveragePooling2D": MakeKerasPooling,
    "GlobalAveragePooling2D": MakeKerasPooling,
    "SimpleRNN": MakeKerasRNN,
    "GRU": MakeKerasRNN,           # GSoC Ex5: enabled GRU
    "LSTM": MakeKerasRNN,          # GSoC Ex5: enabled LSTM
}

# These layers support an optional activation function fused into the operator
mapKerasLayerWithActivation = {
    "Dense": MakeKerasDense,
    "Conv2D": MakeKerasConv,
    "Conv2DTranspose": MakeKerasConvTranspose,   # GSoC Ex5: Conv2DTranspose
}


def add_layer_into_RModel(rmodel, layer_data):
    """
    Add a Keras layer operation to an existing RModel using the SOFIE framework.

    This function takes an existing RModel and a dictionary representing a Keras layer
    and its attributes, and adds the corresponding layer operation to the RModel using
    the SOFIE framework. The function supports various types of Keras layers, including
    those with or without activation functions.

    Parameters:
    rmodel (RModel): An existing RModel to which the layer operation will be added.
    layer_data (dict): A dictionary containing layer information including type,
                      attributes, input, output, and layer data type.

    Returns:
    RModel: The updated RModel after adding the layer operation.

    Raises exception: If the provided layer type or activation function is not supported.
    """
    import numpy as np
    from ROOT.TMVA.Experimental import SOFIE

    def move_operator(op):
        """
        Wrap an operator into a std::unique_ptr to pass it to RModel::AddOperator().
        """
        import ROOT

        ROOT.SetOwnership(op, False)
        return ROOT.std.unique_ptr[type(op)](op)

    keras_version = get_keras_version()

    fLayerType = layer_data["layerType"]

    # reshape and flatten layers don't have weights, but they need constant tensor for the shape
    if fLayerType == "Reshape" or fLayerType == "Flatten":
        Attributes = layer_data["layerAttributes"]
        if keras_version < "2.16":
            LayerName = Attributes["_name"]
        else:
            LayerName = Attributes["name"]

        if fLayerType == "Reshape":
            TargetShape = np.asarray(Attributes["target_shape"]).astype("int64")
            TargetShape = np.insert(TargetShape, 0, 1)
        else:
            if "_build_input_shape" in Attributes.keys():
                input_shape = Attributes["_build_input_shape"]
            elif "_build_shapes_dict" in Attributes.keys():
                input_shape = list(Attributes["_build_shapes_dict"]["input_shape"])
            else:
                raise RuntimeError("Failed to extract build input shape from " + fLayerType + " layer")
            TargetShape = [SOFIE.ConvertShapeToLength(input_shape[1:])]
            TargetShape = np.asarray(TargetShape)

        shape_tensor_name = LayerName + "_shape"
        shape_data = TargetShape.data
        print(TargetShape, shape_data)
        print(len(TargetShape))
        rmodel.AddInitializedTensor["int64_t"](shape_tensor_name, [len(TargetShape)], shape_data)

    # These layers only have one operator - excluding the recurrent layers, in which the activation function(s)
    # are included in the recurrent operator
    if fLayerType in mapKerasLayer.keys():
        Attributes = layer_data["layerAttributes"]
        inputs = layer_data["layerInput"]
        outputs = layer_data["layerOutput"]
        if keras_version < "2.16":
            LayerName = Attributes["_name"]
        else:
            LayerName = Attributes["name"]

        fLayerOutput = outputs[0]
        if fLayerType == "GlobalAveragePooling2D":
            if layer_data["channels_last"]:
                op = SOFIE.ROperator_Transpose("float")([0, 3, 1, 2], inputs[0], LayerName + "PreTrans")
                rmodel.AddOperator(move_operator(op))
                inputs[0] = LayerName + "PreTrans"
            outputs[0] = LayerName + "Squeeze"
            rmodel.AddOperator(move_operator(mapKerasLayer[fLayerType](layer_data)))
            op = SOFIE.ROperator_Reshape(SOFIE.ReshapeOpMode.Squeeze, [2, 3], LayerName + "Squeeze", fLayerOutput)
            rmodel.AddOperator(move_operator(op))

        elif fLayerType == "BatchNormalization":
            if "_build_input_shape" in Attributes.keys():
                num_input_shapes = len(Attributes["_build_input_shape"])
            elif "_build_shapes_dict" in Attributes.keys():
                num_input_shapes = len(list(Attributes["_build_shapes_dict"]["input_shape"]))

            axis = Attributes["axis"]
            axis = axis[0] if isinstance(axis, list) else axis
            if axis < 0:
                axis += num_input_shapes
            fAttrPerm = list(range(0, num_input_shapes))
            fAttrPerm[1] = axis
            fAttrPerm[axis] = 1
            op = SOFIE.ROperator_Transpose("float")(fAttrPerm, inputs[0], LayerName + "PreTrans")
            rmodel.AddOperator(move_operator(op))
            inputs[0] = LayerName + "PreTrans"
            outputs[0] = LayerName + "PostTrans"
            rmodel.AddOperator(move_operator(mapKerasLayer[fLayerType](layer_data)))
            op = SOFIE.ROperator_Transpose("float")(fAttrPerm, LayerName + "PostTrans", fLayerOutput)
            rmodel.AddOperator(move_operator(op))

        elif fLayerType == "MaxPooling2D" or fLayerType == "AveragePooling2D":
            if layer_data["channels_last"]:
                op = SOFIE.ROperator_Transpose("float")([0, 3, 1, 2], inputs[0], LayerName + "PreTrans")
                rmodel.AddOperator(move_operator(op))
                inputs[0] = LayerName + "PreTrans"
                outputs[0] = LayerName + "PostTrans"
            rmodel.AddOperator(move_operator(mapKerasLayer[fLayerType](layer_data)))
            if layer_data["channels_last"]:
                op = SOFIE.ROperator_Transpose("float")([0, 2, 3, 1], LayerName + "PostTrans", fLayerOutput)
                rmodel.AddOperator(move_operator(op))

        else:
            rmodel.AddOperator(move_operator(mapKerasLayer[fLayerType](layer_data)))

        return rmodel

    # These layers require two operators - dense/conv and their activation function
    elif fLayerType in mapKerasLayerWithActivation.keys():
        Attributes = layer_data["layerAttributes"]
        if keras_version < "2.16":
            LayerName = Attributes["_name"]
        else:
            LayerName = Attributes["name"]
        fPActivation = Attributes["activation"]
        LayerActivation = fPActivation.__name__
        if LayerActivation in ["selu", "sigmoid"]:
            rmodel.AddNeededStdLib("cmath")

        # if there is an activation function after the layer
        if LayerActivation != "linear":
            if LayerActivation not in mapKerasLayer.keys():
                raise Exception(
                    "TMVA.SOFIE - parsing keras activation function " + LayerActivation + " is not yet supported"
                )
            outputs = layer_data["layerOutput"]
            inputs = layer_data["layerInput"]
            fActivationLayerOutput = outputs[0]

            # Conv2D and Conv2DTranspose: Keras uses channels_last by default but ONNX uses
            # channels_first, so we need Transpose operators before and after.
            if fLayerType in ("Conv2D", "Conv2DTranspose"):
                if layer_data["channels_last"]:
                    op = SOFIE.ROperator_Transpose("float")([0, 3, 1, 2], inputs[0], LayerName + "PreTrans")
                    rmodel.AddOperator(move_operator(op))
                    inputs[0] = LayerName + "PreTrans"
                    layer_data["layerInput"] = inputs
            outputs[0] = LayerName + fLayerType
            layer_data["layerOutput"] = outputs
            op = mapKerasLayerWithActivation[fLayerType](layer_data)
            rmodel.AddOperator(move_operator(op))
            Activation_layer_input = LayerName + fLayerType
            if fLayerType in ("Conv2D", "Conv2DTranspose"):
                if layer_data["channels_last"]:
                    op = SOFIE.ROperator_Transpose("float")(
                        [0, 2, 3, 1], LayerName + fLayerType, LayerName + "PostTrans"
                    )
                    rmodel.AddOperator(move_operator(op))
                    Activation_layer_input = LayerName + "PostTrans"

            inputs[0] = Activation_layer_input
            outputs[0] = fActivationLayerOutput
            layer_data["layerInput"] = inputs
            layer_data["layerOutput"] = outputs
            rmodel.AddOperator(move_operator(mapKerasLayer[LayerActivation](layer_data)))

        else:
            # linear activation — still need Transpose for Conv2D / Conv2DTranspose
            if fLayerType in ("Conv2D", "Conv2DTranspose"):
                inputs = layer_data["layerInput"]
                outputs = layer_data["layerOutput"]
                fLayerOutput = outputs[0]
                if layer_data["channels_last"]:
                    op = SOFIE.ROperator_Transpose("float")([0, 3, 1, 2], inputs[0], LayerName + "PreTrans")
                    rmodel.AddOperator(move_operator(op))
                    inputs[0] = LayerName + "PreTrans"
                    layer_data["layerInput"] = inputs
                    outputs[0] = LayerName + "PostTrans"
            rmodel.AddOperator(move_operator(mapKerasLayerWithActivation[fLayerType](layer_data)))
            if fLayerType in ("Conv2D", "Conv2DTranspose"):
                if layer_data["channels_last"]:
                    op = SOFIE.ROperator_Transpose("float")([0, 2, 3, 1], LayerName + "PostTrans", fLayerOutput)
                    rmodel.AddOperator(move_operator(op))
        return rmodel
    else:
        raise Exception("TMVA.SOFIE - parsing keras layer " + fLayerType + " is not yet supported")


class PyKeras:
    def Parse(filename, batch_size=1):

        import keras
        import numpy as np
        from ROOT.TMVA.Experimental import SOFIE

        keras_version = get_keras_version()

        if not os.path.exists(filename):
            raise RuntimeError("Model file {} not found!".format(filename))

        keras_model = keras.models.load_model(filename)
        keras_model.load_weights(filename)

        sep = "/"
        if os.name == "nt":
            sep = "\\"

        isep = filename.rfind(sep)
        filename_nodir = filename
        if isep != -1:
            filename_nodir = filename[isep + 1:]

        ttime = time.time()
        gmt_time = time.gmtime(ttime)
        parsetime = time.asctime(gmt_time)

        rmodel = SOFIE.RModel.RModel(filename_nodir, parsetime)

        print("PyKeras: parsing model ", filename)

        layer_iter = 0
        is_functional_model = True if keras_model.__class__.__name__ == "Functional" else False
        for layer in keras_model.layers:
            layer_data = {}
            layer_data["layerType"] = layer.__class__.__name__
            layer_data["layerAttributes"] = layer.__dict__
            if keras_version < "2.16" or is_functional_model:
                if "input_layer" in layer.name:
                    layer_data["layerInput"] = layer.name
                else:
                    layer_data["layerInput"] = (
                        [x.name for x in layer.input] if isinstance(layer.input, list) else [layer.input.name]
                    )
            else:
                if "input_layer" in layer.input.name:
                    layer_data["layerInput"] = [layer.input.name]
                else:
                    if layer_iter == 0:
                        input_layer_name = "tensor_input_" + layer.name
                    else:
                        input_layer_name = "tensor_output_" + keras_model.layers[layer_iter - 1].name
                    layer_data["layerInput"] = [input_layer_name]
            if keras_version < "2.16" or is_functional_model:
                layer_data["layerOutput"] = (
                    [x.name for x in layer.output] if isinstance(layer.output, list) else [layer.output.name]
                )
            else:
                output_layer_name = "tensor_output_" + layer.name
                layer_data["layerOutput"] = (
                    [x.name for x in layer.output] if isinstance(layer.output, list) else [output_layer_name]
                )

            layer_iter += 1
            fLayerType = layer_data["layerType"]
            layer_data["layerDType"] = layer.dtype

            if len(layer.weights) > 0:
                if keras_version < "2.16":
                    layer_data["layerWeight"] = [x.name for x in layer.weights]
                else:
                    layer_data["layerWeight"] = [x.path for x in layer.weights]
            else:
                layer_data["layerWeight"] = []

            # For Conv/Pooling layers we need to know the data format
            if layer_data["layerType"] in [
                "Conv2D", "Conv2DTranspose",                    # GSoC Ex5: added Conv2DTranspose
                "MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D"
            ]:
                layer_data["channels_last"] = True if layer.data_format == "channels_last" else False

            # For recurrent layers extract additional unique information
            if layer_data["layerType"] in ["SimpleRNN", "LSTM", "GRU"]:
                layer_data["layerAttributes"]["activation"] = layer.activation
                layer_data["layerAttributes"]["direction"] = "backward" if layer.go_backwards else "forward"
                layer_data["layerAttributes"]["units"] = layer.units
                layer_data["layerAttributes"]["layout"] = layer.input.shape[0] is None
                layer_data["layerAttributes"]["hidden_size"] = layer.output.shape[-1]

                if layer_data["layerType"] != "SimpleRNN":
                    layer_data["layerAttributes"]["recurrent_activation"] = layer.recurrent_activation

                if layer_data["layerType"] == "GRU":
                    layer_data["layerAttributes"]["linear_before_reset"] = (
                        1 if layer.reset_after and layer.recurrent_activation.__name__ == "sigmoid" else 0
                    )

            if fLayerType == "InputLayer":
                continue

            # Adding required BLAS routines
            if fLayerType == "Dense":
                rmodel.AddBlasRoutines({"Gemm", "Gemv"})
            elif fLayerType == "BatchNormalization":
                rmodel.AddBlasRoutines({"Copy", "Axpy"})
            elif fLayerType in ("Conv1D", "Conv2D", "Conv3D", "Conv2DTranspose"):  # GSoC Ex5
                rmodel.AddBlasRoutines({"Gemm", "Axpy"})

            rmodel = add_layer_into_RModel(rmodel, layer_data)

        # Extracting model weights
        weight = []
        for idx in range(len(keras_model.get_weights())):
            weightProp = {}
            if keras_version < "2.16":
                weightProp["name"] = keras_model.weights[idx].name
            else:
                weightProp["name"] = keras_model.weights[idx].path
            weightProp["dtype"] = keras_model.get_weights()[idx].dtype.name

            # Weight axis transposition:
            # Conv2D kernel shape in Keras:        (kH, kW, C_in,  C_out)
            # Conv2DTranspose kernel in Keras:     (kH, kW, C_out, C_in)  ← note swapped C_in/C_out
            # ONNX/SOFIE expects:
            #   Conv2D         → (C_out, C_in,  kH, kW)  i.e. transpose (3,2,0,1)
            #   Conv2DTranspose→ (C_in,  C_out, kH, kW)  i.e. transpose (3,2,0,1) same axes
            if "conv" in weightProp["name"] and keras_model.weights[idx].shape.ndims == 4:
                weightProp["value"] = keras_model.get_weights()[idx].transpose((3, 2, 0, 1)).copy()
            else:
                weightProp["value"] = keras_model.get_weights()[idx]
            weight.append(weightProp)

        for weightIter in range(len(weight)):
            fWeightTensor = weight[weightIter]
            fWeightName = fWeightTensor["name"]
            fWeightDType = SOFIE.ConvertStringToType(fWeightTensor["dtype"])
            fWeightTensorValue = fWeightTensor["value"]
            fWeightTensorSize = 1
            fWeightTensorShape = []

            if (
                "simple_rnn" in fWeightName
                or "lstm" in fWeightName
                or ("gru" in fWeightName and "bias" not in fWeightName)
            ):
                fWeightTensorShape.append(1)

            for j in range(len(fWeightTensorValue.shape)):
                fWeightTensorShape.append(fWeightTensorValue.shape[j])
                fWeightTensorSize *= fWeightTensorValue.shape[j]

            if fWeightDType == SOFIE.ETensorType.FLOAT:
                fWeightArray = fWeightTensorValue

                # LSTM gate reordering: Keras order is [i, f, c, o], ONNX order is [i, o, f, c]
                if "lstm" in fWeightName:
                    if "kernel" in fWeightName:
                        units = int(fWeightArray.shape[1] / 4)
                        W_f = fWeightArray[:, units: units * 2].copy()
                        W_c = fWeightArray[:, units * 2: units * 3].copy()
                        W_o = fWeightArray[:, units * 3:].copy()
                        fWeightArray[:, units: units * 2] = W_o
                        fWeightArray[:, units * 2: units * 3] = W_f
                        fWeightArray[:, units * 3:] = W_c
                    else:
                        units = int(fWeightArray.shape[0] / 4)
                        W_f = fWeightArray[units: units * 2].copy()
                        W_c = fWeightArray[units * 2: units * 3].copy()
                        W_o = fWeightArray[units * 3:].copy()
                        fWeightArray[units: units * 2] = W_o
                        fWeightArray[units * 2: units * 3] = W_f
                        fWeightArray[units * 3:] = W_c

                if "simple_rnn" in fWeightName or "lstm" in fWeightName or "gru" in fWeightName:
                    if "kernel" in fWeightName:
                        fWeightArray = np.transpose(fWeightArray)
                        fWeightTensorShape[1], fWeightTensorShape[2] = fWeightTensorShape[2], fWeightTensorShape[1]

                    fData = fWeightArray.flatten()

                    if "bias" in fWeightName and len(fData.shape) == 1:
                        fWeightTensorShape[1] *= 2
                        fRbias = fData.copy() * 0
                        fData = np.concatenate((fData, fRbias))

                else:
                    fData = fWeightArray.flatten()
                rmodel.AddInitializedTensor["float"](fWeightName, fWeightTensorShape, fData)
            else:
                raise TypeError("Type error: TMVA SOFIE does not yet support data layer type: " + fWeightDType)

        # Extracting input tensor info
        if keras_version < "2.16":
            fPInputs = keras_model.input_names
        else:
            fPInputs = [x.name for x in keras_model.inputs]

        fPInputShape = (
            keras_model.input_shape if isinstance(keras_model.input_shape, list) else [keras_model.input_shape]
        )
        fPInputDType = []
        for idx in range(len(keras_model.inputs)):
            dtype = keras_model.inputs[idx].dtype.__str__()
            if dtype == "float32":
                fPInputDType.append(dtype)
            else:
                fPInputDType.append(dtype[9:-2])

        if len(fPInputShape) == 1:
            inputName = fPInputs[0]
            inputDType = SOFIE.ConvertStringToType(fPInputDType[0])
            inputShape = list(fPInputShape[0])
            if inputShape[0] is None or inputShape[0] <= 0:
                inputShape[0] = batch_size
            rmodel.AddInputTensorInfo(inputName, inputDType, inputShape)
            rmodel.AddInputTensorName(inputName)
        else:
            for inputName, inputDType, inputShapeTuple in zip(fPInputs, fPInputDType, fPInputShape):
                inputDType = SOFIE.ConvertStringToType(inputDType)
                inputShape = list(inputShapeTuple)
                if inputShape[0] is None or inputShape[0] <= 0:
                    inputShape[0] = batch_size
                rmodel.AddInputTensorInfo(inputName, inputDType, inputShape)
                rmodel.AddInputTensorName(inputName)

        # Adding OutputTensorInfos
        outputNames = []
        if keras_version < "2.16" or is_functional_model:
            for layerName in keras_model.output_names:
                final_layer = keras_model.get_layer(layerName)
                output_layer_name = final_layer.output.name
                outputNames.append(output_layer_name)
        else:
            output_layer_name = "tensor_output_" + keras_model.layers[-1].name
            outputNames.append(output_layer_name)

        rmodel.AddOutputTensorNameList(outputNames)
        return rmodel
