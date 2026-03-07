import math

from .. import get_keras_version


def MakeKerasConvTranspose(layer):
    """
    Create a Keras-compatible Conv2DTranspose (transposed convolution) layer operation
    using the SOFIE framework.

    Conv2DTranspose (also called deconvolution or fractionally-strided convolution)
    is the transpose of a regular convolution. It is commonly used in:
      - Autoencoders (decoder path)
      - Generative Adversarial Networks (GANs)
      - Semantic segmentation (upsampling)
      - Super-resolution networks

    Mathematical operation:
        For input X with shape (N, C_in, H, W) and kernel W with shape
        (C_in, C_out, kH, kW), the output Y has shape (N, C_out, H_out, W_out)
        where H_out and W_out are determined by the stride and padding.

    Keras weight layout:    (kH, kW, C_out, C_in)  — channels last convention
    ONNX/SOFIE weight layout: (C_in, C_out, kH, kW) — channels first convention
    The weight transposition is handled in parser.py during weight extraction.

    Parameters
    ----------
    layer : dict
        Dictionary containing layer information:
          - layerInput    : list of input tensor names
          - layerOutput   : list of output tensor names
          - layerDType    : data type string (must be "float32")
          - layerWeight   : list of weight tensor names [kernel, bias]
          - layerAttributes: dict of Keras layer attributes including:
              * kernel_size    : tuple (kH, kW)
              * strides        : tuple (sH, sW)
              * padding        : "valid" or "same"
              * dilation_rate  : tuple (dH, dW)
              * groups         : int, number of groups for grouped convolution
              * output_padding : tuple or None (extra padding added to output)

    Returns
    -------
    ROperator_ConvTranspose
        A SOFIE framework operator representing the Conv2DTranspose operation.

    Raises
    ------
    RuntimeError
        If padding type is not "valid" or "same", or if dtype is not float.

    Notes
    -----
    ONNX ConvTranspose attributes:
        auto_pad      : "VALID", "SAME_UPPER", "SAME_LOWER", or "NOTSET"
        dilations     : dilation factor for each spatial axis
        group         : number of groups for grouped convolution
        kernel_shape  : spatial dimensions of the kernel
        output_padding: extra padding added to one side of each spatial dimension
        output_shape  : expected output spatial shape (optional)
        pads          : explicit padding [x1_begin, x2_begin, x1_end, x2_end]
        strides       : stride along each spatial axis

    GSoC 2026 Exercise 5 implementation by Abhinav Pandey
    """
    from ROOT.TMVA.Experimental import SOFIE

    keras_version = get_keras_version()

    # --- Extract layer data ---
    finput      = layer["layerInput"]
    foutput     = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    attributes  = layer["layerAttributes"]
    fWeightNames = layer["layerWeight"]

    fLayerInputName  = finput[0]
    fLayerOutputName = foutput[0]

    # Kernel (weight) and bias tensor names
    fKernelName = fWeightNames[0]
    fBiasName   = fWeightNames[1] if len(fWeightNames) > 1 else ""

    # --- Extract Conv2DTranspose attributes ---
    fAttrKernelShape = list(attributes["kernel_size"])    # (kH, kW)
    fAttrStrides     = list(attributes["strides"])        # (sH, sW)
    fAttrDilations   = list(attributes["dilation_rate"])  # (dH, dW)
    fAttrGroup       = int(attributes.get("groups", attributes.get("group", 1)))
    fKerasPadding    = str(attributes["padding"])         # "valid" or "same"

    # output_padding: extra rows/cols added to one side of output
    output_padding_attr = attributes.get("output_padding", None)
    if output_padding_attr is None:
        fAttrOutputPadding = [0, 0]
    else:
        fAttrOutputPadding = list(output_padding_attr)

    # --- Determine ONNX padding ---
    fAttrPads    = [0, 0, 0, 0]  # [top, left, bottom, right]
    fAttrAutopad = "NOTSET"

    if fKerasPadding == "valid":
        # No padding: output is larger than input
        fAttrAutopad = "VALID"
        fAttrPads    = [0, 0, 0, 0]

    elif fKerasPadding == "same":
        # Padding is added so that output_size = input_size * stride
        # For ConvTranspose with "same" padding:
        #   total_padding = max(kernel_size - stride, 0)
        #   pad_before = total_padding // 2
        #   pad_after  = total_padding - pad_before
        fAttrAutopad = "NOTSET"

        pad_h = max(fAttrKernelShape[0] - fAttrStrides[0], 0)
        pad_w = max(fAttrKernelShape[1] - fAttrStrides[1], 0)

        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        # ONNX pads format for 2D: [H_begin, W_begin, H_end, W_end]
        fAttrPads = [pad_top, pad_left, pad_bottom, pad_right]

    else:
        raise RuntimeError(
            "TMVA::SOFIE - RModel Keras Parser does not yet support "
            "Conv2DTranspose layer with padding: " + fKerasPadding
        )

    # --- Create SOFIE ROperator_ConvTranspose ---
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_ConvTranspose["float"](
            fAttrAutopad,       # auto_pad string
            fAttrDilations,     # dilations [dH, dW]
            fAttrGroup,         # group (int)
            fAttrKernelShape,   # kernel_shape [kH, kW]
            fAttrOutputPadding, # output_padding [pH, pW]
            [],                 # output_shape [] (not specified, inferred)
            fAttrPads,          # pads [H_begin, W_begin, H_end, W_end]
            fAttrStrides,       # strides [sH, sW]
            fLayerInputName,    # X  — input tensor name
            fKernelName,        # W  — kernel tensor name
            fBiasName,          # B  — bias tensor name
            fLayerOutputName,   # Y  — output tensor name
        )
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Conv2DTranspose does not yet "
            "support input type " + fLayerDType
        )
