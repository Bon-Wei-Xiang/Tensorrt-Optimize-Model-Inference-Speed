Using NVIDIA TensorRT to Optimize Model Inference Speed
===============================================================

Overview
--------

This guide provides instructions on how to use NVIDIA TensorRT to optimize model inference speed. TensorRT is a high-performance deep learning inference library developed by NVIDIA that leverages the power of NVIDIA GPUs to accelerate inference processes. The provided code demonstrates how to load a TensorRT engine, allocate memory for inputs and outputs, and run inference.

Prerequisites
-------------

To use TensorRT, you need the following:

*   **NVIDIA GPU**: Ensure your system has an NVIDIA GPU.
*   **CUDA Toolkit**: Download and install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).
*   **TensorRT**: Download and install TensorRT from [NVIDIA's website](https://developer.nvidia.com/tensorrt/download).

Installation
------------

Follow these steps to set up your environment:

### Step 1: Install CUDA Toolkit

1.  Visit the [CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads).
2.  Choose your operating system, architecture, distribution, and version.
3.  Follow the provided instructions to download and install the CUDA Toolkit.

### Step 2: Install TensorRT

1.  Visit the [TensorRT download page](https://developer.nvidia.com/tensorrt/download).
2.  Choose your operating system and version.
3.  Follow the provided instructions to download and install TensorRT.

### Step 3: Install Python Packages

Ensure you have the following Python packages installed:

    pip install numpy pycuda

Code Explanation
----------------

### Code 1: Check Model Info

This script loads a TensorRT engine and prints information about the model's tensors.

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)


    with open("best200_5.engine", "rb") as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())

    for idx, i in enumerate(model):
        print(f"block {idx}")
        print(f"layer type : {trt.nptype(model.get_tensor_dtype(i))}")
        print(f"layer shape : {model.get_tensor_shape(i)}")
        if model.binding_is_input(i):
            print("this is input layer")
        else:
            print("this is output layer")
    

### Code 2: TensorRT Inference Engine

This class provides methods to load a TensorRT engine, allocate memory for inputs and outputs, and run inference.

    import numpy as np
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda
    import cv2
    
    
    class HostDeviceMemory:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem
    
        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    
        def __repr__(self):
            return self.__str__()
    
    
    class Tensorrt_engine:
        def __init__(self, engine_path):
            self.Logger = trt.Logger(trt.Logger.INFO)
            self.Engine = self.__load_engine(engine_path=engine_path)
            self.context = self.Engine.create_execution_context()
            self.Stream = cuda.Stream()
            self.__binding, self.__input_layer, self.__output_layer = self.__allocate_buffer()
    
        def __load_engine(self, engine_path):
            with open(engine_path, 'rb') as f, trt.Runtime(logger=self.Logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            return engine
    
        def __allocate_buffer(self):
            binding, input_layer, output_layer = [], [], []
            for i in self.Engine:
                layer_shape = self.Engine.get_tensor_shape(i)
                layer_type = trt.nptype(self.Engine.get_tensor_dtype(i))
                sample = np.empty(layer_shape, layer_type)
                host_mem = cuda.pagelocked_empty_like(sample)
                device_mem = cuda.mem_alloc_like(host_mem)
                binding.append(int(device_mem))
                if self.Engine.binding_is_input(i):
                    input_layer.append(HostDeviceMemory(host_mem=host_mem, device_mem=device_mem))
                else:
                    output_layer.append(HostDeviceMemory(host_mem=host_mem, device_mem=device_mem))
            return binding, input_layer, output_layer
    
        def __call__(self, img):
            for i in self.__input_layer:
                i.host[:] = img[:]
    
            [cuda.memcpy_htod_async(i.device, i.host, self.Stream) for i in self.__input_layer]
            self.context.execute_async_v2(bindings=self.__binding, stream_handle=self.Stream.handle)
            [cuda.memcpy_dtoh_async(i.host, i.device, self.Stream) for i in self.__output_layer]
            self.Stream.synchronize()
            return [i.host for i in self.__output_layer]
    

Model Conversion to TensorRT Engine
-----------------------------------

To convert an ONNX model to a TensorRT engine, you can use the `trtexec.exe` tool that comes with TensorRT. This tool simplifies the conversion process.

### Step 1: Locate `trtexec.exe`

After installing TensorRT, you can find `trtexec.exe` in the TensorRT bin directory, usually located at `TensorRT-8.5.1.7\bin\trtexec.exe`.

### Step 2: Convert ONNX Model

Use the following command to convert an ONNX model to a TensorRT engine:

    trtexec.exe --onnx=path_to_your_model.onnx --saveEngine=path_to_save_engine.trt

Replace `path_to_your_model.onnx` with the path to your ONNX model file and `path_to_save_engine.trt` with the desired path to save the TensorRT engine.

### Dynamic Batch Size for Batch Inference

To enable batch inference, ensure your ONNX model supports dynamic shapes. You can then set the batch size during conversion using `trtexec.exe`:

    trtexec.exe --onnx=path_to_your_model.onnx --saveEngine=path_to_save_engine.trt --minShapes=input:1x3x224x224 --optShapes=input:16x3x224x224 --maxShapes=input:32x3x224x224

Replace `input:1x3x224x224`, `input:16x3x224x224`, and `input:32x3x224x224` with the appropriate minimum, optimal, and maximum input shapes for your model.

Usage Example
-------------

    # Initialize the TensorRT engine
    model = Tensorrt_engine("model_path")
    
    # Preprocess your image
    # image = preprocess_image_function()
    
    # Run inference
    predict = model(image)
    
    # Post-process the inference data
    # postprocess_inference_data(predict)
    

Replace `"model_path"` with the path to your TensorRT engine file. Implement the `preprocess_image_function` and `postprocess_inference_data` according to your specific requirements.

Conclusion
----------

By following this guide, you should be able to set up TensorRT to optimize your model inference speed. Ensure you have the required hardware and software, install the necessary packages, convert your ONNX model to a TensorRT engine using `trtexec.exe`, and use the provided code to load your TensorRT engine and run inference. For further details, refer to the official [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html).