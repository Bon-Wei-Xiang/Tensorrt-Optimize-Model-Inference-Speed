import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)


with open("model_path", "rb") as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())

for idx, i in enumerate(model):
    print(f"block {idx}")
    print(f"layer type : {trt.nptype(model.get_tensor_dtype(i))}")
    print(f"layer shape : {model.get_tensor_shape(i)}")
    if model.binding_is_input(i):
        print("this is input layer")
    else:
        print("this is output layer")
    
