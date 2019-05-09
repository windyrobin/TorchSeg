import onnx
import os
import tensorrt as trt
import example
import pycuda.driver as cuda
import pycuda.autoinit
import common
import time

import numpy as np


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
#G_LOGGER = trt4.infer.ConsoleLogger(trt4.infer.LogSeverity.WARNING)


def get_engine(onnx_file_path, engine_file_path, plugin_factory):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        #with open(engine_file_path, "rb") as f, trt4.infer.create_infer_runtime(G_LOGGER) as runtime:
            #return runtime.deserialize_cuda_engine(f.read(), plugin_factory)
            print(plugin_factory)
            #return runtime.deserialize_cuda_engine(f.read(), plugin_factory)
            return runtime.deserialize_cuda_engine(f.read(), plugin_factory)

#plugin_factory = trt.OnnxPluginFactory(TRT_LOGGER)

plugin_factory= example.Create(TRT_LOGGER)

onnx_file = './bisenet.onnx'
#engine_file = './bisenet.trt'
engine_file = './bisenet.trt'

# Output shapes expected by the post-processor                                                                                                                                                          
output_shapes = [(1, 19, 96, 192)]                                                                                                                                  
# Do inference with TensorRT                                                                                                                                                                            
trt_outputs = []                                                                                                                                                                                        
def print_statics(arr):
    mean = np.mean(arr)
    max = np.max(arr)
    min = np.min(arr)
    std = np.std(arr)
    print('max,min,mean,std:', max, min, mean, std)

with get_engine(onnx_file, engine_file, plugin_factory) as engine, engine.create_execution_context() as context:                                                                                              
    print(engine)
    print(context)
    #print('engine max batch size:', engine.max_batch_size)
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)                                                                                                                                 

    context.debug_sync = True
    # Do inference                                                                                                                                                                                      
    #print('Running inference on image {}...'.format(input_image_path))                                                                                                                                  
    # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
    #inputs[0].host = image
    ##input_data = np.random.random(size=(1, 3, 768, 768*2)).astype(np.float32)
    input_data = np.load('./diffd/torch_data.npy')

    inputs[0].host = input_data
    #input_data = np.array(input_data, dtype=np.float32, order='C')
    #np.copyto(inputs[0].host, input_data.ravel())
    print("input statics:")
    print_statics(input_data)
    begin_time = time.time()
    #trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    trt_outputs = common.do_inference_sync(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    end_time = time.time()
    print("cost time:", round(1000*(end_time-begin_time)))
    for i in range(1):
       print("output statics:")
       print(len(trt_outputs[i]))
       print_statics(trt_outputs[i])


