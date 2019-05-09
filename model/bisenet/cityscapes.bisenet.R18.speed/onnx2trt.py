# encoding=utf8
# install tensorrt 4.0 first
# tar -xzvf TensorRT-xxxx.Ubuntu-16.04.3.cuda-9.0.tar.gz
# add TensorRT-xxx/lib to bashrc
# sudo pip install the tensorrt-xxx-cp27-cp27mu-linux_x86_64.whl  in uff folder
# sudo pip install the whl  in graphsurgeon folder
# sudo pip install the whl  in python folder

from __future__ import print_function
import sys
import json
import os
import common
import tensorrt as trt

import onnx 
#from tensorrt.parsers import uffparser
#import pycuda.driver as cuda


#trt5
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#trt4
#G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    #parser = uffparser.create_uff_parser()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    # Parse the Uff Network
        builder.max_workspace_size = common.GiB(2)
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
        print('parser errors: ', parser.num_errors)
        for i in range(parser.num_errors):
            err = parser.get_error(i)
            print(err)

        tensor = network.get_input(0)
        print(tensor.name)
        print(tensor.shape)

        print("layers: %d".format(network.num_layers))
        print("inputs: %d".format(network.num_inputs))
        print("outputs: %d".format(network.num_outputs))

        ##network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
        return builder.build_cuda_engine(network)


if __name__=='__main__':

    model = onnx.load('./bisenet.onnx')
    # Check that the IR is well formed
    #onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    #onnx.helper.printable_graph(model.graph)

    engine = build_engine('./bisenet.onnx')
    # Build an engine, allocate buffers and create a stream.
    # For more information on buffer allocation, refer to the introductory samples.
    #inputs, outputs, bindings, stream = allocate_buffers(engine)
    #context = engine.create_execution_context()
    
