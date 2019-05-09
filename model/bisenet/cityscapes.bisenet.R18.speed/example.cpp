#include <pybind11/pybind11.h>
#include <NvOnnxParserRuntime.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "onnx create  plugin"; // optional module docstring

    m.def("Create", &nvonnxparser::createRawPluginFactory, "create plugin");
}
