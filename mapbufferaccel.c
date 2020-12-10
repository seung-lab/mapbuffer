#ifndef MAPBUFFER_BINARYSEARCH_H_
#define MAPBUFFER_BINARYSEARCH_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

uint64_t c_eytzinger_binary_search(uint64_t x, uint64_t* array, size_t N) {
    // int64_t block_size = 8; // two cache lines 64 * 2 / 8
    uint64_t k = 1;
    while (k <= (uint64_t)N) {
        // __builtin_prefetch(array + k * block_size);
        // multiply by 2 b/c index is [label, pos, label, pos]
        k = 2 * k + (array[(k - 1) << 1] < x); 
    }
    k >>= ffs(~k);
    return k - 1;
}

static PyObject* eytzinger_binary_search(PyObject* self, PyObject *args) {
    Py_buffer index;
    Py_ssize_t label;

    if (!PyArg_ParseTuple(args, "ny*", &label, &index)) {
        return NULL;
    }
    
    // num bytes / 8 = uint64, then divide by two because 
    // index is [label, pos, label, pos]
    size_t N = (size_t)index.len / 2 / 8;
    uint64_t* bytes = (uint64_t*)index.buf;

    int64_t res = c_eytzinger_binary_search((uint64_t)label, bytes, N);
    return Py_BuildValue("L", res); // L = long long
}

static PyMethodDef mapbufferaccel_methods[] = {
    {"eytzinger_binary_search", (PyCFunction)eytzinger_binary_search, METH_VARARGS, "Binary search on Eytzinger sorted mapbuffer index. Arguments: uint64_t label, uint64* index"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mapbufferaccel_module = {
    PyModuleDef_HEAD_INIT,
    "mapbufferaccel",
    "Accelerated functions for MapBuffer.",
    -1,
    mapbufferaccel_methods
};

PyMODINIT_FUNC PyInit_mapbufferaccel(void) {
    return PyModule_Create(&mapbufferaccel_module);
}

#endif