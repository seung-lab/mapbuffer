#ifndef MAPBUFFER_BINARYSEARCH_H_
#define MAPBUFFER_BINARYSEARCH_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <numpy/arrayobject.h>

#if defined _MSC_VER
# include <intrin.h>
#endif

uint64_t mb_ffs (uint64_t x) {
#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
   return __builtin_ffs(x);
#elif defined _MSC_VER
  /* _BitScanForward
     <https://docs.microsoft.com/en-us/cpp/intrinsics/bitscanforward-bitscanforward64> */
  unsigned long bit;
  if (_BitScanForward (&bit, x)) {
    return bit + 1;
  }
  return 0;
#else 
  if (x == 0) {
    return 0;
  }
  for (uint64_t i = 0; i < 64; i++) {
    if ((x >> i) & 0x1) {
        return i + 1;
    }
  }
  return 0;
#endif
}

uint64_t c_eytzinger_binary_search(uint64_t x, uint64_t* array, size_t N) {
    // int64_t block_size = 8; // two cache lines 64 * 2 / 8
    uint64_t k = 1;
    while (k <= (uint64_t)N) {
        // __builtin_prefetch(array + k * block_size);
        // multiply by 2 b/c index is [label, pos, label, pos]
        k = 2 * k + (array[(k - 1) << 1] < x); 
    }
    k >>= mb_ffs(~k);
    k -= 1;

    if (k >= 0 && array[k << 1] == x) {
        return k;
    }

    return -1;
}

uint32_t c_eytzinger_sort_indices(
    uint32_t* array, uint32_t n, 
    uint32_t i /*=0*/, uint32_t k /*=1*/
) {
    if (k <= n) {
        i = c_eytzinger_sort_indices(array, n, i, 2 * k);
        array[k-1] = i;
        i++;
        i = c_eytzinger_sort_indices(array, n, i, 2 * k + 1);
    }
    return i;
}

// takes an array sorted in ascending order and transforms it
// into eytzinger order
static PyObject* eytzinger_sort_indices(PyObject* self, PyObject *args) {
    PyGILState_STATE state = PyGILState_Ensure();
    import_array();
    
    Py_ssize_t length;
    if (!PyArg_ParseTuple(args, "n", &length)) {
        PyGILState_Release(state);
        Py_RETURN_NONE;
    }
    npy_intp dims[] = {length};
    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_UINT32);

    if (arr == NULL) {
        PyGILState_Release(state);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Numpy array");
        Py_RETURN_NONE;
    }

    uint32_t* data = (uint32_t*)PyArray_DATA(arr);
    c_eytzinger_sort_indices(data, length, 0, 1);
    
    PyGILState_Release(state);
    return (PyObject*)arr;
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
    PyBuffer_Release(&index);
    
    return Py_BuildValue("L", res); // L = long long
}

static PyMethodDef mapbufferaccel_methods[] = {
    {"eytzinger_binary_search", (PyCFunction)eytzinger_binary_search, METH_VARARGS, "Binary search on Eytzinger sorted mapbuffer index. Arguments: uint64_t label, uint64* index"},
    {"eytzinger_sort_indices", (PyCFunction)eytzinger_sort_indices, METH_VARARGS, "Return the Eytzinger layout as a uint32 numpy array for a given sorted array of length N. Arguments: int length"},
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