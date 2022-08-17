#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>


#include "function.h" //this is the function that we want to call

// Docstring
static char module_docstring[] =
    "Example module providing a function to add two numbers";

PyDoc_STRVAR(add_doc, "Add two numbers or arrays.\n\n"
                      "Parameters\n"
                      "----------\n"
                      "a: numpy_array\n"
                      "   or single number\n"
                      "b: numpy_array\n"
                      "   or single number\n"

                      "Returns\n"
                      "----------\n"
                      "result: numpy_array\n"
                      "   added numbers\n\n");

// Declare functions that should go into the module
static PyObject *add(PyObject *self, PyObject *args);

// This is the module itself
static PyMethodDef module_methods[] = {
    {"add", (PyCFunction)add, METH_VARARGS, add_doc}, {NULL, NULL, 0, NULL}};

static struct PyModuleDef mymod_def = {PyModuleDef_HEAD_INIT, "mymod",
                                       module_docstring, -1, module_methods};

PyMODINIT_FUNC PyInit_mymod(void) {
    PyObject *m = PyModule_Create(&mymod_def);
    if (m == NULL)
        return NULL;

    import_array();
    return m;
}

static PyObject *add(PyObject *self, PyObject *args) {
    // Python offers a lot of flexibility so the first thing we need to do
    // is to parse the arguments and make sure we are called with correct
    // parameters
    PyObject *a_obj;
    PyObject *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj))
        return NULL;

    // Create two numpy arrays from the passed objects, if possible numpy will
    // use the underlying buffer, otherwise it will create a copy, for example
    // if data type is different or we pass in a list. The
    // NPY_ARRAY_C_CONTIGUOUS flag ensures that we have contiguous memory.
    PyObject *a_array =
        PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *b_array =
        PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);

    // If parsing of a or b fails we throw an exception in Python
    if (a_array == NULL || b_array == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "Could not convert one of the arguments to a numpy array.");
        return NULL;
    }

    // Dimensions should agree
    if (PyArray_NDIM((PyArrayObject *)a_array) !=
        PyArray_NDIM((PyArrayObject *)b_array)) {
        PyErr_SetString(PyExc_ValueError, "Number of dimensions need to match");
        return NULL;
    }

    auto ndim = PyArray_NDIM(reinterpret_cast<PyArrayObject *>(a_array));
    auto a_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(a_array));
    auto b_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(b_array));

    for (int i = 0; i != ndim; ++i) {
        if (a_shape[i] != b_shape[i]) {
            PyErr_SetString(PyExc_ValueError,
                            "The shape of the arguments need to match.");
            return NULL;
        }
    }

    // Create array for return values
    PyObject *result_array = PyArray_SimpleNew(ndim, a_shape, NPY_DOUBLE);

    // For the C++ function call we need pointers (or another C++ type/data
    // structure)
    double *a = reinterpret_cast<double *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(a_array)));
    double *b = reinterpret_cast<double *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(b_array)));
    double *result = reinterpret_cast<double *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(result_array)));

    // Now call add wih pointers and size
    auto size = PyArray_Size(a_array);
    cpp_add(a, b, result, size);

    // Clean up
    Py_DECREF(a_array);
    Py_DECREF(b_array);

    return result_array;
}
