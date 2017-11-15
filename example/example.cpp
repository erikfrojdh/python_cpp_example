#define  NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include "function.h"


/* Docstrings */
static char module_docstring[] =
    "example: add two doubles";


PyDoc_STRVAR(
    add_doc,
    "Add two numbers.\n\n"
    "Parameters\n"
    "----------\n"
    "a: numpy_array\n"
    "   single number\n"
    "b: numpy_array\n"
    "   single number\n"
    
    "Returns\n"
    "----------\n"
    "result: numpy_array\n"
    "   added number\n\n"
        );


/* Available functions */
static PyObject *add(PyObject *self, PyObject *args);


/* Module specification */
static PyMethodDef module_methods[] = {
    {"add", (PyCFunction)add, METH_VARARGS, add_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef example_def = {
    PyModuleDef_HEAD_INIT,
    "_example",
    module_docstring,
    -1,
    module_methods
};

/* Initialize the module */
PyMODINIT_FUNC
PyInit__example(void)
{
    PyObject *m = PyModule_Create(&example_def);
    if (m == NULL){
        return NULL;
        }

    /* Load `numpy` functionality. */
    import_array();
    return m;
}

static PyObject *add(PyObject *self, PyObject *args){
    
    //PyObject to be extracted from *args
    PyObject *a_obj;
    PyObject *b_obj;

    
    //Check and parse..
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)){
        return NULL;
    }    

    //Numpy array from the parsed objects 
    PyObject *a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);


    //Exception if it fails
    if (a_array == NULL || b_array == NULL ){
        std::cout << "Something went wrong, possibly crappy arguments?" << std::endl; 
        return NULL;
    }
    


    /* Get a pointer to the data as C-types. */
    double *a = (double*)PyArray_DATA((PyArrayObject*)a_array);
    double *b = (double*)PyArray_DATA((PyArrayObject*)b_array);

    
    
    /* Create a numpy array to return to Python */
    int ndim = 1;
    npy_intp dims[1] = { 1 };
    PyObject *result_array = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    

    /* Get a pointer to the data as C-types. */
    double *result = (double*)PyArray_DATA((PyArrayObject*)result_array);

    //Fit the data
    result[0] = add(a[0], b[0]);


    //Clean up
    Py_DECREF(a_array);
    Py_DECREF(b_array);

    return result_array;
}


