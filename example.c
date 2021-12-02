#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"

static PyObject *mat_vec_1d(PyObject *self, PyObject *args) {

    // Inputs
    PyObject *indices; // 1d np.array
    PyObject *other;   // 1d np.array
    PyObject *result;  // 1d np.array

    // Outputs (np.array)
    PyObject *output_array = NULL; // 1d np.array

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &indices, &PyArray_Type, &other, &PyArray_Type, &result)) {
        return NULL;
    }

    PyArrayObject *indices_arr;
    PyArrayObject *other_arr;

    indices_arr = (PyArrayObject *) PyArray_FROM_OTF(indices, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (indices_arr == NULL) {
        Py_XDECREF(indices_arr);
        return NULL;
    }

    other_arr = (PyArrayObject *) PyArray_FROM_OTF(other, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (other_arr == NULL) {
        Py_XDECREF(other_arr);
        return NULL;
    }

    PyObject *result_arr = NULL;

#if NPY_API_VERSION >= 0x0000000c
    result_arr = PyArray_FROM_OTF(result, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    result_arr = PyArray_FROM_OTF(result, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif

    int size = (int) PyArray_SIZE(indices_arr);

    int *indices_data = (int *) PyArray_DATA(indices_arr);
    double *other_data = (double *) PyArray_DATA(other_arr);
    double *result_data = (double *) PyArray_DATA(result_arr);

    // This is where the _multiplication_ happens
    for (int i = 0; i < size; i++) {
        result_data[i] = other_data[indices_data[i]];
    }

    Py_DECREF(indices_arr);
    Py_DECREF(other_arr);
    return result;

fail:
    Py_DECREF(indices_arr);
    Py_DECREF(other_arr);
    Py_XDECREF(result);
    return NULL;
}


static PyObject *mat_vec_1d_dummy(PyObject *self, PyObject *args) {

    // Inputs
    PyObject *matrix; // 1d np.array
    PyObject *vector;   // 1d np.array
    PyObject *result;  // 1d np.array

    // Outputs (np.array)
    PyObject *output_array = NULL; // 1d np.array

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &matrix, &PyArray_Type, &vector, &PyArray_Type, &result)) {
        return NULL;
    }

    PyArrayObject *matrix_arr;
    PyArrayObject *vector_arr;

    matrix_arr = PyArray_FROM_OTF(matrix, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (matrix_arr == NULL) {
        Py_XDECREF(matrix_arr);
        return NULL;
    }

    vector_arr = PyArray_FROM_OTF(vector, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (vector_arr == NULL) {
        Py_XDECREF(vector_arr);
        return NULL;
    }

    PyObject *result_arr = NULL;

#if NPY_API_VERSION >= 0x0000000c
    result_arr = PyArray_FROM_OTF(result, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    result_arr = PyArray_FROM_OTF(result, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif

    int size = PyArray_SIZE(matrix_arr);

    int *matrix_data = PyArray_DATA(matrix_arr);
    double *vector_data = PyArray_DATA(vector_arr);
    double *result_data = PyArray_DATA(result_arr);

    // This is where the _multiplication_ happens
    for (int i = 0; i < size; i += 2) {
        result_data[matrix_data[i]] += vector_data[matrix_data[i + 1]];
    }
    /*
    for (int i=0; i<n; ++i) {
        y[i] = 0.0;
        for (int j = offset_data[i]; j < offset_data[i + 1]; ++j) {
            result_data[i] += vector_data[j] * x[col[j]];
        }
    }
    */

    Py_DECREF(matrix_arr);
    Py_DECREF(vector_arr);
    return result;
fail:
    Py_DECREF(matrix_arr);
    Py_DECREF(vector_arr);
    Py_XDECREF(result);
    return NULL;
}

// This is an example I used to get familiar with the API. Nothing important.
static PyObject * example_wrapper(PyObject *self, PyObject *args) {
    PyObject *arg1=NULL, *arg2=NULL, *out=NULL;
    PyObject *arr1=NULL, *arr2=NULL, *oarr=NULL;

    if (!PyArg_ParseTuple(args, "OOO!", &arg1, &arg2, &PyArray_Type, &out)) return NULL;

    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr1 == NULL) return NULL;

    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr2 == NULL) goto fail;

#if NPY_API_VERSION >= 0x0000000c
    oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif

    if (oarr == NULL) goto fail;

    npy_intp size;
    double *dptr;  /* could make this any variable type */

    size = PyArray_SIZE(arr1);
    dptr = PyArray_DATA(arr1);


    while (size--) {
        dptr++;
    }

    /* code that makes use of arguments */
    /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.
       If an error occurs goto fail.
     */

    Py_DECREF(arr1);
    Py_DECREF(arr2);

#if NPY_API_VERSION >= 0x0000000c
    PyArray_ResolveWritebackIfCopy(oarr);
#endif
    Py_DECREF(oarr);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
#if NPY_API_VERSION >= 0x0000000c
    PyArray_DiscardWritebackIfCopy(oarr);
#endif
    Py_XDECREF(oarr);
    return NULL;
}


// Method definition
static PyMethodDef ModuleMethods[] = {
    {"example", example_wrapper, METH_VARARGS, "This is my example wrapper"},
    {"mat_vec_1d", mat_vec_1d, METH_VARARGS, "This is mat dot vec 1d"},
    {"mat_vec_1d_dummy", mat_vec_1d_dummy, METH_VARARGS, "bla bla bla"},
    {NULL, NULL, 0, NULL}
};

// Inicializacion del modulo
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "example",
    NULL,
    -1,
    ModuleMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_example(void) {
    PyObject *m;
    import_array();
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}