#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"

static PyObject *binary_matrix_vector(PyObject *self, PyObject *args) {
    // This is the function that works for the general case where
    // you can have any number of ones per row.

    // Inputs
    PyObject *matrix;  // 2d np.array of shape (length, 2)
    PyObject *vector;  // 1d np.array
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

    Py_DECREF(matrix_arr);
    Py_DECREF(vector_arr);
    return result;
fail:
    Py_DECREF(matrix_arr);
    Py_DECREF(vector_arr);
    Py_XDECREF(result);
    return NULL;
}



static PyObject *matrix_vector(PyObject *self, PyObject *args) {
    // This is the function that works for the general case where
    // you can have any number of ones per row.

    // Inputs
    PyObject *offset;  // 1d np.array
    PyObject *column;  // 1d np.array
    PyObject *value;   // 1d np.array
    PyObject *vector;  // 1d np.array
    PyObject *result;  // 1d np.array

    // Outputs (np.array)
    PyObject *output_array = NULL; // 1d np.array

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyArray_Type, &offset, &PyArray_Type, &column, &PyArray_Type, &value, &PyArray_Type, &vector, &PyArray_Type, &result)) {
        return NULL;
    }

    PyArrayObject *offset_arr;
    PyArrayObject *column_arr;
    PyArrayObject *value_arr;
    PyArrayObject *vector_arr;

    offset_arr = PyArray_FROM_OTF(offset, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (offset_arr == NULL) {
        Py_XDECREF(offset_arr);
        return NULL;
    }

    column_arr = PyArray_FROM_OTF(column, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (column_arr == NULL) {
        Py_XDECREF(column_arr);
        return NULL;
    }

    value_arr = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (value_arr == NULL) {
        Py_XDECREF(value_arr);
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

    int n = PyArray_SIZE(result_arr);

    int *offset_data = PyArray_DATA(offset_arr);
    int *column_data = PyArray_DATA(column_arr);
    double *value_data = PyArray_DATA(value_arr);
    double *vector_data = PyArray_DATA(vector_arr);
    double *result_data = PyArray_DATA(result_arr);

    // This is where the _multiplication_ happens
    for (int i = 0; i < n; i++) {
        for (int j = offset_data[i]; j < offset_data[i + 1]; ++j) {
            result_data[i] += value_data[j] * vector_data[column_data[j]];
        }
    }

    Py_DECREF(offset_arr);
    Py_DECREF(column_arr);
    Py_DECREF(value_arr);
    Py_DECREF(vector_arr);
    return result;
fail:
    Py_DECREF(offset_arr);
    Py_DECREF(column_arr);
    Py_DECREF(value_arr);
    Py_DECREF(vector_arr);
    Py_XDECREF(result);
    return NULL;
}

// Not exported for now...
static PyObject *dot_matrix_vector_special(PyObject *self, PyObject *args) {
    // This function only works for the case where you have exactly one 1 per row.

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

    indices_arr = PyArray_FROM_OTF(indices, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (indices_arr == NULL) {
        Py_XDECREF(indices_arr);
        return NULL;
    }

    other_arr = PyArray_FROM_OTF(other, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
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

    int size = PyArray_SIZE(indices_arr);

    int *indices_data = PyArray_DATA(indices_arr);
    double *other_data = PyArray_DATA(other_arr);
    double *result_data = PyArray_DATA(result_arr);

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


// Method definition
static PyMethodDef ModuleMethods[] = {
    {"binary_matrix_vector", binary_matrix_vector, METH_VARARGS, "Dot product between a ZeroOneMatrix and a vector."},
    {"matrix_vector", matrix_vector, METH_VARARGS, "Dot product between a SparseMatrix and a vector."},
    {NULL, NULL, 0, NULL}
};

// Inicializacion del modulo
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "sparsedot",
    NULL,
    -1,
    ModuleMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_sparsedot(void) {
    PyObject *m;
    import_array();
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}