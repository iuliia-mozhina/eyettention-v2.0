// python module - interface for python

#include <Python.h>
#include <numpy/arrayobject.h>

#include "swiftstat7_api.c"


#define SWIFTPY_MAJOR 0
#define SWIFTPY_MINOR 2

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

static int model_counter = 0, m_array_size=10;
static swift_model **m;

static int data_counter = 0, ds_array_size = 10;
static swift_dataset **ds;


#ifdef SWIFT_MPI
#include "swiftstat7_mpi.c"

static PyObject*
swift_mpi_load(PyObject *self, PyObject *args, PyObject *keywds)
{

    static char *kwlist[] = {"outputPath", "parmPath", "corpusFile", "fixseqName", "seed", "neval", "verbose", NULL};

    char *outputPath, *parmPath, *corpusFile, *fixseqName;
    int verbose = 0, how_many_evals = 0;
    long seed;
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ssssl|Ib", kwlist, &outputPath, &parmPath, &corpusFile, &fixseqName, &seed, &how_many_evals, &verbose))
        return NULL;

    //unlock(&write_lock);
    swift_load_mpi(outputPath, parmPath, corpusFile, fixseqName, seed, how_many_evals);

    Py_RETURN_NONE;
}


static PyObject*
swift_mpi_update(PyObject *self, PyObject *args, PyObject *keywds)
{

    static char *kwlist[] = {"param", "value", NULL};

    int param_id;
    char *param_name;
    PyObject *param_value;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "sO", kwlist, &param_name, &param_value))
        return NULL;

    if(param_name != NULL) {
        param_id = swift_find_param(param_name);
        if(param_id != -1) {
            if(swift_parameters_meta[param_id].type == PARTYPE_INTEGER) {
                if(!PyInt_Check(param_value)) {
                    PyErr_Format(PyExc_ValueError, "The value for “%s” must be an integer number!", param_name);
                    return NULL;
                }
                swift_update_parameter_mpi(param_id, (int) PyInt_AsLong(param_value));
            } else if(swift_parameters_meta[param_id].type == PARTYPE_DOUBLE) {
                if(!PyFloat_Check(param_value)) {
                    PyErr_Format(PyExc_ValueError, "The value for “%s” must be a floating point number (e.g., float or double)!", param_name);
                    return NULL;
                }
                swift_update_parameter_mpi(param_id, (double) PyFloat_AsDouble(param_value));
            } else {
                PyErr_Format(PyExc_ValueError, "The value for “%s” cannot be set because that parameter type cannot be handled by this module!", param_name);
                return NULL;
            }
        }else{
            PyErr_Format(PyExc_ValueError, "There is no parameter '%s'!", param_name);
            return NULL;
        }
    }

    Py_RETURN_NONE;
}

static PyObject*
swift_mpi_loglik(PyObject *self, PyObject *args)
{
    int i, threads = 0;

    double loglik[N_LOGLIKS];

    swift_eval_mpi(loglik);

    PyObject *list = PyList_New((Py_ssize_t) N_LOGLIKS);
    for(i=0;i<N_LOGLIKS;i++) {
        PyList_SetItem(list, (Py_ssize_t) i, PyFloat_FromDouble(loglik[i]));
    }

    return Py_BuildValue("N", list);
}

static PyObject*
swift_mpi_loglik_lb(PyObject *self, PyObject *args)
{
    int i, threads = 0;

    double loglik[N_LOGLIKS];

    swift_eval_mpi_lb(loglik);

    PyObject *list = PyList_New((Py_ssize_t) N_LOGLIKS);
    for(i=0;i<N_LOGLIKS;i++) {
        PyList_SetItem(list, (Py_ssize_t) i, PyFloat_FromDouble(loglik[i]));
    }

    return Py_BuildValue("N", list);
}

static PyObject*
swift_mpi_finalize(PyObject *self, PyObject *args)
{
    swift_finalize_mpi();

    Py_RETURN_NONE;
}

#endif

static PyObject*
swift_loadmodel(PyObject *self, PyObject *args, PyObject *keywds)
{

    static char *kwlist[] = {"outputPath", "parmPath", "corpusFile", "seed", "verbose", NULL};

    char *outputPath, *parmPath, *corpusFile;
    int verbose = 0;
    long seed;
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "sssl|b", kwlist, &outputPath, &parmPath, &corpusFile, &seed, &verbose))
        return NULL;

    if(model_counter == m_array_size) {
        m_array_size+=10;
        m = (swift_model**) realloc(m, m_array_size * sizeof(swift_model*));
    }

    printf("Load model from output <%s>, parm <%s>, corpus <%s>.\n", outputPath, parmPath, corpusFile);

    //unlock(&write_lock);
    if(!swift_load_model(outputPath, parmPath, corpusFile, seed, &m[model_counter], verbose)) {
        PyErr_Format(PyExc_RuntimeError, "Failed to load model. For details see console.");
        return NULL;
    }

    return Py_BuildValue("i", model_counter++);
}


static unsigned short swift_model_exists(int model_id) {
    if(model_id<0||model_id>=model_counter) {
        PyErr_Format(PyExc_ValueError, "There is no model loaded under %d.", model_id);
        return 0;
    }else if(m[model_id] == NULL) {
        PyErr_Format(PyExc_ValueError, "Model %d has already been unloaded (freed).", model_id);
        return 0;
    }
    return 1;
}

static unsigned short swift_dataset_exists(int data_id) {
    if(data_id<0||data_id>=data_counter) {
        PyErr_Format(PyExc_ValueError, "There is no dataset loaded under <%d>.", data_id);
        return 0;
    }else if(ds[data_id] == NULL) {
        PyErr_Format(PyExc_ValueError, "Dataset <%d> has already been unloaded (freed).", data_id);
        return 0;
    }
    return 1;
}

static PyObject*
swift_getmodel(PyObject *self, PyObject *args, PyObject *keywds)
{

    static char *kwlist[] = {"model", "field", "fields", NULL};

    int model_id = 0;
    PyObject *fields = Py_None;
    char *single_field = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|IsO", kwlist, &model_id, &single_field, &fields))
        return NULL;

    if(fields != Py_None && single_field != NULL) {
        PyErr_SetString(PyExc_ValueError, "Please set only either field (single-field query) or fields (multi-field query passed as a list of field names)!");
        return NULL;
    }

    if(fields != Py_None && !PyList_Check(fields)) {
        PyErr_SetString(PyExc_ValueError, "Argument fields must be a list of fields to return or None to return all fields!");
        return NULL;
    }

    if(!swift_model_exists(model_id))
        return NULL;

    swift_model *mx = m[model_id];

    PyObject *ret = PyDict_New();

    int j, i;

    for(j=0;(fields==Py_None&&j==0)||(single_field!=NULL&&j==0)||(fields!=Py_None&&(Py_ssize_t)j<PyList_Size(fields));j++) {
        char *field_name = NULL;
        if(single_field!=NULL) {
            field_name = single_field;
        }else if(fields!=Py_None) {
            PyObject *field = PyList_GetItem(fields, (Py_ssize_t) j);
            if(!PyString_Check(field)) {
                Py_XDECREF(ret);
                PyErr_Format(PyExc_ValueError, "Field at %d is not a string!", j);
                return NULL;
            }
            field_name = PyString_AsString(field);
        }
        unsigned short field_found = 0;
        // if(field_name == NULL || !strcmp(field_name, "seed")) {
        //     PyDict_SetItemString(ret, "seed", PyInt_FromLong(mx->seed));
        //     field_found = 1;
        // }
        for(i=0;swift_parameters_meta[i].name!=NULL;i++) {
            if(field_name == NULL || !strcmp(field_name, swift_parameters_meta[i].name)) {
                field_found = 1;
                PyObject *pval = NULL;
                if(swift_parameters_meta[i].type == PARTYPE_INTEGER) {
                    pval = PyInt_FromLong(valbyid(mx->params, i, swift_parameter_int));
                } else if(swift_parameters_meta[i].type == PARTYPE_DOUBLE) {
                    pval = PyFloat_FromDouble(valbyid(mx->params, i, swift_parameter_dbl));
                }
                if(pval != NULL) {
                    PyDict_SetItemString(ret, swift_parameters_meta[i].name, pval);
                } else {
                    Py_XDECREF(ret);
                    PyErr_Format(PyExc_ValueError, "Don't know how to handle parameter type for %s!", swift_parameters_meta[i].name);
                    return NULL;
                }
            }
        }
        if(!field_found) {
            Py_XDECREF(ret);
            PyErr_Format(PyExc_ValueError, "Unknown field %s!", field_name);
            return NULL;
        }
    }

    if(single_field!=NULL) {
        PyObject *item = PyDict_GetItemString(ret, single_field);
        Py_XINCREF(item);
        Py_XDECREF(ret);
        return item;
    }

    return ret;
}

static PyObject*
swift_updatemodel(PyObject *self, PyObject *args, PyObject *keywds)
{

    static char *kwlist[] = {"param", "value", "model", NULL};

    int model_id = 0, param_id;
    char *param_name;
    PyObject *param_value;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "sO|I", kwlist, &param_name, &param_value, &model_id))
        return NULL;

    if(!swift_model_exists(model_id))
        return NULL;

    if(param_name != NULL) {
        param_id = swift_find_param(param_name);
        if(param_id != -1) {
            if(swift_parameters_meta[param_id].type == PARTYPE_INTEGER) {
                if(!PyInt_Check(param_value)) {
                    PyErr_Format(PyExc_ValueError, "The value for “%s” must be an integer number!", param_name);
                    return NULL;
                }
                setvalbyid(m[model_id]->params, param_id, swift_parameter_int, (int) PyInt_AsLong(param_value));
            } else if(swift_parameters_meta[param_id].type == PARTYPE_DOUBLE) {
                if(!PyFloat_Check(param_value)) {
                    PyErr_Format(PyExc_ValueError, "The value for “%s” must be a floating point number (e.g., float or double)!", param_name);
                    return NULL;
                }
                setvalbyid(m[model_id]->params, param_id, swift_parameter_dbl, PyFloat_AsDouble(param_value));
            } else {
                PyErr_Format(PyExc_ValueError, "The value for “%s” cannot be set because that parameter type cannot be handled by this module!", param_name);
                return NULL;
            }
        }else{
            PyErr_Format(PyExc_ValueError, "There is no parameter '%s'!", param_name);
            return NULL;
        }
    }

    Py_RETURN_NONE;
}

static PyObject*
swift_loaddata(PyObject *self, PyObject *args, PyObject *keywds)
{
    static char *kwlist[] = {"fixseqName", "verbose", NULL};
    int verbose = 0;

    char *fixseqName;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|b", kwlist,
                                     &fixseqName, &verbose))
        return NULL;

    printf("Load fixseqs from <%s> into <%d>.\n", fixseqName, data_counter);

    //lock(&write_lock);

    if(data_counter == ds_array_size) {
        ds_array_size+=10;
        ds = (swift_dataset**) realloc(ds, ds_array_size * sizeof(swift_dataset*));
    }

    if(!swift_load_data(fixseqName, &ds[data_counter], verbose)) {
        PyErr_SetString(PyExc_RuntimeError, "Loading data failed. See console for details.");
        return NULL;
    }

    //printf("Nfix=%d, Ntrials=%d\n",ds[data_counter]->subsets[0]->Nfix,ds[data_counter]->subsets[0]->Ntrials);
    
    //unlock(&write_lock);

    return Py_BuildValue("i", data_counter++);

}

static PyObject*
swift_loglik(PyObject *self, PyObject *args, PyObject *keywds)
{
    static char *kwlist[] = {"model", "data", "threads", "validate", NULL};
    int data_id = 0, model_id = 0, i, threads = 0, validate = 1;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|IIIb", kwlist,
                                     &model_id, &data_id, &threads, &validate))
        return NULL;

    double loglik[N_LOGLIKS];

    if(!swift_dataset_exists(data_id))
        return NULL;

    if(!swift_model_exists(model_id))
        return NULL;

    if(validate && !swift_validate(ds[data_id], m[model_id])) {
        PyErr_SetString(PyExc_RuntimeError, "This dataset could not be validated. Please check the dataset or turn of validation.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    swift_eval_all(m[model_id], ds[data_id], loglik, threads, 0);
    Py_END_ALLOW_THREADS

    PyObject *list = PyList_New((Py_ssize_t) N_LOGLIKS);
    for(i=0;i<N_LOGLIKS;i++) {
        PyList_SetItem(list, (Py_ssize_t) i, PyFloat_FromDouble(loglik[i]));
    }

    return Py_BuildValue("N", list);

}


static PyObject*
swift_validatedata(PyObject *self, PyObject *args, PyObject *keywds)
{
    static char *kwlist[] = {"model", "data", NULL};
    int data_id = 0, model_id = 0;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|II", kwlist,
                                     &model_id, &data_id))
        return NULL;

    if(!swift_dataset_exists(data_id))
        return NULL;

    if(!swift_model_exists(model_id))
        return NULL;

    

    return Py_BuildValue("b", swift_validate(ds[data_id], m[model_id]));

}


static PyObject*
swift_freemodel(PyObject *self, PyObject *args, PyObject *keywds)
{

    static char *kwlist[] = {"model", NULL};

    int model_id = 0;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|I", kwlist, &model_id))
        return NULL;

    if(!swift_model_exists(model_id))
        return NULL;

    //lock(&write_lock);
    free_swift_model(m[model_id]);
    m[model_id] = NULL;
    //unlock(&write_lock);
    Py_RETURN_NONE;
}

static PyObject*
swift_nlogliks(PyObject *self, PyObject *args)
{
    return Py_BuildValue("I", N_LOGLIKS);
}

static PyObject*
swift_freedata(PyObject *self, PyObject *args, PyObject *keywds)
{

    static char *kwlist[] = {"data", NULL};

    int data_id=0;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|I", kwlist, &data_id))
        return NULL;

    if(!swift_dataset_exists(data_id))
        return NULL;

    //lock(&write_lock);
    free_swift_dataset(ds[data_id]);
    //unlock(&write_lock);
    ds[data_id] = NULL;

    Py_RETURN_NONE;
}

static PyObject*
swift_generatedata(PyObject *self, PyObject *args, PyObject *keywds)
{
    static char *kwlist[] = {"path", "model", "dir", "snr", NULL};

    int model_id=0;
    char *path, *directory = "../SIM";
    unsigned int snr = 0;
    unsigned short fixseq = 1;
    unsigned int threads = 0;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|IsIbI", kwlist, &path, &model_id, &directory, &snr, &fixseq, &threads))
        return NULL;

    if(!swift_model_exists(model_id))
        return NULL;

    if(snr < 0 || snr > nsentences(m[model_id]->corpus)) {
        char msg[300];
        sprintf(msg, "Corpus contains %d items. Item no. must be between 1 and %d (%u given)", nsentences(m[model_id]->corpus), nsentences(m[model_id]->corpus), snr);
        PyErr_SetString(PyExc_ValueError, msg);
        return NULL;
    }

    if(snr == 0)
        swift_generate_all(m[model_id], directory, path, threads, fixseq, 0);
    else
        swift_generate_single(m[model_id], directory, path, snr, 0);
    //unlock(&read_lock);

    Py_RETURN_NONE;
}

static PyObject*
swift_cite(PyObject *self, PyObject *args, PyObject *keywds)
{
    static char *kwlist[] = {"style", "format", NULL};

    char *style = NULL, *format = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|ss", kwlist, &style, &format))
        return NULL;

    if(format == NULL)
        format = "plain";

    if(strcmp("plain", format) && strcmp("markdown", format)) {
        PyErr_Format(PyExc_ValueError, "Invalid format '%s'. Supported formats are 'plain' and 'markdown'.", format);
        return NULL;
    }

    int i;
    if(style == NULL) {
        // if style is not set, get all of the styles as a dict
        PyObject *ret = PyDict_New();
        for(i=0;swift_citations[i].style!=NULL;i++) {
            char *val = NULL;
            if(!strcmp(format, "plain"))
                val = swift_citations[i].plain;
            else if(!strcmp(format, "markdown"))
                val = swift_citations[i].markdown;
            if(val != NULL)
                PyDict_SetItemString(ret, swift_citations[i].style, PyString_FromString(val));
        }
        return ret;
    }else{
        swift_citation *citation_style = swift_find_citation_style(style);
        if(citation_style != NULL) {
            char *ret = NULL;
            if(!strcmp(format, "plain"))
                ret = citation_style->plain;
            else if(!strcmp(format, "markdown"))
                ret = citation_style->markdown;
            if(ret == NULL) {
                PyErr_Format(PyExc_ValueError, "The citation style '%s' is not available in the requested format '%s'.", style, format);
                return NULL;
            }else{
                return Py_BuildValue("s", ret);
            }
        }else{
            char tmp[400] = {0};
            for(i=0;swift_citations[i].style!=NULL;i++) {
                if(i==0)
                    sprintf(tmp, "%s", swift_citations[i].style);
                else if(swift_citations[i+1].style!=NULL)
                    sprintf(&tmp[strlen(tmp)], ", %s", swift_citations[i].style);
                else
                    sprintf(&tmp[strlen(tmp)], ", and %s", swift_citations[i].style);
            }
            PyErr_Format(PyExc_ValueError, "The citation style '%s' is not available. Supported styles are %s.", style, tmp);
            return NULL;
        }
    }



}


static version module_version;

static PyObject*
swift_getversion(PyObject *self, PyObject *args)
{
    char variant_string[512];
    swift_variant_string(variant_string);
    return Py_BuildValue("{s:(IIs),s:(II),s:(II)}", "swift", swift_version.major, swift_version.minor, variant_string, "api", swift_api_version.major, swift_api_version.minor, "module", module_version.major, module_version.minor);
}

static PyObject*
swift_cores(PyObject *self, PyObject *args)
{
    #ifdef DISABLE_THREADS
    return Py_BuildValue("I", 1);
    #else
    return Py_BuildValue("I", omp_get_max_threads());
    #endif
}



static PyMethodDef SwiftMethods[] = {
#ifdef SWIFT_MPI

    {"mpiload", (PyCFunction)swift_mpi_load, METH_VARARGS | METH_KEYWORDS, "Load model configuration, store, and return internal model ID."},
    {"mpiupdate", (PyCFunction)swift_mpi_update, METH_VARARGS | METH_KEYWORDS, "Update a model parameter."},
    {"mpieval", (PyCFunction)swift_mpi_loglik, METH_VARARGS, "Evaluate likelihood of a dataset given a model."},
    {"mpievallb", (PyCFunction)swift_mpi_loglik_lb, METH_VARARGS, "Evaluate likelihood of a dataset given a model. This is load-balanced."},
    {"mpifinalize", (PyCFunction)swift_mpi_finalize, METH_VARARGS, "Finalize MPI cluster"},

#endif

    {"loadmodel", (PyCFunction)swift_loadmodel, METH_VARARGS | METH_KEYWORDS, "Load model configuration, store, and return internal model ID."},
    {"getmodel", (PyCFunction)swift_getmodel, METH_VARARGS | METH_KEYWORDS, "Return model specification."},
    {"updatemodel", (PyCFunction)swift_updatemodel, METH_VARARGS | METH_KEYWORDS, "Update a model parameter."},
    {"freemodel", (PyCFunction)swift_freemodel, METH_VARARGS | METH_KEYWORDS, "Free model. This clears the memory allocated for the model. Future access to this model using its internal ID will result in a segfault and abort the application!"},
    {"generate", (PyCFunction)swift_generatedata, METH_VARARGS | METH_KEYWORDS, "Generate data using the model."},
    {"loaddata", (PyCFunction)swift_loaddata, METH_VARARGS | METH_KEYWORDS, "Load empirical data (subsets), store, and return internal data ID."},
    {"nlogliks", (PyCFunction)swift_nlogliks, METH_VARARGS, "Get number of likelihoods. This is the length of the array returned by eval()."},
    {"eval", (PyCFunction)swift_loglik, METH_VARARGS | METH_KEYWORDS, "Evaluate likelihood of a dataset given a model."},
    {"version", (PyCFunction)swift_getversion, METH_VARARGS, "Get version information about the module."},
    {"freedata", (PyCFunction)swift_freedata, METH_VARARGS | METH_KEYWORDS, "Free data. This clears the memory allocated for the data. Future access to these data using their internal ID will result in a segfault and abort the application!"},
    {"citation", (PyCFunction)swift_cite, METH_VARARGS | METH_KEYWORDS , "Return citations for SWIFT in all available formats."},
    {"cores", (PyCFunction)swift_cores, METH_VARARGS , "Get available compute cores."},
    {"validate", (PyCFunction)swift_validatedata, METH_VARARGS | METH_KEYWORDS , "Validate a loaded dataset against a loaded corpus (model)."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initswiftstat7(void)
{
    module_version.major = SWIFTPY_MAJOR;
    module_version.minor = SWIFTPY_MINOR;

    #ifdef SWIFT_MPI
    swift_init_mpi();
    #endif

    char module_version_string[100], version_string[1024];
    #ifdef SWIFT_MPI
    const char addinfo[] = " MPI";
    #else
    const char addinfo[] = "";
    #endif
    sprintf(module_version_string, "Python module v%d.%d%s", module_version.major, module_version.minor, addinfo);
    swift_complete_version_string(version_string, module_version_string);

    fprintf(stderr, "%s loaded\n", version_string);

    #ifdef DISABLE_THREADS
    fprintf(stderr, "Note: No multithreading! Parallelized methods (eval, generate) will not employ parallel computing. To enable multithreading, please recompile the module with OpenMP (e.g., -fopenmp).\n");
    #endif

    (void) Py_InitModule("swiftstat7", SwiftMethods);
    import_array();
    m = (swift_model**) calloc(m_array_size, sizeof(swift_model*));
    ds = (swift_dataset**) calloc(ds_array_size, sizeof(swift_dataset*));
    
}

