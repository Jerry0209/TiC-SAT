#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ElasticTrace.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "cpu/o3/probe/elastic_trace.hh"

#include <string>
#include "base/types.hh"
#include <string>
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ElasticTrace");
    py::class_<ElasticTraceParams, ProbeListenerObjectParams, std::unique_ptr<ElasticTraceParams, py::nodelete>>(m, "ElasticTraceParams")
        .def(py::init<>())
        .def("create", &ElasticTraceParams::create)
        .def_readwrite("dataDepTraceFile", &ElasticTraceParams::dataDepTraceFile)
        .def_readwrite("depWindowSize", &ElasticTraceParams::depWindowSize)
        .def_readwrite("instFetchTraceFile", &ElasticTraceParams::instFetchTraceFile)
        .def_readwrite("startTraceInst", &ElasticTraceParams::startTraceInst)
        .def_readwrite("traceVirtAddr", &ElasticTraceParams::traceVirtAddr)
        ;

    py::class_<ElasticTrace, ProbeListenerObject, std::unique_ptr<ElasticTrace, py::nodelete>>(m, "ElasticTrace")
        ;

}

static EmbeddedPyBind embed_obj("ElasticTrace", module_init, "ProbeListenerObject");
