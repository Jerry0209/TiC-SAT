#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/InstPBTrace.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "cpu/inst_pb_trace.hh"

#include <string>
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_InstPBTrace");
    py::class_<InstPBTraceParams, InstTracerParams, std::unique_ptr<InstPBTraceParams, py::nodelete>>(m, "InstPBTraceParams")
        .def(py::init<>())
        .def("create", &InstPBTraceParams::create)
        .def_readwrite("file_name", &InstPBTraceParams::file_name)
        ;

    py::class_<Trace::InstPBTrace, Trace::InstTracer, std::unique_ptr<Trace::InstPBTrace, py::nodelete>>(m, "Trace_COLONS_InstPBTrace")
        ;

}

static EmbeddedPyBind embed_obj("InstPBTrace", module_init, "InstTracer");
