#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/MemTraceProbe.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "mem/probes/mem_trace.hh"

#include "sim/system.hh"
#include <string>
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_MemTraceProbe");
    py::class_<MemTraceProbeParams, BaseMemProbeParams, std::unique_ptr<MemTraceProbeParams, py::nodelete>>(m, "MemTraceProbeParams")
        .def(py::init<>())
        .def("create", &MemTraceProbeParams::create)
        .def_readwrite("system", &MemTraceProbeParams::system)
        .def_readwrite("trace_compress", &MemTraceProbeParams::trace_compress)
        .def_readwrite("trace_file", &MemTraceProbeParams::trace_file)
        .def_readwrite("with_pc", &MemTraceProbeParams::with_pc)
        ;

    py::class_<MemTraceProbe, BaseMemProbe, std::unique_ptr<MemTraceProbe, py::nodelete>>(m, "MemTraceProbe")
        ;

}

static EmbeddedPyBind embed_obj("MemTraceProbe", module_init, "BaseMemProbe");
