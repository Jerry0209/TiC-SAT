#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/TraceCPU.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "cpu/trace/trace_cpu.hh"

#include <string>
#include <string>
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_TraceCPU");
    py::class_<TraceCPUParams, BaseCPUParams, std::unique_ptr<TraceCPUParams, py::nodelete>>(m, "TraceCPUParams")
        .def(py::init<>())
        .def("create", &TraceCPUParams::create)
        .def_readwrite("dataTraceFile", &TraceCPUParams::dataTraceFile)
        .def_readwrite("enableEarlyExit", &TraceCPUParams::enableEarlyExit)
        .def_readwrite("freqMultiplier", &TraceCPUParams::freqMultiplier)
        .def_readwrite("instTraceFile", &TraceCPUParams::instTraceFile)
        .def_readwrite("progressMsgInterval", &TraceCPUParams::progressMsgInterval)
        .def_readwrite("sizeLoadBuffer", &TraceCPUParams::sizeLoadBuffer)
        .def_readwrite("sizeROB", &TraceCPUParams::sizeROB)
        .def_readwrite("sizeStoreBuffer", &TraceCPUParams::sizeStoreBuffer)
        ;

    py::class_<TraceCPU, BaseCPU, std::unique_ptr<TraceCPU, py::nodelete>>(m, "TraceCPU")
        ;

}

static EmbeddedPyBind embed_obj("TraceCPU", module_init, "BaseCPU");
