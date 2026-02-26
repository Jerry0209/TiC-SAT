#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmNativeTrace.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/nativetrace.hh"

namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmNativeTrace");
    py::class_<ArmNativeTraceParams, NativeTraceParams, std::unique_ptr<ArmNativeTraceParams, py::nodelete>>(m, "ArmNativeTraceParams")
        .def(py::init<>())
        .def("create", &ArmNativeTraceParams::create)
        .def_readwrite("stop_on_pc_error", &ArmNativeTraceParams::stop_on_pc_error)
        ;

    py::class_<Trace::ArmNativeTrace, Trace::NativeTrace, std::unique_ptr<Trace::ArmNativeTrace, py::nodelete>>(m, "Trace_COLONS_ArmNativeTrace")
        ;

}

static EmbeddedPyBind embed_obj("ArmNativeTrace", module_init, "NativeTrace");
