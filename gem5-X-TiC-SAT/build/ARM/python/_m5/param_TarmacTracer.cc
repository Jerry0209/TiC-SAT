#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/TarmacTracer.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/tracers/tarmac_tracer.hh"

#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_TarmacTracer");
    py::class_<TarmacTracerParams, InstTracerParams, std::unique_ptr<TarmacTracerParams, py::nodelete>>(m, "TarmacTracerParams")
        .def(py::init<>())
        .def("create", &TarmacTracerParams::create)
        .def_readwrite("end_tick", &TarmacTracerParams::end_tick)
        .def_readwrite("start_tick", &TarmacTracerParams::start_tick)
        ;

    py::class_<Trace::TarmacTracer, Trace::InstTracer, std::unique_ptr<Trace::TarmacTracer, py::nodelete>>(m, "Trace_COLONS_TarmacTracer")
        ;

}

static EmbeddedPyBind embed_obj("TarmacTracer", module_init, "InstTracer");
