#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/MinorFUTiming.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "cpu/minor/func_unit.hh"

#include <string>
#include "base/types.hh"
#include "base/types.hh"
#include "cpu/timing_expr.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "cpu/minor/func_unit.hh"
#include <vector>
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_MinorFUTiming");
    py::class_<MinorFUTimingParams, SimObjectParams, std::unique_ptr<MinorFUTimingParams, py::nodelete>>(m, "MinorFUTimingParams")
        .def(py::init<>())
        .def("create", &MinorFUTimingParams::create)
        .def_readwrite("description", &MinorFUTimingParams::description)
        .def_readwrite("extraAssumedLat", &MinorFUTimingParams::extraAssumedLat)
        .def_readwrite("extraCommitLat", &MinorFUTimingParams::extraCommitLat)
        .def_readwrite("extraCommitLatExpr", &MinorFUTimingParams::extraCommitLatExpr)
        .def_readwrite("mask", &MinorFUTimingParams::mask)
        .def_readwrite("match", &MinorFUTimingParams::match)
        .def_readwrite("opClasses", &MinorFUTimingParams::opClasses)
        .def_readwrite("srcRegsRelativeLats", &MinorFUTimingParams::srcRegsRelativeLats)
        .def_readwrite("suppress", &MinorFUTimingParams::suppress)
        ;

    py::class_<MinorFUTiming, SimObject, std::unique_ptr<MinorFUTiming, py::nodelete>>(m, "MinorFUTiming")
        ;

}

static EmbeddedPyBind embed_obj("MinorFUTiming", module_init, "SimObject");
