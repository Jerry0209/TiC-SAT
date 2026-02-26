#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/TarmacParser.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/tracers/tarmac_parser.hh"

#include "base/types.hh"
#include "base/addr_range.hh"
#include <string>
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_TarmacParser");
    py::class_<TarmacParserParams, InstTracerParams, std::unique_ptr<TarmacParserParams, py::nodelete>>(m, "TarmacParserParams")
        .def(py::init<>())
        .def("create", &TarmacParserParams::create)
        .def_readwrite("cpu_id", &TarmacParserParams::cpu_id)
        .def_readwrite("exit_on_diff", &TarmacParserParams::exit_on_diff)
        .def_readwrite("exit_on_insn_diff", &TarmacParserParams::exit_on_insn_diff)
        .def_readwrite("ignore_mem_addr", &TarmacParserParams::ignore_mem_addr)
        .def_readwrite("mem_wr_check", &TarmacParserParams::mem_wr_check)
        .def_readwrite("path_to_trace", &TarmacParserParams::path_to_trace)
        .def_readwrite("start_pc", &TarmacParserParams::start_pc)
        ;

    py::class_<Trace::TarmacParser, Trace::InstTracer, std::unique_ptr<Trace::TarmacParser, py::nodelete>>(m, "Trace_COLONS_TarmacParser")
        ;

}

static EmbeddedPyBind embed_obj("TarmacParser", module_init, "InstTracer");
