#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmSemihosting.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/semihosting.hh"

#include <string>
#include "base/types.hh"
#include "base/types.hh"
#include <string>
#include <string>
#include <string>
#include <time.h>
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmSemihosting");
    py::class_<ArmSemihostingParams, SimObjectParams, std::unique_ptr<ArmSemihostingParams, py::nodelete>>(m, "ArmSemihostingParams")
        .def(py::init<>())
        .def("create", &ArmSemihostingParams::create)
        .def_readwrite("cmd_line", &ArmSemihostingParams::cmd_line)
        .def_readwrite("mem_reserve", &ArmSemihostingParams::mem_reserve)
        .def_readwrite("stack_size", &ArmSemihostingParams::stack_size)
        .def_readwrite("stderr", &ArmSemihostingParams::stderr)
        .def_readwrite("stdin", &ArmSemihostingParams::stdin)
        .def_readwrite("stdout", &ArmSemihostingParams::stdout)
        .def_readwrite("time", &ArmSemihostingParams::time)
        ;

    py::class_<ArmSemihosting, SimObject, std::unique_ptr<ArmSemihosting, py::nodelete>>(m, "ArmSemihosting")
        ;

}

static EmbeddedPyBind embed_obj("ArmSemihosting", module_init, "SimObject");
