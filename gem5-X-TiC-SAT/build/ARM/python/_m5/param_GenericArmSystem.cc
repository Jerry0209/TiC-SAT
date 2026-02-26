#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/GenericArmSystem.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/system.hh"

#include "base/types.hh"
#include <string>
#include "enums/ArmMachineType.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_GenericArmSystem");
    py::class_<GenericArmSystemParams, ArmSystemParams, std::unique_ptr<GenericArmSystemParams, py::nodelete>>(m, "GenericArmSystemParams")
        .def(py::init<>())
        .def("create", &GenericArmSystemParams::create)
        .def_readwrite("atags_addr", &GenericArmSystemParams::atags_addr)
        .def_readwrite("dtb_filename", &GenericArmSystemParams::dtb_filename)
        .def_readwrite("early_kernel_symbols", &GenericArmSystemParams::early_kernel_symbols)
        .def_readwrite("enable_context_switch_stats_dump", &GenericArmSystemParams::enable_context_switch_stats_dump)
        .def_readwrite("machine_type", &GenericArmSystemParams::machine_type)
        .def_readwrite("panic_on_oops", &GenericArmSystemParams::panic_on_oops)
        .def_readwrite("panic_on_panic", &GenericArmSystemParams::panic_on_panic)
        ;

    py::class_<GenericArmSystem, ArmSystem, std::unique_ptr<GenericArmSystem, py::nodelete>>(m, "GenericArmSystem")
        ;

}

static EmbeddedPyBind embed_obj("GenericArmSystem", module_init, "ArmSystem");
