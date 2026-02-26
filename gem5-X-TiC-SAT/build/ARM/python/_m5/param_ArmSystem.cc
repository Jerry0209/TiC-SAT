#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmSystem.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/system.hh"

#include <vector>
#include <string>
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "arch/arm/semihosting.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmSystem");
    py::class_<ArmSystemParams, SystemParams, std::unique_ptr<ArmSystemParams, py::nodelete>>(m, "ArmSystemParams")
        .def(py::init<>())
        .def("create", &ArmSystemParams::create)
        .def_readwrite("auto_reset_addr_64", &ArmSystemParams::auto_reset_addr_64)
        .def_readwrite("boot_loader", &ArmSystemParams::boot_loader)
        .def_readwrite("flags_addr", &ArmSystemParams::flags_addr)
        .def_readwrite("gic_cpu_addr", &ArmSystemParams::gic_cpu_addr)
        .def_readwrite("have_large_asid_64", &ArmSystemParams::have_large_asid_64)
        .def_readwrite("have_lpae", &ArmSystemParams::have_lpae)
        .def_readwrite("have_security", &ArmSystemParams::have_security)
        .def_readwrite("have_virtualization", &ArmSystemParams::have_virtualization)
        .def_readwrite("highest_el_is_64", &ArmSystemParams::highest_el_is_64)
        .def_readwrite("m5ops_base", &ArmSystemParams::m5ops_base)
        .def_readwrite("multi_proc", &ArmSystemParams::multi_proc)
        .def_readwrite("phys_addr_range_64", &ArmSystemParams::phys_addr_range_64)
        .def_readwrite("reset_addr_64", &ArmSystemParams::reset_addr_64)
        .def_readwrite("semihosting", &ArmSystemParams::semihosting)
        ;

    py::class_<ArmSystem, System, std::unique_ptr<ArmSystem, py::nodelete>>(m, "ArmSystem")
        ;

}

static EmbeddedPyBind embed_obj("ArmSystem", module_init, "System");
