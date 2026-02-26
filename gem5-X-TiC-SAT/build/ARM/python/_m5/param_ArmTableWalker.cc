#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmTableWalker.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/table_walker.hh"

#include "base/types.hh"
#include "sim/system.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmTableWalker");
    py::class_<ArmTableWalkerParams, MemObjectParams, std::unique_ptr<ArmTableWalkerParams, py::nodelete>>(m, "ArmTableWalkerParams")
        .def(py::init<>())
        .def("create", &ArmTableWalkerParams::create)
        .def_readwrite("is_stage2", &ArmTableWalkerParams::is_stage2)
        .def_readwrite("num_squash_per_cycle", &ArmTableWalkerParams::num_squash_per_cycle)
        .def_readwrite("sys", &ArmTableWalkerParams::sys)
        .def_readwrite("port_port_connection_count", &ArmTableWalkerParams::port_port_connection_count)
        ;

    py::class_<ArmISA::TableWalker, MemObject, std::unique_ptr<ArmISA::TableWalker, py::nodelete>>(m, "ArmISA_COLONS_TableWalker")
        ;

}

static EmbeddedPyBind embed_obj("ArmTableWalker", module_init, "MemObject");
