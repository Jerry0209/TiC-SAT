#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmTLB.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/tlb.hh"

#include "base/types.hh"
#include "sim/system.hh"
#include "arch/arm/table_walker.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmTLB");
    py::class_<ArmTLBParams, BaseTLBParams, std::unique_ptr<ArmTLBParams, py::nodelete>>(m, "ArmTLBParams")
        .def(py::init<>())
        .def("create", &ArmTLBParams::create)
        .def_readwrite("is_stage2", &ArmTLBParams::is_stage2)
        .def_readwrite("size", &ArmTLBParams::size)
        .def_readwrite("sys", &ArmTLBParams::sys)
        .def_readwrite("walker", &ArmTLBParams::walker)
        ;

    py::class_<ArmISA::TLB, BaseTLB, std::unique_ptr<ArmISA::TLB, py::nodelete>>(m, "ArmISA_COLONS_TLB")
        ;

}

static EmbeddedPyBind embed_obj("ArmTLB", module_init, "BaseTLB");
