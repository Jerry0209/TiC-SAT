#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmStage2MMU.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/stage2_mmu.hh"

#include "arch/arm/tlb.hh"
#include "sim/system.hh"
#include "arch/arm/tlb.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmStage2MMU");
    py::class_<ArmStage2MMUParams, SimObjectParams, std::unique_ptr<ArmStage2MMUParams, py::nodelete>>(m, "ArmStage2MMUParams")
        .def(py::init<>())
        .def("create", &ArmStage2MMUParams::create)
        .def_readwrite("stage2_tlb", &ArmStage2MMUParams::stage2_tlb)
        .def_readwrite("sys", &ArmStage2MMUParams::sys)
        .def_readwrite("tlb", &ArmStage2MMUParams::tlb)
        ;

    py::class_<ArmISA::Stage2MMU, SimObject, std::unique_ptr<ArmISA::Stage2MMU, py::nodelete>>(m, "ArmISA_COLONS_Stage2MMU")
        ;

}

static EmbeddedPyBind embed_obj("ArmStage2MMU", module_init, "SimObject");
