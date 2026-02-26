#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmISA.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/isa.hh"

#include "enums/DecoderFlavour.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "arch/arm/pmu.hh"
#include "sim/system.hh"
#include "enums/VecRegRenameMode.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmISA");
    py::class_<ArmISAParams, SimObjectParams, std::unique_ptr<ArmISAParams, py::nodelete>>(m, "ArmISAParams")
        .def(py::init<>())
        .def("create", &ArmISAParams::create)
        .def_readwrite("decoderFlavour", &ArmISAParams::decoderFlavour)
        .def_readwrite("fpsid", &ArmISAParams::fpsid)
        .def_readwrite("id_aa64afr0_el1", &ArmISAParams::id_aa64afr0_el1)
        .def_readwrite("id_aa64afr1_el1", &ArmISAParams::id_aa64afr1_el1)
        .def_readwrite("id_aa64dfr0_el1", &ArmISAParams::id_aa64dfr0_el1)
        .def_readwrite("id_aa64dfr1_el1", &ArmISAParams::id_aa64dfr1_el1)
        .def_readwrite("id_aa64isar0_el1", &ArmISAParams::id_aa64isar0_el1)
        .def_readwrite("id_aa64isar1_el1", &ArmISAParams::id_aa64isar1_el1)
        .def_readwrite("id_aa64mmfr0_el1", &ArmISAParams::id_aa64mmfr0_el1)
        .def_readwrite("id_aa64mmfr1_el1", &ArmISAParams::id_aa64mmfr1_el1)
        .def_readwrite("id_isar0", &ArmISAParams::id_isar0)
        .def_readwrite("id_isar1", &ArmISAParams::id_isar1)
        .def_readwrite("id_isar2", &ArmISAParams::id_isar2)
        .def_readwrite("id_isar3", &ArmISAParams::id_isar3)
        .def_readwrite("id_isar4", &ArmISAParams::id_isar4)
        .def_readwrite("id_isar5", &ArmISAParams::id_isar5)
        .def_readwrite("id_mmfr0", &ArmISAParams::id_mmfr0)
        .def_readwrite("id_mmfr1", &ArmISAParams::id_mmfr1)
        .def_readwrite("id_mmfr2", &ArmISAParams::id_mmfr2)
        .def_readwrite("id_mmfr3", &ArmISAParams::id_mmfr3)
        .def_readwrite("impdef_nop", &ArmISAParams::impdef_nop)
        .def_readwrite("midr", &ArmISAParams::midr)
        .def_readwrite("pmu", &ArmISAParams::pmu)
        .def_readwrite("system", &ArmISAParams::system)
        .def_readwrite("vecRegRenameMode", &ArmISAParams::vecRegRenameMode)
        ;

    py::class_<ArmISA::ISA, SimObject, std::unique_ptr<ArmISA::ISA, py::nodelete>>(m, "ArmISA_COLONS_ISA")
        ;

}

static EmbeddedPyBind embed_obj("ArmISA", module_init, "SimObject");
