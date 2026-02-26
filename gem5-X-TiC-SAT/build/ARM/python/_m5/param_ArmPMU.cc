#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmPMU.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/pmu.hh"

#include "base/types.hh"
#include "base/types.hh"
#include "dev/platform.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmPMU");
    py::class_<ArmPMUParams, SimObjectParams, std::unique_ptr<ArmPMUParams, py::nodelete>>(m, "ArmPMUParams")
        .def(py::init<>())
        .def("create", &ArmPMUParams::create)
        .def_readwrite("cycleEventId", &ArmPMUParams::cycleEventId)
        .def_readwrite("eventCounters", &ArmPMUParams::eventCounters)
        .def_readwrite("platform", &ArmPMUParams::platform)
        .def_readwrite("pmuInterrupt", &ArmPMUParams::pmuInterrupt)
        ;

    py::class_<ArmISA::PMU, SimObject, std::unique_ptr<ArmISA::PMU, py::nodelete>>(m, "ArmISA_COLONS_PMU")
        .def("addEventProbe", &ArmISA::PMU::addEventProbe)
        .def("addSoftwareIncrementEvent", &ArmISA::PMU::addSoftwareIncrementEvent)
        ;

}

static EmbeddedPyBind embed_obj("ArmPMU", module_init, "SimObject");
