#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/EnergyCtrl.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/energy_ctrl.hh"

#include "sim/dvfs_handler.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_EnergyCtrl");
    py::class_<EnergyCtrlParams, BasicPioDeviceParams, std::unique_ptr<EnergyCtrlParams, py::nodelete>>(m, "EnergyCtrlParams")
        .def(py::init<>())
        .def("create", &EnergyCtrlParams::create)
        .def_readwrite("dvfs_handler", &EnergyCtrlParams::dvfs_handler)
        ;

    py::class_<EnergyCtrl, BasicPioDevice, std::unique_ptr<EnergyCtrl, py::nodelete>>(m, "EnergyCtrl")
        ;

}

static EmbeddedPyBind embed_obj("EnergyCtrl", module_init, "BasicPioDevice");
