#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/RealViewCtrl.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/rv_ctrl.hh"

#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_RealViewCtrl");
    py::class_<RealViewCtrlParams, BasicPioDeviceParams, std::unique_ptr<RealViewCtrlParams, py::nodelete>>(m, "RealViewCtrlParams")
        .def(py::init<>())
        .def("create", &RealViewCtrlParams::create)
        .def_readwrite("idreg", &RealViewCtrlParams::idreg)
        .def_readwrite("proc_id0", &RealViewCtrlParams::proc_id0)
        .def_readwrite("proc_id1", &RealViewCtrlParams::proc_id1)
        ;

    py::class_<RealViewCtrl, BasicPioDevice, std::unique_ptr<RealViewCtrl, py::nodelete>>(m, "RealViewCtrl")
        ;

}

static EmbeddedPyBind embed_obj("RealViewCtrl", module_init, "BasicPioDevice");
