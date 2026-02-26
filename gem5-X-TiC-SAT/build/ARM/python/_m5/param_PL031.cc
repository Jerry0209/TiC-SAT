#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/PL031.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/rtc_pl031.hh"

#include <time.h>
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_PL031");
    py::class_<PL031Params, AmbaIntDeviceParams, std::unique_ptr<PL031Params, py::nodelete>>(m, "PL031Params")
        .def(py::init<>())
        .def("create", &PL031Params::create)
        .def_readwrite("time", &PL031Params::time)
        ;

    py::class_<PL031, AmbaIntDevice, std::unique_ptr<PL031, py::nodelete>>(m, "PL031")
        ;

}

static EmbeddedPyBind embed_obj("PL031", module_init, "AmbaIntDevice");
