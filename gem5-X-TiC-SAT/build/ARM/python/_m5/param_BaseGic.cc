#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/BaseGic.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/base_gic.hh"

#include "dev/platform.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_BaseGic");
    py::class_<BaseGicParams, PioDeviceParams, std::unique_ptr<BaseGicParams, py::nodelete>>(m, "BaseGicParams")
        .def_readwrite("platform", &BaseGicParams::platform)
        ;

    py::class_<BaseGic, PioDevice, std::unique_ptr<BaseGic, py::nodelete>>(m, "BaseGic")
        ;

}

static EmbeddedPyBind embed_obj("BaseGic", module_init, "PioDevice");
