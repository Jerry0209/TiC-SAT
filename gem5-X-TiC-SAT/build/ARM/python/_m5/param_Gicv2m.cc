#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/Gicv2m.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/gic_v2m.hh"

#include <vector>
#include "dev/arm/gic_v2m.hh"
#include "dev/arm/base_gic.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_Gicv2m");
    py::class_<Gicv2mParams, PioDeviceParams, std::unique_ptr<Gicv2mParams, py::nodelete>>(m, "Gicv2mParams")
        .def(py::init<>())
        .def("create", &Gicv2mParams::create)
        .def_readwrite("frames", &Gicv2mParams::frames)
        .def_readwrite("gic", &Gicv2mParams::gic)
        .def_readwrite("pio_delay", &Gicv2mParams::pio_delay)
        ;

    py::class_<Gicv2m, PioDevice, std::unique_ptr<Gicv2m, py::nodelete>>(m, "Gicv2m")
        ;

}

static EmbeddedPyBind embed_obj("Gicv2m", module_init, "PioDevice");
