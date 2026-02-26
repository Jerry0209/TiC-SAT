#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/HDLcd.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/hdlcd.hh"

#include "enums/ImageFormat.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "sim/clock_domain.hh"
#include "base/types.hh"
#include "base/vnc/vncinput.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_HDLcd");
    py::class_<HDLcdParams, AmbaDmaDeviceParams, std::unique_ptr<HDLcdParams, py::nodelete>>(m, "HDLcdParams")
        .def(py::init<>())
        .def("create", &HDLcdParams::create)
        .def_readwrite("enable_capture", &HDLcdParams::enable_capture)
        .def_readwrite("frame_format", &HDLcdParams::frame_format)
        .def_readwrite("pixel_buffer_size", &HDLcdParams::pixel_buffer_size)
        .def_readwrite("pixel_chunk", &HDLcdParams::pixel_chunk)
        .def_readwrite("pxl_clk", &HDLcdParams::pxl_clk)
        .def_readwrite("virt_refresh_rate", &HDLcdParams::virt_refresh_rate)
        .def_readwrite("vnc", &HDLcdParams::vnc)
        .def_readwrite("workaround_dma_line_count", &HDLcdParams::workaround_dma_line_count)
        .def_readwrite("workaround_swap_rb", &HDLcdParams::workaround_swap_rb)
        ;

    py::class_<HDLcd, AmbaDmaDevice, std::unique_ptr<HDLcd, py::nodelete>>(m, "HDLcd")
        ;

}

static EmbeddedPyBind embed_obj("HDLcd", module_init, "AmbaDmaDevice");
