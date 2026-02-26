import _m5.core
import m5.util

buildEnv = m5.util.SmartDict([('USE_SYSTEMC', False), ('USE_FENV', True), ('SS_COMPATIBLE_FP', False), ('TARGET_ISA', 'arm'), ('TARGET_GPU_ISA', 'hsail'), ('CP_ANNOTATE', False), ('USE_POSIX_CLOCK', True), ('USE_KVM', False), ('USE_TUNTAP', True), ('PROTOCOL', 'MOESI_CMP_directory'), ('HAVE_PROTOBUF', True), ('HAVE_PERF_ATTR_EXCLUDE_HOST', 1), ('USE_PNG', False)])

compileDate = _m5.core.compileDate
_globals = globals()
for key,val in _m5.core.__dict__.iteritems():
    if key.startswith('flag_'):
        flag = key[5:]
        _globals[flag] = val
del _globals
