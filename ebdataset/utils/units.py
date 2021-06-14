from pint import UnitRegistry as _UR

_reg = _UR()

wunits = _reg.wraps

## Global time management namespace
second = s = _reg.s
millisecond = ms = _reg.ms
microsecond = us = _reg.us
nanosecond = ns = _reg.ns
killosecond = ks = _reg.ks
hertz = Hz = _reg.Hz
millihertz = mhertz = mHz = _reg.mHz
kilohertz = khertz = kHz = _reg.kHz
