"""Inert reader for the authors' dill pickles (Roth2024_PLoSCompBiol). Nothing in the files is executed and no library
function named in them is ever called: every class or function the pickle refers to is replaced by a stand-in that only
records its arguments (dict, list, tuple, set, str, int, float, bool, bytes are the only real types). Afterwards the
recorded numpy arrays are rebuilt with numpy.frombuffer from their raw bytes, and pandas frames from their stored blocks.
"""
import pickle
import numpy as np

_REAL = {"dict": dict, "list": list, "tuple": tuple, "set": set, "frozenset": frozenset, "str": str, "int": int,
         "float": float, "bool": bool, "bytes": bytes, "bytearray": bytearray, "complex": complex}


class Stub:
    def __init__(self, *args, **kw):
        self._args = args

    def __new__(cls, *args, **kw):
        o = object.__new__(cls)
        o._args = args
        return o

    def __call__(self, *args, **kw):
        o = Stub()
        o._args = args
        o._called = self
        return o

    def __setstate__(self, state):
        self._state = state

    def __repr__(self):
        return f"Stub({getattr(self, '_name', '?')})"


def _make(module, name):
    c = type(name, (Stub,), {"_name": f"{module}.{name}"})
    return c


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" and name in _REAL:
            return _REAL[name]
        return _make(module, name)


def load(path):
    with open(path, "rb") as f:
        return _U(f).load()


def name_of(o):
    return getattr(type(o), "_name", None) if isinstance(o, Stub) else None


def to_array(o):
    """Rebuild a numpy ndarray recorded by numpy's _reconstruct from its state."""
    st = getattr(o, "_state", None)
    assert st is not None and len(st) >= 5, "not an ndarray state"
    _, shape, dt, fortran, raw = st[:5]
    dtype_args = dt._args
    dtype = np.dtype(dtype_args[0])
    if dtype.kind == "O":
        flat = list(raw) if isinstance(raw, (list, tuple)) else []
        a = np.empty(len(flat), dtype=object)
        for i, v in enumerate(flat):
            a[i] = v
        return a.reshape(shape, order="F" if fortran else "C")
    bo = None
    s = getattr(dt, "_state", None)
    if s is not None and len(s) > 1 and s[1] in ("<", ">", "|", "="):
        dtype = dtype.newbyteorder(s[1] if s[1] != "|" else "=")
    return np.frombuffer(raw, dtype=dtype).reshape(shape, order="F" if fortran else "C").copy()


def is_array(o):
    n = name_of(o)
    return bool(n and n.endswith("multiarray._reconstruct") and isinstance(getattr(o, "_state", None), tuple) and len(o._state) >= 5)


def _idx(ix):
    d = ix._args[1] if len(ix._args) > 1 else None
    if isinstance(d, dict) and "data" in d:
        return to_array(d["data"]) if is_array(d["data"]) else list(d["data"])
    if isinstance(d, dict) and "start" in d:
        return np.arange(d["start"], d["stop"], d["step"])
    raise ValueError("unknown index")


def frame(df):
    """Rebuild a pandas DataFrame recorded in the pickle into {'columns': [...], 'index': array, 'data': {column: 1-D array}}."""
    mgr = df._state["_mgr"]
    blocks, axes = mgr._args
    cols, index = _idx(axes[0]), _idx(axes[1])
    data = {}
    for b in blocks:
        vals, place, _ = b._args
        v = to_array(vals) if is_array(vals) else vals
        if is_array(place):
            pl = [int(i) for i in to_array(place)]
        elif isinstance(place, Stub):
            pl = list(range(place._args[0], place._args[1], place._args[2]))
        else:
            pl = list(place)
        for r, c in enumerate(pl):
            data[cols[c]] = v[r]
    return dict(columns=list(cols), index=np.asarray(index), data=data)


def scalar(o):
    """numpy scalar recorded by numpy.core.multiarray.scalar(dtype, bytes)."""
    if isinstance(o, Stub):
        dt, raw = o._args
        return np.frombuffer(raw, dtype=np.dtype(dt._args[0]))[0]
    return o
