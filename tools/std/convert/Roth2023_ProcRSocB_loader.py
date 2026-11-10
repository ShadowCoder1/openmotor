"""Restricted reader for the authors' dill pickles (Roth2023_ProcRSocB). The pickles embed the authors' class and a numba
function by value; nothing from them is executed: classes and functions become inert stand-ins, and only numpy and pandas
objects are rebuilt. Anything else raises."""
import importlib
import pickle
import builtins


class Dummy:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.__dict__.update(s if isinstance(s, dict) else {"_state": s})


def _inert(*a, **k):
    return Dummy()


def _create_array(f, args, state, npdict=None):
    a = f(*args)
    a.__setstate__(state)
    if npdict is not None:
        a.__dict__.update(npdict)
    return a


def _import_module(name, obj=None):
    if name.split(".")[0] in ("numpy", "pandas"):
        m = importlib.import_module(name)
        return getattr(m, obj) if obj else m
    return Dummy()


_SAFE_TYPES = ("type", "object", "slice", "dict", "list", "tuple", "int", "float", "str", "bool", "set")


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pandas.core.indexes.numeric":
            return super().find_class("pandas", "Index")
        if module.startswith(("numpy", "pandas")) or module == "collections":
            return super().find_class(module, name)
        if module.startswith("_frozen_importlib"):
            return Dummy
        if module.startswith(("numba", "scipy")):
            return _inert
        if module == "dill._dill":
            if name == "_create_array":
                return _create_array
            if name == "_import_module":
                return _import_module
            if name == "_create_type":
                return lambda *a, **k: Dummy
            if name == "_load_type":
                return lambda n: getattr(builtins, n) if n in _SAFE_TYPES else Dummy
            return _inert
        if module == "builtins" and name == "__dict__":
            return {}
        if module == "builtins" and name in ("dict", "list", "set", "tuple", "slice", "object", "frozenset", "complex", "bytearray"):
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load(path):
    with open(path, "rb") as f:
        return _U(f).load()
