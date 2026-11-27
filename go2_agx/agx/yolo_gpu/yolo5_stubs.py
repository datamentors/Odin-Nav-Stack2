# Lets the YOLOv5 repo code (models/, utils/) run INFERENCE in the Odin-perception CUDA venv without installing
# its plotting/training extras (ultralytics, pandas, seaborn, matplotlib, scipy, psutil, thop) into that venv
# (it is Florence-2's venv: we do not want pip to touch it). Only this process sees these empty modules.
import importlib.abc, importlib.machinery, importlib.util, sys, types

STUB_ROOTS = ("ultralytics", "pandas", "seaborn", "matplotlib", "scipy", "psutil", "thop")


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub(self.__name__ + "." + name)

    def __call__(self, *a, **k):
        return _Stub("call")


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    roots = STUB_ROOTS

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _Stub(spec.name)
        m.__path__ = []
        m.__version__ = "0.0-stub"   # YOLOv5 checks ultralytics.__version__ and runs "pip install" if missing!
        return m

    def exec_module(self, module):
        pass


def _no_pip(cmd):
    # YOLOv5 calls os.system("pip install ...") / check_requirements when something looks missing:
    # never let it install into this venv (it once started pulling another torch + CUDA and filled the disk)
    if "pip" in str(cmd):
        print("[yolo5_stubs] BLOCKED: %s" % cmd, file=sys.stderr, flush=True)
        return 0
    return _real_system(cmd)


def install():
    global _real_system
    import os
    _real_system = os.system
    os.system = _no_pip
    os.environ["PIP_NO_INDEX"] = "1"          # belt and braces: pip cannot download anything from here
    import packaging.version  # noqa: F401  (YOLOv5 uses packaging.version; ultralytics normally imports it)
    finder = _Finder()
    # only stub what is NOT installed: the detector service may run a tracker (BoT-SORT) that needs the real scipy
    finder.roots = tuple(r for r in STUB_ROOTS if importlib.util.find_spec(r) is None)
    sys.meta_path.insert(0, finder)
    import torch
    patches = importlib.import_module("ultralytics.utils.patches")
    patches.torch_load = lambda *a, **k: torch.load(*a, **{**k, "weights_only": False})
