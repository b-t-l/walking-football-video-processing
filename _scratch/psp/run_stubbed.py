"""Test harness (not part of the app): run `python -m player_positions` on a machine without torch & co.
Heavy libraries that are missing get stubbed in THIS process only; the player_positions code itself never uses them."""
import importlib.abc, importlib.machinery, os, runpy, sys, types
from unittest import mock

STUB_TOP = {"torch", "torchvision", "cv2", "supervision", "ultralytics", "moviepy", "shapely", "inference", "roboflow",
            "PIL", "sklearn", "scipy", "matplotlib", "reportlab", "numba", "tensorflow", "onnxruntime", "transformers"}

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUB_TOP:
            try:                                   # use the real one when it is installed
                sys.meta_path.remove(self)
                real = importlib.util.find_spec(name)
            except Exception:
                real = None
            finally:
                sys.meta_path.insert(0, self)
            if real is not None:
                return None
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; m.__name__ = spec.name
        return m
    def exec_module(self, module): pass

import importlib.util
sys.meta_path.insert(0, _Finder())
sys.path.insert(0, os.getcwd())
sys.argv = ["player_positions"] + sys.argv[1:]
runpy.run_module("player_positions", run_name="__main__")
