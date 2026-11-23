import importlib
import pkgutil

from sweep_bench import tasks

for _module in pkgutil.iter_modules(tasks.__path__, tasks.__name__ + "."):
    importlib.import_module(_module.name)
