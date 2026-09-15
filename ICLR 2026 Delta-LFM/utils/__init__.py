from .options import args
import importlib

def import_from_dotted_path(dotted_path):
    module_path, attr_name = dotted_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)



