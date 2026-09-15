import yaml
import os, re

# https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
# for read in special characters


def load_yaml(file):

    yaml_loader = yaml.SafeLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))


    with open(file) as f:
        data = yaml.safe_load(f)

    # expand ${DATA_DIR} / ${WORK_DIR} style placeholders in string values
    def _expand(v):
        if isinstance(v, str):
            return os.path.expandvars(v)
        if isinstance(v, dict):
            return {k: _expand(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_expand(x) for x in v]
        return v

    return _expand(data)



