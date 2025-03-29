import importlib
from copy import deepcopy
from os import path as osp

from app.pipe.components.codeformer.basicsr.utils import scandir
from app.pipe.components.codeformer.basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]# import all the arch modules
_arch_modules = [importlib.import_module(f'app.pipe.components.codeformer.basicsr.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    return net
