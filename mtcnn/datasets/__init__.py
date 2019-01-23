from .wider_face import WiderFace
from .celeba import CelebA

import sys
def get_by_name(name, *args, **kwargs):
    """Get Dataset instance by name

    Args:
        name (str): Name of a avaliable dataset class
    """
    this_module = sys.modules[__name__]
    if hasattr(this_module, name):
        return getattr(this_module, name)(*args, **kwargs)

    else:
        raise AttributeError("No queue named %s." % name)