from pkg_resources import resource_filename
with open(resource_filename("gpdist", "VERSION"), "r") as version_file:
    __version__ = version_file.read().strip()
del resource_filename

from .add_boundary_distance import AddBoundaryDistance
from .add_distance import AddDistance
from .add_pre_post_cleft_distance import AddPrePostCleftDistance
from .balance_by_threshold import BalanceByThreshold
from .balance_global_by_threshold import BalanceGlobalByThreshold
from .combine_distances import CombineDistances
from .tanh_saturate import TanhSaturate
