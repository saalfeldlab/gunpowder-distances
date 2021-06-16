import logging
import numpy as np
import collections
from scipy.ndimage.morphology import distance_transform_edt
from gunpowder.array import Array
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class CombineDistances(BatchFilter):
    '''Combine signed distances that were computed of different labels. Note, that the result is only guaranteed to
    be correct if the combined labels were not overlapping or touching.

    Args:

        distance_array_keys(tuple of :class:``ArrayKey``): Tuple containing the :class:``ArrayKey`` that should be
            combined.

        target_distance_array_key(:class:``ArrayKey``): The :class:``ArrayKey`` for the combined distance transform.
            This may also be a :class:``ArrayKey`` from the distance_array_keys that should be overwritten.

        mask_array_keys(tuple of :class:``ArrayKey``, optional): The :class:``ArrayKey`` of the masks corresponding
            to the distance_array_keys

        target_mask_array_key(:class: ``ArrayKey``, optional): The :class:``ArrayKey`` for the combined masks. This may
            also be a :class:``ArrayKey`` from the mask_array_keys that should be overwritten.
    '''

    def __init__(
            self,
            distance_array_keys,
            target_distance_array_key,
            mask_array_keys=None,
            target_mask_array_key=None
            ):

        self.distance_array_keys = distance_array_keys
        self.target_distance_array_key = target_distance_array_key
        self.mask_array_keys = mask_array_keys
        self.target_mask_array_key = target_mask_array_key
        if self.mask_array_keys is None:
            assert self.target_mask_array_key is None, (
                "target_mask_array_key cannot be computed without specifiying mask_array_keys in CombineDistances")
        if self.target_mask_array_key is None:
            assert self.mask_array_keys is None, (
                "missing target_mask_array_key to combine the masks from mask_array_keys in CombineDistances")

    def setup(self):

        voxel_size = self.spec[self.distance_array_keys[0]].voxel_size
        for distance_array_key in self.distance_array_keys:
            assert distance_array_key in self.spec, (
                "Upstream does not provide %s needed by "
                "CombineDistances"%distance_array_key)
            assert self.spec[distance_array_key].voxel_size == voxel_size, \
                "Voxel sizes of distances to be combined by CombineDistances do not match: {0:}, {1:}".format(
                    self.spec[distance_array_key].voxel_size, voxel_size)

        if self.mask_array_keys is not None:
            voxel_size = self.spec[self.mask_array_keys[0]].voxel_size
            for mask_array_key in self.mask_array_keys:
                assert mask_array_key in self.spec, (
                    "Upstream does not provide %s needed by "
                    "CombineDistances"%mask_array_key)
                assert self.spec[mask_array_key].voxel_size == voxel_size, \
                    "Voxel sizes of masks to be combined by CombineDistances do not match: {0:}, {1:}".format(
                        self.spec[mask_array_key].voxel_size, voxel_size)

        if self.target_distance_array_key not in self.spec:
            spec = self.spec[self.distance_array_keys[0]].copy()
            self.provides(self.target_distance_array_key, spec)

        if self.target_mask_array_key is not None:
            if self.target_mask_array_key not in self.spec:
                spec = self.spec[self.mask_array_keys[0]].copy()
                self.provides(self.target_mask_array_key, spec)

    def process(self, batch, request):

        if (self.target_distance_array_key not in request) and (self.target_mask_array_key not in request):
            return

        distances = []
        for distance_array_key in self.distance_array_keys:
            distances.append(batch.arrays[distance_array_key].data)
        combined_distance = np.max(tuple(distances), axis=0)
        spec = self.spec[self.target_distance_array_key].copy()
        spec.roi = request[self.target_distance_array_key].roi
        batch.arrays[self.target_distance_array_key] = Array(combined_distance, spec)
        del combined_distance

        if self.target_mask_array_key is not None:
            masks = []
            for mask_array_key in self.mask_array_keys:
                masks.append(batch.arrays[mask_array_key].data)
            combined_mask = np.max(tuple(masks), axis=0)
            spec = self.spec[self.target_mask_array_key].copy()
            spec.roi = request[self.target_mask_array_key].roi
            batch.arrays[self.target_mask_array_key] = Array(combined_mask, spec)
