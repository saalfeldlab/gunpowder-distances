import logging
import numpy as np
import collections
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.morphology import distance_transform_edt
from gunpowder.array import Array
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddDistance(BatchFilter):
    '''Add a volume with vectors pointing away from the closest boundary.

    The vectors are the spacial gradients of the distance transform, i.e., the
    distance to the boundary between labels or the background label (0).

    Args:

        label_array_key(:class:``ArrayKeys``): The volume type to read the
            labels from.

        distance_array_key(:class:``ArrayKeys``, optional): The volume type
            to generate containing the values of the distance transform.

        boundary_array_key(:class:``ArrayKeys``, optional): The volume type
            to generate containing a boundary labeling. Note this volume will
            be doubled as it encodes boundaries between voxels.
    '''

        # gradient_array_key(:class:``VolumeType``): The volume type to
        #     generate containing the gradients.
        #
        # normalize(string, optional): ``None``, ``'l1'``, or ``'l2'``. Specifies
        #     if and how to normalize the gradients.
        #
        # scale(string, optional): ``None`` or ``exp``. If ``exp``, distance
        #     gradients will be scaled by ``beta*e**(-d*alpha)``, where ``d`` is
        #     the distance to the boundary.
        #
        # scale_args(tuple, optional): For ``exp`` a tuple with the values of
        #     ``alpha`` and ``beta``.

    def __init__(
            self,
            label_array_key,
            distance_array_key,
            normalize=None,
            normalize_args=None,
            label_id=1,
            factor=1):

        self.label_array_key = label_array_key
        self.distance_array_key = distance_array_key
        self.normalize = normalize
        self.normalize_args = normalize_args
        if not isinstance(label_id, collections.Iterable):
            label_id = (label_id,)
        self.label_id = label_id
        self.factor = factor

    def setup(self):

        assert self.label_array_key in self.spec, (
            "Upstream does not provide %s needed by "
            "AddBoundaryDistance"%self.label_array_key)

        spec = self.spec[self.label_array_key].copy()
        spec.dtype = np.float32
        spec.voxel_size *= self.factor
        self.provides(self.distance_array_key, spec)

    def prepare(self, request):

        if self.distance_array_key in request:
            del request[self.distance_array_key]

    def process(self, batch, request):

        if not self.distance_array_key in request:
            return

        voxel_size = self.spec[self.label_array_key].voxel_size
        binary_label = np.logical_or.reduce([batch.arrays[self.label_array_key].data == lid for lid in
                                             self.label_id])

        dim = len(binary_label.shape)
        if binary_label.std() == 0:
            max_distance = min(dim*vs for dim, vs in zip(binary_label.shape, voxel_size))
            if np.sum(binary_label) == 0:
                distances = - np.ones(binary_label.shape, dtype=np.float32) * max_distance
            else:
                distances = np.ones(binary_label.shape, dtype=np.float32) * max_distance
        else:

            distances = distance_transform_edt(binary_label, sampling=tuple(float(v) for v in voxel_size))
            distances -= distance_transform_edt(np.logical_not(binary_label), sampling=tuple(float(v) for v in
                                                                                             voxel_size))
        #distances = np.expand_dims(distances, 0)
        if isinstance(self.factor, tuple):
            slices = tuple(slice(None, None, k) for k in self.factor)
        else:
            slices = tuple(slice(None, None, self.factor) for _ in range(dim))

        distances = distances[slices]
        if self.normalize is not None:
            distances = self.__normalize(distances, self.normalize, self.normalize_args)

        spec = self.spec[self.distance_array_key].copy()
        spec.roi = request[self.distance_array_key].roi
        batch.arrays[self.distance_array_key] = Array(distances, spec)


    def __normalize(self, distances, norm, normalize_args):
        if norm == 'tanh':
            scale = normalize_args
            return np.tanh(distances/scale)


