import logging
import numpy as np

from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.morphology import distance_transform_edt
from gunpowder.array import Array
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddBoundaryDistance(BatchFilter):
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
            boundary_array_key=None,
            normalize=None,
            normalize_args=None):

        self.label_array_key = label_array_key
        self.distance_array_key = distance_array_key
        self.boundary_array_key = boundary_array_key
        self.normalize = normalize
        self.normalize_args = normalize_args

    def setup(self):

        assert self.label_array_key in self.spec, (
            "Upstream does not provide %s needed by "
            "AddBoundaryDistance"%self.label_array_key)

        spec = self.spec[self.label_array_key].copy()
        spec.dtype = np.float32
        self.provides(self.distance_array_key, spec)
        if self.boundary_array_key is not None:
            spec.voxel_size /= 2
            self.provides(self.boundary_array_key, spec)

    def prepare(self, request):

        if self.distance_array_key in request:
            del request[self.distance_array_key]

        if (self.boundary_array_key is not None and
            self.boundary_array_key in request):
            del request[self.boundary_array_key]

    def process(self, batch, request):

        if not self.distance_array_key in request:
            return

        labels = batch.arrays[self.label_array_key].data
        voxel_size = self.spec[self.label_array_key].voxel_size

        # get boundaries between label regions
        boundaries = self.__find_boundaries(labels)

        # mark boundaries with 0 (not 1)
        boundaries = 1.0 - boundaries

        if np.sum(boundaries == 0) == 0:
            max_distance = min(dim*vs for dim, vs in zip(labels.shape, voxel_size))
            if np.sum(labels) == 0:
                distances = - np.ones(labels.shape, dtype=np.float32)*max_distance
            else:
                distances = np.ones(labels.shape, dtype=np.float32)*max_distance
        else:

            # get distances (voxel_size/2 because image is doubled)
            distances = distance_transform_edt(
                boundaries,
                sampling=tuple(float(v)/2 for v in voxel_size))
            distances = distances.astype(np.float32)

            # restore original shape
            downsample = (slice(None, None, 2),)*len(voxel_size)
            distances = distances[downsample]

            # todo: inverted distance
            distances[labels == 0] = - distances[labels == 0]
        distances = np.expand_dims(distances, 0)


        if self.normalize is not None:
            distances = self.__normalize(distances, self.normalize, self.normalize_args)

        #if self.scale is not None:
        #    self.__scale(gradients, distances, self.scale, self.scale_args)

        spec = self.spec[self.distance_array_key].copy()
        spec.roi = request[self.distance_array_key].roi
        batch.arrays[self.distance_array_key] = Array(distances, spec)

        if (
                self.boundary_array_key is not None and
                self.boundary_array_key in request):

            # add one more face at each dimension, as boundary map has shape
            # 2*s - 1 of original shape s
            grown = np.ones(tuple(s + 1 for s in boundaries.shape))
            grown[tuple(slice(0, s) for s in boundaries.shape)] = boundaries
            spec.voxel_size = voxel_size/2
            logger.debug("voxel size of boundary volume: %s", spec.voxel_size)
            batch.arrays[self.boundary_array_key] = Array(grown, spec)

    def __find_boundaries(self, labels):

        # labels: 1 1 1 1 0 0 2 2 2 2 3 3       n
        # shift :   1 1 1 1 0 0 2 2 2 2 3       n - 1
        # diff  :   0 0 0 1 0 1 0 0 0 1 0       n - 1
        # bound.: 00000001000100000001000      2n - 1

        logger.debug("computing boundaries for %s", labels.shape)

        dims = len(labels.shape)
        in_shape = labels.shape
        out_shape = tuple(2*s - 1 for s in in_shape)
        out_slices = tuple(slice(0, s) for s in out_shape)

        boundaries = np.zeros(out_shape, dtype=np.bool)

        logger.debug("boundaries shape is %s", boundaries.shape)

        for d in range(dims):

            logger.debug("processing dimension %d", d)

            shift_p = [slice(None)]*dims
            shift_p[d] = slice(1, in_shape[d])

            shift_n = [slice(None)]*dims
            shift_n[d] = slice(0, in_shape[d] - 1)

            diff = (labels[shift_p] - labels[shift_n]) != 0

            logger.debug("diff shape is %s", diff.shape)

            target = [slice(None, None, 2)]*dims
            target[d] = slice(1, out_shape[d], 2)

            logger.debug("target slices are %s", target)

            boundaries[target] = diff

        return boundaries

    def __normalize(self, distances, norm, normalize_args):

        if norm == 'tanh':
            scale = normalize_args
            return np.tanh(distances/scale)

    #def __normalize(self, gradients, norm):
#
    #    dims = gradients.shape[0]
#
    #    if norm == 'l1':
    #        factors = sum([np.abs(gradients[d]) for d in range(dims)])
    #    elif norm == 'l2':
    #        factors = np.sqrt(
    #                sum([np.square(gradients[d]) for d in range(dims)]))
    #    else:
    #        raise RuntimeError('norm %s not supported'%norm)
#
    #    factors[factors < 1e-5] = 1
    #    gradients /= factors
#
    #def __scale(self, gradients, distances, scale, scale_args):
#
    #    dims = gradients.shape[0]
#
    #    if scale == 'exp':
    #        alpha, beta = self.scale_args
    #        factors = np.exp(-distances*alpha)*beta
#
    #    gradients *= factors
