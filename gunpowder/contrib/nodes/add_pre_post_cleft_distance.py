import logging
import numpy as np

from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.morphology import distance_transform_edt
from gunpowder.array import Array
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class AddPrePostCleftDistance(BatchFilter):
    '''Add a volume with vectors pointing away from the closest boundary.

    The vectors are the spacial gradients of the distance transform, i.e., the
    distance to the boundary between labels or the background label (0).

    Args:

        label_array_key(:class:``VolumeType``): The volume type to read the
            labels from.

        distance_array_key(:class:``VolumeType``, optional): The volume type
            to generate containing the values of the distance transform.

        boundary_array_key(:class:``VolumeType``, optional): The volume type
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
            cleft_array_key,
            label_array_key,

            presyn_distance_array_key,
            postsyn_distance_array_key,
            cleft_to_presyn_neuron_id,
            cleft_to_postyn_neuron_id,
            normalize=None,
            normalize_args=None,
            bg_value=0,
            include_cleft=True
    ):

        self.cleft_array_key = cleft_array_key
        self.label_array_key = label_array_key

        self.presyn_distance_array_key = presyn_distance_array_key
        self.postsyn_distance_array_key = postsyn_distance_array_key
        self.cleft_to_presyn_neuron_id = cleft_to_presyn_neuron_id
        self.cleft_to_postsyn_neuron_id = cleft_to_postyn_neuron_id
        self.normalize = normalize
        self.normalize_args = normalize_args
        self.bg_value = bg_value
        self.include_cleft = include_cleft

    def setup(self):

        assert self.label_array_key in self.spec, (
                "Upstream does not provide %s needed by "
                "AddBoundaryDistance"%self.label_array_key)
        assert self.cleft_array_key in self.spec, (
                "Upstream does not provide %s needed by "
                "AddPrePostCleftDistance"%self.cleft_array_key)

        spec = self.spec[self.label_array_key].copy()
        spec.dtype = np.float32
        if self.presyn_distance_array_key is not None:
            self.provides(self.presyn_distance_array_key, spec.copy())
        if self.postsyn_distance_array_key is not None:
            self.provides(self.postsyn_distance_array_key, spec.copy())

    def prepare(self, request):

        if self.presyn_distance_array_key is not None and self.presyn_distance_array_key in request:
            del request[self.presyn_distance_array_key]
        if self.postsyn_distance_array_key is not None and self.postsyn_distance_array_key in request:
            del request[self.postsyn_distance_array_key]

    def process(self, batch, request):

        if (self.presyn_distance_array_key not in request and
                self.postsyn_distance_array_key not in request):
            return

        clefts = batch.arrays[self.cleft_array_key].data
        labels = batch.arrays[self.label_array_key].data
        voxel_size = self.spec[self.cleft_array_key].voxel_size
        max_distance = min(dim * vs for dim, vs in zip(clefts.shape, voxel_size))

        if (self.presyn_distance_array_key is not None and
                self.presyn_distance_array_key in request):
            presyn_distances = -np.ones(clefts.shape, dtype=np.float) * max_distance
        if (self.postsyn_distance_array_key is not None and
                self.postsyn_distance_array_key in request):
            postsyn_distances = -np.ones(clefts.shape, dtype=np.float) * max_distance
        if (self.presyn_distance_array_key is not None and
            self.presyn_distance_array_key in request) or (self.postsyn_distance_array_key is not None and
                                                           self.postsyn_distance_array_key in request):
            contained_cleft_ids = np.unique(clefts)
            for cleft_id in contained_cleft_ids:
                if cleft_id != self.bg_value:
                    d = -distance_transform_edt(clefts != cleft_id, sampling=voxel_size)
                    if (self.presyn_distance_array_key is not None and
                            self.presyn_distance_array_key in request):
                        try:
                            pre_neuron_id = np.array(list(self.cleft_to_presyn_neuron_id[cleft_id]))
                            pre_mask = np.any(labels[...,None] == pre_neuron_id[None,...],axis=-1)
                            if self.include_cleft:
                                pre_mask = np.any([pre_mask, clefts==cleft_id], axis=0)
                            presyn_distances[pre_mask] = np.max((presyn_distances, d), axis=0)[pre_mask]
                        except KeyError:
                            logger.warning("No Key in Pre Dict %s" %str(cleft_id))
                    if (self.postsyn_distance_array_key is not None and
                            self.postsyn_distance_array_key in request):
                        try:
                            post_neuron_id = np.array(list(self.cleft_to_postsyn_neuron_id[cleft_id]))
                            post_mask = np.any(labels[..., None] == post_neuron_id[None, ...], axis=-1)
                            if self.include_cleft:
                                post_mask = np.any([post_mask, clefts==cleft_id], axis=0)
                            postsyn_distances[post_mask] = np.max((postsyn_distances, d), axis=0)[post_mask]
                        except KeyError:
                            logger.warning("No Key in Post Dict %s" %str(cleft_id))
            if (self.presyn_distance_array_key is not None and
                    self.presyn_distance_array_key in request):
                #presyn_distances = np.expand_dims(presyn_distances, 0)
                if self.normalize is not None:
                    presyn_distances = self.__normalize(presyn_distances, self.normalize, self.normalize_args)
                pre_spec = self.spec[self.presyn_distance_array_key].copy()
                pre_spec.roi = request[self.presyn_distance_array_key].roi
                batch.arrays[self.presyn_distance_array_key] = Array(presyn_distances, pre_spec)

            if (self.postsyn_distance_array_key is not None and
                    self.postsyn_distance_array_key in request):

                #postsyn_distances = np.expand_dims(postsyn_distances, 0)
                if self.normalize is not None:
                    postsyn_distances = self.__normalize(postsyn_distances, self.normalize, self.normalize_args)
                post_spec = self.spec[self.postsyn_distance_array_key].copy()
                post_spec.roi = request[self.postsyn_distance_array_key].roi
                batch.arrays[self.postsyn_distance_array_key] = Array(postsyn_distances, post_spec)


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
