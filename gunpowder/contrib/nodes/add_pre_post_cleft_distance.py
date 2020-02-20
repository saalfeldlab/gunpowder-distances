import logging
import numpy as np
import collections.abc

from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
from scipy.ndimage import generate_binary_structure
from gunpowder.array import Array
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class AddPrePostCleftDistance(BatchFilter):
    '''Add two arrays with cleft distances being masked using their pre-and postsynaptic sites, respectively.

    Computes the cleft distance and then uses the dictionaries mapping cleft id to pre- and postsynaptic neuron ids
    to generate two masks per cleft - one for the pre- and one for the postsynaptic sites for the values in the cleft
    distance map.

    Args:
        cleft_array_key(:class:``ArrayKey``): The :class:``ArrayKey`` to read the cleft labels from.

        label_array_key(:class:``ArrayKey``): The :class:``ArrayKey`` to read the
            neuron labels from.

        cleft_distance_array_key(:class:``ArrayKey``): The class ``ArrayKey`` to generate containing the values of
            the distance transform masked to the cleft.

        presyn_distance_array_key(:class:``ArrayKey``): The class ``ArrayKey`` to generate containing the values of
            the distance transform masked to the presynaptic sites.

        postsyn_distance_array_key(:class:``ArrayKey``): The class ``ArrayKey`` to generate containting the values of
            the distance transform  masked to the postsynaptic sites.

        cleft_mask_array_key(:class:``ArrayKey``): The class ``ArrayKey`` to update in order to compensate for
            windowing artifacts after distance transform for the cleft array.

        presyn_mask_array_key(:class:``ArrayKey``): The class ``ArrayKey`` to update in order to compensate for
            windowing artifacts after distance transform for the presynaptic array.

        postsyn_mask_array_key(:class:``ArrayKey``): The class ``ArrayKey`` to update in order to compensate for
            windowing artifacts after distance transform for the postsynaptic array.

        cleft_to_presyn_neuron_id(dict): The dictionary that maps cleft ids to corresponding presynaptic neuron ids.

        cleft_to_postsyn_neuron_id(dict): The dictionary that maps cleft ids to corresponding presynaptic neuron ids.

        bg_value(int or tuple, optional): The background value in the cleft array, can be several. (default: 0)

        include_cleft(boolean, optional): whether to include the whole cleft as part of the label when  calculating
            the masked distance transform (default: True)

        add_constant(scalar, optional): constant value to add to distance transform (default: None, i.e. nothing is
            added)

        max_distance(scalar, tuple, optional): maximal distance that computed distances will be clipped to. For a
            single value this is the absolute value of the minimal and maximal distance. A tuple should be given as (
            minimal_distance, maximal_distance) (default: None, i.e. no clipping)

    '''

    def __init__(
            self,
            cleft_array_key,
            label_array_key,
            cleft_distance_array_key,
            presyn_distance_array_key,
            postsyn_distance_array_key,
            cleft_mask_array_key,
            presyn_mask_array_key,
            postsyn_mask_array_key,
            cleft_to_presyn_neuron_id,
            cleft_to_postyn_neuron_id,
            cleft_to_presyn_neuron_id_filtered=None,
            cleft_to_postsyn_neuron_id_filtered=None,
            bg_value=0,
            include_cleft=True,
            add_constant=None,
            max_distance=None
    ):

        self.cleft_array_key = cleft_array_key
        self.label_array_key = label_array_key
        self.cleft_mask_array_key = cleft_mask_array_key
        self.presyn_mask_array_key = presyn_mask_array_key
        self.postsyn_mask_array_key = postsyn_mask_array_key

        self.cleft_distance_array_key = cleft_distance_array_key
        self.presyn_distance_array_key = presyn_distance_array_key
        self.postsyn_distance_array_key = postsyn_distance_array_key
        self.cleft_to_presyn_neuron_id = cleft_to_presyn_neuron_id
        self.cleft_to_postsyn_neuron_id = cleft_to_postyn_neuron_id
        self.cleft_to_presyn_neuron_id_filtered = cleft_to_presyn_neuron_id_filtered
        self.cleft_to_postsyn_neuron_id_filtered = cleft_to_postsyn_neuron_id_filtered
        if not isinstance(bg_value, collections.abc.Iterable):
            bg_value = (bg_value,)
        self.bg_value = bg_value
        assert not isinstance(self.bg_value, set)
        self.include_cleft = include_cleft
        self.max_distance = max_distance
        self.add_constant = add_constant

    def setup(self):

        assert self.label_array_key in self.spec, (
                "Upstream does not provide %s needed by "
                "AddBoundaryDistance"%self.label_array_key)
        assert self.cleft_array_key in self.spec, (
                "Upstream does not provide %s needed by "
                "AddPrePostCleftDistance"%self.cleft_array_key)

        spec = self.spec[self.label_array_key].copy()
        spec.dtype = np.float32
        if self.cleft_distance_array_key is not None:
            self.provides(self.cleft_distance_array_key, spec.copy())
        if self.presyn_distance_array_key is not None:
            self.provides(self.presyn_distance_array_key, spec.copy())
        if self.postsyn_distance_array_key is not None:
            self.provides(self.postsyn_distance_array_key, spec.copy())

    def prepare(self, request):

        if self.cleft_distance_array_key is not None and self.cleft_distance_array_key in request:
            del request[self.cleft_distance_array_key]
        if self.presyn_distance_array_key is not None and self.presyn_distance_array_key in request:
            del request[self.presyn_distance_array_key]
        if self.postsyn_distance_array_key is not None and self.postsyn_distance_array_key in request:
            del request[self.postsyn_distance_array_key]

    def process(self, batch, request):

        if (self.cleft_distance_array_key not in request and
                self.presyn_distance_array_key not in request and
                self.postsyn_distance_array_key not in request):
            return

        voxel_size = self.spec[self.cleft_array_key].voxel_size
        clefts = batch.arrays[self.cleft_array_key].data
        bg_mask = np.isin(clefts,self.bg_value)
        clefts[bg_mask] = self.bg_value[0]
        labels = batch.arrays[self.label_array_key].data
        cleft_mask_total = batch.arrays[self.cleft_mask_array_key].data
        pre_mask_total = batch.arrays[self.presyn_mask_array_key].data
        post_mask_total = batch.arrays[self.postsyn_mask_array_key].data
        if (self.cleft_distance_array_key is not None and self.cleft_distance_array_key in request) or (
                self.presyn_distance_array_key is not None and self.presyn_distance_array_key in request) or (
                self.postsyn_distance_array_key is not None and self.postsyn_distance_array_key in request):

            constant_label = clefts.std() == 0

            tmp = np.zeros(np.array(clefts.shape) + np.array((2,)* clefts.ndim), dtype=clefts.dtype)
            slices = tmp.ndim * (slice(1, -1),)
            tmp[slices] = np.ones(clefts.shape, dtype=clefts.dtype)
            distances = distance_transform_edt(binary_erosion(tmp, border_value=1,
                                                   structure=generate_binary_structure(tmp.ndim, tmp.ndim)),
                                                   sampling=voxel_size)
            if self.max_distance is None:
                logger.warning("Without a max distance to clip to constant batches will always be completely masked "
                                    "out")
            else:
                actual_max_distance = np.max(distances)
                if self.max_distance > actual_max_distance:
                    logger.warning("The given max distance {0:} to clip to is higher than the maximal distance ({"
                                        "1:}) that can be contained in a batch of size {2:}".format(self.max_distance,
                                                                                                   actual_max_distance,
                                                                                                clefts.shape))
            if self.bg_value[0] in clefts:
                distances += 1
                distances *= -1
            cleft_distances = distances[slices]
            presyn_distances = distances[slices]
            postsyn_distances = distances[slices]
            if not constant_label:
                contained_cleft_ids = np.unique(clefts)
                for cleft_id in contained_cleft_ids:
                    if cleft_id != self.bg_value[0]:
                        d = self.__signed_distance(clefts == cleft_id, sampling=voxel_size)
                        if (self.cleft_distance_array_key is not None and self.cleft_distance_array_key in request):
                            if (cleft_id in self.cleft_to_presyn_neuron_id) or (
                                    cleft_id in self.cleft_to_postsyn_neuron_id):
                                cleft_distances = np.max((cleft_distances, d), axis=0)
                        if (self.presyn_distance_array_key is not None and
                                self.presyn_distance_array_key in request):
                            try:
                                pre_neuron_id = np.array(list(self.cleft_to_presyn_neuron_id[cleft_id]))
                                pre_mask = np.any(labels[..., None] == pre_neuron_id[None, ...], axis=-1)
                                if self.include_cleft:
                                    pre_mask = np.any([pre_mask, clefts == cleft_id], axis=0)
                                presyn_distances[pre_mask] = np.max((presyn_distances, d), axis=0)[pre_mask]
                            except KeyError as e:
                                if self.cleft_to_presyn_neuron_id_filtered is not None:
                                    if cleft_id in self.cleft_to_presyn_neuron_id_filtered:
                                        logger.info("Key {0:} filtered out from Pre Dict".format(cleft_id))
                                    else:
                                        logger.error("No Key in Pre Dict %s" % str(cleft_id))
                                        raise e

                        if (self.postsyn_distance_array_key is not None and
                                self.postsyn_distance_array_key in request):
                            try:
                                post_neuron_id = np.array(list(self.cleft_to_postsyn_neuron_id[cleft_id]))
                                post_mask = np.any(labels[..., None] == post_neuron_id[None, ...], axis=-1)
                                if self.include_cleft:
                                    post_mask = np.any([post_mask, clefts == cleft_id], axis=0)
                                postsyn_distances[post_mask] = np.max((postsyn_distances, d), axis=0)[post_mask]
                            except KeyError as e:
                                if self.cleft_to_postsyn_neuron_id_filtered is not None:
                                    if cleft_id in self.cleft_to_postsyn_neuron_id_filtered:
                                        logger.info("Key {0:} filtered out from Post Dict".format(cleft_id))
                                    else:
                                        logger.error("No Key in Post Dict %s" % str(cleft_id))
                                        raise e

            if self.max_distance is not None:
                if self.add_constant is None:
                    add = 0
                else:
                    add = self.add_constant

                if self.cleft_distance_array_key is not None and self.cleft_distance_array_key in request:
                    cleft_distances = self.__clip_distance(cleft_distances, (-self.max_distance-add,
                                                                             self.max_distance-add))

                if self.presyn_distance_array_key is not None and self.presyn_distance_array_key in request:
                    presyn_distances = self.__clip_distance(presyn_distances, (-self.max_distance-add,
                                                            self.max_distance-add))

                if self.postsyn_distance_array_key is not None and self.postsyn_distance_array_key in request:
                    postsyn_distances = self.__clip_distance(postsyn_distances, (-self.max_distance-add,
                                                                                 self.max_distance-add))

            if self.add_constant is not None and not constant_label:
                if self.cleft_distance_array_key is not None and self.cleft_distance_array_key in request:
                    cleft_distances += self.add_constant
                if self.presyn_distance_array_key is not None and self.presyn_distance_array_key in request:
                    presyn_distances += self.add_constant

                if self.postsyn_distance_array_key is not None and self.postsyn_distance_array_key in request:
                    postsyn_distances += self.add_constant

            if self.cleft_distance_array_key is not None and self.cleft_distance_array_key in request:
                cleft_mask_total = self.__constrain_distances(cleft_mask_total, cleft_distances, self.spec[
                    self.cleft_mask_array_key].voxel_size)
            if self.presyn_distance_array_key is not None and self.presyn_distance_array_key in request:
                pre_mask_total = self.__constrain_distances(pre_mask_total, presyn_distances, self.spec[
                    self.presyn_mask_array_key].voxel_size)
            if self.postsyn_distance_array_key is not None and self.postsyn_distance_array_key in request:
                post_mask_total = self.__constrain_distances(post_mask_total, postsyn_distances, self.spec[
                    self.postsyn_mask_array_key].voxel_size)

            if (self.cleft_distance_array_key is not None and
                self.cleft_distance_array_key in request):
                cleft_spec = self.spec[self.cleft_distance_array_key].copy()
                cleft_spec.roi = request[self.cleft_distance_array_key].roi
                batch.arrays[self.cleft_distance_array_key] = Array(cleft_distances, cleft_spec)
                batch.arrays[self.cleft_mask_array_key] = Array(cleft_mask_total, cleft_spec)


            if (self.presyn_distance_array_key is not None and
                    self.presyn_distance_array_key in request):
                #presyn_distances = np.expand_dims(presyn_distances, 0)
                pre_spec = self.spec[self.presyn_distance_array_key].copy()
                pre_spec.roi = request[self.presyn_distance_array_key].roi
                batch.arrays[self.presyn_distance_array_key] = Array(presyn_distances, pre_spec)
                batch.arrays[self.presyn_mask_array_key] = Array(pre_mask_total, pre_spec)

            if (self.postsyn_distance_array_key is not None and
                    self.postsyn_distance_array_key in request):

                #postsyn_distances = np.expand_dims(postsyn_distances, 0)
                post_spec = self.spec[self.postsyn_distance_array_key].copy()
                post_spec.roi = request[self.postsyn_distance_array_key].roi
                batch.arrays[self.postsyn_distance_array_key] = Array(postsyn_distances, post_spec)
                batch.arrays[self.postsyn_mask_array_key] = Array(post_mask_total, post_spec)

    @staticmethod
    def __signed_distance(label, **kwargs):
        # calculate signed distance transform relative to a binary label. Positive distance inside the object,
        # negative distance outside the object. This function estimates signed distance by taking the difference
        # between the distance transform of the label ("inner distances") and the distance transform of
        # the complement of the label ("outer distances"). To compensate for an edge effect, .5 (half a pixel's
        # distance) is added to the positive distances and subtracted from the negative distances.
        inner_distance = distance_transform_edt(binary_erosion(label, border_value=1,
                                                               structure=generate_binary_structure(label.ndim,
                                                                                                   label.ndim)),
                                                               **kwargs)
        outer_distance = distance_transform_edt(np.logical_not(label), **kwargs)
        result = inner_distance - outer_distance

        return result

    def __constrain_distances(self, mask, distances, mask_sampling):
        # remove elements from the mask where the label distances exceed the distance from the boundary

        tmp = np.zeros(np.array(mask.shape) + np.array((2,)*mask.ndim), dtype=mask.dtype)
        slices = tmp.ndim * (slice(1, -1), )
        tmp[slices] = mask
        boundary_distance = distance_transform_edt(binary_erosion(tmp,
                                                                  border_value=1,
                                                                  structure=generate_binary_structure(tmp.ndim,
                                                                                                      tmp.ndim)),
                                                                  sampling=mask_sampling)
        boundary_distance = boundary_distance[slices]
        if self.max_distance is not None:
            if self.add_constant is None:
                add = 0
            else:
                add = self.add_constant
            boundary_distance = self.__clip_distance(boundary_distance, (-self.max_distance-add, self.max_distance-add))

        mask_output = mask.copy()
        if self.max_distance is not None:
            logging.debug("Total number of masked in voxels before distance masking {0:}".format(np.sum(mask_output)))
            mask_output[(abs(distances) >= boundary_distance) *
                        (distances >= 0) *
                        (boundary_distance < self.max_distance - add)] = 0
            logging.debug("Total number of masked in voxels after postive distance masking {0:}".format(np.sum(
                mask_output)))
            mask_output[(abs(distances) >= boundary_distance + 1) *
                        (distances < 0) *
                        (boundary_distance + 1 < self.max_distance - add)] = 0
            logging.debug("Total number of masked in voxels after negative distance masking {0:}".format(np.sum(
                mask_output)))
        else:
            logging.debug("Total number of masked in voxels before distance masking {0:}".format(np.sum(mask_output)))
            mask_output[np.logical_and(abs(distances) >= boundary_distance, distances >= 0)] = 0
            logging.debug("Total number of masked in voxels after postive distance masking {0:}".format(np.sum(
                mask_output)))
            mask_output[np.logical_and(abs(distances) >= boundary_distance + 1, distances < 0)] = 0
            logging.debug("Total number of masked in voxels after negative distance masking {0:}".format(np.sum(
                mask_output)))
        return mask_output

    @staticmethod
    def __clip_distance(distances, max_distance):
        if not isinstance(max_distance, tuple):
            max_distance = (-max_distance, max_distance)
        distances = np.clip(distances, max_distance[0], max_distance[1])
        return distances

