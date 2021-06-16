from .batch_filter import BatchFilter
from gunpowder.array import ArrayKeys, Array
import collections
import itertools
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BalanceGlobalByThreshold(BatchFilter):
    '''Creates a scale volume to balance the loss between positive and negative
    labels.

    Args:

        labels (:class:``VolumeType``): A volume containing binary labels.

        scales (:class:``VolumeType``): A volume with scales to be created. This
            new volume will have the same ROI and resolution as `labels`.

        mask (:class:``VolumeType``, optional): An optional mask (or list of
            masks) to consider for balancing. Every voxel marked with a 0 will
            not contribute to the scaling and will have a scale of 0 in
            `scales`.

        slab (tuple of int, optional): A shape specification to perform the
            balancing in slabs of this size. -1 can be used to refer to the
            actual size of the label volume. For example, a slab of::

                (2, -1, -1, -1)

            will perform the balancing for every each slice `(0:2,:)`,
            `(2:4,:)`, ... individually.
    '''

    def __init__(self, labels, scales, frac_pos, frac_neg, threshold=0, mask=None, slab=None):

        self.labels = labels
        self.scales = scales
        if mask is None:
            self.masks = []
        elif not isinstance(mask, collections.Iterable):
            self.masks = [mask]
        else:
            self.masks = mask

        self.slab = slab
        self.threshold = threshold
        self.frac_pos = frac_pos
        self.frac_neg = frac_neg

        self.skip_next = False

    def setup(self):

        assert self.labels in self.spec, (
            "Asked to balance labels %s, which are not provided."%self.labels)

        for mask in self.masks:
            assert mask in self.spec, (
                "Asked to apply mask %s to balance labels, but mask is not "
                "provided."%mask)

        spec = self.spec[self.labels].copy()
        spec.dtype = np.float32
        self.provides(self.scales, spec)
        self.enable_autoskip()

    def process(self, batch, request):

        if self.skip_next:
            self.skip_next = False
            return

        labels = batch.arrays[self.labels]

        # initialize error scale with 1s
        error_scale = np.ones(labels.data.shape, dtype=np.float32)

        # set error_scale to 0 in masked-out areas
        for identifier in self.masks:
            mask = batch.arrays[identifier]
            assert labels.data.shape == mask.data.shape, (
                "Shape of mask %s %s does not match %s %s"%(
                    mask,
                    mask.data.shape,
                    self.labels,
                    labels.data.shape))
            error_scale *= mask.data
        self.__balance(
                    labels.data,
                    error_scale,
                    self.frac_pos,
                    self.frac_neg
                )

        spec = self.spec[self.scales].copy()
        spec.roi = labels.spec.roi
        batch.arrays[self.scales] = Array(error_scale, spec)

    def __balance(self, labels, scale, frac_pos, frac_neg):

        labels = labels > self.threshold

        # compute the class weights for positive and negative samples
        w_pos = 1.0 / (2.0 * frac_pos)
        w_neg = 1.0 / (2.0 * frac_neg)

        # scale the masked-in scale with the class weights
        scale *= (labels >= 0.5) * w_pos + (labels < 0.5) * w_neg
