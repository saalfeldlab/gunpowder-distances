import logging
import numpy as np

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class TanhSaturate(BatchFilter):
    '''Saturate the values of an array to be floats between -1 and 1 by applying the tanh function.

    Args:

        array (:class:`ArrayKey`):

            The key of the array to modify.

        factor (scalar, optional):

            The factor to divide by before applying the tanh, controls how quickly the values saturate to -1, 1.
    '''

    def __init__(self, array, scale=None):

        self.array = array
        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1.

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]

        array.data = np.tanh(array.data/self.scale)
