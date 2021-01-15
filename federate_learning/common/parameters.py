from typing import List

import numpy as np

NDArrayList = List[np.ndarray]
ValueList = List[list]


class Parameters(object):
    def __init__(self,
                 ndarray_list: NDArrayList = None,
                 list_of_values: ValueList = None
                 ):
        if ndarray_list:
            self.weights = ndarray_list
        else:
            self.weights = [np.array(l) for l in list_of_values]

    def to_list(self):
        return [w.tolist() for w in self.weights]

    def to_ndarray(self):
        return self.weights

    def get_shape(self):
        return [w.shape for w in self.weights]
