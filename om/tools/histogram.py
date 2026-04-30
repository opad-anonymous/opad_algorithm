import numpy as np
from collections import Counter

from om.models.model import DiscreteDistributionModel


class ArrayHistogram(DiscreteDistributionModel):
    def __init__(self):
        self.histogram = Counter()
        self.shape = None
        self.dtype = None
        self.all_entries_count = 0

    def add_array(self, array):
        """
        Adds a numpy array to the histogram, increasing its count.

        Args:
        - array (np.ndarray): The numpy array to add to the histogram.
        """
        if self.shape is None:
            self.shape = array.shape
            self.dtype = array.dtype
        else:
            assert array.shape == self.shape
            assert array.dtype == self.dtype

        # Convert array to a hashable byte representation
        array_bytes = array.tobytes()
        self.histogram[array_bytes] += 1
        self.all_entries_count += 1

    def calc_normalization_factor(self):
        return self.all_entries_count

    def calc_unnormalized_prob(self, array):
        """
        Gets the count of a numpy array in the histogram. That is, just counts

        Args:
        - array (np.ndarray): The numpy array to query in the histogram.

        Returns:
        - count (int): The count of the array in the histogram.
        """
        array_bytes = array.tobytes()
        return self.histogram[array_bytes]

    def calc_neg_log_unnormalized_prob(self, x):
        return -np.log(self.calc_unnormalized_prob(x))

    def generate_all_states(self):
        """
        Gets all unique arrays currently in the histogram.

        Returns:
        - unique_arrays (List[np.ndarray]): A list of unique numpy arrays in the histogram.
        """
        # Convert the byte representation back to arrays
        return [np.frombuffer(key, dtype=self.dtype).reshape(self.shape) for key in self.histogram.keys()]

    def __repr__(self):
        result = "ArrayHistogram:\n"
        for key, count in self.histogram.items():
            array = np.frombuffer(key, dtype=self.dtype).reshape(self.shape)
            result += f"Array:\t{array}\t {count}\n"
        return result
