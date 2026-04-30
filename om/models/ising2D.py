import numpy as np
import itertools
from om.models.model import DiscreteDistributionModel


class IsingModel2D(DiscreteDistributionModel):
    """
    Assuming that the interaction strength is constant and there is no external magnetic field
    """
    def __init__(self, N, J=1, h=0, beta=1.0):
        self.N = N  # Size of the lattice (N x N)
        self.J = J  # Interaction strength
        self.h = h  # external magnetic field (assuming that it is fixed)
        self.beta = beta  # Inverse temperature

    def calc_neg_log_unnormalized_prob(self, x):
        """
        Calculate the energy of a given N x N configuration, x (i.e. x is sigma).
        where H(sigma) = -sum_{i,j in {1..N}X{1..N}} (J * [ sigma_{i,j}sigma_{i+1, j} + sigma_{i,j}sigma_{i, j+1}])
                         -sum_{i,j in {1..N}X{1..N}} mu * (h * sigma_{i,j})  # assuming mu=1 and h=h_{i,j} is fixed
        """
        mu=1

        energy = 0
        for i in range(self.N):
            for j in range(self.N):
                s_ij = x[i, j]
                assert s_ij in {-1, 1}
                # Periodic boundary conditions
                s_right = x[i, (j + 1) % self.N]
                s_down = x[(i + 1) % self.N, j]
                energy -= self.J * s_ij * (s_right + s_down)  # interaction term
                energy -= mu * self.h * s_ij

        energy *= self.beta
        return energy

    def generate_init_state(self):
        return np.ones((self.N, self.N))

    def generate_all_states(self):
        """Generate all possible configurations for an N x N system."""
        all_states = itertools.product([-1, 1], repeat=self.N * self.N)
        return [np.reshape(np.array(state), (self.N, self.N)) for state in all_states]

