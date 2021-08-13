# SYSTEM IMPORTS
from abc import ABC, abstractmethod
import numpy as np
import pymm


# PYTHON PROJECT IMPORTS
from .posterior import Posterior


class LazyPymmPosterior(Posterior):
    def __init__(self,
                 num_params: int,
                 shelf,
                 K: int = 15,
                 dtype: np.dtype = np.float32) -> None:
        super().__init__(num_params, dtype=dtype)
        self.shelf = shelf
        self.shelf.K = K

        self.shelf.mu = pymm.ndarray((self.num_params,1), dtype=self.dtype)
        self.shelf.sec_moment_uncentered = pymm.ndarray((self.num_params, 1),
                                                              dtype=self.dtype)

        self.shelf.diag = pymm.ndarray((self.num_params, 1), dtype=self.dtype)
        # self.shelf.cov = self.shelf.ndarray((self.num_params, self.num_params),
        #                                     dtype=self.dtype)
        self.shelf.D_hat = pymm.ndarray((self.num_params, K), dtype=self.dtype)
        self.shelf.D_hat_start = 0

        self.shelf.num_samples = 0

    def update(self,
               theta: np.ndarray) -> None:
        theta = theta.reshape(-1,1)
        self.shelf.num_samples += 1
        self.shelf.mu += theta

        self.shelf.sec_moment_uncentered += theta**2

        self.shelf.D_hat[:,self.D_hat_start] = (theta-(self.sec_moment_uncentered/self.num_samples)).reshape(-1)
        self.shelf.D_hat_start = (self.D_hat_start + 1) % self.K

    def finalize(self) -> None:
        self.mu /= self.num_samples
        self.diag = self.sec_moment_uncentered - (self.mu**2)
        # self.cov = (self.D_hat @ self.D_hat.T) / (self.K - 1)

    @property
    def nbytes(self) -> int:
        return self.shelf.mu.nbytes + self.shelf.sec_moment_uncentered.nbytes +\
               self.shelf.diag.nbytes + self.shelf.D_hat.nbytes + 4 + 4 + 4

    def sample(self) -> np.ndarray:
        z1: np.ndarray = np.random.normal(loc=0, scale=1,
                                          size=(self.num_params,1))
        z2: np.ndarray = np.random.normal(loc=0, scale=1,
                                          size=(self.shelf.K, 1))

        return self.shelf.mu + (1 / np.sqrt(2)) * np.sqrt(self.shelf.diag) * z1 +\
               (1 / (np.sqrt(2 * (self.shelf.K-1)))) * self.shelf.D_hat.dot(z2)
