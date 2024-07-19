import math
import numpy as np

def fexp(x, p):
    return np.sign(x) * (np.abs(x) ** p)


class SuperQuadrics:
    def __init__(self, size, shape, resolution=100):
        self.a1, self.a2, self.a3 = size
        self.e1, self.e2 = shape
        self.N = resolution
        self.x, self.y, self.z, self.eta, self.omega = self.sample_equal_distance_on_sq()

    def sq_surface(self, eta, omega):
        x = self.a1 * fexp(np.cos(eta), self.e1) * fexp(np.cos(omega), self.e2)
        y = self.a2 * fexp(np.cos(eta), self.e1) * fexp(np.sin(omega), self.e2)
        z = self.a3 * fexp(np.sin(eta), self.e1)
        return x, y, z

    def sample_equal_distance_on_sq(self):
        eta = np.linspace(-np.pi / 2, np.pi / 2, self.N)
        omega = np.linspace(-np.pi, np.pi, self.N)
        eta, omega = np.meshgrid(eta, omega)
        x, y, z = self.sq_surface(eta, omega)
        return x, y, z, eta, omega