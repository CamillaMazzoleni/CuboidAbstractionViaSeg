import numpy as np
from mayavi import mlab
from superquadric_class import SuperQuadrics
import matplotlib.pyplot as plt
from PIL import Image
from tvtk.tools import visual
import os
from matplotlib.colors import Normalize
import json


def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


class Visualizer:
    def __init__(self):
        self.elements = {}
        self.fig = mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))

    def add_superquadric(self, name: str, scalings: np.array = np.array([1.0, 1.0, 1.0]),
                         exponents: np.array = np.array([2.0, 2.0]), translation: np.array = np.array([0.0, 0.0, 0.0]),
                         rotation: np.array = np.eye(3), color: np.array = np.array([0, 255, 0]),
                         resolution: int = 100, visible: bool = True):
        """Adds a superquadric mesh to the scene."""

        # Initialize the SuperQuadrics class
        sq = SuperQuadrics(scalings, exponents, resolution)
        vertices = np.stack((sq.x.flatten(), sq.y.flatten(), sq.z.flatten()), axis=-1)

        triangles = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                triangles.append([i * resolution + j, (i + 1) * resolution + j + 1, (i + 1) * resolution + j])
                triangles.append([i * resolution + j, i * resolution + j + 1, (i + 1) * resolution + j + 1])

        vertices = np.array(vertices)
        triangles = np.array(triangles)

        # Apply color
        color = color / 255.0  # Normalize the color

        # Create a Mayavi mesh
        mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles,
                                    color=tuple(color), representation='surface')

        # Apply rotation and translation
        rotated_vertices = np.dot(vertices, rotation.T)
        mesh.mlab_source.set(x=rotated_vertices[:, 0] + translation[0],
                             y=rotated_vertices[:, 1] + translation[1],
                             z=rotated_vertices[:, 2] + translation[2])
        
        self.elements[name] = mesh

        return mesh, vertices


    def show(self):
        mlab.show()

def main():
    output_folder = 'superquadric_views'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    visualizer = Visualizer()
    data = load_json('/Users/cmazzoleni/Documents/GitHub/Cuboidabstractionviaseg/json_files/ffe1c487f7b9909bfebad4f49b26ec52.json')

    for i, component in enumerate(data["components"]):
        scale = np.array(component["scale"])
        rotation = np.array(component["rotation"])
        position = np.array(component["position"])
        epsilon1, epsilon2 = component["epsilon1"], component["epsilon2"]
        exponents = np.array([epsilon1, epsilon2])
        visualizer.add_superquadric(f"sq_{i}", scalings=scale, exponents=exponents, translation=position, rotation=rotation, color=np.array([0, 255, 0]))

    visualizer.show()

if __name__ == "__main__":
    main()