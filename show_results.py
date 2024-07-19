import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
from superquadric_class import SuperQuadrics
from PIL import Image
import os
import json


def load_and_visualize_npy(file_path):
    points = np.load(file_path)
    
    if points.shape[1] < 3:
        raise ValueError("Expected point cloud data with at least 3 columns for x, y, z coordinates, but got shape {}".format(points.shape))
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.set_title('3D Scatter Plot of Point Cloud')
    
   
    ax.view_init(elev=-45, azim=90, roll=0)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig("pointcloud.png")
    plt.show()

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

class Visualizer:
    def __init__(self):
        self.elements = {}
        self.fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))

    def add_superquadric(self, name: str, scalings: np.array = np.array([1.0, 1.0, 1.0]),
                         exponents: np.array = np.array([2.0, 2.0]), translation: np.array = np.array([0.0, 0.0, 0.0]),
                         rotation: np.array = np.eye(3), color: np.array = np.array([0, 255, 0]),
                         resolution: int = 100, visible: bool = True):
        sq = SuperQuadrics(scalings, exponents, resolution)
        vertices = np.stack((sq.x.flatten(), sq.y.flatten(), sq.z.flatten()), axis=-1)

        triangles = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                triangles.append([i * resolution + j, (i + 1) * resolution + j + 1, (i + 1) * resolution + j])
                triangles.append([i * resolution + j, i * resolution + j + 1, (i + 1) * resolution + j + 1])

        vertices = np.array(vertices)
        triangles = np.array(triangles)
        color = color / 255.0

        mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles,
                                    color=tuple(color), representation='surface')

        rotated_vertices = np.dot(vertices, rotation.T)
        mesh.mlab_source.set(x=rotated_vertices[:, 0] + translation[0],
                             y=rotated_vertices[:, 1] + translation[1],
                             z=rotated_vertices[:, 2] + translation[2])
        
        self.elements[name] = mesh
        return mesh, vertices

    def save_visualization(self, filename):
        mlab.view(azimuth=0, elevation=0, distance=2, focalpoint = (0,0,0), figure=self.fig)
        screenshot = mlab.screenshot(figure=self.fig, mode='rgb', antialiased=True)
        plt.imsave(filename, screenshot)
        mlab.show()
        

def main():
    output_folder = 'superquadric_views'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = '/Users/cmazzoleni/Documents/GitHub/Cuboidabstractionviaseg/ShapeNetNormal4096/chair/1a8bbf2994788e2743e99e0cae970928.npy'
    load_and_visualize_npy(file_path)

    visualizer = Visualizer()
    data = load_json('/Users/cmazzoleni/Documents/GitHub/Cuboidabstractionviaseg/output_folder/1a8bbf2994788e2743e99e0cae970928.json')

    for i, component in enumerate(data["components"]):
        scale = np.array(component["scale"])
        rotation = np.array(component["rotation"])
        position = np.array(component["position"])
        epsilon1, epsilon2 = component["epsilon1"], component["epsilon2"]
        exponents = np.array([epsilon1, epsilon2])
        visualizer.add_superquadric(f"sq_{i}", scalings=scale, exponents=exponents, translation=position, rotation=rotation, color=np.array([0, 255, 0]))

    visualizer.save_visualization("superquadric.png")

    # Combine the images
    pointcloud_image = Image.open("pointcloud.png")
    superquadric_image = Image.open("superquadric.png")

    combined_image = Image.new('RGB', (pointcloud_image.width + superquadric_image.width, pointcloud_image.height))
    combined_image.paste(pointcloud_image, (0, 0))
    combined_image.paste(superquadric_image, (pointcloud_image.width, 0))
    combined_image.save("comparison.png")
    combined_image.show()

if __name__ == "__main__":
    main()
