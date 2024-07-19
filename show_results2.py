import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
from superquadric_class import SuperQuadrics
from PIL import Image
import os
import json
import open3d as o3d
from os import walk
import random
from enum import Enum

class ObjectClass(Enum):
    CHAIR = 'chair'
    TABLE = 'table'
    AIRPLANE = 'airplane'


def load_npy(file_path):
    return np.load(file_path)

def load_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points), np.asarray(pcd.colors)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def visualize_pointcloud(points, title, save_path, elev=20, azim=30):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.set_title(title)
    #remove axis
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    #remove grid
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)  # Set consistent view angle
    plt.savefig(save_path)
    plt.close()

class Visualizer:
    def __init__(self):
        self.fig = mlab.figure(size=(550, 550), bgcolor=(1, 1, 1))

    def add_superquadric(self, scale, rotation, translation, exponents, color):
        sq = SuperQuadrics(scale, exponents, 100)
        vertices = np.stack((sq.x.flatten(), sq.y.flatten(), sq.z.flatten()), axis=-1)
        triangles = []
        for i in range(99):
            for j in range(99):
                triangles.append([i * 100 + j, (i + 1) * 100 + j + 1, (i + 1) * 100 + j])
                triangles.append([i * 100 + j, i * 100 + j + 1, (i + 1) * 100 + j + 1])
        vertices = np.array(vertices)
        triangles = np.array(triangles)
        mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles, color=color)
        rotated_vertices = np.dot(vertices, rotation.T)
        mesh.mlab_source.set(x=rotated_vertices[:, 0] + translation[0],
                             y=rotated_vertices[:, 1] + translation[1],
                             z=rotated_vertices[:, 2] + translation[2])
        return mesh

    def save_visualization(self, save_path, elev=0, azim=0):
        mlab.view(azimuth=azim, elevation=elev, distance=2, focalpoint = (0,0,0), figure=self.fig)
        screenshot = mlab.screenshot(figure=self.fig, mode='rgb', antialiased=True)
        
        plt.imsave(save_path, screenshot)
        

def visualize_cuboids(json_data, save_path, elev=20, azim=30):
    visualizer = Visualizer()
    for component in json_data['components']:
        scale = np.array(component['scale'])
        rotation = np.array(component['rotation'])
        translation = np.array(component['position'])
        exponents = np.array([component['epsilon1'], component['epsilon2']])
        visualizer.add_superquadric(scale, rotation, translation, exponents, color=(0, 1, 0))
    visualizer.save_visualization(save_path, elev=elev, azim=azim)

def visualize_ply(file_path, title, save_path, elev=20, azim=30):
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    points = np.asarray(pcd.points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=elev, azim=azim)  # Set consistent view angle
    plt.savefig(save_path)
    plt.close()

def main():
    object_class = ObjectClass.TABLE
    input_init_path = f'/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/ShapeNetNormal4096/{object_class.value}/'
    base_init_path = f'/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/output_folder/{object_class.value}/'
    base_result_folder = f'/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/result_folder/{object_class.value}/'
    combined_result_folder = f'/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/result_folder/{object_class.value}/combined/'
    print (base_init_path)
    f = []
    for (dirpath, dirnames, filenames) in walk(base_init_path):
        f.extend(filenames)
    
    #print(f)
    #select 10 random objects from the list f
    id_list = []
    for i in range (10):
        id_list.append(random.choice(f))

    object_id = '1a8bbf2994788e2743e99e0cae970928'

    #do this for all objects in id_list
    for i, object_id in enumerate(id_list):
        print (object_id)
        if object_id.endswith('.json'):
            object_id = object_id[:-5]
        input_path = os.path.join(input_init_path, f'{object_id}.npy')
        segment_path = os.path.join(base_init_path, f'{object_id}_cubepred.ply')
        cuboids_path = os.path.join(base_init_path, f'{object_id}.json')

        input_points = load_npy(input_path)
        cuboids_data = load_json(cuboids_path)
        segment_points, segment_colors = load_ply(segment_path)
        # Set consistent view angles
        elev = 180
        azim = 0

        visualize_pointcloud(input_points, 'Input Point Cloud', os.path.join(base_result_folder,f'input{i}.png'), elev=-90, azim=90)
        #visualize_ply(segment_path, 'Segmented Point Cloud', 'segment.png', elev= elev, azim= azim)
        visualize_cuboids(cuboids_data, os.path.join(base_result_folder,f'cuboids{i}.png'), elev= elev, azim= azim)

        input_image = Image.open(os.path.join(base_result_folder,f'input{i}.png'))
        #segment_image = Image.open('segment.png')
        cuboids_image = Image.open(os.path.join(base_result_folder,f'cuboids{i}.png'))

        #width = input_image.width + segment_image.width + cuboids_image.width
        width = input_image.width + cuboids_image.width
        #height = max (input_image.height, segment_image.height, cuboids_image.height)
        height = max (input_image.height, cuboids_image.height)
        combined_image = Image.new('RGB', (width, height))
        combined_image.paste(input_image, (0, 0))
        #combined_image.paste(segment_image, (input_image.width, 0))
        #combined_image.paste(cuboids_image, (input_image.width + segment_image.width, 0))
        combined_image.paste(cuboids_image, (input_image.width, 0))
        combined_image.save(os.path.join(combined_result_folder,f'combined{i}.png'))
        combined_image.show()

if __name__ == "__main__":
    main()
