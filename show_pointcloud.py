import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_and_visualize_npy(file_path):
    # Load the .npy file
    points = np.load(file_path)
    
    # Check the shape of the loaded points
    points = np.load(file_path)
    
    # Check the shape of the loaded points
    if points.shape[1] < 3:
        raise ValueError("Expected point cloud data with at least 3 columns for x, y, z coordinates, but got shape {}".format(points.shape))
    
    # Extract x, y, z coordinates (assuming they are the first three columns)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.set_title('3D Scatter Plot of Point Cloud')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    
    # Add a color bar which maps values to colors
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    
    # Show the plot
    plt.show()

# Example usage
file_path = '/Users/cmazzoleni/Documents/GitHub/Cuboidabstractionviaseg/ShapeNetNormal4096/table/ffe1c487f7b9909bfebad4f49b26ec52.npy'
load_and_visualize_npy(file_path)
