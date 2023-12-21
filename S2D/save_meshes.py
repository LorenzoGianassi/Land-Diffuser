import numpy as np  # Import the numpy library for numerical operations
import trimesh  # Import the trimesh library for working with 3D meshes
import _pickle as pickle  # Import the _pickle library for pickling and unpickling data
import os  # Import the os library for file and directory operations

def save_meshes(predictions, save_path_Meshes, n_meshes, template_path):
    """
    Save 3D mesh predictions as PLY files.

    Args:
    - predictions: A list of predicted 3D meshes.
    - save_path_Meshes: The directory where the PLY files will be saved.
    - n_meshes: The number of meshes to save.
    - template_path: The path to a template mesh used for defining the mesh structure.

    This function loads a template 3D mesh, and for each prediction in the list, it creates a
    trimesh object and exports it as a PLY file in the specified directory.
    """
    tri = trimesh.load(template_path, process=False)  # Load the template mesh
    triangles = tri.faces  # Extract the faces (triangles) from the template mesh

    for i in range(n_meshes):
        # Create a trimesh object from the prediction and the template faces
        tri_mesh = trimesh.Trimesh(np.asarray(np.squeeze(predictions[i])), np.asarray(triangles),process=False)

        # Export the trimesh object as a PLY file with a numbered filename
        tri_mesh.export(os.path.join(save_path_Meshes, "tst{0:03}.ply".format(i)))
