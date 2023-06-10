import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision import transforms

import h5py

SYMSOL_I = {"tet", "cube", "icosa", "cone", "cyl"}
SYMSOL_II = {"tetX", "cylO", "sphereX"}


class BottleCapDataset(Dataset):
    # Adapted from: https://www.tensorflow.org/datasets/catalog/symmetric_solids.
    # Args:
    #   data_dir: str, path to directory containing dataset (e.g. ../bingham/bingham_vis/bc_ds_h5).
    #   dataset: str, name of dataset (e.g. Arrow_cube_data.hdf5).
    #   subset: list of str, subset of shapes to use (e.g. ["Arrow", "Diamond"]).
    #   neg_samples: int, number of negative samples to use per image.
    def __init__(self, data_dir, subset, neg_samples, preprocess=True, type="train"):
        assert neg_samples > 0

        # Location of the h5 files
        self.data_dir = data_dir + "/" + type

        # Set of files to use (e.g. "Diamond")
        self.objects = sorted(subset)

        # Fule file names
        self.object_file_names = [f"{obj}_cube_data.hdf5" for obj in self.objects]

        # Number of files
        self.n_objects = len(subset)

        # Load the first dataset by default
        self.datasets = None

        self.components = ["Bottle", "Cap"]
        
        # Number of images per component'

        with h5py.File(f"{self.data_dir}/{self.object_file_names[0]}", "r") as f:
            self.n_images = len(f["Bottle"]['images'])

            # Number of bottle components (probably 2--bottle and cap)
            self.n_components = len(f.keys())

        # Total number of images in the dataset
        self.length = self.n_objects * self.n_components * self.n_images


        # print("Length is: ", self.length)
        # print("n_objects is: ", self.n_objects)
        # print("n_components is: ", self.n_components)
        # print("n_images is: ", self.n_images)

        self.neg_samples = neg_samples

        if preprocess:
            self.preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.preprocess = transforms.ToTensor()

    
    def idx_to_local(self, idx):
        l_file = self.length // self.n_objects
        l_component = self.length // (self.n_objects * self.n_components)

        # Current file
        file_idx = idx // l_file
        # Component index
        component_idx = (idx % l_file) // l_component

        image_idx = (idx % l_file) % l_component

        return (file_idx, component_idx, image_idx)



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # If it hasn't been opened yet, open the dataset
        if self.datasets is None:
            self.datasets = [h5py.File(f"{self.data_dir}/{self.object_file_names[i]}", "r") for i in range(self.n_objects)]
        
        # Get the correct indices
        file_idx, component_idx, image_idx = self.idx_to_local(idx)
        # print("File idx is: ", file_idx)
        # print("Component idx is: ", component_idx)
        # print("Image idx is: ", image_idx)

        # Get the image and rotation
        img = self.datasets[file_idx][self.components[component_idx]]["images"][image_idx]
        # print("Angles length is: ", self.datasets[file_idx][self.components[component_idx]]["angles"][0].shape)
        R = self.datasets[file_idx][self.components[component_idx]]["angles"][image_idx]

        img = self.preprocess(img)

        fake_Rs = Rotation.random(self.neg_samples).as_matrix()
        R_fake_Rs = np.concatenate([R[None], fake_Rs])

        return (img, R_fake_Rs.reshape(-1, 9))
