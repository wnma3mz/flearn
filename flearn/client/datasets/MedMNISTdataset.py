import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/dataset.py
INFO = {
    "pathmnist": {
        "description": "PathMNIST: A dataset based on a prior study for predicting survival from colorectal cancer histology slides, which provides a dataset NCT-CRC-HE-100K of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset CRC-VAL-HE-7K of 7,180 image patches from a different clinical center. 9 types of tissues are involved, resulting a multi-class classification task. We resize the source images of 3 x 224 x 224 into 3 x 28 x 28, and split NCT-CRC-HE-100K into training and valiation set with a ratio of 9:1.",
        "url": "https://zenodo.org/record/4269852/files/pathmnist.npz?download=1",
        "MD5": "a8b06965200029087d5bd730944a56c1",
        "task": "multi-class",
        "label": {
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium",
        },
        "n_channels": 3,
        "n_samples": {"train": 89996, "val": 10004, "test": 7180},
        "license": "CC BY 4.0",
    },
    "chestmnist": {
        "description": "ChestMNIST: A dataset based on NIH-ChestXray14 dataset, comprising 112,120 frontal-view X-ray images of 30,805 unique patients with the text-mined 14 disease image labels, which could be formulized as multi-label binary classification task. We use the official data split, and resize the source images of 1 x 1024 x 1024 into 1 x 28 x 28.",
        "url": "https://zenodo.org/record/4269852/files/chestmnist.npz?download=1",
        "MD5": "02c8a6516a18b556561a56cbdd36c4a8",
        "task": "multi-label, binary-class",
        "label": {
            "0": "atelectasis",
            "1": "cardiomegaly",
            "2": "effusion",
            "3": "infiltration",
            "4": "mass",
            "5": "nodule",
            "6": "pneumonia",
            "7": "pneumothorax",
            "8": "consolidation",
            "9": "edema",
            "10": "emphysema",
            "11": "fibrosis",
            "12": "pleural",
            "13": "hernia",
        },
        "n_channels": 1,
        "n_samples": {"train": 78468, "val": 11219, "test": 22433},
        "license": "CC0 1.0",
    },
    "dermamnist": {
        "description": "DermaMNIST: A dataset based on HAM10000, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images labeled as 7 different categories, as a multi-class classification task. We split the images into training, validation and test set with a ratio of 7:1:2. The source images of 3 x 600 x 450 are resized into 3 x 28 x 28.",
        "url": "https://zenodo.org/record/4269852/files/dermamnist.npz?download=1",
        "MD5": "0744692d530f8e62ec473284d019b0c7",
        "task": "multi-class",
        "label": {
            "0": "actinic keratoses and intraepithelial carcinoma",
            "1": "basal cell carcinoma",
            "2": "benign keratosis-like lesions",
            "3": "dermatofibroma",
            "4": "melanoma",
            "5": "melanocytic nevi",
            "6": "vascular lesions",
        },
        "n_channels": 3,
        "n_samples": {"train": 7007, "val": 1003, "test": 2005},
        "license": "CC BY-NC 4.0",
    },
    "octmnist": {
        "description": "OCTMNIST: A dataset based on a prior dataset of 109,309 valid optical coherence tomography (OCT) images for retinal diseases. 4 types are involved, leading to a multi-class classification task. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are single-channel, and their sizes range from (384-1,536) x (277-512). We center-crop the images and resize them into 1 x 28 x 28.",
        "url": "https://zenodo.org/record/4269852/files/octmnist.npz?download=1",
        "MD5": "c68d92d5b585d8d81f7112f81e2d0842",
        "task": "multi-class",
        "label": {
            "0": "choroidal neovascularization",
            "1": "diabetic macular edema",
            "2": "drusen",
            "3": "normal",
        },
        "n_channels": 1,
        "n_samples": {"train": 97477, "val": 10832, "test": 1000},
        "license": "CC BY 4.0",
    },
    "pneumoniamnist": {
        "description": "PneumoniaMNIST: A dataset based on a prior dataset of 5,856 pediatric chest X-ray images. The task is binary-class classification of pneumonia and normal. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are single-channel, and their sizes range from (384-2,916) x (127-2,713). We center-crop the images and resize them into 1 x 28 x 28.",
        "url": "https://zenodo.org/record/4269852/files/pneumoniamnist.npz?download=1",
        "MD5": "28209eda62fecd6e6a2d98b1501bb15f",
        "task": "binary-class",
        "label": {"0": "normal", "1": "pneumonia"},
        "n_channels": 1,
        "n_samples": {"train": 4708, "val": 524, "test": 624},
        "license": "CC BY 4.0",
    },
    "retinamnist": {
        "description": "RetinaMNIST: A dataset based on DeepDRiD, a dataset of 1,600 retina fundus images. The task is ordinal regression for 5-level grading of diabetic retinopathy severity. We split the source training set with a ratio of 9:1 into training and validation set, and use the source validation set as test set. The source images of 3 x 1,736 x 1,824 are center-cropped and resized into 3 x 28 x 28",
        "url": "https://zenodo.org/record/4269852/files/retinamnist.npz?download=1",
        "MD5": "bd4c0672f1bba3e3a89f0e4e876791e4",
        "task": "ordinal regression",
        "label": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
        "n_channels": 3,
        "n_samples": {"train": 1080, "val": 120, "test": 400},
        "license": "CC BY 4.0",
    },
    "breastmnist": {
        "description": "BreastMNIST: A dataset based on a dataset of 780 breast ultrasound images. It is categorized into 3 classes: normal, benign and malignant. As we use low-resolution images, we simplify the task into binary classification by combing normal and benign as negative, and classify them against malignant as positive. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images of 1 x 500 x 500 are resized into 1 x 28 x 28.",
        "url": "https://zenodo.org/record/4269852/files/breastmnist.npz?download=1",
        "MD5": "750601b1f35ba3300ea97c75c52ff8f6",
        "task": "binary-class",
        "label": {"0": "malignant", "1": "normal, benign"},
        "n_channels": 1,
        "n_samples": {"train": 546, "val": 78, "test": 156},
        "license": "CC BY 4.0",
    },
    "organmnist_axial": {
        "description": "OrganMNIST_Axial: A dataset based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into grey scale with a abdominal window; we then crop 2D images from the center slices of the 3D bounding boxes in axial views (planes). The images are resized into 1 x 28 x 28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url": "https://zenodo.org/record/4269852/files/organmnist_axial.npz?download=1",
        "MD5": "866b832ed4eeba67bfb9edee1d5544e6",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen",
        },
        "n_channels": 1,
        "n_samples": {"train": 34581, "val": 6491, "test": 17778},
        "license": "CC BY 4.0",
    },
    "organmnist_coronal": {
        "description": "OrganMNIST_Coronal: A dataset based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into grey scale with a abdominal window; we then crop 2D images from the center slices of the 3D bounding boxes in coronal views (planes). The images are resized into 1 x 28 x 28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url": "https://zenodo.org/record/4269852/files/organmnist_coronal.npz?download=1",
        "MD5": "0afa5834fb105f7705a7d93372119a21",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen",
        },
        "n_channels": 1,
        "n_samples": {"train": 13000, "val": 2392, "test": 8268},
        "license": "CC BY 4.0",
    },
    "organmnist_sagittal": {
        "description": "OrganMNIST_Sagittal: A dataset based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into grey scale with a abdominal window; we then crop 2D images from the center slices of the 3D bounding boxes in sagittal views (planes). The images are resized into 1 x 28 x 28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url": "https://zenodo.org/record/4269852/files/organmnist_sagittal.npz?download=1",
        "MD5": "e5c39f1af030238290b9557d9503af9d",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen",
        },
        "n_channels": 1,
        "n_samples": {"train": 13940, "val": 2452, "test": 8829},
        "license": "CC BY 4.0",
    },
}


class MedMNIST(Dataset):

    flag = ...

    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False
    ):
        """dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation

        """
        super(MedMNIST, self).__init__()

        self.info = INFO[self.flag]
        self.root = root

        if download:
            self.download()

        if not os.path.exists(os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split == "train":
            self.img = npz_file["train_images"]
            self.label = npz_file["train_labels"]
        elif self.split == "val":
            self.img = npz_file["val_images"]
            self.label = npz_file["val_labels"]
        elif self.split == "test":
            self.img = npz_file["test_images"]
            self.label = npz_file["test_labels"]

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.img.shape[0]

    def __repr__(self):
        """Adapted from torchvision."""
        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__

        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        body.append("Split: {}".format(self.split))
        body.append("Task: {}".format(self.info["task"]))
        body.append("Number of channels: {}".format(self.info["n_channels"]))
        body.append("Meaning of labels: {}".format(self.info["label"]))
        body.append("Number of samples: {}".format(self.info["n_samples"]))
        body.append("Description: {}".format(self.info["description"]))
        body.append("License: {}".format(self.info["license"]))

        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url

            download_url(
                url=self.info["url"],
                root=self.root,
                filename="{}.npz".format(self.flag),
                md5=self.info["MD5"],
            )
        except:
            raise RuntimeError(
                "Something went wrong when downloading! "
                + "Go to the homepage to download manually. "
                + "https://github.com/MedMNIST/MedMNIST"
            )


class PathMNIST(MedMNIST):
    flag = "pathmnist"


class OCTMNIST(MedMNIST):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST):
    flag = "chestmnist"


class DermaMNIST(MedMNIST):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST):
    flag = "retinamnist"


class BreastMNIST(MedMNIST):
    flag = "breastmnist"


class OrganMNISTAxial(MedMNIST):
    flag = "organmnist_axial"


class OrganMNISTCoronal(MedMNIST):
    flag = "organmnist_coronal"


class OrganMNISTSagittal(MedMNIST):
    flag = "organmnist_sagittal"
