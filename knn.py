import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import models
from PIL import Image
import pandas as pd
import logging
#import optuna
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info("Using device: %s", device)

# Define categories and image size
categories = ['7-malignant-bcc', '1-benign-melanocytic nevus', '6-benign-other',
              '14-other-non-neoplastic/inflammatory/infectious', '8-malignant-scc',
              '9-malignant-sccis', '10-malignant-ak', '3-benign-fibrous papule',
              '4-benign-dermatofibroma', '2-benign-seborrheic keratosis',
              '5-benign-hemangioma', '11-malignant-melanoma',
              '13-other-melanocytic lesion with possible re-excision (severe/spitz nevus, aimp)',
              '12-malignant-other']
img_size = 224  # Image size set to 224

# Compose the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the augmentation pipeline
augmentation_transforms = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
])

class CustomImageDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        print("Loading image paths...")
        self.image_paths = self._get_image_paths()
        print("Image paths loaded. Total images: %d" % len(self.image_paths))

    def _get_image_paths(self):
        valid_paths = []
        for root_dir in self.root_dirs:
            for label_dir in os.listdir(root_dir):
                label_path = os.path.join(root_dir, label_dir)
                if os.path.isdir(label_path):
                    label = int(label_dir)  # Assuming folder names are integers corresponding to labels
                    for file in os.listdir(label_path):
                        if file.endswith(".png"):
                            img_path = os.path.join(label_path, file)
                            valid_paths.append((img_path, label))
        logging.info("Total valid paths found: %d", len(valid_paths))
        return valid_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

class AugmentedImageDataset(Dataset):
    def __init__(self, original_dataset, augmented_dirs, transform=None):
        self.original_dataset = original_dataset
        self.augmented_dirs = augmented_dirs
        self.transform = transform
        print("Loading augmented image paths...")
        self.augmented_paths = self._get_augmented_paths()
        print("Augmented image paths loaded. Total augmented images: %d" % len(self.augmented_paths))

    def _get_augmented_paths(self):
        augmented_paths = []
        for augmented_dir in self.augmented_dirs:
            for root, _, files in os.walk(augmented_dir):
                for file in files:
                    if file.endswith(".png"):
                        img_path = os.path.join(root, file)
                        label = int(os.path.basename(root))
                        augmented_paths.append((img_path, label))
        return augmented_paths

    def __len__(self):
        return len(self.original_dataset) + len(self.augmented_paths)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            img_path, label = self.augmented_paths[idx - len(self.original_dataset)]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)

def extract_features(model, dataloader):
    features = []
    print("Extracting features...")
    total_images = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            output = model(images.to(device))
            features.append(output.squeeze().cpu().numpy())
            
            # Print progress every 500 images
            if (batch_idx + 1) * batch_size % 500 == 0 or (batch_idx + 1) * batch_size == total_images:
                progress = (batch_idx + 1) * batch_size
                print(f"Processed {progress} images out of {total_images}")

    print("Feature extraction complete. Total features: %d" % len(features))
    return np.vstack(features)


def reduce_dimensionality(features, n_components=50):
    print("Reducing dimensionality...")
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    print("Dimensionality reduction complete. Reduced features shape: %s" % str(reduced_features.shape))
    return reduced_features

def initialize_clusters(reduced_features, n_clusters):
    print("Initializing clusters...")
    random_indices = random.sample(range(len(reduced_features)), n_clusters)
    centroids = reduced_features[random_indices]
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(centroids)
    _, labels = knn.kneighbors(reduced_features)
    print("Clusters initialized.")
    return labels.flatten(), centroids

def update_centroids(clustered_images, reduced_features, n_clusters):
    print("Updating centroids...")
    new_centroids = []
    for i in range(n_clusters):
        cluster_indices = clustered_images[i]
        if len(cluster_indices) > 0:
            cluster_features = reduced_features[cluster_indices]
            new_centroids.append(cluster_features.mean(axis=0))
        else:
            new_centroids.append(np.zeros(reduced_features.shape[1]))
    print("Centroids updated.")
    return np.array(new_centroids)

def knn_clustering(reduced_features, n_clusters, max_iterations=10):
    print("Performing KNN clustering...")
    labels, centroids = initialize_clusters(reduced_features, n_clusters)
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        clustered_images = {i: [] for i in range(n_clusters)}
        for i, label in enumerate(labels):
            clustered_images[label].append(i)

        new_centroids = update_centroids(clustered_images, reduced_features, n_clusters)
        if np.allclose(centroids, new_centroids):
            print("Convergence reached.")
            break
        centroids = new_centroids

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(centroids)
        _, labels = knn.kneighbors(reduced_features)
        labels = labels.flatten()

    print("KNN clustering complete.")
    return clustered_images, centroids, labels

def save_clusters(dataloader, labels, output_cluster_dir):
    print("Saving clusters...")
    if not os.path.exists(output_cluster_dir):
        os.makedirs(output_cluster_dir)

    batch_idx = 0
    for images, _ in dataloader:
        for i, img in enumerate(images):
            cluster_label = labels[batch_idx * len(images) + i]
            cluster_dir = os.path.join(output_cluster_dir, f"cluster_{cluster_label}")
            if not os.path.exists(cluster_dir):
                os.makedirs(cluster_dir)
            
            img_path = os.path.join(cluster_dir, f"img_{batch_idx}_{i}.png")
            save_image(img, img_path)
        batch_idx += 1

    print(f"Clusters saved to {output_cluster_dir}")

if __name__ == "__main__":
    # Step 1: Load datasets
    root_dirs = ["images1", "images2", "images3", "images4"]
    augmented_dirs = ["augmented_images", "augmented_images1"]

    print("Loading original dataset...")
    original_dataset = CustomImageDataset(root_dirs, transform=transform)
    print("Original dataset loaded.")

    print("Loading augmented dataset...")
    augmented_dataset = AugmentedImageDataset(original_dataset, augmented_dirs, transform=transform)
    dataloader = DataLoader(augmented_dataset, batch_size=32, shuffle=False)
    print("Augmented dataset loaded.")

    # Step 2: Feature extraction using a pre-trained ResNet18 model
    print("Loading pre-trained ResNet18 model...")
    resnet = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(resnet.children())[:-1])  # Remove final classification layer
    model.to(device)
    model.eval()

    print("Extracting features...")
    features = extract_features(model, dataloader)

    # Step 3: Dimensionality reduction using PCA
    reduced_features = reduce_dimensionality(features)

    # Step 4: Perform KNN clustering
    n_clusters = 10
    print(f"Performing KNN clustering with {n_clusters} clusters...")
    clustered_images, centroids, labels = knn_clustering(reduced_features, n_clusters)

    # Step 5: Save the clusters
    output_cluster_dir = "./clusters"
    save_clusters(dataloader, labels, output_cluster_dir)
