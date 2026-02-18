import torch
import rasterio as rio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from rasterio.io import MemoryFile
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision.transforms import v2 as transforms
from collections import Counter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
torch.set_printoptions(precision=6, sci_mode=False)
import time
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score


def calculate_optimal_offsets(image_path, patch_size, stride):
    """
    Calculate offsets to distribute leftover pixels evenly when tiling an image.

    Args:
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
        patch_size (int): Size of the square patches (in pixels).
        stride (int): Step size between patches (in pixels).

    Returns:
        (int, int): offset_left, offset_top
    """

    with rio.open(image_path) as img:
        img_width = img.width
        img_height = img.height

        if patch_size > img_width or patch_size > img_height:
            raise ValueError("Patch size cannot be larger than the image dimensions.")

        if stride <= 0:
            raise ValueError("Stride must be a positive integer.")

        leftover_width = img_width - ((img_width - patch_size) // stride * stride + patch_size)
        leftover_height = img_height - ((img_height - patch_size) // stride * stride + patch_size)

        offset_left = leftover_width // 2
        offset_top = leftover_height // 2
    
    return offset_left, offset_top

def match_rasters(raster_to_change_path, raster_path):
    raster_to_change = rioxarray.open_rasterio(raster_to_change_path, masked=True)
    raster = rioxarray.open_rasterio(raster_path, masked=True)
    raster_to_change = raster_to_change.drop_vars("band").squeeze()
    raster = raster.drop_vars("band").squeeze()

    def print_raster(raster):
        print(
            f"shape: {raster.rio.shape}\n"
            f"resolution: {raster.rio.resolution()}\n"
            f"bounds: {raster.rio.bounds()}\n"
            f"sum: {raster.sum().item()}\n"
            f"CRS: {raster.rio.crs}\n"
        )

    print("Matching this Raster:\n----------------\n")
    print_raster(raster_to_change)
    print("To this Raster:\n----------------\n")
    print_raster(raster)

    raster_matched = raster_to_change.rio.reproject_match(raster)

    print("Matched Raster:\n-------------------\n")
    print_raster(raster_matched)
    print("To this Raster:\n----------------\n")
    print_raster(raster)

    # for debug use
    #return raster_matched.rio.to_raster("debug.tif", driver="GTiff", compress="LZW")

    # Write the aligned raster to a rasterio.MemoryFile
    with MemoryFile() as memfile:
        with memfile.open(
                driver="GTiff",
                height=raster_matched.rio.shape[0],
                width=raster_matched.rio.shape[1],
                count=1,
                dtype=raster_matched.dtype,
                crs=raster_matched.rio.crs,
                transform=raster_matched.rio.transform(),
        ) as dataset:
            dataset.write(raster_matched.values, 1)

        return memfile.open()

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def get_normalization_parameters(training_loader):
    mean = 0
    std = 0
    total_pixels = 0

    # get mean and std for each batch in loader
    for patches, _ in training_loader:

        # flatten the patches tensor to (batch_size, C, H*W)
        pixels = patches.view(patches.size(0), patches.size(1), -1)

        mean += pixels.mean(dim=(0, 2)) * patches.size(0)
        std += pixels.std(dim=(0, 2)) * patches.size(0)

        total_pixels += patches.size(0)

    # get the resulting mean and std
    mean /= total_pixels
    std /= total_pixels

    return mean, std

def normalize_loader(original_loader, means, stds):

    for batch in original_loader:

        # loader with labels
        if len(batch) == 2:
            # create new Torch Dataset with labels
            class NormalizedDataset(Dataset):
                def __init__(self, original_loader, means, stds):
                    self.original_loader = original_loader
                    self.means = means
                    self.stds = stds
                    self.normalization = transforms.Normalize(mean=self.means, std=self.stds)

                def __len__(self):
                    return len(self.original_loader.dataset)

                def __getitem__(self, idx):
                    features, label = self.original_loader.dataset[idx]

                    norm_features = self.normalization(features)

                    return norm_features, label

        # loader without labels
        else:
            # create new Torch Dataset without labels
            class NormalizedDataset(Dataset):
                def __init__(self, original_loader, means, stds):
                    self.original_loader = original_loader
                    self.means = means
                    self.stds = stds
                    self.normalization = transforms.Normalize(mean=self.means, std=self.stds)

                def __len__(self):
                    return len(self.original_loader.dataset)

                def __getitem__(self, idx):
                    features = self.original_loader.dataset[idx]

                    norm_features = self.normalization(features)

                    return norm_features

        norm_dataset = NormalizedDataset(original_loader, means, stds)
        norm_loader = DataLoader(norm_dataset, batch_size=original_loader.batch_size, shuffle=False)

        return norm_loader

class FeaturePatchesDataset(Dataset):
    def __init__(self, image_path, patch_size, stride, offset_left=0, offset_top=0):
        """
            Dataset for extracted featured patches from an image.

            Args:
                image_path (str): File path to the feature image.
                patch_size (int): Size of the square patches (in pixels), e.g. '32' will generate 32x32 patches.
                stride (int): Step size between patches (in pixels).
                offset_left (int | 'best'): Number of pixels to ignore from the left edge of the image.
                    If '0' - no offset will be used (default).
                    If 'best' - evenly distributed optimal offset from both sides of the image will be calculated.
                offset_top (int | 'best'): Number of pixels to ignore from the top edge of the image.
                    If '0' - no offset will be used (default).
                    If 'best' - evenly distributed optimal offset from top and bottom of the image will be calculated.
        """

        self.image_path = image_path
        self.patch_size = patch_size
        self.stride = stride

        if offset_left == 'best':
            self.offset_left, _ = calculate_optimal_offsets(self.image_path, self.patch_size, self.stride)
        else:
            self.offset_left = offset_left
        if offset_top == 'best':
            _, self.offset_top = calculate_optimal_offsets(self.image_path, self.patch_size, self.stride)
        else:
            self.offset_top = offset_top

        # open the imagery
        with rio.open(image_path) as src_features:
            self.src_features = src_features

            # adjust dimensions based on offsets
            self.width = src_features.width - self.offset_left
            self.height = src_features.height - self.offset_top

        # precompute patch positions (accounting for offsets)
        self.patches = [
            (row + self.offset_top, col + self.offset_left)
            for row in range(0, self.height - patch_size + 1, stride)
            for col in range(0, self.width - patch_size + 1, stride)
        ]

        print(f"Total patches loaded: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        row, col = self.patches[idx]

        # extract the image patch features
        with rio.open(self.image_path) as src_features:
            window = rio.windows.Window(col, row, self.patch_size, self.patch_size)
            patch_features = src_features.read(window=window)

        return torch.tensor(patch_features, dtype=torch.float32)

class LabeledPatchesDataset(Dataset):
    def __init__(self, image_path, reference_path, patch_size, stride, offset_left=0, offset_top=0, background_label=0):
        """
            Dataset for extracting labeled patches from an image. Patch label corresponds to the label \
            of the central pixel of the patch derived from reference raster. If the central pixel label corresponds to
            provided background label, this patch is skipped.

            Args:
                image_path (str): Path to the feature image.
                reference_path (str): Path to the reference raster.
                patch_size (int): Size of the square patches (in pixels), e.g. '32' will generate 32x32 patches.
                stride (int): Step size between patches (in pixels).
                offset_left (int | 'best'): Number of pixels to ignore from the left edge of the image.
                    If '0' - no offset will be used (default).
                    If 'best' - evenly distributed optimal offset from both sides of the image will be calculated.
                offset_top (int | 'best'): Number of pixels to ignore from the top edge of the image.
                    If '0' - no offset will be used.
                    If 'best' - evenly distributed optimal offset from top and bottom of the image will be calculated.
                background_label: label corresponding to the background (nodata) of the reference raster (default=0).
        """

        self.image_path = image_path
        self.patch_size = patch_size
        self.stride = stride
        self.background_label = background_label

        if offset_left == 'best':
            self.offset_left, _ = calculate_optimal_offsets(self.image_path, self.patch_size, self.stride)
        else:
            self.offset_left = offset_left
        if offset_top == 'best':
            _, self.offset_top = calculate_optimal_offsets(self.image_path, self.patch_size, self.stride)
        else:
            self.offset_top = offset_top

        # open reference raster
        self.src_labels = rio.open(reference_path)

        # open the imagery
        with rio.open(image_path) as src_features:
            self.src_features = src_features

            # ensure the dimensions and CRS match
            if self.src_features.width != self.src_labels.width \
                    or self.src_features.height != self.src_labels.height \
                    or self.src_features.crs != self.src_labels.crs:
                print("Dimensions or CRS do not match, aligning the reference raster to match the features raster...")

                # Use match_rasters to align the reference raster
                self.src_labels = match_rasters(reference_path, self.image_path)

            # adjust dimensions based on offsets
            self.width = src_features.width - self.offset_left
            self.height = src_features.height - self.offset_top

            # precompute patch positions
            self.labeled_patches = []
            self.labeled_patches_count = 0

            for row in range(0, self.height - patch_size + 1, stride):
                for col in range(0, self.width - patch_size + 1, stride):

                    # extract the label for central pixel from the reference raster
                    center_row = row + self.offset_top + self.patch_size // 2
                    center_col = col + self.offset_left + self.patch_size // 2

                    central_window = rio.windows.Window(center_col, center_row, 1, 1)
                    label = self.src_labels.read(1, window=central_window).item()

                    # Only include patches with valid labels (non-background)
                    if label != self.background_label:
                        self.labeled_patches.append((row + self.offset_top, col + self.offset_left, label))
                        self.labeled_patches_count += 1

        print(f"Total ground truth patches generated: {self.labeled_patches_count}")

        # print the counts of unique labels
        labeled_patches_arr = np.array(self.labeled_patches)
        unique_labels, counts = np.unique(labeled_patches_arr[:, 2], return_counts=True)
        print("Unique Labels:", unique_labels)
        print("Counts:", counts)

    def __len__(self):
        return len(self.labeled_patches)

    def __getitem__(self, idx):
        row, col, label = self.labeled_patches[idx]

        # Extract the image patch
        with rio.open(self.image_path) as src_features:
            window = rio.windows.Window(col, row, self.patch_size, self.patch_size)
            patch_features = src_features.read(window=window)

        # Return the patch and its label
        return torch.tensor(patch_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def generate_feature_patches_loader(image_path, patch_size, stride, batch_size, offset_left=0, offset_top=0, shuffle=False):
    """
    Generate DataLoaders for image patches and optionally for ground truth labels.

    Args:
        image_path (str): Path to the image (features).
        patch_size (int): Size of the square patches (in pixels).
        stride (int): Step size between patches (in pixels).
        batch_size (int): Number of patches per batch.
        offset_left (int | 'best'): Offset from the left edge of the image.
        offset_top (int | 'best'): Offset from the top edge of the image.
        reference_path (str, optional): Path to the reference raster (labels). Default is None.
        background_label (int, optional): Label value to ignore as background. Default is None.

    Returns:
        If reference_path and background_label are provided:
            (DataLoader, DataLoader): Features DataLoader and Ground Truth DataLoader.
        Otherwise:
            DataLoader: Features DataLoader only.
    """

    features_dataset = FeaturePatchesDataset(image_path=image_path,
                                             patch_size=patch_size,
                                             stride=stride,
                                             offset_left=offset_left,
                                             offset_top=offset_top)

    return DataLoader(features_dataset, batch_size=batch_size, shuffle=shuffle)

def generate_labeled_patches_loader(image_path, reference_path, patch_size, stride, batch_size, offset_left=0, offset_top=0, background_label=0, shuffle=False):
    """
    Generate DataLoaders for labeled image patches.

    Args:
        image_path (str): Path to the image (features).
        patch_size (int): Size of the square patches (in pixels).
        stride (int): Step size between patches (in pixels).
        batch_size (int): Number of patches per batch.
        offset_left (int | 'best'): Offset from the left edge of the image.
        offset_top (int | 'best'): Offset from the top edge of the image.
        reference_path (str, optional): Path to the reference raster (labels). Default is None.
        background_label (int, optional): Label value to ignore as background. Default is None.

    Returns:
        If reference_path and background_label are provided:
            (DataLoader, DataLoader): Features DataLoader and Ground Truth DataLoader.
        Otherwise:
            DataLoader: Features DataLoader only.
    """

    labeled_dataset = LabeledPatchesDataset(image_path=image_path,
                                            reference_path=reference_path,
                                            patch_size=patch_size,
                                            stride=stride,
                                            offset_left=offset_left,
                                            offset_top=offset_top,
                                            background_label=background_label)

    return DataLoader(labeled_dataset, batch_size=batch_size, shuffle=shuffle)

def compute_label_mapping(gt_loader):
    """
    Compute a label mapping from original labels to a continuous range [0, N] based on the training dataset.
    
    Args:
        gt_loader (DataLoader): Training DataLoader with original labels.

    Returns:
        dict: Mapping of original labels to new labels.
    """
    all_labels = []
    for _, label in gt_loader:
        all_labels.extend(label.numpy())

    original_labels, counts = np.unique(all_labels, return_counts=True)

    label_mapping = {label: new_label for new_label, label in enumerate(original_labels)}

    return label_mapping

def label_remapping(gt_loader, label_mapping):
    """
    Remap labels using a predefined mapping.

    Args:
        gt_loader (DataLoader): DataLoader with original labels.
        label_mapping (dict): Precomputed mapping from original labels to new labels.

    Returns:
        DataLoader: DataLoader with remapped labels.
    """

    class RemappedDataset(Dataset):
        def __init__(self, original_loader, label_mapping):
            self.original_loader = original_loader
            self.label_mapping = label_mapping

        def __len__(self):
            return len(self.original_loader.dataset)

        def __getitem__(self, idx):
            features, label = self.original_loader.dataset[idx]

            remapped_label = torch.tensor(self.label_mapping.get(label.item(), -1))

            return features, remapped_label

    # Create the remapped dataset
    remapped_dataset = RemappedDataset(gt_loader, label_mapping)
    gt_loader_remapped = DataLoader(remapped_dataset, batch_size=gt_loader.batch_size, shuffle=False)

    all_labels = []
    for _, label in gt_loader:
        all_labels.extend(label.numpy())

    original_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"Original unique label values: {original_labels}, Counts: {counts}")

    all_remapped_labels = []
    for _, label in gt_loader_remapped:
        all_remapped_labels.extend(label.numpy())

    remapped_labels, remap_counts = np.unique(all_remapped_labels, return_counts=True)
    print(f"Remapped unique label values: {remapped_labels}, Counts: {remap_counts}")

    return gt_loader_remapped

class AugmentedLabeledPatchesDataset(Dataset):
    def __init__(self, image_path, reference_path, patch_size, stride, transform=None,
                 offset_left=0, offset_top=0, background_label=0,
                 label_mapping=None, mean=None, std=None):
        self.image_path = image_path
        self.patch_size = patch_size
        self.stride = stride
        self.background_label = background_label
        self.transform = transform
        self.label_mapping = label_mapping
        self.mean = mean
        self.std = std

        if offset_left == 'best':
            self.offset_left, _ = calculate_optimal_offsets(image_path, patch_size, stride)
        else:
            self.offset_left = offset_left
        if offset_top == 'best':
            _, self.offset_top = calculate_optimal_offsets(image_path, patch_size, stride)
        else:
            self.offset_top = offset_top

        self.src_labels = rio.open(reference_path)

        with rio.open(image_path) as src_features:
            self.src_features = src_features

            if self.src_features.width != self.src_labels.width \
                    or self.src_features.height != self.src_labels.height \
                    or self.src_features.crs != self.src_labels.crs:
                print("Dimensions or CRS do not match, aligning the reference raster to match the features raster...")
                self.src_labels = match_rasters(reference_path, image_path)

            self.width = src_features.width - self.offset_left
            self.height = src_features.height - self.offset_top

            self.labeled_patches = []
            self.labeled_patches_count = 0

            for row in range(0, self.height - patch_size + 1, stride):
                for col in range(0, self.width - patch_size + 1, stride):
                    center_row = row + self.offset_top + patch_size // 2
                    center_col = col + self.offset_left + patch_size // 2
                    central_window = rio.windows.Window(center_col, center_row, 1, 1)
                    label = self.src_labels.read(1, window=central_window).item()

                    if label != self.background_label:
                        self.labeled_patches.append((row + self.offset_top, col + self.offset_left, label))
                        self.labeled_patches_count += 1

        print(f"Total ground truth patches generated: {self.labeled_patches_count}")
        labels_np = np.array([x[2] for x in self.labeled_patches])
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        print("Unique Labels:", unique_labels)
        print("Counts:", counts)

        self.normalizer = transforms.Normalize(mean=mean, std=std) if mean is not None and std is not None else None

    def __len__(self):
        return len(self.labeled_patches)

    def __getitem__(self, idx):
        row, col, label = self.labeled_patches[idx]
        with rio.open(self.image_path) as src_features:
            window = rio.windows.Window(col, row, self.patch_size, self.patch_size)
            patch_features = src_features.read(window=window)

        patch_tensor = torch.tensor(patch_features, dtype=torch.float32)

        if self.normalizer:
            patch_tensor = self.normalizer(patch_tensor)

        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        remapped_label = self.label_mapping[label] if self.label_mapping else label

        return patch_tensor, torch.tensor(remapped_label, dtype=torch.long)

def generate_augmented_loader_with_sampler(image_path, reference_path, patch_size, stride, batch_size,
                                           offset_left=0, offset_top=0, background_label=0,
                                           transform=None, label_mapping=None, mean=None, std=None,
                                           alpha=1.0):
    dataset = AugmentedLabeledPatchesDataset(
        image_path=image_path,
        reference_path=reference_path,
        patch_size=patch_size,
        stride=stride,
        offset_left=offset_left,
        offset_top=offset_top,
        background_label=background_label,
        transform=transform,
        label_mapping=label_mapping,
        mean=mean,
        std=std
    )

    # Extract raw or remapped labels
    labels = [label for _, _, label in dataset.labeled_patches]
    if label_mapping:
        labels = [label_mapping[label] for label in labels]

    class_counts = np.bincount(labels)
    class_weights = (1.0 / (class_counts + 1e-6)) ** alpha
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights)*2, replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return loader
    
def cnn_training(model, train_patches_norm, test_patches_norm, num_epochs, criterion, learning_rate, optimizer, output):
    start = time.time()

    # track losses and accuracies
    train_loss, test_loss = [], []
    train_accuracy, test_accuracy = [], []
    train_f1, test_f1 = [], []

    for epoch in range(1, num_epochs+1):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
                model.train()
                dataloader = train_patches_norm
            else:
                model.train(False)
                model.eval()
                dataloader = test_patches_norm
    
            step = 0
            correct = 0
            total = 0
            # iterate over data
            for x, y in dataloader:
                step += 1
                x = x.cuda()  
                y = y.cuda()
    
                # forward pass
                if phase == 'train':
                    # clear gradients
                    optimizer.zero_grad()
                    # forward pass
                    outputs = model(x)
                    # calculate loss
                    loss = criterion(outputs, y.long())
                    # backward pass
                    loss.backward()
                    # optimization step
                    optimizer.step()
                else:
                    # testing
                    with torch.no_grad():
                        # forward pass
                        outputs = model(x)
                        # calculate loss
                        loss = criterion(outputs, y.long())
    
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                batch_accuracy = correct/total
                if phase == 'train':
                    train_loss.append(loss.item())
                    if step % 20 == 0:
                        print(f'Epoch: {epoch} Step: {step} Training Loss: {loss.item():.4f} Accuracy: {batch_accuracy:.4f}')
                else:
                    test_loss.append(loss.item())
                    #if step % 10 == 0:
                    print(f'Epoch: {epoch} Step: {step} Testing Loss: {loss.item():.4f} Accuracy: {batch_accuracy:.4f}')
            
            # train and test accuracy for epoch
            if phase == 'train': 
                model.eval()
                correct = 0
                total = 0
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for features, labels in train_patches_norm:
                        features = features.cuda()
                        labels = labels.cuda()
                        
                        outputs = model(features)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
    
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
    
                # compute epoch accuracy and weighted F1-score
                epoch_train_accuracy = correct / total
                f1_weighted = f1_score(all_labels, all_preds, average='weighted', labels=np.unique(all_labels))
                train_accuracy.append(epoch_train_accuracy)
                train_f1.append(f1_weighted)
                print(f'Epoch: {epoch} Training Accuracy: {epoch_train_accuracy:.4f}, Weighted F1-Score: {f1_weighted:.4f}')
                
            else:
                model.eval()
                correct = 0
                total = 0
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for features, labels in test_patches_norm:
                        features = features.cuda()
                        labels = labels.cuda()
                        
                        outputs = model(features)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
    
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
    
                # compute epoch accuracy and weighted F1-score
                epoch_test_accuracy = correct / total
                f1_weighted = f1_score(all_labels, all_preds, average='weighted', labels=np.unique(all_labels))
                test_accuracy.append(epoch_test_accuracy)
                test_f1.append(f1_weighted)
                print(f'Epoch: {epoch} Testing Accuracy: {epoch_test_accuracy:.4f}, Weighted F1-Score: {f1_weighted:.4f}')
    
        # model checkpointing
        if (epoch_test_accuracy>=0.40) and ((epoch_train_accuracy-epoch_test_accuracy)<=0.05) and (epoch_train_accuracy>=epoch_test_accuracy):
            checkpoint = {
                "epoch_num": epoch,
                "model_state": model.state_dict(),
                "train_accuracy": epoch_train_accuracy,
                "test_accuracy": epoch_test_accuracy,
                "gap": epoch_train_accuracy-epoch_test_accuracy
            }
            output = output.replace(".pth", f"_epoch{epoch}.pth")
            torch.save(checkpoint, output)
            print(f"New model saved at epoch {epoch}.")
            
    print('---')
    time_elapsed = time.time() - start
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

def extract_embeddings(train_patches, test_patches, cnn_model):
    cnn_model.eval()
    train_embeddings = list()
    train_labels = list()
    test_embeddings = list()
    test_labels = list()

    with torch.no_grad():
        for feature,label in train_patches:
            feature = feature.cuda()
            embedding = cnn_model.get_embedding_raw_fc(feature)
        
            train_embeddings.append(embedding.cpu().numpy())
            train_labels.append(label.cpu().numpy())

    train_embeddings = np.concatenate(train_embeddings, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    with torch.no_grad():
        for feature,label in test_patches:
            feature = feature.cuda()
            embedding = cnn_model.get_embedding_raw_fc(feature)
            
            test_embeddings.append(embedding.cpu().numpy())
            test_labels.append(label.cpu().numpy())

    test_embeddings = np.concatenate(test_embeddings, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    return train_embeddings, train_labels, test_embeddings, test_labels

def aggregate_morphometrics(train_morphometrics_patches, test_morphometrics_patches, train_embeddings, test_embeddings):
    train_urbanform = list()
    train_labels = list()
    test_urbanform = list()
    test_labels = list()

    for feature,label in train_morphometrics_patches:
        train_urbanform.append(feature.cpu().numpy())
        train_labels.append(label.cpu().numpy())

    train_urbanform = np.concatenate(train_urbanform, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    for feature,label in test_morphometrics_patches:
        test_urbanform.append(feature.cpu().numpy())
        test_labels.append(label.cpu().numpy())

    test_urbanform = np.concatenate(test_urbanform, axis=0)
    y_test = np.concatenate(test_labels, axis=0)

    # aggregation mean,min,max,std,median
    mean_train_urbanform = train_urbanform.mean(axis=(2,3))
    min_train_urbanform = train_urbanform.min(axis=(2,3))
    max_train_urbanform = train_urbanform.max(axis=(2,3))
    std_train_urbanform = train_urbanform.std(axis=(2,3))
    med_train_urbanform = np.median(train_urbanform, axis=(2, 3))
    
    mean_test_urbanform = test_urbanform.mean(axis=(2,3))
    min_test_urbanform = test_urbanform.min(axis=(2,3))
    max_test_urbanform = test_urbanform.max(axis=(2,3))
    std_test_urbanform = test_urbanform.std(axis=(2,3))
    med_test_urbanform = np.median(test_urbanform, axis=(2, 3))

    # merge
    X_train = np.hstack((train_embeddings,mean_train_urbanform,min_train_urbanform,max_train_urbanform,std_train_urbanform,med_train_urbanform))
    X_test = np.hstack((test_embeddings,mean_test_urbanform,min_test_urbanform,max_test_urbanform,std_test_urbanform,med_test_urbanform))

    return X_train, X_test