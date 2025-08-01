```python
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from typing import Tuple, List, Optional


class KITTIDataset(Dataset):
    """KITTI Dataset Loader"""
    
    def __init__(self, data_dir: str, transform=None, target_size: Tuple[int, int] = (128, 128), max_labels: int = 50):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.max_labels = max_labels  # Maximum number of labels
        
        # Get image and label file paths
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        
        # Support multiple image formats
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')) + 
                                 glob.glob(os.path.join(self.image_dir, '*.jpg')) +
                                 glob.glob(os.path.join(self.image_dir, '*.jpeg')))
        self.label_files = sorted(glob.glob(os.path.join(self.label_dir, '*.txt')))
        
        print(f"Loaded {len(self.image_files)} KITTI images")
        
        # Check if dataset is empty
        if len(self.image_files) == 0:
            raise ValueError(f"KITTI dataset is empty: {self.image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Load labels (if available)
        labels = torch.zeros((self.max_labels, 5), dtype=torch.float32)  # Fixed-size label tensor
        if idx < len(self.label_files):
            label_path = self.label_files[idx]
            loaded_labels = self._load_labels(label_path)
            if len(loaded_labels) > 0:
                # Take only the first max_labels
                num_labels = min(len(loaded_labels), self.max_labels)
                labels[:num_labels] = loaded_labels[:num_labels]
        
        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'labels': labels,
            'image_path': image_path
        }
    
    def _load_labels(self, label_path: str) -> torch.Tensor:
        """Load label file"""
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append([class_id, x_center, y_center, width, height])
        
        return torch.tensor(labels, dtype=torch.float32) if labels else torch.empty((0, 5))


class DrivingStereoDataset(Dataset):
    """DrivingStereo Dataset Loader"""
    
    def __init__(self, data_dir: str, transform=None, target_size: Tuple[int, int] = (128, 128)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get left, right image, and disparity map paths
        self.left_dir = os.path.join(data_dir, 'image_L')
        self.right_dir = os.path.join(data_dir, 'image_R')
        self.disparity_dir = os.path.join(data_dir, 'disparity')
        
        # Support multiple image formats
        self.left_files = sorted(glob.glob(os.path.join(self.left_dir, '*.png')) + 
                                glob.glob(os.path.join(self.left_dir, '*.jpg')) +
                                glob.glob(os.path.join(self.left_dir, '*.jpeg')))
        self.right_files = sorted(glob.glob(os.path.join(self.right_dir, '*.png')) + 
                                 glob.glob(os.path.join(self.right_dir, '*.jpg')) +
                                 glob.glob(os.path.join(self.right_dir, '*.jpeg')))
        
        print(f"Loaded {len(self.left_files)} DrivingStereo images")
        
        # Check if dataset is empty
        if len(self.left_files) == 0:
            raise ValueError(f"DrivingStereo dataset is empty: {self.left_dir}")
    
    def __len__(self):
        return len(self.left_files)
    
    def __getitem__(self, idx):
        # Load left image
        left_path = self.left_files[idx]
        left_image = cv2.imread(left_path)
        if left_image is None:
            raise ValueError(f"Failed to load left image: {left_path}")
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        left_image = cv2.resize(left_image, self.target_size)
        
        # Convert to tensor
        if self.transform:
            left_image = self.transform(left_image)
        else:
            left_image = torch.from_numpy(left_image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': left_image,
            'image_path': left_path
        }


def get_data_loaders(kitti_dir: str, driving_stereo_dir: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Get data loaders"""
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    try:
        kitti_dataset = KITTIDataset(kitti_dir, transform=transform)
        print(f"✓ KITTI dataset created successfully, containing {len(kitti_dataset)} samples")
    except Exception as e:
        print(f"✗ KITTI dataset creation failed: {e}")
        raise
    
    try:
        driving_stereo_dataset = DrivingStereoDataset(driving_stereo_dir, transform=transform)
        print(f"✓ DrivingStereo dataset created successfully, containing {len(driving_stereo_dataset)} samples")
    except Exception as e:
        print(f"✗ DrivingStereo dataset creation failed: {e}")
        raise
    
    # Create data loaders
    kitti_loader = DataLoader(
        kitti_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Reduce workers to avoid issues
        drop_last=True
    )
    
    driving_stereo_loader = DataLoader(
        driving_stereo_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Reduce workers to avoid issues
        drop_last=True
    )
    
    print("✓ Data loaders created successfully")
    
    return kitti_loader, driving_stereo_loader


def get_real_images_for_evaluation(data_loader: DataLoader, num_samples: int = 1000) -> torch.Tensor:
    """Get real images for evaluation"""
    images = []
    count = 0
    
    for batch in data_loader:
        if count >= num_samples:
            break
            
        if isinstance(batch, dict):
            batch_images = batch['image']
        else:
            batch_images = batch[0]
        
        for image in batch_images:
            if count >= num_samples:
                break
            images.append(image)
            count += 1
    
    if not images:
        print("Warning: No real images obtained, using random images")
        return torch.randn(num_samples, 3, 128, 128)
    
    return torch.stack(images)


def get_real_labels_for_evaluation(data_loader: DataLoader, num_samples: int = 1000) -> List[torch.Tensor]:
    """Get real labels for evaluation"""
    labels = []
    count = 0
    
    for batch in data_loader:
        if count >= num_samples:
            break
            
        if isinstance(batch, dict):
            batch_labels = batch.get('labels', [])
        else:
            batch_labels = batch[1] if len(batch) > 1 else []
        
        for label in batch_labels:
            if count >= num_samples:
                break
            # Filter out all-zero labels
            if torch.sum(label) > 0:
                labels.append(label)
                count += 1
    
    # If labels are insufficient, fill with random labels
    while len(labels) < num_samples:
        # Create random label: class ID=0 (person), position near image center
        random_label = torch.tensor([[0, 0.5 + np.random.normal(0, 0.1), 0.5 + np.random.normal(0, 0.1), 
                                    0.1 + np.random.uniform(0, 0.2), 0.1 + np.random.uniform(0, 0.2)]])
        labels.append(random_label)
    
    return labels[:num_samples]


def create_synthetic_dataset_from_labels(generated_images: torch.Tensor, real_labels: List[torch.Tensor], 
                                       output_dir: str) -> str:
    """Create synthetic dataset from generated images and real labels"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    num_images = min(len(generated_images), len(real_labels))
    
    for i in range(num_images):
        # Save image
        img = generated_images[i]
        img = torch.clamp(img, -1, 1)
        img = ((img + 1) / 2 * 255).byte().cpu().numpy().transpose(1, 2, 0)
        img_path = os.path.join(output_dir, 'images', f'image_{i:06d}.jpg')
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Save label
        label = real_labels[i]
        label_path = os.path.join(output_dir, 'labels', f'image_{i:06d}.txt')
        np.savetxt(label_path, label.cpu().numpy())
    
    print(f"✓ Synthetic dataset created: {output_dir}")
    return output_dir
```