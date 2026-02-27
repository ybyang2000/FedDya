import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import transforms, datasets
import random

# ==========================================
# 1. MNIST DATASET
# ==========================================
COLORS = np.array([
    [1.0, 0.0, 0.0],  # 0: Red
    [0.0, 1.0, 0.0],  # 1: Green
    [0.0, 0.0, 1.0],  # 2: Blue
    [1.0, 1.0, 0.0],  # 3: Yellow
    [1.0, 0.0, 1.0],  # 4: Magenta
    [0.0, 1.0, 1.0],  # 5: Cyan
    [1.0, 0.5, 0.0],  # 6: Orange
    [0.5, 0.0, 0.5],  # 7: Purple
    [0.5, 0.5, 0.5],  # 8: Gray
    [0.0, 0.5, 0.5]   # 9: Teal
])

ANGLES = [i * 36 for i in range(10)]

class ConfiguredColoredMNIST(Dataset):
    def __init__(self, original_dataset, sample_configs, transform=None):
        self.original_data = original_dataset.data
        self.original_targets = original_dataset.targets
        self.sample_configs = sample_configs
        self.transform = transform

    def __len__(self):
        return len(self.sample_configs)

    def __getitem__(self, index):
        # 1. Retrieve the allocation configuration of this sample
        img_idx, color_idx, angle_idx = self.sample_configs[index]
        # 2. Obtain the original grayscale image and number labels
        img = Image.fromarray(self.original_data[img_idx].numpy(), mode='L')
        digit_label = self.original_targets[img_idx].item()
        # 3. Image rotation
        angle = ANGLES[angle_idx]
        img_rotated = img.rotate(angle, resample=Image.BILINEAR)
        # 4. Image coloring
        mask = np.array(img_rotated).astype(np.float32) / 255.0
        mask = mask[..., np.newaxis]
        fg_color = COLORS[color_idx]
        # The background color should be randomly selected, as long as it is not equal to the foreground color.
        available_bg_indices = [i for i in range(10) if i != color_idx]
        bg_color = COLORS[random.choice(available_bg_indices)]
        colored_img_np = mask * fg_color + (1.0 - mask) * bg_color
        
        colored_img_tensor = torch.tensor(colored_img_np, dtype=torch.float32).permute(2, 0, 1)
        if self.transform:
            colored_img_tensor = self.transform(colored_img_tensor)
            
        new_label = [digit_label, color_idx, angle_idx]
        return colored_img_tensor, torch.tensor(new_label, dtype=torch.long)


class GlobalConfiguredTestDataset(Dataset):
    def __init__(self, test_dataset, transform):
        self.original_data = test_dataset.data
        self.original_targets = test_dataset.targets
        self.transform = transform
        self.length = len(test_dataset)
        
        random.seed(42)
        self.sample_configs = []
        for i in range(self.length):
            color_idx = random.randint(0, 9)
            angle_idx = random.randint(0, 9)
            self.sample_configs.append((i, color_idx, angle_idx))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_idx, color_idx, angle_idx = self.sample_configs[index]
        img = Image.fromarray(self.original_data[img_idx].numpy(), mode='L')
        digit_label = self.original_targets[img_idx].item()
        
        angle = ANGLES[angle_idx]
        img_rotated = img.rotate(angle, resample=Image.BILINEAR)
        
        mask = np.array(img_rotated).astype(np.float32) / 255.0
        mask = mask[..., np.newaxis]
        
        fg_color = COLORS[color_idx]
        available_bg_indices = [i for i in range(10) if i != color_idx]
        bg_color = COLORS[random.choice(available_bg_indices)]
        
        colored_img_np = mask * fg_color + (1.0 - mask) * bg_color
        
        colored_img_tensor = torch.tensor(colored_img_np, dtype=torch.float32).permute(2, 0, 1)
        if self.transform:
            colored_img_tensor = self.transform(colored_img_tensor)
            
        new_label = [digit_label, color_idx, angle_idx]
        return colored_img_tensor, torch.tensor(new_label, dtype=torch.long)


# Main Function: MNIST
def get_mnist(dataset_root, args):
    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset_raw = datasets.MNIST(root=dataset_root, train=True, download=True)
    
    indices_by_class = {cls: [] for cls in range(10)}
    for i, target in enumerate(train_dataset_raw.targets):
        indices_by_class[target.item()].append(i)

    client_volume_proportions = np.random.dirichlet([args.alpha1] * args.num_clients)
    client_sample_counts = (client_volume_proportions * args.total_train_samples).astype(int)

    client_datasets = []
    
    for client_id in range(args.num_clients):
        digit_class_dist = np.random.dirichlet([args.alpha2] * 10)
        color_class_dist = np.random.dirichlet([args.alpha2] * 10)
        angle_class_dist = np.random.dirichlet([args.alpha2] * 10)
        
        num_samples_for_client = client_sample_counts[client_id]
        client_configs = []
        
        for _ in range(num_samples_for_client):
            chosen_digit = np.random.choice(10, p=digit_class_dist)
            chosen_color = np.random.choice(10, p=color_class_dist)
            chosen_angle = np.random.choice(10, p=angle_class_dist)
            img_idx = random.choice(indices_by_class[chosen_digit])
            client_configs.append((img_idx, chosen_color, chosen_angle))
            
        client_dataset = ConfiguredColoredMNIST(train_dataset_raw, client_configs, transform)
        client_datasets.append(client_dataset)

    train_loaders = {i: DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True) for i in range(args.num_clients)}
    v_train_dataset = ConcatDataset(client_datasets)
    v_train_loader = DataLoader(v_train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset_raw = datasets.MNIST(root=dataset_root, train=False, download=True)
    global_test_dataset = GlobalConfiguredTestDataset(test_dataset_raw, transform)
    v_test_loader = DataLoader(global_test_dataset, batch_size=args.batch_size, shuffle=False)

    test_loaders = {}
    test_data_size_per_client = len(global_test_dataset) // args.num_clients
    indices = list(range(len(global_test_dataset)))
    
    for i in range(args.num_clients):
        start_idx = i * test_data_size_per_client
        end_idx = start_idx + test_data_size_per_client if i != args.num_clients - 1 else len(global_test_dataset)
        subset_indices = indices[start_idx:end_idx]
        client_subset = Subset(global_test_dataset, subset_indices)
        test_loaders[i] = DataLoader(client_subset, batch_size=args.batch_size, shuffle=False)

    return train_loaders, test_loaders, v_train_loader, v_test_loader




# ==========================================
# 2. CIFAR-10 DATASET
# ==========================================
ANIMAL_CLASSES = [2, 3, 4, 5, 6, 7]  # Birds, cats, deer, dogs, frogs, horses
VEHICLE_CLASSES = [0, 1, 8, 9]      # Airplanes, cars, ships, trucks

ANIMAL_MAP = {original_label: new_label for new_label, original_label in enumerate(ANIMAL_CLASSES)}
VEHICLE_MAP = {original_label: new_label for new_label, original_label in enumerate(VEHICLE_CLASSES)}

ANIMAL_MAP_REV = {v: k for k, v in ANIMAL_MAP.items()}
VEHICLE_MAP_REV = {v: k for k, v in VEHICLE_MAP.items()}

# ---------------------------------------------------------------------------------
class GeneratedPairedDataset(Dataset):
    def __init__(self, original_dataset, image_index_pairs, transform=None):
        self.original_data = original_dataset.data
        self.original_targets = np.array(original_dataset.targets)
        self.image_index_pairs = image_index_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_index_pairs)

    def __getitem__(self, index):
        animal_original_idx, vehicle_original_idx = self.image_index_pairs[index]
        left_image = self.original_data[animal_original_idx]
        animal_label = self.original_targets[animal_original_idx]
        right_image = self.original_data[vehicle_original_idx]
        vehicle_label = self.original_targets[vehicle_original_idx]
        combined_image = np.concatenate((left_image, right_image), axis=1)
        new_label = [ANIMAL_MAP[animal_label], VEHICLE_MAP[vehicle_label]]
        if self.transform:
            combined_image = self.transform(combined_image)
        return combined_image, torch.tensor(new_label, dtype=torch.long)

class GlobalPairedTestDataset(Dataset):
    def __init__(self, test_dataset, transform):
        self.animal_indices = [i for i, t in enumerate(test_dataset.targets) if t in ANIMAL_CLASSES]
        self.vehicle_indices = [i for i, t in enumerate(test_dataset.targets) if t in VEHICLE_CLASSES]
        
        rng = random.Random(42)
        rng.shuffle(self.animal_indices)
        rng.shuffle(self.vehicle_indices)

        self.length = min(len(self.animal_indices), len(self.vehicle_indices))
        self.original_data = test_dataset.data
        self.original_targets = np.array(test_dataset.targets)
        self.transform = transform

    def __len__(self): 
        return self.length

    def __getitem__(self, index):
        animal_idx = self.animal_indices[index]
        vehicle_idx = self.vehicle_indices[index] 
        left_image = self.original_data[animal_idx]
        right_image = self.original_data[vehicle_idx]
        animal_label = self.original_targets[animal_idx]
        vehicle_label = self.original_targets[vehicle_idx]
        combined_image = np.concatenate((left_image, right_image), axis=1)
        new_label = [ANIMAL_MAP[animal_label], VEHICLE_MAP[vehicle_label]]
        if self.transform: 
            combined_image = self.transform(combined_image)
        return combined_image, torch.tensor(new_label, dtype=torch.long)


# Main Function: CIFAR-10
# ---------------------------------------------------------------------------------
def get_cifar10(dataset_root, args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset_raw = datasets.CIFAR10(root=dataset_root, train=True, download=True)
    
    animal_indices_by_class = {cls: [] for cls in ANIMAL_CLASSES}
    vehicle_indices_by_class = {cls: [] for cls in VEHICLE_CLASSES}
    for i, target in enumerate(train_dataset_raw.targets):
        if target in ANIMAL_CLASSES:
            animal_indices_by_class[target].append(i)
        elif target in VEHICLE_CLASSES:
            vehicle_indices_by_class[target].append(i)

    client_volume_proportions = np.random.dirichlet([args.alpha1] * args.num_clients)
    client_sample_counts = (client_volume_proportions * args.total_train_samples).astype(int)

    client_datasets = []
    for client_id in range(args.num_clients):
        animal_class_dist = np.random.dirichlet([args.alpha2] * len(ANIMAL_CLASSES))
        vehicle_class_dist = np.random.dirichlet([args.alpha2] * len(VEHICLE_CLASSES))
        
        num_samples_for_client = client_sample_counts[client_id]
        client_image_pairs = []
        
        animal_original_labels = list(animal_indices_by_class.keys())
        vehicle_original_labels = list(vehicle_indices_by_class.keys())

        for _ in range(num_samples_for_client):
            chosen_animal_class = np.random.choice(animal_original_labels, p=animal_class_dist)
            chosen_vehicle_class = np.random.choice(vehicle_original_labels, p=vehicle_class_dist)
            animal_idx = random.choice(animal_indices_by_class[chosen_animal_class])
            vehicle_idx = random.choice(vehicle_indices_by_class[chosen_vehicle_class])
            client_image_pairs.append((animal_idx, vehicle_idx))
            
        client_dataset = GeneratedPairedDataset(train_dataset_raw, client_image_pairs, transform)
        client_datasets.append(client_dataset)

    train_loaders = {i: DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True) for i in range(args.num_clients)}

    v_train_dataset = ConcatDataset(client_datasets)
    v_train_loader = DataLoader(v_train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dataset_raw = datasets.CIFAR10(root=dataset_root, train=False, download=True)

    global_test_dataset = GlobalPairedTestDataset(test_dataset_raw, transform)

    v_test_loader = DataLoader(global_test_dataset, batch_size=args.batch_size, shuffle=False)

    test_loaders = {}
    test_data_size_per_client = len(global_test_dataset) // args.num_clients
    indices = list(range(len(global_test_dataset)))
    for i in range(args.num_clients):
        start_idx = i * test_data_size_per_client
        end_idx = start_idx + test_data_size_per_client if i != args.num_clients - 1 else len(global_test_dataset)
        subset_indices = indices[start_idx:end_idx]
        client_subset = Subset(global_test_dataset, subset_indices)
        test_loaders[i] = DataLoader(client_subset, batch_size=args.batch_size, shuffle=False)

    return train_loaders, test_loaders, v_train_loader, v_test_loader



# ==========================================
# 3. CIFAR-100 DATASET
# ==========================================
COARSE_LABELS = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables', 
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores', 
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_sized_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 
    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
]

COARSE_MAP = {name: i for i, name in enumerate(COARSE_LABELS)}

fine_to_coarse_map = [
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  # 0-9
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,  # 10-19
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  # 20-29
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,  # 30-39
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,  # 40-49
    16, 4, 17,  4,  2,  0, 17,  4, 18, 17,  # 50-59
    10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  # 60-69
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,  # 70-79
    16, 19,  2,  4,  6, 19,  5,  5,  8, 19,  # 80-89
    18, 1,  2, 15,  6,  0, 17,  8, 14, 13   # 90-99
]
# Task 1
TASK1_COARSE_LABELS = [COARSE_MAP['large_carnivores'], COARSE_MAP['large_omnivores_and_herbivores']]
# Task 2
TASK2_COARSE_LABELS = [COARSE_MAP['medium_sized_mammals'], COARSE_MAP['small_mammals']]
# Task 3
TASK3_COARSE_LABELS = [COARSE_MAP['household_electrical_devices']]
# Task 4
TASK4_COARSE_LABELS = [COARSE_MAP['household_furniture']]

# --- Create a new consecutive label mapping for each task ---
def create_label_map(task_coarse_labels, fine_to_coarse):
    fine_labels_in_task = sorted([i for i, coarse in enumerate(fine_to_coarse) if coarse in task_coarse_labels])
    return {original_label: new_label for new_label, original_label in enumerate(fine_labels_in_task)}

TASK1_MAP = create_label_map(TASK1_COARSE_LABELS, fine_to_coarse_map)
TASK2_MAP = create_label_map(TASK2_COARSE_LABELS, fine_to_coarse_map)
TASK3_MAP = create_label_map(TASK3_COARSE_LABELS, fine_to_coarse_map)
TASK4_MAP = create_label_map(TASK4_COARSE_LABELS, fine_to_coarse_map)


class FourCornerDataset(Dataset):
    """
    Images from four tasks are combined into the four corners of a single 64x64 image.
    """
    def __init__(self, original_dataset, image_index_quads, transform=None):
        self.original_data = original_dataset.data
        self.original_targets = np.array(original_dataset.targets) 
        self.image_index_quads = image_index_quads
        self.transform = transform

    def __len__(self):
        return len(self.image_index_quads)

    def __getitem__(self, index):
        idx1, idx2, idx3, idx4 = self.image_index_quads[index]

        img_tl = self.original_data[idx1] # Top-Left
        img_tr = self.original_data[idx2] # Top-Right
        img_bl = self.original_data[idx3] # Bottom-Left
        img_br = self.original_data[idx4] # Bottom-Right

        label_tl_orig = self.original_targets[idx1]
        label_tr_orig = self.original_targets[idx2]
        label_bl_orig = self.original_targets[idx3]
        label_br_orig = self.original_targets[idx4]

        combined_image = np.zeros((64, 64, 3), dtype=np.uint8)
        combined_image[0:32, 0:32, :] = img_tl
        combined_image[0:32, 32:64, :] = img_tr
        combined_image[32:64, 0:32, :] = img_bl
        combined_image[32:64, 32:64, :] = img_br

        new_labels = [
            TASK1_MAP[label_tl_orig], TASK2_MAP[label_tr_orig],
            TASK3_MAP[label_bl_orig], TASK4_MAP[label_br_orig]
        ]
        
        if self.transform:
            combined_image = self.transform(combined_image)
            
        return combined_image, torch.tensor(new_labels, dtype=torch.long)


class GlobalFourCornerTestDataset(Dataset):
    """
    The global test set contains a fixed 5000 samples.
    The categories of the four corners of each sample are randomly 
    selected from the category pool corresponding to the task.
    """
    def __init__(self, test_dataset, transform, num_samples=5000):
        self.transform = transform
        self.num_samples = num_samples
        random.seed(42)
        np.random.seed(42)
        test_indices_by_fine_label = {i: [] for i in range(100)}
        for idx, target in enumerate(test_dataset.targets):
            test_indices_by_fine_label[target].append(idx)

        task_fine_labels = {
            1: list(TASK1_MAP.keys()), 2: list(TASK2_MAP.keys()),
            3: list(TASK3_MAP.keys()), 4: list(TASK4_MAP.keys())
        }

        self.image_index_quads = []
        for _ in range(self.num_samples):
            chosen_fine_label1 = random.choice(task_fine_labels[1])
            chosen_fine_label2 = random.choice(task_fine_labels[2])
            chosen_fine_label3 = random.choice(task_fine_labels[3])
            chosen_fine_label4 = random.choice(task_fine_labels[4])
            
            idx1 = random.choice(test_indices_by_fine_label[chosen_fine_label1])
            idx2 = random.choice(test_indices_by_fine_label[chosen_fine_label2])
            idx3 = random.choice(test_indices_by_fine_label[chosen_fine_label3])
            idx4 = random.choice(test_indices_by_fine_label[chosen_fine_label4])
            
            self.image_index_quads.append((idx1, idx2, idx3, idx4))

        self.original_data = test_dataset.data
        self.original_targets = np.array(test_dataset.targets)
        
    def __len__(self): 
        return self.num_samples

    def __getitem__(self, index):
        idx1, idx2, idx3, idx4 = self.image_index_quads[index]
        
        img_tl = self.original_data[idx1]; img_tr = self.original_data[idx2]
        img_bl = self.original_data[idx3]; img_br = self.original_data[idx4]

        label_tl_orig = self.original_targets[idx1]; label_tr_orig = self.original_targets[idx2]
        label_bl_orig = self.original_targets[idx3]; label_br_orig = self.original_targets[idx4]

        combined_image = np.zeros((64, 64, 3), dtype=np.uint8)
        combined_image[0:32, 0:32, :] = img_tl; combined_image[0:32, 32:64, :] = img_tr
        combined_image[32:64, 0:32, :] = img_bl; combined_image[32:64, 32:64, :] = img_br

        new_labels = [
            TASK1_MAP[label_tl_orig], TASK2_MAP[label_tr_orig],
            TASK3_MAP[label_bl_orig], TASK4_MAP[label_br_orig]
        ]
        
        if self.transform: 
            combined_image = self.transform(combined_image)
            
        return combined_image, torch.tensor(new_labels, dtype=torch.long)

# Main Function: CIFAR-100
def get_cifar100(dataset_root, args):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 标准化参数
    ])

    train_dataset_raw = datasets.CIFAR100(root=dataset_root, train=True, download=True)

    train_indices_by_fine_label = {i: [] for i in range(100)}
    for idx, target in enumerate(train_dataset_raw.targets):
        train_indices_by_fine_label[target].append(idx)

    task_fine_labels = {
        1: list(TASK1_MAP.keys()), 2: list(TASK2_MAP.keys()),
        3: list(TASK3_MAP.keys()), 4: list(TASK4_MAP.keys())
    }

    client_volume_proportions = np.random.dirichlet([args.alpha1] * args.num_clients)
    client_sample_counts = (client_volume_proportions * args.total_train_samples).astype(int)
    
    client_datasets = []
    for client_id in range(args.num_clients):
        task_class_dists = {
            1: np.random.dirichlet([args.alpha2] * len(task_fine_labels[1])),
            2: np.random.dirichlet([args.alpha2] * len(task_fine_labels[2])),
            3: np.random.dirichlet([args.alpha2] * len(task_fine_labels[3])),
            4: np.random.dirichlet([args.alpha2] * len(task_fine_labels[4]))
        }
        
        num_samples_for_client = client_sample_counts[client_id]
        client_image_quads = []
        
        for _ in range(num_samples_for_client):
            chosen_fine_label1 = np.random.choice(task_fine_labels[1], p=task_class_dists[1])
            chosen_fine_label2 = np.random.choice(task_fine_labels[2], p=task_class_dists[2])
            chosen_fine_label3 = np.random.choice(task_fine_labels[3], p=task_class_dists[3])
            chosen_fine_label4 = np.random.choice(task_fine_labels[4], p=task_class_dists[4])
            
            idx1 = random.choice(train_indices_by_fine_label[chosen_fine_label1])
            idx2 = random.choice(train_indices_by_fine_label[chosen_fine_label2])
            idx3 = random.choice(train_indices_by_fine_label[chosen_fine_label3])
            idx4 = random.choice(train_indices_by_fine_label[chosen_fine_label4])
            
            client_image_quads.append((idx1, idx2, idx3, idx4))
            
        client_dataset = FourCornerDataset(train_dataset_raw, client_image_quads, transform)
        client_datasets.append(client_dataset)

    train_loaders = {i: DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True, num_workers=4) for i in range(args.num_clients)}
    v_train_dataset = ConcatDataset(client_datasets)
    v_train_loader = DataLoader(v_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset_raw = datasets.CIFAR100(root=dataset_root, train=False, download=True)
    
    global_test_dataset = GlobalFourCornerTestDataset(test_dataset_raw, transform)
    v_test_loader = DataLoader(global_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    test_loaders = {}
    if len(global_test_dataset) > 0:
        test_data_size_per_client = len(global_test_dataset) // args.num_clients
        indices = list(range(len(global_test_dataset)))
        for i in range(args.num_clients):
            start_idx = i * test_data_size_per_client
            end_idx = start_idx + test_data_size_per_client if i != args.num_clients - 1 else len(global_test_dataset)
            subset_indices = indices[start_idx:end_idx]
            client_subset = Subset(global_test_dataset, subset_indices)
            test_loaders[i] = DataLoader(client_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loaders, test_loaders, v_train_loader, v_test_loader



# ==========================================
# 4. FLAME DATASET
# ==========================================
class FireDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.images = []
        self.labels = []  
        self.primary_class_to_idx = {}  
        self.secondary_class_to_idx = {}  
        self._load_data()

    def _load_data(self):
        split = 'train' if self.train else 'test'
        split_dir = os.path.join(self.root, split)

        # Dimension 1 label mapping (0: forest, 1: lake in forest, 2: snow in forest1/2)
        primary_classes = ['forest', 'lake in forest', 'snow in forest1', 'snow in forest2']
        self.primary_class_to_idx = {'forest': 0, 'lake in forest': 1, 'snow in forest1': 2, 'snow in forest2': 2}

        # Dimension 2 label mapping (0: forest, 1: lake in forest, 2: snow in forest1/2)
        self.secondary_class_to_idx = {'forest': 0, 'lake in forest': 0, 'snow in forest1': 0, 'snow in forest2': 1}

        valid_extensions = ['.jpg', '.jpeg', '.png']
        for cls in primary_classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue 
            class_count = 0
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path) and os.path.splitext(img_name)[-1].lower() in valid_extensions:
                    self.images.append(img_path)
                    primary_label = self.primary_class_to_idx[cls]
                    secondary_label = self.secondary_class_to_idx[cls]
                    self.labels.append([primary_label, secondary_label])
                    class_count += 1
            print(f"Class {cls} - Loaded {class_count} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (254, 254), (0, 0, 0))  # Error placeholder image
            with open("error_log.txt", "a") as log_file:
                log_file.write(f"Error loading image: {img_path}\n")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# Main Function: FLME
def get_flame(dataset_root, args):
    num_clients = args.num_clients
    alpha1 = args.alpha1
    alpha2 = args.alpha2
    batch_size = args.batch_size
    total_train_samples = 3000 
    dataset_root = os.path.join(dataset_root, "flame")
    transform = transforms.Compose([
        transforms.Resize((254, 254)),
        transforms.ToTensor(),
    ])

    train_dataset = FireDataset(root=dataset_root, train=True, transform=transform)
    test_dataset = FireDataset(root=dataset_root, train=False, transform=transform)

# Part 1: Organizing the Training Set Index by Primary Label (First Dimension) 
# The primary labels in FLAME Dataset are mapped to 0, 1, and 2, for a total of 3 categories.
    num_classes = 3
    indices_by_primary_class = {cls: [] for cls in range(num_classes)}
    for idx in range(len(train_dataset)):
        primary_label = train_dataset.labels[idx][0]
        indices_by_primary_class[primary_label].append(idx)
        
    for cls, indices in indices_by_primary_class.items():
        if not indices:
            raise ValueError(f"Error: There are no samples in the main category {cls}, so sampling cannot be performed.")

    # Part 2: Calculate the data volume distribution for each client (controlled by alpha1)
    client_volume_proportions = np.random.dirichlet([alpha1] * num_clients)
    client_sample_counts = (client_volume_proportions * total_train_samples).astype(int)

    client_indices = {i: [] for i in range(num_clients)}

    # Part 3: Allocate data to each client (controlled by alpha2) 
    for client_id in range(num_clients):
        class_dist = np.random.dirichlet([alpha2] * num_classes)
        num_samples_for_client = client_sample_counts[client_id]
        
        for _ in range(num_samples_for_client):
            chosen_primary_class = np.random.choice(num_classes, p=class_dist)
            idx = random.choice(indices_by_primary_class[chosen_primary_class])
            client_indices[client_id].append(idx)

    client_datasets = []
    train_loaders = {}
    
    for i in range(num_clients):
        client_subset = Subset(train_dataset, client_indices[i])
        client_datasets.append(client_subset)
        train_loaders[i] = DataLoader(client_subset, batch_size=batch_size, shuffle=True)

    v_train_dataset = ConcatDataset(client_datasets)
    v_train_loader = DataLoader(v_train_dataset, batch_size=batch_size, shuffle=True)
    v_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loaders = {}
    test_data_size_per_client = len(test_dataset) // num_clients
    all_test_indices = list(range(len(test_dataset)))
    
    random.seed(42)
    random.shuffle(all_test_indices)
    
    for i in range(num_clients):
        start_idx = i * test_data_size_per_client
        end_idx = start_idx + test_data_size_per_client if i != num_clients - 1 else len(test_dataset)
        
        client_test_subset = Subset(test_dataset, all_test_indices[start_idx:end_idx])
        test_loaders[i] = DataLoader(client_test_subset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loaders, v_train_loader, v_test_loader




def get_dataset(dataset_root, dataset, args):
    if dataset == 'mnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar10(dataset_root, args)
    elif dataset == 'cifar100':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar100(dataset_root, args)
    elif dataset == 'flame':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_flame(dataset_root, args)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_train_loader, v_test_loader