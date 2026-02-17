

import os 
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder 
from PIL import Image, ImageFile, PngImagePlugin
import torch.distributed as dist 
 
from data.data_augmentation_rgb import DataAugmentationDINO 
from data.samplers import InfiniteSampler 
 
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 16 * (1024**2)  # 16MB

class ZScoreNormalize:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: Tensor [1, H, W]
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + self.eps)

class RGBDatasetWithAugmentation(Dataset):
    def __init__(self, root='dataset/remote', split='train',  augmentation=None):
        
        self.split = split

        folder_path = os.path.join(root, split) 

        self.base_dataset = ImageFolder(
            root=folder_path,
            transform=None  
        )

        self.to_gray = transforms.Grayscale(num_output_channels=1)

        self.augmentation = augmentation


        self.normalize = transforms.Compose([      
            transforms.ToTensor(),
            ZScoreNormalize()
        ])

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = self.to_gray(img)  
        # Eğitim modunda augmentation uygula
        if self.split == 'train' and self.augmentation is not None:
            crops = self.augmentation(img) #Les crops sont normalisés via l'augmentation
            img = self.normalize(img)
            
            return img, crops
        else:
            img = self.normalize(img)  
            return img, label
    
    def __len__(self):
        return len(self.base_dataset)
    
    def get_classes(self):
        return self.base_dataset.classes
 
class RGBDatasetLoader:
    def __init__(self, cfg,advance=0, seed=42,  is_dino=True , is_infsampler=True ): 
        assert dist.is_initialized(), "Distributed computing is not initialized! " 

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.data_dir = cfg.dataset.dataset_path
        self.num_workers = cfg.train.num_workers 
        self.mean = cfg.dataset.mean
        self.std = cfg.dataset.std 
        self.resize = getattr(cfg.dataset, "resize", False)
        self.size = getattr(cfg.dataset, "size", (224,224))
        
        batch_size = getattr(cfg.train, "batch_size_per_gpu", None)
        if batch_size:
            self.batch_size=batch_size
        else: 
            self.global_batch_size = getattr(cfg.train, "global_batch_size", None)

            self.batch_size=self.global_batch_size//self.world_size
         
        
        augmentation=None 
        if is_dino:
            augmentation = DataAugmentationDINO(
                cfg=cfg
            ) 
        train_dataset = RGBDatasetWithAugmentation(
            root=self.data_dir,
            split='train',
            augmentation=augmentation, 
            mean=self.mean,
            std=self.std,
            resize=self.resize,
            size=self.size
        )
        print('train_dataset len: ',len(train_dataset) ) 
        valid_split = 'valid'
        if not os.path.exists(os.path.join(self.data_dir, 'valid')) and os.path.exists(os.path.join(self.data_dir, 'test')):
            valid_split = 'test'
            #print(f"'valid' directory not found, using '{valid_split}' directory instead.")
        
        valid_dataset = RGBDatasetWithAugmentation(
            root=self.data_dir,
            split=valid_split, 
            mean=self.mean,
            std=self.std,
            resize=self.resize
        )
        sampler=None
        if is_infsampler: 
            sampler = InfiniteSampler(
                        shuffle=cfg.dataset.shuffle,
                        advance=advance,
                        sample_count=len(train_dataset),
                        seed=seed,
                        rank=self.rank,
                        world_size=self.world_size
                    )
        self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=sampler
            ) 
         
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
         
        self.classes = train_dataset.get_classes()
        self.num_classes = len(self.classes)
        
    def get_train_loader(self):
        return self.train_loader
    
    def get_valid_loader(self):
        return self.valid_loader
    
    def get_loaders(self):
        return self.train_loader, self.valid_loader
    
    def get_classes(self):
        return self.classes
    
    def get_num_classes(self):
        return self.num_classes
