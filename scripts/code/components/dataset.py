import os
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from code.logging import logger  # Importing logger module for logging

class CarvanaDataset(Dataset):
    """
    Dataset class for the Carvana dataset.
    
    Args:
        root_path (str): Root directory of the dataset.
        test (bool, optional): If True, initializes the dataset in test mode.
    """
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if test:
            logger.info("Initializing Carvana dataset in test mode.")  # Logging initialization in test mode
            self.images = sorted([root_path+"/manual_test/"+i for i in os.listdir(root_path+"/manual_test/")])
            self.masks = sorted([root_path+"/manual_test_masks/"+i for i in os.listdir(root_path+"/manual_test_masks/")])
        else:
            logger.info("Initializing Carvana dataset in train mode.")  # Logging initialization in train mode
            self.images = sorted([root_path+"/train/"+i for i in os.listdir(root_path+"/train/")])
            self.masks = sorted([root_path+"/train_masks/"+i for i in os.listdir(root_path+"/train_masks/")])
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.
        
        Args:
            index (int): Index of the item to retrieve.
        
        Returns:
            tuple: Tuple containing transformed image and mask.
        """
        logger.info(f"Loading item at index {index}.")  # Logging loading of item at specified index
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        transformed_img = self.transform(img)
        transformed_mask = self.transform(mask)

        logger.info(f"Item at index {index} loaded and transformed.")  # Logging loading and transformation completion
        return transformed_img, transformed_mask

    def __len__(self):
        """
        Returns the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        length = len(self.images)
        logger.info(f"Length of dataset: {length}.")  # Logging length of dataset
        return length
