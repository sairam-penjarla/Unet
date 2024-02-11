import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from code.components.model_architecture import UNet
from scripts.code.components.dataset import CarvanaDataset
from code.logging import logger  # Importing logger module for logging

class ModelInferencing():
    """
    Class for model inferencing.
    """
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def pred_show_image_grid(self, data_path, model_pth):
        """
        Predict and display a grid of images with original and predicted masks.

        Args:
            data_path (str): Path to the data directory.
            model_pth (str): Path to the model checkpoint.
        """
        logger.info("Starting prediction and display of image grid.")  # Logging start of prediction and display
        model = UNet(in_channels=3, num_classes=1).to(self.device)
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(self.device)))
        image_dataset = CarvanaDataset(data_path, test=True)
        images = []
        orig_masks = []
        pred_masks = []

        for img, orig_mask in image_dataset:
            logger.info("Performing inference on an image.")  # Logging inference on an image
            img = img.float().to(self.device)
            img = img.unsqueeze(0)

            pred_mask = model(img)

            img = img.squeeze(0).cpu().detach()
            img = img.permute(1, 2, 0)

            pred_mask = pred_mask.squeeze(0).cpu().detach()
            pred_mask = pred_mask.permute(1, 2, 0)
            pred_mask[pred_mask < 0]=0
            pred_mask[pred_mask > 0]=1

            orig_mask = orig_mask.cpu().detach()
            orig_mask = orig_mask.permute(1, 2, 0)

            images.append(img)
            orig_masks.append(orig_mask)
            pred_masks.append(pred_mask)

        images.extend(orig_masks)
        images.extend(pred_masks)
        fig = plt.figure()
        for i in range(1, 3*len(image_dataset)+1):
            fig.add_subplot(3, len(image_dataset), i)
            plt.imshow(images[i-1], cmap="gray")
        plt.show()
        logger.info("Prediction and display of image grid completed.")  # Logging completion of prediction and display

    def single_image_inference(self, image_pth, model_pth):
        """
        Perform inference on a single image and display original and predicted masks.

        Args:
            image_pth (str): Path to the image file.
            model_pth (str): Path to the model checkpoint.
        """
        logger.info("Performing inference on a single image.")  # Logging inference on a single image
        model = UNet(in_channels=3, num_classes=1).to(self.device)
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(self.device)))

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

        img = transform(Image.open(image_pth)).float().to(self.device)
        img = img.unsqueeze(0)
    
        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0]=0
        pred_mask[pred_mask > 0]=1

        fig = plt.figure()
        for i in range(1, 3): 
            fig.add_subplot(1, 2, i)
            if i == 1:
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(pred_mask, cmap="gray")
        plt.show()
        logger.info("Inference on a single image completed.")  # Logging completion of inference
