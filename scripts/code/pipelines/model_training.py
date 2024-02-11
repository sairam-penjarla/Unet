import torch
from tqdm import tqdm
from torch import optim, nn
from code.components.dataset import CarvanaDataset
from code.components.model_architecture import UNet
from torch.utils.data import DataLoader, random_split
from code.logging import logger  # Importing logger module for logging

class ModelTraining():
    """
    Class for model training.
    """
    def __init__(self, config: dict) -> None:
        """
        Initializes the ModelTraining class.

        Args:
            config (dict): Configuration dictionary containing parameters and paths.
        """
        self.LEARNING_RATE = config['params']['lr']
        self.BATCH_SIZE = config['params']['batch_size']
        self.EPOCHS = config['params']['epochs']
        self.DATA_PATH = config['paths']['data_path']
        self.MODEL_SAVE_PATH = config['paths']['model_path']
        self.torch_manual_seed  = config['params']['torch_manual_seed']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        """
        Executes the training process.
        """
        logger.info("Starting model training.")  # Logging start of model training
        train_dataset = CarvanaDataset(self.DATA_PATH)
        generator = torch.Generator().manual_seed(self.torch_manual_seed)

        train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

        train_dataloader = DataLoader(
                                    dataset = train_dataset,
                                    batch_size = self.BATCH_SIZE,
                                    shuffle = True
                                    )
        val_dataloader = DataLoader(
                                    dataset = val_dataset,
                                    batch_size = self.BATCH_SIZE,
                                    shuffle = True
                                    )
        model = UNet(in_channels=3, num_classes=1).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in tqdm(range(self.EPOCHS)):
            model.train()
            train_running_loss = 0
            for idx, img_mask in enumerate(tqdm(train_dataloader)):
                img = img_mask[0].float().to(self.device)
                mask = img_mask[1].float().to(self.device)

                y_pred = model(img)
                optimizer.zero_grad()

                loss = criterion(y_pred, mask)
                train_running_loss += loss.item()
                
                loss.backward()
                optimizer.step()

            train_loss = train_running_loss / (idx + 1)

            model.eval()
            val_running_loss = 0
            with torch.no_grad():
                for idx, img_mask in enumerate(tqdm(val_dataloader)):
                    img = img_mask[0].float().to(self.device)
                    mask = img_mask[1].float().to(self.device)
                    
                    y_pred = model(img)
                    loss = criterion(y_pred, mask)

                    val_running_loss += loss.item()

                val_loss = val_running_loss / (idx + 1)

            print("-"*30)
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
            print("-"*30)

        torch.save(model.state_dict(), self.MODEL_SAVE_PATH)
        logger.info("Model training completed.")  # Logging completion of model training
