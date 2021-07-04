from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms as T

class MNISTDataModule(pl.LightningDataModule):
    
    def __init__(self, root_dir):
        super(MNISTDataModule, self).__init__()
        self.root_dir = root_dir
        
        self.batch_size = 128
        self.val_size = 1000
        
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5), (0.5))
        ])

    def prepare_data(self):
        MNIST(self.root_dir, train=True, download=True)
        MNIST(self.root_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = MNIST(
                self.root_dir, train=True, transform=self.transforms
            )
            self.val_dataset = MNIST(
                self.root_dir, train=False, transform=self.transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_size
        )
