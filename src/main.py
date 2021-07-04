import pytorch_lightning as pl
from model import MNISTClassifier
from datamodule import MNISTDataModule
from pytorch_lightning.callbacks import EarlyStopping
from app import create_app
import os

def main():
    MNIST_dm = MNISTDataModule("data")

    if not os.path.exists("checkpoints/model.ckpt"):
        model = MNISTClassifier()
        print("starting training.")
        trainer = pl.Trainer(
            max_epochs=10,
            callbacks=[EarlyStopping('val_loss', patience=3)],
            progress_bar_refresh_rate=20
        )

        trainer.fit(model, datamodule=MNIST_dm)
        print(f"Training accuracy is {model.train_acc.compute()}")
        print(f"Validation accuracy is {model.val_acc.compute()}")
    else:
        print("loading from checkpoint.")
        model = MNISTClassifier.load_from_checkpoint("checkpoints/model.ckpt")
    
    print("launching app.")
    model.eval()
    app = create_app(model)

if __name__ == '__main__':
    main()
