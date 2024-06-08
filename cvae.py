# Run with GPU
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from utils import SmilesDataset, image_grid
from plotters import LatentPlotter
from models import CVAE
from PIL import Image
from torchvision import models

torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

# Define Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[0, 5, 10, 19, 28], use_gpu=True):
        super(PerceptualLoss, self).__init__()
        self.feature_layers = feature_layers
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.vgg = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features.to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0
        x = torch.cat((x, x, x), 1) if x.size(1) == 1 else x
        y = torch.cat((y, y, y), 1) if y.size(1) == 1 else y
        x, y = x.to(self.device), y.to(self.device)
        for i, layer in enumerate(self.vgg):
            x = layer(x.clone())
            y = layer(y.clone())
            if i in self.feature_layers:
                loss += F.mse_loss(x, y)
        return loss

def main(LATENT_SIZE, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, RESIZE, BETA):
    EVAL_FREQ = 5

    # prepare the data
    transform = transforms.Compose(
                    [
                    transforms.Resize((RESIZE,RESIZE),interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    ])
    
    transform_eval = transforms.Compose(
                    [
                    transforms.Resize((RESIZE,RESIZE),interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                    ])

    dataset = SmilesDataset(csv_file="datasets/smiles_dataset.txt", root_dir="datasets/images/", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataset_eval = SmilesDataset(csv_file="datasets/smiles_dataset.txt", root_dir="datasets/images/", transform=transform_eval)
    dataloader_eval = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    for sample in dataloader:
        input_shape = sample["image"].shape
        break
    
    # prepare the network and plotting functions
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cvae = CVAE(input_size=input_shape, latent_size=LATENT_SIZE).to(device)
    latent_plot = LatentPlotter(ndim=LATENT_SIZE, file_path="results/latent_variance.txt")
    optimizer = torch.optim.Adam(cvae.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9) # step size was 30 before

    # Perceptual loss
    perceptual_loss = PerceptualLoss(use_gpu=torch.backends.mps.is_available())

    # training loop
    cvae.train()
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = 0.0
        for sample in dataloader:
            optimizer.zero_grad()

            images = sample["image"].float().to(device)  # Convert to float32 before moving to device
            encodings = sample["encode"].float().to(device)  # Convert to float32 before moving to device

            outdict = cvae(images, encodings)
            pz, qz, z, x, xhat = outdict["pz"], outdict["qz"], outdict["z"], outdict["x"], outdict["xhat"]
            latent_plot(z_values=z.detach().cpu().numpy())

            # compute loss
            MSE = torch.mean((xhat - x)**2)
            KLD = -BETA * torch.mean(torch.mean(1 + torch.log(qz.sigma**2) - qz.mu**2 - torch.exp(qz.sigma**2)))
            perceptual_loss_value = perceptual_loss(xhat, x)
            loss = MSE + KLD + perceptual_loss_value

            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item()

        latent_plot.reset()
        scheduler.step()
        
        # Loop to generate images (done every EVAL_FREQ)
        if (epoch % EVAL_FREQ == 0) or (epoch == 1):
            cvae.eval()
            for sample in dataloader_eval:
                optimizer.zero_grad()
                images = sample["image"].float().to(device)  # Convert to float32 before moving to device
                encodings = sample["encode"].float().to(device)  # Convert to float32 before moving to device
                outdict = cvae(images, encodings)
                x, xhat = outdict["x"], outdict["xhat"]
                break
            
            images = []
            for img in xhat:
                img = (255*img).cpu().detach().numpy().astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))
                pil_image = Image.fromarray(img).convert("RGB")
                images.append(pil_image)
            
            grid = image_grid(images, 4, 8, RESIZE, RESIZE)
            grid.save(f"results/iterations/iter_{epoch}.png")
            

            if epoch == 1:
                images = []
                for img in x:
                    img = (255*img).cpu().detach().numpy().astype(np.uint8)
                    img = np.transpose(img, (1, 2, 0))
                    pil_image = Image.fromarray(img).convert("RGB")
                    images.append(pil_image)
                grid = image_grid(images, 4, 8, RESIZE, RESIZE)
                grid.save(f"results/iterations/original.png")

        # save a snapshot of the netwrok every 100 epochs
        if epoch % 100 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': cvae.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        }, "model_savepoints/model.pt")

        # update lr scheduler
        last_lr = scheduler.get_last_lr()[0]

        # print some info to the terminal and to the traning results file
        print(f"epoch:{epoch}, loss:{train_loss}, mse: {MSE}, kld: {KLD}, perceptual_loss: {perceptual_loss_value}, lr: {last_lr}")
        with open('results/training.txt', 'a') as f:
            print(epoch,train_loss,MSE.item(),KLD.item(),perceptual_loss_value.item(),last_lr, file=f)

    # always save on the last iteration
    torch.save({
                            'epoch': epoch,
                            'model_state_dict': cvae.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_loss,
                            }, "model_savepoints/model.pt")
    
if __name__ == "__main__":

    import argparse
    # parsing user input
    # example: python cvae.py --n_epochs=100 --lr=0.00002 --beta=0.5 --batch_size=8 --latent_size=100 --resize=20 
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", help="Number of epochs (defaults 100)", default=100, type=int)
    parser.add_argument("--lr", help="Learning rate (defaults 0.00002)", default=0.0002, type=float)
    parser.add_argument("--beta", help="Beta constant (defaults 0.5)", default=0.5, type=float)
    parser.add_argument("--batch_size", help="Batch size (defaults 8)", default=8, type=int)
    parser.add_argument("--latent_size", help="Size of the latent space (defaults to 100)", default=100, type=int)
    parser.add_argument("--resize", help="Pixels of the resized image (defaults to 20)", default=20, type=int)
    args = parser.parse_args()

    main(LATENT_SIZE=args.latent_size, NUM_EPOCHS=args.n_epochs, LEARNING_RATE=args.lr, BATCH_SIZE=args.batch_size, RESIZE=args.resize, BETA=args.beta)
