import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('training.txt', delim_whitespace=True, header=None, 
                   names=["epoch", "train_loss", "MSE", "KLD", "perceptual_loss", "last_lr"])

# Plot the perceptual loss
plt.figure(figsize=(10, 6))
plt.plot(data["epoch"], data["perceptual_loss"], label="Perceptual Loss")
plt.title("Perceptual Loss")
plt.xlabel("Epoch")
plt.ylabel("Perceptual Loss")
plt.legend()
plt.grid(True)
plt.show()
