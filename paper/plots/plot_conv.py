from qml_benchmarks.hyperparam_search_utils import read_data
import matplotlib.pyplot as plt
import numpy as np
from qml_benchmarks.models.quanvolutional_neural_network import QuanvolutionalNeuralNetwork
import seaborn as sns

#### mnist_cg data plot

# note: you need to download the mnist_cg data
X32,y32 = read_data("datasets-for-plots/mnist_cg/mnist_pixels_3-5_32x32_train-20.csv")
X32 = np.reshape(X32, (X32.shape[0], 32, 32))

X16,y16 = read_data("datasets-for-plots/mnist_cg/mnist_pixels_3-5_16x16_train-20.csv")
X16 = np.reshape(X16, (X16.shape[0], 16, 16))

X8,y8 = read_data("datasets-for-plots/mnist_cg/mnist_pixels_3-5_8x8_train-20.csv")
X8 = np.reshape(X8, (X16.shape[0], 8, 8))

X4,y4 = read_data("datasets-for-plots/mnist_cg/mnist_pixels_3-5_4x4_train-20.csv")
X4 = np.reshape(X4, (X4.shape[0], 4, 4))


# Create subplots
fig, axes = plt.subplots(2, 4, figsize=(17, 9),
                        tight_layout=True)  # Adjust the figsize as needed
idx3 = 7
idx5 = -8

images3 = [-X32[idx3], -X16[idx3], -X8[idx3], -X4[idx3]]
images5 = [-X32[idx5], -X16[idx5], -X8[idx5], -X4[idx5]]

# Plot each image in a horizontal line
for i in range(4):
    axes[0][i].imshow(images3[i], cmap='gray')
    axes[0][i].axis('off')  # Turn off axis labels for clarity
    axes[1][i].imshow(images5[i], cmap='gray')
    axes[1][i].axis('off')  # Turn off axis labels for clarity

plt.savefig('figures/mnist_cg.png')

### Bars and stripes plot

plt.clf()

X,y = read_data('datasets-for-plots/bars_and_stripes/bars_and_stripes_16_x_16_0.5noise.csv')
fig, axes = plt.subplots(ncols=4, figsize=(8,8))

axes[0].axis('off')
axes[0].imshow(np.reshape(-X[0], (16,16)), cmap='gray')
axes[1].axis('off')
axes[1].imshow(np.reshape(-X[4], (16,16)), cmap='gray')
axes[2].axis('off')
axes[2].imshow(np.reshape(-X[6], (16,16)), cmap='gray')
axes[3].axis('off')
axes[3].imshow(np.reshape(-X[3], (16,16)), cmap='gray')

plt.savefig('figures/bars_and_stripes.png', bbox_inches='tight')

#### quanv layer plot

plt.clf()

X,y = read_data("datasets-for-plots/mnist_cg/mnist_pixels_3-5_16x16_train-20.csv")

model = QuanvolutionalNeuralNetwork(n_qchannels=3)
model.initialize(16*16)

data = np.concatenate((X[-5:], X[:5]))
X_out = model.batched_quanv_layer(model.transform(data))

idx3 = 8
idx5 = 1
fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(15,7))
axes[0][0].axis('off')
axes[0][0].imshow(np.reshape(-data[idx3], (16,16)), cmap='gray')
axes[1][0].axis('off')
axes[1][0].imshow(np.reshape(-data[idx5], (16,16)), cmap='gray')
for i in range(0,3):
    axes[0][i+1].imshow(X_out[idx3,:,:,i].T, cmap='gray')
    axes[0][i+1].axis('off')
    axes[1][i+1].imshow(X_out[idx5,:,:,i].T, cmap='gray')
    axes[1][i+1].axis('off')

plt.savefig("figures/quanv_map.png", bbox_inches='tight')





