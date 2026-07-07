# IMGS789 - Machine Learning for Difficult Data
# Max Bakalos
# 7/3/2026

# Project (Topic #5): Few-Shot Learning with Prototypical Network

#############################################################
# Setup

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
import time
from collections import defaultdict
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef

# import matplotlib
# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

##########################################################################################
# FUNCTIONS

def checkGPU():
    # Make sure PyTorch is configured correctly
    print("PyTorch Version: %s" % torch.__version__)
    # Check that the GPU is working with PyTorch
    cuda_available = torch.cuda.is_available()
    print("CUDA is available? : %r" % cuda_available)
    if cuda_available:
        print("Device: " + torch.cuda.get_device_name(0))
    device = torch.device("cuda" if cuda_available else "cpu")
    print("Using:", device)
    # Return Device String: "cpu" or "gpu"
    return device
    

def image_show(im, colormap="gray", title=None):
    if torch.is_tensor(im):
        im = im.squeeze().cpu()

    plt.imshow(im, cmap=colormap)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()
    
def load_data(data_set):
    # load entire dataset into tensors
    images = []
    labels = []
    
    for i in range(len(data_set)):
        im, lbl = data_set[i]
        images.append(im)
        labels.append(lbl)
    
    return torch.stack(images), torch.tensor(labels)

def class_indexer(dataset, not_included=None):
    class_indices = defaultdict(list)
    
    labels = dataset._flat_character_images

    for idx, (image_file_name, label) in enumerate(labels):
        class_indices[label].append(idx)

    # Remove unwanted class indices
    if not_included == None:
        not_included = [] # replace none with empty list
    for n in not_included:
        del class_indices[n]

    return class_indices


def episode_samples(image_data, label_data, class_indices, n_way=5, n_support=5, n_query=10):
    # Initialize Support/Query Inputs and Labels
    support_images = []
    support_labels = []
    support_ep_labels = []
    query_images = []
    query_labels = []
    query_ep_labels = []

    # Select random classes (# of classes chosen = n_way)
    selected_classes = random.sample( list(class_indices.keys()), n_way )

    for episode_label, class_label in enumerate(selected_classes):

        # Pick random image data indeces from this class
        x_indices = random.sample( class_indices[class_label], (n_support + n_query) )

        # Support set (1st part of selected group of indeces)
        for i in x_indices[:n_support]:
            # x_im, label = dataset[i]
            x_im = image_data[i]
            label = label_data[i]

            support_images.append(x_im)
            support_labels.append(label)
            support_ep_labels.append(episode_label)

        # Query set (2nd part of selected group of indeces)
        for i in x_indices[n_support:]:
            # x_im, label = dataset[i]
            x_im = image_data[i]
            label = label_data[i]

            query_images.append(x_im)
            query_labels.append(label)
            query_ep_labels.append(episode_label)

    # Format Selected Episode Data into Stacks
    support_images = torch.stack(support_images)
    query_images = torch.stack(query_images)
    support_labels = torch.tensor(support_labels)
    query_labels = torch.tensor(query_labels)
    support_ep_labels = torch.tensor(support_ep_labels)
    query_ep_labels = torch.tensor(query_ep_labels)

    return ( support_images, support_labels, support_ep_labels,
             query_images, query_labels, query_ep_labels, 
             torch.tensor(selected_classes) )

def prototypes(data_features, labels, num_classes):
    proto = []

    for c in range(num_classes):
        class_emb = data_features[labels == c]

        if class_emb.shape[0] == 0:
            raise ValueError(f"Empty class in episode: {c}")

        proto.append(class_emb.mean(dim=0))

    return torch.stack(proto)

def test_model(model, shot, way, query_samples_per_class, data_set, image_data, label_data, class_indices, iterations, device):
        
    # way_test = 3
    # test_runs_amount = 10
    # episodes_per_test_run = 100
    # acc_tot = 0
    
    # pred_tot = torch.empty(0, dtype=torch.long, device=device)
    # epLabel_tot = torch.empty(0, dtype=torch.long, device=device)
    yLabel_tot = torch.empty(0, dtype=torch.long, device=device)
    predY_tot = torch.empty(0, dtype=torch.long, device=device)
    
    avg_loss = 0
    
    accuracy = np.zeros(iterations)
    
    # Put Network in Evaluation Mode
    model.eval()
    
    with torch.no_grad():
        
        # For each iteration...
        for i in range(iterations):
            
            # Get Support & Query Episode Samples
            te_support_x, te_support_y, te_support_ep, te_query_x, te_query_y, te_query_ep, te_classes = episode_samples( image_data, label_data, class_indices, n_way=way, n_support=shot, n_query=query_samples_per_class )
            # Move to GPU
            te_support_x = te_support_x.to(device)
            te_support_y = te_support_y.to(device)
            te_support_ep = te_support_ep.to(device)
            te_query_x = te_query_x.to(device)
            te_query_y = te_query_y.to(device)
            te_query_ep = te_query_ep.to(device)
    
            # Pass forward through the network to get latent space feature representations of data
            te_support_features = model(te_support_x)
            te_query_features = model(te_query_x)
    
            # Find the Prototypes
            support_prototypes = prototypes(te_support_features, te_support_ep, way)
    
            # Find the distance to each prototype
            distances = torch.cdist(te_query_features, support_prototypes)
            
            # Calculate Loss
            loss = loss_criterion(lgSftmx(-distances), te_query_ep)
            
            # Predicted class => shortest distance
            predictions = torch.argmin(distances, dim=1)
            
            # Accuracy: % of true predictions
            accuracy[i] = (predictions == te_query_ep).float().mean()
            
            
            # Add onto running loss sum
            avg_loss += loss.item()
            # # Add onto running accuracy
            # acur_tot += accuracy[i]
            
            # Append Y Label Total
            yLabel_tot = torch.cat((yLabel_tot, te_query_y))
            # Append Total Y Predictions
            te_classes = te_classes.to(device)
            predY_tot = torch.cat((predY_tot, te_classes[predictions]))
            
        
        avg_loss = avg_loss / iterations
        # acur_tot = acur_tot / iterations
        
        # Accuracy Mean & Standard Deviation
        avg_accuracy = np.mean(accuracy) # average
        std_accuracy = np.std(accuracy, ddof=1) # standard deviation (ddof=1 makes it the sample standard deviation)
        
        SEM =  std_accuracy / np.sqrt(iterations) # standard error of the mean
        confidence_interval_95_percent = 1.96 * SEM # C.I.
        # lo_conf_bound_acur = avg_accuracy - confidence_interval_95_percent
        # hi_conf_bound_acur = avg_accuracy + confidence_interval_95_percent
        
        # Matthews Correlation Coefficient
        MCC = matthews_corrcoef( yLabel_tot.cpu().numpy(), predY_tot.cpu().numpy() )
        
        # # Confusion Matrix
        # test_labels_numeric = torch.sort(te_classes)[0].tolist()
        # test_labels_text = [ data_set._characters[x] for x in test_labels_numeric ]
        # conf_mat = confusion_matrix(yLabel_tot.cpu().numpy(), predY_tot.cpu().numpy(),
        #                             labels=test_labels_numeric, normalize='true')
        
        return ( avg_loss, 
                 avg_accuracy, std_accuracy, 
                 confidence_interval_95_percent,
                 MCC  
                  )


########################################################################

class EpisodicBatchSampler(torch.utils.data.Sampler):
    def __init__(self, class_indices, n_way, n_episodes):
        self.class_indices = class_indices
        self.classes = list(class_indices.keys())
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield random.sample(self.classes, self.n_way)

##################################################################################
# CNN Architecture

# class LatentSpacerNetV2(nn.Module):
#     def __init__(self):
#         super(LatentSpacerNetV2, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 7, padding=3) # 6 feature maps
#         self.conv2 = nn.Conv2d(32, 64, 5, padding=2) # 16 feature maps
#         self.conv3 = nn.Conv2d(64, 64, 5, padding=2) # 32 feature maps(120, 84)
#         self.pool = nn.MaxPool2d(2)
#         self.embedding = nn.Linear(64*13*13, 128) # vector latent-space embedding

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1)
#         x = self.embedding(x) # embed into latent space
#         x = F.normalize(x, p=2, dim=1) # normalize results
#         return x

class LatentSpacerNetV2(nn.Module):
    def __init__(self):
        super(LatentSpacerNetV2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3) # 6 feature maps
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2) # 16 feature maps
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2) # 32 feature maps(120, 84)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((4,4)) # adaptive average pooling (each feature image is reduced to its average value)
        self.embedding = nn.Linear(64*4*4, 128) # vector latent-space embedding

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.avgpool(x) # average pool
        x = torch.flatten(x, 1) # flatten into vector
        x = self.embedding(x) # embed into latent space
        x = F.normalize(x, p=2, dim=1) # normalize results
        return x
    

###################################################################################################
# %% MAIN

# def main():
device = checkGPU()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LOAD DATA

batch_size = 8

# Transformation to apply to each input sample image
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: 1.0 - x)]) # invert images so that text (important part) is white and background is black

# Load Data from Built-In Datasets (Fashion MNIST --> 28x28 grayscale images of clothes)

# Training Set ("Background")
train_set = torchvision.datasets.Omniglot(
    root="data", 
    background=True, # backgorund set (train)
    download=True, 
    transform=data_transform)

# Testing Set ("Evaluation")
test_set = torchvision.datasets.Omniglot(
    root="data", 
    background=False, # evaluation set (test)
    download=True, 
    transform=data_transform)

# # Create DataLoaders
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# Pre-load the Dataset so images don't need to be loaded from folders during training
print("Pre-Loading Training Data")
tr_images, tr_labels = load_data(train_set)
print("Pre-Loading Testing Data")
te_images, te_labels = load_data(test_set)

# Move The Dataset to the GPU
tr_images = tr_images.to(device)
te_images = te_images.to(device)
tr_labels = tr_labels.to(device)
te_labels = te_labels.to(device)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Support-Query Split
print("Indexing Classes for Support-Query Split")
train_class_indices = class_indexer(train_set)
test_class_indices = class_indexer(test_set)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training Setup

net = LatentSpacerNetV2().to(device) # put the network onto the GPU if available
# Check if GPU is used
print("Convolutional Neural Network Device: %s" % next(net.parameters()).device)

# Loss Function
lgSftmx = nn.LogSoftmax(dim=1)
loss_criterion = nn.NLLLoss()

# Optimization Algorithm: Stochastic Gradient Descent (SGD)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam( net.parameters(), lr=6e-4 )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training

print("Training...")

# Start Training Timer
start = time.time()

total_epochs = 20 # total number of epochs
episodes_per_epoch = 100 # number of episodes within each epoch


way = 5 # number of classes
shot = 5 # number of samples per class
query_samples_per_class = 15 # number of samples in the query set
# the number of samples in the support set are [way × shot]

way_test = 5 # number of classes for the test (while still training)
test_iterations = 50 # number of tests to average (while still training)

# # Setup Plot
# plt.ion()
# fig, ax = plt.subplots()
# loss_line = ax.plot([], [])[0]
# ax.set(xlim=(0, total_epochs-1), ylim=(0, 1))
# plt.show()
# epoch_losses = []

# Put Network in Training Mode
net.train()



# For each epoch...
for epoch in range(total_epochs):
    running_loss = 0.0 # reset running loss
    running_accuracy = 0.0 # reset running accuracy
    
    # For each episode
    for i in range(episodes_per_epoch):
        # GEt Support & Query Episode Samples
        tr_support_x, tr_support_y, tr_support_ep, tr_query_x, tr_query_y, tr_query_ep, tr_classes = episode_samples( tr_images, tr_labels, train_class_indices, n_way=way, n_support=shot, n_query=query_samples_per_class )
        # Move to GPU
        tr_support_x = tr_support_x.to(device)
        tr_support_y = tr_support_y.to(device)
        tr_support_ep = tr_support_ep.to(device)
        tr_query_x = tr_query_x.to(device)
        tr_query_y = tr_query_y.to(device)
        tr_query_ep = tr_query_ep.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Pass forward through the network to get latent space feature representations of data
        tr_support_features = net(tr_support_x)
        tr_query_features = net(tr_query_x)

        # Find the Prototypes
        support_prototypes = prototypes(tr_support_features, tr_support_ep, way)

        # Find the distance to each prototype
        distances = torch.cdist(tr_query_features, support_prototypes)
        
        loss = loss_criterion(lgSftmx(-distances), tr_query_ep)

        # Backpropagate the loss
        loss.backward() # backpropagation

        # Move towards a smaller loss
        optimizer.step() # gradient descent
        
        # Add onto running loss sum
        running_loss += loss.item()
        
        # Metrics ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        
        # Predicted class => shortest distance
        predictions = torch.argmin(distances, dim=1)
        
        # Accuracy: % of true predictions
        accuracy = (predictions == tr_query_ep).float().mean()
        
        # Add onto running accuracy
        running_accuracy += accuracy.item()

    avg_epoch_loss = running_loss / episodes_per_epoch
    
    print('[Epoch %d, Episodes %d ]\n- Train  |  Loss: %.3f  Accuracy: %.2f%%' %
      (epoch + 1,
       episodes_per_epoch,
       avg_epoch_loss,
       100*running_accuracy/ episodes_per_epoch))
    
    # Test Model (just to keep track of progress, DO NOT USE TO CHANGE MODEL!!)
    ( te_avg_loss, 
      te_avg_acur, te_std_acur, te_CI, 
      te_MCC  
       ) = test_model(net, shot, way_test, query_samples_per_class, 
                                      test_set, te_images, te_labels, 
                                      test_class_indices, test_iterations, device)
    
    # Test Results
    print("- Test   |  Loss: %.3f  Accuracy: %.2f%%" % (te_avg_loss, 100*te_avg_acur))
    # ConfusionMatrixDisplay(te_conf_mat, display_labels=test_labels_text).plot()
    plt.draw()
    
    # # Update Plot
    # epoch_losses.append(avg_epoch_loss) # y-data update
    # loss_line.set_data(np.arange(0,epoch+1), epoch_losses) # update plot data
    # fig.canvas.draw_idle() # draw
    # plt.pause(0.1)

# Confusion Matrix
# ConfusionMatrixDisplay(te_conf_mat, display_labels=test_labels_text).plot()

# End training timer
end = time.time()
print("Training Time: %f seconds" % (end - start))

print('Finished Training')


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TEST

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("|||||||||||||||||||||||||||||||||||||||||")

# Start Testing Timer
start = time.time()

way_final_test = 5
final_test_iterations = 1000

print("Testing with %d iterations... " % final_test_iterations)
# loss_tot, acur_tot, conf_mat, test_labels_text = test_model(net, shot, way_final_test, query_samples_per_class, 
#                                                             test_set, te_images, te_labels, 
#                                                             test_class_indices, final_test_iterations, device)
( avg_loss, 
  avg_acur, std_acur, CI_95perc, 
  MCC
   ) = test_model(net, shot, way_final_test, query_samples_per_class, 
                                  test_set, te_images, te_labels, 
                                  test_class_indices, final_test_iterations, device)

    
# %% Final Results
print("_______________________\nTEST RESULTS")
print("Loss: %.3f" % (avg_loss))
print("Accuracy: %.2f%% ± %.3f%%  |  Standard Deviation: %.2f%%" % (100*avg_acur, 100*CI_95perc, 100*std_acur))
# print("Standard Deviation of Accuracy: %.2f%%" % (100 *std_acur))
# print("Average Accuracy: %.2f%%" % (100 *avg_acur))
print("MCC: %.3f" % MCC)
# ConfusionMatrixDisplay(conf_mat, display_labels=test_labels_text).plot();


# End testing timer
end = time.time()
print("Testing Time: %f seconds" % (end - start))

print('Finished Testing')

# if __name__ == "__main__":
#     main()