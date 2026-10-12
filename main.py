import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


transform =transforms.ToTensor()

train_data=datasets.MNIST(root='./data' ,train=True,download=True,transform=transform)
test_data=datasets.MNIST(root='./data' ,train=False,download=True,transform=transform)
#print(test_data)

#test batch
train_loader=DataLoader(train_data,batch_size=10,shuffle=True)
test_loader=DataLoader(test_data,batch_size=10,shuffle=False)


class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #input_channels(image) out_channels(filters-features) kernal_size(filter size) stride(filter step)
        self.conv_layer_1=nn.Conv2d(1,6,3,1,padding=0)
        self.conv_layer_2=nn.Conv2d(6,16,3,1,padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 *5, 120) #16 feature maps*h*w || 5*5 is the h*w after conv1,conv2 and their max pooling layers 
        #input 28*28 -> [conv_layer_1 => 26*26 ->pooling=>13*13] ->[conv_layer_2=> 11*11 -> pooling=>5*5]//padding zero
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #10 is 0-9 output 

    def forward(self,X):
        X=F.relu(self.conv_layer_1(X))
        X=F.max_pool2d(X,2) #without stride, pytorch defaults stride to the kernel size
        #2nd pass
        X=F.relu(self.conv_layer_2(X))
        X=F.max_pool2d(X,2) 

        X = torch.flatten(X, 1)

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X

torch.manual_seed(67)
model=BasicCNN()
#print(model)

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



def train_model(model, train_loader, criterion, optimizer, epochs=5):
    train_losses = []
    train_accuracies = []
    train_errors = []


    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct_predictions = 0
        total_incorrect_predictions = 0
        total_samples = 0
        #for images, true_labels in train_loader:
        for batch_index, (images, true_labels) in enumerate(train_loader):

            y_pred = model(images)
            loss = criterion(y_pred, true_labels)
            optimizer.zero_grad()
            loss.backward() #backprop
            optimizer.step() #ppdate weights

 
   
            predicted_labels = y_pred.argmax(dim=1)
            # metrics
            batch_size = true_labels.size(0)
            batch_correct = (predicted_labels == true_labels).sum().item()
            batch_incorrect = (predicted_labels != true_labels).sum().item()

            total_loss += loss.item() * batch_size
            total_correct_predictions += batch_correct
            total_incorrect_predictions += batch_incorrect
            total_samples += batch_size

        epoch_loss = total_loss / total_samples
        epoch_accuracy = 100 * total_correct_predictions / total_samples
        epoch_error = 100 * total_incorrect_predictions / total_samples

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        train_errors.append(epoch_error)

        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Average loss: {epoch_loss:.4f}")
        print(f"Training error: {epoch_error:.2f}%")
        print(f"Training accuracy: {epoch_accuracy:.2f}%")
        print("-" * 50)

        epoch_accuracy = 100 * total_correct_predictions / total_samples

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
    return train_losses, train_accuracies

def test_model(model, test_loader, criterion):


    model.eval()

    total_loss = 0.0
    total_correct_predictions = 0
    total_incorrect_predictions = 0
    total_samples = 0

    with torch.no_grad():

        for batch_index, (input_images, true_labels) in enumerate(test_loader):

            predicted_scores = model(input_images)
            loss = criterion(predicted_scores, true_labels)
            predicted_labels = predicted_scores.argmax(dim=1)
            batch_size = true_labels.size(0)
            batch_correct = (
                predicted_labels == true_labels
            ).sum().item()
            batch_incorrect = (
                predicted_labels != true_labels
            ).sum().item()
            batch_error = 100 * batch_incorrect / batch_size
            total_loss += loss.item() * batch_size
            total_correct_predictions += batch_correct
            total_incorrect_predictions += batch_incorrect
            total_samples += batch_size
 
    test_loss = total_loss / total_samples
    test_accuracy = 100 * total_correct_predictions / total_samples
    test_error = 100 * total_incorrect_predictions / total_samples

    print("\nTest Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Error: {test_error:.2f}%")
    return test_loss, test_accuracy, test_error


#for flask app, probabilities
def predict(model, image_tensor):

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence, probs[0].tolist()

if __name__ == "__main__":
    print(model)
    #train
    train_losses, train_accuracies = train_model(
        model, train_loader, criterion, optimizer, epochs=5
    )
    #test
    test_loss, test_accuracy, test_error = test_model(
        model, test_loader, criterion
    )
    # save
    #torch.save(model.state_dict(), './weights/mnist_cnn_weights.pth')
