import torch
from torchvision.models import vgg16, VGG16_Weights 
from torch import nn 
import torch.optim as optim
import os
from timefrequencydataset import TimeFrequencyMapDataset
from torch.utils.data import random_split, DataLoader

def load_modified_pretrained_vgg16(num_classes):
    class ModifiedVGG16(nn.Module):
        def __init__(self, features, avgpool, classifier):
            super().__init__()
            self.features = features
            self.avgpool = avgpool
            self.classifier = classifier
            
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1) 
            x = x.transpose(0, 1)
            x = self.classifier(x)
            return x

    model = vgg16(weights=VGG16_Weights.DEFAULT)

    # Set first layer to take 1 channel
    original_weight = model.features[0].weight.data
    new_first_layer = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    new_first_layer.weight.data = original_weight.mean(dim=1, keepdim=True)

    # Remove final maxpooling and onwards
    layers = list(model.features.children())[1:-1]
    features = nn.Sequential(new_first_layer, *layers)

    # Freeze pretrained layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Add global average pooling
    avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # Add fully connected layers
    classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

    model = ModifiedVGG16(features, avgpool, classifier)
    model = model.double()

    return model

def train_model(model, dataset, epochs):
    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size 
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset)
    test_dataset = DataLoader(test_dataset)

    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        train_loss = 0

        for input_maps, labels in train_dataloader:
            optimizer.zero_grad()

            outputs = model(input_maps)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch: {epoch}, train_loss: {train_loss}")

if __name__ == "__main__":

    model = load_modified_pretrained_vgg16(5)
    
    data_path = os.path.join("..", "data")

    dataset = TimeFrequencyMapDataset(data_path)

    train_model(model, dataset, 25)

