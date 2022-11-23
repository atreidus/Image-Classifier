import numpy as np
import torch
from PIL import Image
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    parser.add_argument('--data_dir', 
                        type=str, 
                        help='Point to get data directory as str.',
                        required=True)
    parser.add_argument('--arch', 
                        type=str, 
                        help='select the architecture you working on',
                        default = 'vgg16')
    parser.add_argument('--device', 
                        type=str, 
                        help='select the device you working on',
                        default = 'cuda')
    parser.add_argument('--epochs', 
                        type=float, 
                        help='select the number of epochs',
                        default = 5)
    parser.add_argument('--lr', 
                        type=float, 
                        help='select the value of learning rate',
                        default = 0.001)
    parser.add_argument('--hidden_units', 
                        type=float, 
                        help='select the number of hidden units',
                        default = 2560)
    args = parser.parse_args()
    return args

       
def create_model(arch ,hidden_units ,device):
    
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.require_grad = False
    model.classifier = nn.Sequential(nn.Linear(25088,hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(hidden_units,102),
                                     nn.LogSoftmax(dim=1))

    model.to(device)
    return model

def valid(model, valid_loader, criterion, device):
    test_loss = 0
    acc = 0
    for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            ps = torch.exp(logps)
            equals = (labels.data == ps.max(dim=1)[1])
            acc += torch.mean(equals.type(torch.FloatTensor)).item()
    return acc,test_loss

def train_model(model, train_loader, test_loader, criterion, learn_r, epochs, device):
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_r)
    running_loss = 0
    print_every = 5
    steps = 0
    epochs = 5

    for epoch in range(epochs):
    
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                acc,test_loss = valid(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Test accuracy: {acc/len(valid_loader):.3f}")
            running_loss = 0
            model.train()
    return model  

def validation(model, test_loader, device):
    
    test_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(output, labels).item()
    print('Accuracy on test images is: %d%%' % (100 * correct / total))
    
def save_checkpoint(model,train_dataset):
    
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'epoch': epochs,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict()
                 }

    torch.save(checkpoint, 'checkpoint.pth')

args = arg_parser()
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(260),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(260),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)

criterion = nn.NLLLoss()
model = create_model(args.arch, args.hidden_units, args.device)
trained_model = train_model(model, train_loader, test_loader, criterion, args.lr, args.epochs, args.device)
validate_model(trained_model, test_loader, device)
save_checkpoint(trained_model, train_dataset)