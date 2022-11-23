import numpy as np
import torch
from PIL import Image
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import json
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        required=True)
    parser.add_argument('--path', 
                        type=str, 
                        help='path of json cat to names.',
                        required=True)
    parser.add_argument('--image_path', dest='image_path', action='store', 
                        default='./flowers/valid/100/image_07895.jpg')
    parser.add_argument('--device', 
                        type=str, 
                        help='select the device you working on',
                        default = 'cpu')
    parser.add_argument('--data_dir', 
                        type=str, 
                        help='Point to get data directory as str.',
                        required=True)
     parser.add_argument('--top_k', 
                         help='number of top K most likely classes', 
                         default=5, 
                         type=int)
    args = parser.parse_args()
    return args

def load_checkpoint(filename='checkpoint.pth'):
    
    checkpoint = torch.load(filename)
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)

    image_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                          std=(0.229, 0.224, 0.225))
                                    ])
    
    pil_image = image_transforms(pil_image)
    
    return pil_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    image = process_image(image_path)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model.forward(image)
        probs, labels = torch.topk(output, topk)        
        probs = probs.exp()
        class_to_idx_rev = {model.class_to_idx[k]: k for k in model.class_to_idx}
        classes = []
        for label in labels.numpy()[0]:
            classes.append(class_to_idx_rev[label])
        return probs.numpy()[0], classes

probs, classes = predict(test_dir + '/1/image_06743.jpg', model, args.top_k, args.device)
print(probs, classes)
args = arg_parser()

def cat_to_name(path='cat_to_name.json'):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

model = load_checkpoint(args.checkpoint)
probs, classes = predict(args.image_path, model, args.top_k, args.device)
cat_to_name = cat_to_name(args.path)
test_image = Image.open(args.image_path)
img_name = cat_to_name['100']
names = [cat_to_name[img_class] for img_class in classes]
plt.subplot(2, 1, 1)
plt.title(img_name)
plt.imshow(test_image)
plt.subplot(2, 1, 2)
plt.barh(names, probs)
plt.xlabel('Probabilities')