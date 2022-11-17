import numpy as np
import torch
import os
import time
import cv2

from torchvision import transforms
from torch.nn import functional as F
from torch import topk

from model import build_model
from api.route.class_names import class_names

seed = 42
np.random.seed(seed)

# Construct the argument parser.
import argparse
import warnings
warnings.simplefilter("ignore", UserWarning)

def argInit():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-img', '--img_path', type=str, default="../1663523018348634.jpg",
        help='Image Path for predict'
    )
    args = vars(parser.parse_args())
    return args

def getModel():
    # Define computation device.
    device = 'cpu'
    # Class names.
    # Initialize model, switch to eval model, load trained weights.
    model = build_model(
        pretrained=False,
        fine_tune=False, 
        num_classes=196
    ).to(device)
    model = model.eval()
    #print(model)
    model.load_state_dict(torch.load('outputs/model.pth', map_location=device)['model_state_dict'])
    # Hook the feature extractor.
    # https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    model._modules.get('features').register_forward_hook(hook_feature)
    # Get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    return model, device

# Define the transforms, resize => tensor => normalize.
transform = transforms.Compose([
     transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    ])

# Run for all the test images.
def predict(image_path):
    #Load model
    model, device = getModel()    
    # Read the image.
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_class = image_path.split(os.path.sep)[-2]
    orig_image = image.copy()
    height, width, _ = orig_image.shape
    # Apply the image transforms.
    image_tensor = transform(image)
    # Add batch dimension.
    image_tensor = image_tensor.unsqueeze(0).to(device)
    # Forward pass through model.
    start_time = time.time()
    outputs = model(image_tensor.to(device))
    end_time = time.time()
    # Get the softmax probabilities.
    probs = F.softmax(outputs).to(device).data.squeeze()
    # Get the class indices of top k probabilities.
    class_idx = topk(probs, 1)[1].int()
    pred_class_name = str(class_names[int(class_idx)])
    
    print(f'\nprediction class is\n{pred_class_name}')

if __name__ == '__main__':
    args = argInit()
    img_path = args['img_path']
    predict(img_path)