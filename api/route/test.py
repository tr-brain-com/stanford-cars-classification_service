import numpy as np
import torch
import glob as glob
import os
import time
import cv2

from torchvision import transforms
from torch.nn import functional as F
from torch import topk
from tqdm import tqdm

from model import build_model
from api.route.class_names import class_names

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

os.makedirs('route/outputs/cam_results/', exist_ok=True)

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
print(model)
model.load_state_dict(torch.load('outputs/model.pth')['model_state_dict'])



# Hook the feature extractor.
# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
model._modules.get('features').register_forward_hook(hook_feature)
# Get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

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

counter = 0
correct_count = 0
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second. 
# Run for all the test images.
all_images = glob.glob('input/car_data/test/*/*.jpg', recursive=True)
print(all_images[:5])
for image_path in tqdm(all_images, total=len(all_images)):
    # Read the image.
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_class = image_path.split(os.path.sep)[-2]
    orig_image = image.copy()
    height, width, _ = orig_image.shape
    # Apply the image transforms.
    image_tensor = transform(image)
    # Add batch dimension.
    image_tensor = image_tensor.unsqueeze(0)
    # Forward pass through model.
    start_time = time.time()
    outputs = model(image_tensor.to(device))
    end_time = time.time()
    # Get the softmax probabilities.
    probs = F.softmax(outputs).data.squeeze()
    # Get the class indices of top k probabilities.
    class_idx = topk(probs, 1)[1].int()
    pred_class_name = str(class_names[int(class_idx)])
    if gt_class == pred_class_name:
        correct_count += 1
    
    counter += 1
    #print(f"Image: {counter}")

print(f"Total number of test images: {len(all_images)}")
print(f"Total correct predictions: {correct_count}")
print(f"Accuracy: {correct_count/len(all_images)*100:.3f}")
