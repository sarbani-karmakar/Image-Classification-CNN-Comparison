import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
exec_path = os.getcwd()
# Load ResNet50 with pretrained ImageNet weights (auto-downloads ~100MB on first run)
model = torchvision.models.resnet50(weights= torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()
# Get the exact preprocessing ResNet50 was trained with, and class labels, straight from the weights metadata
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
classes = weights.meta["categories"]
images = ["godzilla.jpg","house.jpg","giraffe.jpg"]

for image_name in images:
    img = Image.open(os.path.join(exec_path,image_name)).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0],dim=0)

    top5_prob, top5_idx = torch.topk(probabilities,5)

    print(f"\n---{image_name}(ResNet50)")
    for prob,idx in zip(top5_prob,top5_idx):
        print(f"{classes[idx]}:{prob.item()*100:.2f}%")
