import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os

exec_path = os.getcwd()

# Load the raw MobileNetV2 architecture and your existing weights file
model = torchvision.models.mobilenet_v2(weights=None)
state_dict = torch.load(
    os.path.join(exec_path, "mobilenet_v2-b0353104.pth"),
    map_location="cpu"
)
model.load_state_dict(state_dict)
model.eval()

# Load ImageNet class labels (bundled inside the imageai package)
classes_path = os.path.join(
    os.path.dirname(os.path.dirname(torch.__file__)),  # site-packages
    "imageai", "Classification", "imagenet_classes.txt"
)
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# THE EXPERIMENT: direct squish resize, no center-crop
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # squish whole image into 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

images = ["house.jpg", "giraffe.jpg", "godzilla.jpg"]

for image_name in images:
    img = Image.open(os.path.join(exec_path, image_name)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_idx = torch.topk(probabilities, 5)

    print(f"\n--- {image_name} (squish resize, no crop) ---")
    for prob, idx in zip(top5_prob, top5_idx):
        print(f"{classes[idx]} : {prob.item() * 100:.2f}%")