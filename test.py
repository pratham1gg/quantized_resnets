import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time

# -------------------------
# Config
# -------------------------
IMAGENET_ROOT = "/home/pf4636/imagenet2"
BATCH_SIZE = 256
NUM_WORKERS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Transforms (standard ImageNet)
# -------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # uint8 -> float32 [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -------------------------
# Dataset (IMPORTANT PART)
# -------------------------
dataset = datasets.ImageNet(
    root=IMAGENET_ROOT,
    split="val",
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print("Validation images:", len(dataset))

# -------------------------
# Model
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()
model.to(device)

# -------------------------
# Accuracy helper
# -------------------------
def topk_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.item())
        return res

# -------------------------
# Evaluation loop
# -------------------------
top1_correct = 0
top5_correct = 0
total = 0

torch.cuda.synchronize() if device.type == "cuda" else None
start = time.time()

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        c1, c5 = topk_accuracy(outputs, labels)
        top1_correct += c1
        top5_correct += c5
        total += labels.size(0)

torch.cuda.synchronize() if device.type == "cuda" else None
elapsed = time.time() - start

print("===================================")
print(f"Top-1 Accuracy: {100 * top1_correct / total:.2f}%")
print(f"Top-5 Accuracy: {100 * top5_correct / total:.2f}%")
print(f"Throughput: {total / elapsed:.2f} images/s")
print("===================================")
