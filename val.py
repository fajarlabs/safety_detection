import argparse
import torch
from pathlib import Path
from utils.general import check_dataset, check_img_size, non_max_suppression, scale_coords
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import create_dataloader
from utils.general import increment_path
from models.common import DetectMultiBackend

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Model weights path')
parser.add_argument('--data', type=str, default='data.yaml', help='Dataset YAML file path')
parser.add_argument('--img-size', type=int, default=640, help='Inference image size')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--device', type=str, default='', help='Device to use: "cpu" or "cuda"')
parser.add_argument('--conf-thres', type=float, default=0.001, help='Confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
args = parser.parse_args()

# Load the model
device = select_device(args.device)
model = DetectMultiBackend(args.weights, device=device)
stride, names = model.stride, model.names
img_size = check_img_size(args.img_size, s=stride)

# Load the dataset
dataset = create_dataloader(args.data, img_size, args.batch_size, stride, single_cls=False)[0]

# Initialize metrics
stats, ap, ap_class = [], [], []

# Evaluation loop
for batch_i, (img, targets, paths, shapes) in enumerate(dataset):
    img = img.to(device, non_blocking=True).float() / 255.0
    targets = targets.to(device)

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres)

    # Metrics
    for si, pred in enumerate(pred):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        if nl:
            correct = torch.zeros(pred.shape[0], nl, dtype=torch.bool, device=device)
            ap, ap_class = ap_per_class(*correct)

# Print results
print(f"mAP: {ap.mean():.3f}")
