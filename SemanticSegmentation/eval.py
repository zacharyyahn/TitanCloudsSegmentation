#Imports
from __future__ import print_function
import os
import time
import json
import argparse
import numpy as np
import soundfile as sf
from torchvision.transforms import v2 as T
import utils.extra as extra

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import dataset
from utils.params import Params

#Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name",
    type=str,
    help="The specific file, e.g. resnet50_v0"
)
parser.add_argument(
    "threshold",
    type=float,
    help="Confidence threshold for considering a pixel as park of the mask, e.g. 0.9"
)
args = parser.parse_args()

#Fetch model hyperparameters
params = Params("saved_models/" + args.model_name + "_hparams.yaml", "DEFAULT")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_import = __import__('.'.join(['models', params.model]),  fromlist=['object'])
model = model_import.net(params).to(device)
model.load_state_dict(torch.load("saved_models/" + args.model_name + ".ckpt"))

loss_function = nn.BCELoss()
val = model_import.val

transforms = [
    T.ToDtype(torch.float, scale=True),
    T.ToPureTensor(),
    T.Resize((512, 512))
    #T.RandomHorizontalFlip(0.5)
]

#Prepare data
test_data = dataset.CloudsDataset(params, transforms=T.Compose(transforms), root="../Dataset/test/", split="test")
num_epochs = params.epochs

test_loader = DataLoader(
        test_data, 
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=extra.collate_fn
    )

#Count how many parameters are in the model
total_params = sum(p.numel() for p in model.parameters())

print("Preparing to test", args.model_name)
print("Number of parameters:", total_params)
print("Threshold:", args.threshold)

#Do the test
loss, scores = val(model, device, test_loader, loss_function, args.threshold)

print('Loss: %.4f, Avg. MSE: %.4f, Avg. Score %.4f, Avg. IoU %.4f, Avg. Acc %.4f, Avg. Prec, %.4f, Avg. Rec %.4f' % (loss, scores["avg_mse"], scores["avg_score"], scores["avg_iou"], scores["avg_acc"], scores["avg_prec"], scores["avg_recall"]))
print("Global Accuracy: %0.3f, Global Precision: %0.3f, Global Recall: %0.3f" % (scores["global_acc"], scores["global_prec"], scores["global_rec"]))
f = open("eval_log.txt", "a")
f.write("\n---- Threshold: %0.2f ----" % args.threshold)
f.write('\nLoss: %.4f, Avg. MSE: %.4f, Avg. IoU %.4f, Avg. Acc %.4f, Avg. Prec, %.4f, Avg. Rec %.4f' % (loss, scores["avg_mse"], scores["avg_iou"], scores["avg_acc"], scores["avg_prec"], scores["avg_recall"]))
f.write("\nAvg. TP: %0.4f, Avg. FP: %0.4f, Avg. FN: %0.4f" %(scores["avg_tp"], scores["avg_fp"], scores["avg_fn"]))
f.close()
