from __future__ import print_function
import os
import time
import shutil
import json
import argparse
import numpy as np
import shutil
import utils.extra as extra
import torch
import torch.nn as nn
import torch.optim as optim
import time
#from extracode.engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import v2 as T

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import dataset
from utils.params import Params

#Fetch model hyperparameters
params = Params("hparams.yaml", "DEFAULT")
checkpoint_dir = "saved_models"

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

#save the params for reproducibility, version number increases separately for each transform type
version = int(len([file for file in os.listdir("saved_models") if file[:file.find("_")] == params.model]) / 2)
shutil.copy("hparams.yaml", checkpoint_dir + "/" + params.model + "_v" + str(version) + "_hparams.yaml")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

model_import = __import__('.'.join(['models', params.model]),  fromlist=['object'])
model = model_import.net(params).to(device)

#If we want to continue training a model, load it in here
if params.resume != "":
    model.load_state_dict(torch.load("saved_models/" + params.resume + ".ckpt"))

train = model_import.train
val = model_import.val

#Load model, train function, eval function
loss_function = nn.BCELoss() #BCE may be better
optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

transforms = [
    T.ToDtype(torch.float, scale=True),
    T.ToPureTensor(),
    T.Resize((512, 512))
    #T.RandomHorizontalFlip(0.5)
]

#Prepare data
train_data = dataset.CloudsDataset(params, transforms=T.Compose(transforms), root="../Dataset/train/",  split="train")
val_data = dataset.CloudsDataset(params, transforms=T.Compose(transforms), root="../Dataset/train/", split="val",)
num_epochs = params.epochs

train_loader = DataLoader(
        train_data, 
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=extra.collate_fn
    )
val_loader = DataLoader(
        val_data,
        batch_size=params.batch_size, #we always use a batch size of 10 so that we can combine the fragments
        collate_fn=extra.collate_fn
        #shuffle=False,
    )

start_epoch = 0
print("Preparing to train", params.model, "for", num_epochs, "epochs.")
if params.resume != "":
    print("Resuming at epoch", params.resume_epoch)
    start_epoch = params.resume_epoch - 1
print("Batch size:", params.batch_size, ", lr:", params.lr)
global_start = time.perf_counter()

#Training and validation
train_losses, train_mses, val_losses, val_mses = [], [], [], []
for epoch in range(start_epoch, num_epochs):
    # train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)
    # evaluate(model, val_loader, device)
    start_time = time.perf_counter()
    print("---- Epoch Number: %s ----" % (epoch + 1))

    #Train
    train(model, device, train_loader, optimizer, loss_function)
        # Evaluate on both the training and validation set. 
    train_loss, train_scores = val(model, device, train_loader, loss_function, 0.5)
    print('\rEpoch: [%d/%d], Train loss: %.6f, Train MSE: %.6f, Train Avg. IoU: %0.4f' % (epoch+1, num_epochs, train_loss, train_scores["avg_mse"], train_scores["avg_iou"]))

    #Validation
    val_loss, val_scores = val(model, device, val_loader, loss_function, 0.5)
    print('Epoch: [%d/%d], Valid loss: %.6f, Valid MSE: %.6f, Valid Avg. IoU: %0.4f' % (epoch+1, num_epochs, val_loss, val_scores["avg_mse"], val_scores["avg_iou"]))
    
    # Collect some data for logging purposes. 
    train_losses.append(float(train_loss))
    train_mses.append(train_scores["avg_mse"])
    val_losses.append(float(val_loss))
    val_mses.append(val_scores["avg_mse"])
    end_time = time.perf_counter()
    print("Took %0.1f seconds" % (end_time - start_time))

    #If we've found the best model so far
    if val_loss == np.min(val_losses):

        print("Found the current best model for this training, saving....")

        global_end = time.perf_counter()
        torch.save(model.state_dict(), checkpoint_dir + "/" + params.model +"_v" + str(version) + ".ckpt")

        plt.clf()    
        plt.cla()
        plt.plot(train_mses, label="Training MSE")
        plt.plot(val_mses, label="Validation MSE")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.savefig("figures/" + params.model + "_v" + str(version) + "_MSE.png")
            
        plt.clf()   
        plt.cla() 
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("figures/" + params.model + "_v" + str(version) + "_loss.png")

        logs ={
            "model": params.model,
            "best_val_epoch": int(np.argmax(val_losses)+1),
            "lr": params.lr,
            "batch_size":params.batch_size,
            "start_time":global_start,
            "end_time":global_end,
            "notes": "Initial testing"
        }

        with open(checkpoint_dir + "/" + params.model +"_v" + str(version) + "_logs.json", 'w') as f:
            json.dump(logs, f)
    
plt.plot(train_mses, label="Training Accuracy")
plt.plot(val_mses, label="Validation Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("figures/" + params.model + "_v" + str(version) + "_MSE.png")
    
plt.clf()    
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("figures/" + params.model + "_v" + str(version) + "_loss.png")

    # Save model
    # valid_losses.append(valid_loss.item())
    # if np.argmin(valid_losses) == epoch:
    #     print('Saving the best model at %d epochs!' % epoch)
    #     torch.save(cnn.state_dict(), 'best_model.ckpt')
#Log