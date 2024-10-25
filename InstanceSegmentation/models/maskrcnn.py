#Adapted from transfer learning skeleton https://medium.com/swlh/music-genre-classification-using-transfer-learning-pytorch-ea1c23e36eb8
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class net(nn.Module):
    def __init__(self, params):
        super(net, self).__init__()

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        num_classes = 2

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
    
    def forward(self, im, targets):
        
        return self.model(im, targets)
    
def train(model, device, train_loader, optimizer, loss_function):
    model.train()
    for (images, targets) in tqdm(train_loader, leave=False):
        #print("They have shapes", im.shape, mask.shape)
        images = list(torch.stack((image, image, image)).squeeze().to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        #Forward pass
        optimizer.zero_grad()
        loss_dict = model.forward(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()

        #Backward pass
        optimizer.step()


def val(model, device, loader, loss_function, threshold):
    model.eval()
    losses = []
    mses = []
    scores = []
    ious = []
    accs = []
    recalls = []
    precs = []
    tps = []
    fps = []
    fns = []
    global_tp = 0
    global_fp = 0
    global_fn = 0
    tot = 0
    with torch.no_grad():
        for (images, targets) in tqdm(loader, leave=False):
            images = list(torch.stack((image, image, image)).squeeze().to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            #Has 100 mask predictions and bounding box predictions
            out = model(images, None)


            for index, item in enumerate(out):
                this_scores = item["scores"]

                #If there is more than one actual mask, sum them all into one
                act_masks = targets[index]["masks"]
                act_mask = torch.zeros((512, 512))
                for mask_num in range(len(targets[index]["masks"])):
                    act_mask = torch.add(act_mask, act_masks[mask_num])
                act_mask = 1.0 * (act_mask >= 1)#.detach().cpu().numpy() #set them all back to 1

                #Finally, get the predictions and stack the masks in the same way. Only keep masks with confidence > 0.5
                wanted_masks = torch.where(item["scores"] > 0.5)
                pred_mask = torch.zeros((512, 512))
                for index in wanted_masks[0]:
                    this_mask = item["masks"][index].squeeze()
                    if this_mask.shape[0] == 3:
                        this_mask = this_mask[0, :, :].squeeze()
                    pred_mask = torch.add(pred_mask, this_mask)
                pred_mask = 1.0 * (pred_mask >= threshold)#.detach().cpu().numpy()
                
                iou = iou_score(pred_mask, act_mask, threshold)
                ious.append(iou) if iou is not None else None
                mses.append(mean_squared_error(pred_mask, act_mask))
                losses.append(loss_function(pred_mask, act_mask).item())
                acc, prec, rec, tp, fp, fn = acc_prec_recall(pred_mask, act_mask, threshold)

                #Append the metrics if it was a case where there was a cloud and a label
                accs.append(acc) if acc is not None else None
                precs.append(prec) if prec is not None else None
                recalls.append(rec) if rec is not None else None
                tps.append(tp) if acc is not None else None
                fps.append(fp) if acc is not None else None
                fns.append(fn) if acc is not None else None


                #Now calculate the global precision and recall based on how many clouds were identified
                num_acts = len(act_masks)
                num_preds = len(wanted_masks[0])
                if num_acts > num_preds:
                    global_tp += num_preds
                    global_fn += (num_acts - num_preds)
                if num_acts < num_preds:
                    global_tp += num_acts
                    global_fp += (num_preds - num_acts)
                if num_acts == num_preds:
                    global_tp += num_preds
                tot += num_acts
     
    scores_dict = {
        "avg_mse": np.mean(mses),
        "avg_score": np.mean(scores) if len(scores) > 0 else 0.0,
        "avg_iou": np.mean(ious),
        "avg_acc": np.mean(accs),
        "avg_prec": np.mean(precs),
        "avg_recall": np.mean(recalls),
        "avg_tp": np.mean(tps),
        "avg_fp": np.mean(fps),
        "avg_fn": np.mean(fns),
        "global_prec": global_tp / (global_tp + global_fp),
        "global_rec": global_tp / (global_tp + global_fn),
        "global_acc": (global_tp + (tot - global_tp - global_fn - global_fp)) / tot
    }
    valid_loss = np.mean(losses)
    return valid_loss, scores_dict

def iou_score(pred, actual, threshold):
    #We only want to count the instances where the model guesses a cloud is there and it is correct, beause IoU is per pixel not for the whole set
    if torch.mean(pred) == 0.0 or torch.mean(actual) == 0.0:
        return None
    pred = 1.0 * (pred >= threshold)
    intersection = torch.multiply(pred, actual)
    union = 1.0 * (torch.add(pred, actual) >= 1)

    #If there was no label and no cloud was predicted, IoU is 1.0
    
    return torch.sum(intersection) / torch.sum(union)

#Calculate the precision, recall, and accuracy. Only could these metrics for cases where the model correctly identified a cloud. This is
#different from the global precision, recall, and accuracy calcuated elsewhere, which says how many of the clouds in the test set the model
#correctly found. Instead, here we are looking at per-pixel metrics.
def acc_prec_recall(pred, actual, threshold):
    if torch.mean(pred) == 0.0 and torch.mean(actual) == 0.0:
        return None, None, None, None, None, None
    pred = 1.0 * (pred >= threshold)
    acc = None if torch.mean(pred) == 0 or torch.mean(actual) == 0 else torch.sum(1.0 * (pred == actual)) / (pred.shape[0] * pred.shape[1])
    tp = torch.sum(torch.multiply(1.0 * (pred == actual), pred))
    fp = torch.sum(torch.multiply(1.0 * (pred != actual), pred))
    fn = torch.sum(torch.multiply(1.0 * (pred != actual), actual))
    prec = None if torch.mean(pred) == 0 or torch.mean(actual) == 0 else tp / (tp + fp)
    rec = None if torch.mean(pred) == 0 or torch.mean(actual) == 0 else tp / (tp + fn)
    return acc, prec, rec, tp, fp, fn