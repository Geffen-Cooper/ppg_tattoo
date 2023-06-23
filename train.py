import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from tqdm import tqdm
import time

from datasets import *
from models import *
from setup_funcs import *

import time
import logging
import datetime
import sys
import copy
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(linewidth=np.nan)

def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length

def cosine_lr(epoch,base_lr,total_epochs,warmup_epochs):
    if epoch < warmup_epochs:
        lr = _warmup_lr(base_lr,warmup_epochs,epoch)
        return lr
    else:
        e = epoch - warmup_epochs
        es = total_epochs - warmup_epochs
        lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr



def train(model,loss_fn,eval_fn,optimizer,log_name,epochs,ese,accum_bs,device,save_model_ckpt,
          train_loader,val_loader,test_loader,batch_parser,logger,val_metric_handle,warmup_epochs=5):

    # start tensorboard session
    writer = SummaryWriter("saved_data/runs/" + log_name+"_"+str(time.time()))

    # log training parameters
    print("===========================================")
    for k,v in zip(locals().keys(),locals().values()):
        writer.add_text(f"locals/{k}", f"{v}")
        print(f"locals/{k}", f"{v}")
    print("===========================================")


    # ================== training loop ==================
    batch_iter = 0
    model.train()
    model = model.to(device)
    checkpoint_path = "saved_data/checkpoints/" + log_name + ".pth"
    # best_val_acc = 0
    # best_val_f1 = 0
    best_val_metric = val_metric_handle.best_value_init()
    lowest_loss = 1e6

    print(f"****************************************** Training Started ******************************************")
    # load datasets
    
    num_epochs_worse = 0
    for param_group in optimizer.param_groups:
        base_lr = param_group['lr']
        break

    for e in range(epochs):
        for param_group in optimizer.param_groups:
            lr = cosine_lr(e,base_lr,epochs,warmup_epochs)
            writer.add_scalar("Metric/lr",lr,e)
            param_group["lr"] = lr

        if num_epochs_worse == ese:
            break
        for batch_idx, batch in enumerate(train_loader):
            # stop training, run on the test set
            if num_epochs_worse == ese:
                break

            # generic batch processing
            data,target = batch_parser(batch,device)

            # forward pass
            output = model(data)

            # generic loss (divide by accumulation batch size, typically just 1)
            train_loss = loss_fn(output,target)/accum_bs
            writer.add_scalar("Metric/train_loss", train_loss, batch_iter)

            # backward pass
            train_loss.backward()
            # print(data)
            # print(train_loss)
            # print(output,target)
            # print(model.conv6.weight.grad)
            # exit()

            # take a step after sufficient number of steps (just 1 typically)
            if ((batch_iter + 1) % accum_bs == 0):
                optimizer.step()
                optimizer.zero_grad()

            # logging
            if batch_idx % 20 == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}'.format(
                    e, batch_idx * train_loader.batch_size + len(data), len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader), train_loss))
            batch_iter += 1

        # at end of epoch evaluate on the validation set
        val_metric, val_loss = validate(model, val_loader, device, loss_fn, eval_fn, batch_parser,logger,val_metric_handle)
        writer.add_scalar(f"Metric/val_{str(val_metric_handle)}", val_metric, e)
        writer.add_scalar("Metric/val_loss", val_loss, e)

        # logging
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val {}: {:.3f}, val loss: {:.3f}'.format(
            e, batch_idx * train_loader.batch_size + len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), train_loss, str(val_metric_handle), val_metric, val_loss))

        # check if to save new chckpoint
        # if best_val_acc < val_acc:
        if val_metric_handle.is_better(best_val_metric,val_metric):
            logger.info("==================== best validation metric ====================")
            logger.info("epoch: {}, val {}: {}, val loss: {}".format(e, str(val_metric_handle), val_metric, val_loss))
            best_val_metric = val_metric
            lowest_loss = val_loss
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                f"val_{str(val_metric_handle)}": val_metric,
                'val_loss': val_loss,
            }, checkpoint_path)
            num_epochs_worse = 0
        else:
            logger.info(f"info: {num_epochs_worse} num epochs without improving")
            num_epochs_worse += 1

        # check for early stopping
        if num_epochs_worse == ese:
            logger.info(f"Stopping training because accuracy did not improve after {num_epochs_worse} epochs")
            break

    # evaluate on test set
    logger.info(f"Best val {str(val_metric_handle)}: {best_val_metric}")

    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    if test_loader is not None:
        test_metric, test_loss = validate(model, test_loader, device, loss_fn, eval_fn, batch_parser,logger,val_metric_handle)
        logger.info(f"Test {str(val_metric_handle)}: {test_metric}")
    

    logger.info("========================= Training Finished =========================")
    if save_model_ckpt == False:
        os.remove(checkpoint_path)

def validate(model,val_loader,device,loss_fn,eval_fn,batch_parser,logger,val_metric_handle):
    model.eval()
    model = model.to(device)

    val_loss = 0

    # collect all labels and predictions, then feed to val metric specific function
    with torch.no_grad():
        predictions = []
        labels = []

        for batch_idx, batch in enumerate(val_loader):
            # parse the batch and send to device
            data,target = batch_parser(batch,device)

            # get model output
            out = model(data.float())

            # get the loss
            val_loss += loss_fn(out, target)

            # parse the output and target for the prediction and label
            # (out and target may be tuples if we have multiple losses for example)
            prediction, label = eval_fn(out,target)

            predictions.append(prediction)
            labels.append(label)
        
        val_loss /= (len(val_loader))
        val_metric = val_metric_handle.get_val_metric(predictions,labels)

        return val_metric, val_loss

# ===================================== Main =====================================

def print_model_size(model):
    num_parameters = 0
    param_size = 0
    for param in model.parameters():
        num_parameters += param.nelement()
        param_size += param.nelement()*param.element_size()

    print("Number of parameters:", num_parameters)
    print("Model size (KB):",param_size/1e3)


# ============================== validation metrics ==============================
class AccValMetric():
    def best_value_init(self):
        return 0.0
    
    def get_val_metric(self,predictions,labels):
        total = 0
        num_correct = 0
        for prediction,label in zip(predictions,labels):
            if len(prediction.shape) > 1:
                total += len(label)
                num_correct += (prediction == label).float().sum()
            else:
                total += 1
                num_correct += (prediction == label).float()
            
        return num_correct/total
    
    def is_better(self,best,curr):
        return curr > best
    
    def __str__(self):
        return "Accuracy"

class F1ValMetric():
    def __init__(self,class_names,verbose=True,logger=None):
        self.class_names = class_names
        self.verbose = verbose
        self.logger = logger
    
    def best_value_init(self):
        return 0.0

    def get_val_metric(self,predictions,labels):
        num_classes = len(self.class_names)
        TP = torch.zeros(num_classes) # the number of times a class was predicted and it was right
        FP = torch.zeros(num_classes) # the number of times a class was predicted and it was wrong
        FN = torch.zeros(num_classes) # the number of times a class was not predicted and it was wrong
        TP_plus_FP = torch.zeros(num_classes) # the number of times each class was predicted
        TP_plus_FN = torch.zeros(num_classes) # the number of times a class appears (ground truth)

        for prediction,label in zip(predictions,labels):
            for c_i in range(num_classes):
                recs_idxs = (label == c_i).nonzero().view(-1) # TP + FN 
                precs_idxs = (prediction == c_i).nonzero().view(-1) # TP + FP

                # number of times this class was predicted and was right
                TP[c_i] += (prediction[precs_idxs] == label[precs_idxs]).float().sum()

                # number of times this class was predicted and it was wrong
                FP[c_i] += (prediction[precs_idxs] != label[precs_idxs]).float().sum()

                # number of times this class was not predicted and it was wrong
                FN[c_i] += (prediction[recs_idxs] != label[recs_idxs]).float().sum()

                # number of times a class was predicted
                TP_plus_FP[c_i] += len(precs_idxs)

                # number of times a class appeared
                TP_plus_FN[c_i] += len(recs_idxs)

        # print(precision_recall_fscore_support(labels,preds))
        if self.verbose:
            self.logger.info("\nPrecision")
        precision = TP/(TP+FP)
        for i in range(num_classes):
            name = self.class_names[i]
            if self.verbose:
                self.logger.info(f"class {name}: {round(precision[i].item(),3)} ({TP[i]}/{(TP+FP)[i]})")

        if self.verbose:
            self.logger.info("\nRecall")
        recall = TP/(TP+FN)
        for i in range(num_classes):
            name = self.class_names[i]
            if self.verbose:
                self.logger.info(f"class {name}: {round(recall[i].item(),3)} ({TP[i]}/{(TP+FN)[i]})")

        if self.verbose:
            self.logger.info("\nF1 score")

        f1 = 2*(precision*recall)/(precision+recall+1e-6)
        f1 = torch.nan_to_num(f1) # set nans to zero
        if self.verbose:
            self.logger.info(f"MACRO: {f1.mean()}")

        return f1.mean()
    
    def is_better(self,best,curr):
        return curr > best
    
    def __str__(self):
        return "F1 Macro"



if __name__ == "__main__":
    # ===================== train on artery and vein =====================
    # setup session
    logname = "PPG"
    logger = init_logger(logname)
    # init_seeds(432)

    # load data
    train_loader, val_loader, test_loader = load_ppg_dataset(16,50,0,0.8,0.2,14,rescale=[-1,1],fc=True)

    # create model
    model = PPGClassifierFC()

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0004)
    # optimizer = torch.optim.Adam(model.parameters(),lr=0.1)

    # train model
    train_params = {'model': model, 'loss_fn': PPG_LOSS(), 'eval_fn': PPG_EVAL(), 'optimizer': optimizer, 'log_name': logname,
                    'epochs': 50, 'ese': 20, 'accum_bs': 1, 'device': 'cpu', 'save_model_ckpt': True, 'train_loader': train_loader,
                    'val_loader': val_loader, 'test_loader': test_loader, 'batch_parser': ppg_parser, 'logger': logger, 
                    'val_metric_handle': F1ValMetric(['low','high'],True,logger), 'warmup_epochs': 5}

    train(**train_params)

    # ===================== test on both =====================

    test_loader = load_ppg_dataset(128,50,0,0,None,None,rescale=[-1,1],fc=True)
    model.load_state_dict(torch.load('saved_data/checkpoints/PPG.pth')['model_state_dict'])

    validate(model,test_loader,'cpu',PPG_LOSS(),PPG_EVAL(),ppg_parser,logger,F1ValMetric(['low','high'],True,logger))
