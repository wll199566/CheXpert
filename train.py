"""
This script contains the function to train and validate our CNN model on CheXpert data.
"""
# import the necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from utils import torch_utils

def train(epoch, model, optimizer, scheduler, criterion, loader, device, log_interval, log_callback, ckpt_foldername):
    """
    Args:
        - epoch: the current epoch index
        - model: the deep learning model defined in the main.py
        - optimizer: optimizer defined in the main.py
        - scheduler: scheduler defined in the main.py
        - criterion: the dictionary of loss functions defined in the main.py
        - loader: data loader defined in the main.py
        - device: cpu or gpu
        - log_interval: the number of batch at which we write the training information into the log file
        - log_callback: the function defined for writing the training information into the log file
        - ckpt_foldername: the checkpoint folder name for storing the .ckpt file for this epoch
    Returns:
        - training_loss: the training loss for this epoch
        - training_accuracy: the training accuracy for this epoch    
    """
    
    # set up the initialization environment
    start_time = time.time()
    model.train()

    # initialize the statistics
    running_loss = 0.0  # running loss for log_interval
    epoch_loss = 0.0  # loss for this epoch
    total = 0  # total samples
    # the number of samples which are predicted correctly for each class
    correct_no_finding = 0 
    correct_en_card = 0
    correct_card = 0
    correct_lung_op = 0
    correct_lung_le = 0
    correct_edema = 0
    correct_cons = 0
    correct_pneu = 0
    correct_atelec = 0 
    correct_pneurax = 0
    correct_ple_eff = 0
    correct_ple_other = 0
    correct_frac = 0
    correct_sup_dev = 0

    # the output of the dataloader is a dictionary containing batch of images
    # and corresponding 14 labels
    # image is a uint8 tensor (convert to float!!) of shape [batch_size, 3, 320, 320]
    # each label is a nn.LongTensor type tensor of shape (batch_size, 1) (need to squeeze)
    # for each label: Here, in each one-hot vector: 
    # the first element is negative (0.0), 
    # the second element is positive (1.0) 
    # and the last one is uncertainty (-1.0).
    for batch_idx, samples in enumerate(loader):
        
        image = samples["image"].float().to(device)
        no_finding_label = samples["No Finding"].squeeze().to(device)
        en_card_label = samples["Enlarged Cardiomediastinum"].squeeze().to(device)
        card_label = samples["Cardiomegaly"].squeeze().to(device)
        lung_op_label = samples["Lung Opacity"].squeeze().to(device)
        lung_le_label = samples["Lung Lesion"].squeeze().to(device)
        edema_label = samples["Edema"].squeeze().to(device)
        cons_label = samples["Consolidation"].squeeze().to(device)
        pneu_label = samples["Pneumonia"].squeeze().to(device)
        atelec_label = samples["Atelectasis"].squeeze().to(device)
        pneurax_label = samples["Pneumothorax"].squeeze().to(device)
        ple_eff_label = samples["Pleural Effusion"].squeeze().to(device)
        ple_other_label = samples["Pleural Other"].squeeze().to(device)
        frac_label = samples["Fracture"].squeeze().to(device)
        sup_dev_label = samples["Support Devices"].squeeze().to(device)

        # input all the input vectors into the model
        # the output dimension is [batch_size, 41]
        preds = model(image)
        
        # compute the loss
        # since we have several classes, we need to compute for each class, respectively
        # and then sum them up and get the average loss
        # all criterions are CrossEntropyLoss
        loss_no_finding = criterion["No Finding"](preds[:, :2], no_finding_label)
        loss_en_card = criterion["Enlarged Cardiomediastinum"](preds[:, 2:5], en_card_label)
        loss_card = criterion["Cardiomegaly"](preds[:, 5:8], card_label)
        loss_lung_op = criterion["Lung Opacity"](preds[:, 8:11], lung_op_label)
        loss_lung_le = criterion["Lung Lesion"](preds[:, 11:14], lung_le_label)
        loss_edema = criterion["Edema"](preds[:, 14:17], edema_label)
        loss_cons = criterion["Consolidation"](preds[:, 17:20], cons_label)
        loss_pneu = criterion["Pneumonia"](preds[:, 20:23], pneu_label)
        loss_atelec = criterion["Atelectasis"](preds[:, 23:26], atelec_label)
        loss_pneurax = criterion["Pneumothorax"](preds[:, 26:29], pneurax_label)
        loss_ple_eff = criterion["Pleural Effusion"](preds[:, 29:32], ple_eff_label)
        loss_ple_other = criterion["Pleural Other"](preds[:, 32:35], ple_other_label)
        loss_frac = criterion["Fracture"](preds[:, 35:38], frac_label)
        loss_sup_dev = criterion["Support Devices"](preds[:, 38:41], sup_dev_label)

        loss = (1/14)*(loss_no_finding + loss_en_card + loss_card + loss_lung_op + loss_lung_le + loss_edema + loss_cons + loss_pneu + loss_atelec + loss_pneurax + loss_ple_eff + loss_ple_other + loss_frac + loss_sup_dev)
        
        # compute loss
        running_loss += loss.item()
        epoch_loss += loss.item()
        
        # compute accuracy
        total += image.shape[0]  # which the the batch size
        # compute the number of corrects for each class
        correct_no_finding += torch.sum(torch.max(preds[:, :2], dim=1)[1] == no_finding_label).item()
        correct_en_card += torch.sum(torch.max(preds[:, 2:5], dim=1)[1] == en_card_label).item()
        correct_card += torch.sum(torch.max(preds[:, 5:8], dim=1)[1] == card_label).item()
        correct_lung_op += torch.sum(torch.max(preds[:, 8:11], dim=1)[1] == lung_op_label).item()
        correct_lung_le += torch.sum(torch.max(preds[:, 11:14], dim=1)[1] == lung_le_label).item()
        correct_edema += torch.sum(torch.max(preds[:, 14:17], dim=1)[1] == edema_label).item()
        correct_cons += torch.sum(torch.max(preds[:, 17:20], dim=1)[1] == cons_label).item()
        correct_pneu += torch.sum(torch.max(preds[:, 20:23], dim=1)[1] == pneu_label).item()
        correct_atelec += torch.sum(torch.max(preds[:, 23:26], dim=1)[1] == atelec_label).item()
        correct_pneurax += torch.sum(torch.max(preds[:, 26:29], dim=1)[1] == pneurax_label).item()
        correct_ple_eff += torch.sum(torch.max(preds[:, 29:32], dim=1)[1] == ple_eff_label).item()
        correct_ple_other += torch.sum(torch.max(preds[:, 32:35], dim=1)[1] == ple_other_label).item()
        correct_frac += torch.sum(torch.max(preds[:, 35:38], dim=1)[1] == frac_label).item()
        correct_sup_dev += torch.sum(torch.max(preds[:, 38:41], dim=1)[1] == sup_dev_label).item()
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            # to get the time for log_interval batches
            elapse = time.time() - start_time
            # to write some trainig information into the log file
            log_callback("Epoch: {} \t Training process".format(epoch))
            
            log_callback()
            
            log_callback("Epoch: {0} \t"
                         "Time: {1}s / {2} batches, avg_time: {3}\n".format(
                         epoch, elapse, log_interval, elapse / log_interval ))

            
            log_callback("Train Epoch: {} [{}/{} ({:.0f}%)]\tTraining Loss: {:.6f}".format(
                epoch, batch_idx * image.shape[0], len(loader.dataset), 100. * batch_idx / len(loader), running_loss / log_interval))
            
            log_callback()

            # reset the start_time
            start_time = time.time()
            # reset the running loss
            running_loss = 0.0

    # to save the training model as a checkpoint 
    # Note here how the scheduler works!!!!!!!!!!!!!
    torch_utils.save(ckpt_foldername + "/CheXpert_" + str(epoch) + ".cpkt", epoch, model, optimizer, scheduler)    
    
    # to construct a dictionary to store all the training accuracy
    training_accuracy = {"No Finding" : correct_no_finding / total, 
                         "Enlarged Cardiomediastinum" : correct_en_card / total, 
                         "Cardiomegaly" : correct_card / total,
                         "Lung Opacity" : correct_lung_op / total,
                         "Lung Lesion" : correct_lung_le / total,
                         "Edema" : correct_edema / total,
                         "Consolidation" : correct_cons / total,
                         "Pneumonia" : correct_pneu / total,
                         "Atelectasis" : correct_atelec / total,
                         "Pneumothorax" : correct_pneurax / total,
                         "Pleural Effusion" : correct_ple_eff / total,
                         "Pleural Other" : correct_ple_other / total,
                         "Fracture" : correct_frac / total,
                         "Support Devices" : correct_sup_dev / total}

    # Return the training epoch loss and training epoch accuracy
    return epoch_loss / len(loader), training_accuracy

def validation(epoch, model, criterion, loader, device, log_callback):
    """
    Args:
        - epoch: the epoch
        - model: deep learning model we want to evaluation
        - criterion: loss function defined in main.py
        - loader: validation data loader 
        - device: cpu or gpu
        - log_callback: the function defined for writing the training information into the log file
    Returns:
        - valid_loss: average validation loss for this epoch
        - valid_accuarcy: the dictionary of validation accuracy for each class at this epoch
    """
    # intialize the environment
    start_time = time.time()
    model.eval()

    # initialize the statistics
    running_loss = 0.0  # running loss for log_interval
    total = 0  # total samples
    # the number of samples which are predicted correctly for each class
    correct_no_finding = 0 
    correct_en_card = 0
    correct_card = 0
    correct_lung_op = 0
    correct_lung_le = 0
    correct_edema = 0
    correct_cons = 0
    correct_pneu = 0
    correct_atelec = 0 
    correct_pneurax = 0
    correct_ple_eff = 0
    correct_ple_other = 0
    correct_frac = 0
    correct_sup_dev = 0

    # Note here we don't need to keep track of gradients
    with torch.no_grad():
        # the output of the dataloader is a dictionary containing batch of images
        # and corresponding 14 labels
        # image is a uint8 tensor (convert to float!!) of shape [batch_size, 3, 320, 320]
        # each label is a nn.LongTensor type tensor of shape (batch_size, 1) (need to squeeze)
        # for each label: Here, in each one-hot vector: 
        # the first element is negative (0.0), 
        # the second element is positive (1.0) 
        # and the last one is uncertainty (-1.0).
        for batch_idx, samples in enumerate(loader):
            
            image = samples["image"].float().to(device)
            no_finding_label = samples["No Finding"].squeeze().to(device)
            en_card_label = samples["Enlarged Cardiomediastinum"].squeeze().to(device)
            card_label = samples["Cardiomegaly"].squeeze().to(device)
            lung_op_label = samples["Lung Opacity"].squeeze().to(device)
            lung_le_label = samples["Lung Lesion"].squeeze().to(device)
            edema_label = samples["Edema"].squeeze().to(device)
            cons_label = samples["Consolidation"].squeeze().to(device)
            pneu_label = samples["Pneumonia"].squeeze().to(device)
            atelec_label = samples["Atelectasis"].squeeze().to(device)
            pneurax_label = samples["Pneumothorax"].squeeze().to(device)
            ple_eff_label = samples["Pleural Effusion"].squeeze().to(device)
            ple_other_label = samples["Pleural Other"].squeeze().to(device)
            frac_label = samples["Fracture"].squeeze().to(device)
            sup_dev_label = samples["Support Devices"].squeeze().to(device)

            # compute the preds
            preds = model(image)

            # to compute the loss
            # compute the loss
            # since we have several classes, we need to compute for each class, respectively
            # and then sum them up and get the average loss
            # all criterions are CrossEntropyLoss

            # Note here we use the same loss function as that in training
            # since there is no uncertainty label in the validation set
            # the production of it is always 0 
            loss_no_finding = criterion["No Finding"](preds[:, :2], no_finding_label)
            loss_en_card = criterion["Enlarged Cardiomediastinum"](preds[:, 2:5], en_card_label)
            loss_card = criterion["Cardiomegaly"](preds[:, 5:8], card_label)
            loss_lung_op = criterion["Lung Opacity"](preds[:, 8:11], lung_op_label)
            loss_lung_le = criterion["Lung Lesion"](preds[:, 11:14], lung_le_label)
            loss_edema = criterion["Edema"](preds[:, 14:17], edema_label)
            loss_cons = criterion["Consolidation"](preds[:, 17:20], cons_label)
            loss_pneu = criterion["Pneumonia"](preds[:, 20:23], pneu_label)
            loss_atelec = criterion["Atelectasis"](preds[:, 23:26], atelec_label)
            loss_pneurax = criterion["Pneumothorax"](preds[:, 26:29], pneurax_label)
            loss_ple_eff = criterion["Pleural Effusion"](preds[:, 29:32], ple_eff_label)
            loss_ple_other = criterion["Pleural Other"](preds[:, 32:35], ple_other_label)
            loss_frac = criterion["Fracture"](preds[:, 35:38], frac_label)
            loss_sup_dev = criterion["Support Devices"](preds[:, 38:41], sup_dev_label)
    
            loss = (1/14)*(loss_no_finding + loss_en_card + loss_card + loss_lung_op + loss_lung_le + loss_edema + loss_cons + loss_pneu + loss_atelec + loss_pneurax + loss_ple_eff + loss_ple_other + loss_frac + loss_sup_dev) 
    
            # compute loss
            running_loss += loss.item()

            # compute accuracy
            total += image.shape[0]
                        
            # compute the number of corrects for each class
            # Note here, what is different is that we only use first two labels (negative and positive)
            # and get the maximum between them
            correct_no_finding += torch.sum(torch.max(preds[:, :2], dim=1)[1] == no_finding_label).item()
            correct_en_card += torch.sum(torch.max(preds[:, 2:4], dim=1)[1] == en_card_label).item()
            correct_card += torch.sum(torch.max(preds[:, 5:7], dim=1)[1] == card_label).item()
            correct_lung_op += torch.sum(torch.max(preds[:, 8:10], dim=1)[1] == lung_op_label).item()
            correct_lung_le += torch.sum(torch.max(preds[:, 11:13], dim=1)[1] == lung_le_label).item()
            correct_edema += torch.sum(torch.max(preds[:, 14:16], dim=1)[1] == edema_label).item()
            correct_cons += torch.sum(torch.max(preds[:, 17:19], dim=1)[1] == cons_label).item()
            correct_pneu += torch.sum(torch.max(preds[:, 20:22], dim=1)[1] == pneu_label).item()
            correct_atelec += torch.sum(torch.max(preds[:, 23:25], dim=1)[1] == atelec_label).item()
            correct_pneurax += torch.sum(torch.max(preds[:, 26:28], dim=1)[1] == pneurax_label).item()
            correct_ple_eff += torch.sum(torch.max(preds[:, 29:31], dim=1)[1] == ple_eff_label).item()
            correct_ple_other += torch.sum(torch.max(preds[:, 32:34], dim=1)[1] == ple_other_label).item()
            correct_frac += torch.sum(torch.max(preds[:, 35:37], dim=1)[1] == frac_label).item()
            correct_sup_dev += torch.sum(torch.max(preds[:, 38:40], dim=1)[1] == sup_dev_label).item()


        elapse = time.time() - start_time
        
        # write the information into the log file
        log_callback("Epoch: {}\t Validation process".format(epoch))
        
        log_callback()    
        
        log_callback("Epoch: {0} \t"
                     "Time: {1}s / {2} batches, avg_time: {3}\n".format(
                     epoch, elapse, len(loader), elapse / len(loader) ))
        
        log_callback("Train Epoch: {} [{}/{} ({:.0f}%)]\tValidation Loss: {:.6f}".format(
            epoch, batch_idx * image.shape[0], len(loader.dataset), 100. * batch_idx / len(loader), running_loss / len(loader)))
        
        log_callback()
    
        # to construct a dictionary to store all the validation accuracy
        validation_accuracy = {"No Finding" : correct_no_finding / total, 
                             "Enlarged Cardiomediastinum" : correct_en_card / total, 
                             "Cardiomegaly" : correct_card / total,
                             "Lung Opacity" : correct_lung_op / total,
                             "Lung Lesion" : correct_lung_le / total,
                             "Edema" : correct_edema / total,
                             "Consolidation" : correct_cons / total,
                             "Pneumonia" : correct_pneu / total,
                             "Atelectasis" : correct_atelec / total,
                             "Pneumothorax" : correct_pneurax / total,
                             "Pleural Effusion" : correct_ple_eff / total,
                             "Pleural Other" : correct_ple_other / total,
                             "Fracture" : correct_frac / total,
                             "Support Devices" : correct_sup_dev / total}
    

        # return the validation loss and validation accuracy
        return running_loss / len(loader), validation_accuracy




                     

    