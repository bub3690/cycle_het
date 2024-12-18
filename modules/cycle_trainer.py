
import time
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision


# import wandb



def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def train_cycle(args, model, num_classes, num_regions, head_n,
           ema_mode, teacher, momentum_schedule, coef_schedule, it,
          EPOCHS, epoch, train_dataloader, optimizer, classify_criterion, DEVICE):
    #여기에 num_classes_list, num_regions_list, n_head도
    
    model.train()
    MSE = torch.nn.MSELoss()
    
    # coefficient scheduler from  0 to 0.5 
    coff = coef_schedule[it]

    
    train_correct = [0,] * num_regions
    train_mae = [0,] * num_regions
    
    
    total_train_loss = 0
    total_mse_loss = 0
    
    for idx, (sample1, sample2, label,label_float, filename,file_index) in enumerate(train_dataloader):
        sample1, sample2, label,label_float = sample1.to(DEVICE),sample2.to(DEVICE), label.to(DEVICE), label_float.to(DEVICE)
        optimizer.zero_grad()
        
        #pred = model(img, head_n=head_n, projection=True)
        
        pred_s, feat_s  = model(sample2, head_n=head_n, projection=True)        
        pred_t, feat_t  = teacher(sample1, head_n=head_n, projection=True)
        
        
        
        # 출력값 : [batch_size, 6, 4] . 6개의 value를 가진 4개의 class
        
        #loss for classification per 6 position
        loss_classification = 0
        loss_mse = 0
        #print(pred.shape)
        for i in range(num_regions):
            pos_probs = pred_s
            pos_probs = pos_probs[:,i,:]
            
            pos_feat_s = feat_s[:,i,:]
            pos_feat_t = feat_t[:,i,:]
            
            #print(pos_probs[0].argmax(), label[0,i])
            
            loss_classifier = classify_criterion(pos_probs, label[:,i])
            loss_const = MSE(pos_feat_s, pos_feat_t)
            
            loss_classification += ( (1-coff)*loss_classifier + coff*loss_const)
            loss_mse += loss_const # 기록용
            
            prediction = pos_probs.max(1,keepdim=True)[1]
            
            train_correct[i] += prediction.eq(label[:,i].view_as(prediction)).sum().item()
            train_mae[i] += torch.abs(prediction - label[:,i].view_as(prediction)).sum().item()
        
        loss_classification = loss_classification / 6
        
        #loss for regression per 6 position
        # for i in range(6):
        #     pos_max = pred[:,i,:].argmax(dim=1)
        #     loss_regression += regress_criterion(pos_max, label_float[:,i])
        
        # loss_regression = loss_regression / 6
        
        #loss = alpha * loss_classification + (1-alpha) * loss_regression
        loss = loss_classification
        
        loss.backward()
        optimizer.step()
        
        
        total_train_loss += loss.item()
        total_mse_loss += loss_mse.item() # 기록용
    
        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1   
    
    total_train_loss = total_train_loss / len(train_dataloader)
    total_mse_loss = total_mse_loss / len(train_dataloader)
    
    
    train_acc_list = [0,] * num_regions
    for i in range(num_regions):
        train_acc_list[i] = 100. * train_correct[i] / len(train_dataloader.dataset)
        print(f"EPOCH {epoch} / {EPOCHS}, Position {i} Train ACC : {train_acc_list[i]:.2f}")
    
    train_mae_list = [0,] * num_regions
    for i in range(num_regions):
        train_mae_list[i] = train_mae[i] / len(train_dataloader.dataset)
        print(f"EPOCH {epoch} / {EPOCHS}, Position {i} Train MAE : {train_mae_list[i]:.2f}")
    
    
    #mean train acc
    print(f"EPOCH {epoch} / {EPOCHS}, Mean Train ACC : {np.mean(train_acc_list):.2f}")
    print(f"EPOCH {epoch} / {EPOCHS}, Mean Train MAE : {np.mean(train_mae_list):.2f}")
    print(f"EPOCH {epoch} / {EPOCHS}, Loss : {total_train_loss:.2f}")
    print(f"EPOCH {epoch} / {EPOCHS}, Consistency Loss : {total_mse_loss:.4f}")
    
    return total_train_loss, np.mean(train_acc_list), np.mean(train_mae_list), total_mse_loss


def train_noisylabel(args, model, num_classes, num_regions, head_n,
           ema_mode, teacher, momentum_schedule, coef_schedule, it,
          EPOCHS, epoch, train_dataloader, optimizer, classify_criterion, DEVICE):
    """
    Joint learning of teacher and student with sample selection based on confidence.
    """
    model.train()
    teacher.train()  # Teacher is now trainable
    MSE = torch.nn.MSELoss()

    # Coefficient for loss balancing
    coff = coef_schedule[it]

    train_correct = [0] * num_regions
    train_mae = [0] * num_regions

    total_train_loss = 0
    total_student_loss = 0
    total_teacher_loss = 0

    for idx, (sample1, sample2, label, label_float, filename, file_index) in enumerate(train_dataloader):
        sample1, sample2, label, label_float = sample1.to(DEVICE), sample2.to(DEVICE), label.to(DEVICE), label_float.to(DEVICE)
        optimizer.zero_grad()

        # Predictions and features for student and teacher
        pred_s, feat_s = model(sample2, head_n=head_n, projection=True)
        pred_t, feat_t = teacher(sample1, head_n=head_n, projection=True)

        k = 16  # Number of top-k samples to select
        loss_student = 0
        loss_teacher = 0

        for i in range(num_regions):
            pos_probs_s = pred_s[:, i, :]  # Student's predictions for the i-th region
            pos_probs_t = pred_t[:, i, :]  # Teacher's predictions for the i-th region
            pos_feat_s = feat_s[:, i, :]   # Student's features for the i-th region
            pos_feat_t = feat_t[:, i, :]   # Teacher's features for the i-th region

            # Confidence-based Sample Selection for Student
            confidence_scores_t, _ = pos_probs_t.max(dim=-1)  # Max probability as confidence score
            topk_indices = torch.topk(confidence_scores_t, k=min(k, sample1.size(0))).indices  # Select top-k samples

            # Filtered samples for student
            filtered_samples_s = pos_probs_s[topk_indices]
            filtered_targets = label[:, i][topk_indices]
            filtered_feat_s = pos_feat_s[topk_indices]
            filtered_feat_t = pos_feat_t[topk_indices]

            # Student Loss: Classification + Consistency
            loss_s_classifier = classify_criterion(filtered_samples_s, filtered_targets)
            loss_const_s = MSE(filtered_feat_s, filtered_feat_t)
            loss_student +=  loss_s_classifier 

            # Teacher Loss: Classification Only (on all samples)
            loss_t_classifier = classify_criterion(pos_probs_t, label[:, i])
            loss_const_t = MSE(pos_feat_s, pos_feat_t)  # Optional, add consistency loss for teacher
            loss_teacher += loss_t_classifier

            # Update metrics for the region
            prediction = pos_probs_s.max(1, keepdim=True)[1]
            train_correct[i] += prediction.eq(label[:, i].view_as(prediction)).sum().item()
            train_mae[i] += torch.abs(prediction - label[:, i].view_as(prediction)).sum().item()

        # Normalize losses by number of regions
        loss_student /= num_regions
        loss_teacher /= num_regions

        # Combine both losses
        total_loss = loss_student + loss_teacher
        total_loss.backward()
        optimizer.step()

        # Record losses
        total_train_loss += total_loss.item()
        total_student_loss += loss_student.item()
        total_teacher_loss += loss_teacher.item()

    # Finalize Metrics
    total_train_loss /= len(train_dataloader)
    total_student_loss /= len(train_dataloader)
    total_teacher_loss /= len(train_dataloader)

    train_acc_list = [100.0 * train_correct[i] / len(train_dataloader.dataset) for i in range(num_regions)]
    train_mae_list = [train_mae[i] / len(train_dataloader.dataset) for i in range(num_regions)]

    # Logging
    for i in range(num_regions):
        print(f"EPOCH {epoch} / {EPOCHS}, Position {i} Train ACC : {train_acc_list[i]:.2f}")
        print(f"EPOCH {epoch} / {EPOCHS}, Position {i} Train MAE : {train_mae_list[i]:.2f}")

    print(f"EPOCH {epoch} / {EPOCHS}, Mean Train ACC : {np.mean(train_acc_list):.2f}")
    print(f"EPOCH {epoch} / {EPOCHS}, Mean Train MAE : {np.mean(train_mae_list):.2f}")
    print(f"EPOCH {epoch} / {EPOCHS}, Total Loss : {total_train_loss:.2f}")
    print(f"EPOCH {epoch} / {EPOCHS}, Student Loss : {total_student_loss:.2f}")
    print(f"EPOCH {epoch} / {EPOCHS}, Teacher Loss : {total_teacher_loss:.2f}")

    return total_train_loss, np.mean(train_acc_list), np.mean(train_mae_list), total_student_loss





def evaluate(args, model, num_classes, num_regions, head_n, EPOCHS, epoch, valid_dataloader, optimizer, classify_criterion, DEVICE):
    #여기에 num_classes_list, num_regions_list, n_head도
    model.eval()
    
    valid_correct = [0,] * num_regions
    valid_mae = [0,] * num_regions
    
    
    total_valid_loss = 0
    with torch.no_grad():
        for idx, (img, label,label_float, filename,file_index) in enumerate(valid_dataloader):
            img, label,label_float = img.to(DEVICE), label.to(DEVICE), label_float.to(DEVICE)
            pred = model(img, head_n=head_n)
            # 출력값 : [batch_size, 6, 4] . 6개의 value를 가진 4개의 class
            
            #loss for classification per 6 position
            loss_classification = 0
            #print(pred.shape)
            for i in range(num_regions):
                pos_probs = pred
                pos_probs = pos_probs[:,i,:]
                #print(pos_probs[0].argmax(), label[0,i])
                
                loss_classifier = classify_criterion(pos_probs, label[:,i])
                    
                loss_classification += loss_classifier
                
                prediction = pos_probs.max(1,keepdim=True)[1]
                
                valid_correct[i] += prediction.eq(label[:,i].view_as(prediction)).sum().item()
                valid_mae[i] += torch.abs(prediction - label[:,i].view_as(prediction)).sum().item()
            
            loss_classification = loss_classification / 6
            
            #loss for regression per 6 position
            # for i in range(6):
            #     pos_max = pred[:,i,:].argmax(dim=1)
            #     loss_regression += regress_criterion(pos_max, label_float[:,i])
            
            # loss_regression = loss_regression / 6
            
            #loss = alpha * loss_classification + (1-alpha) * loss_regression
            loss = loss_classification
            
            
            total_valid_loss += loss.item()
    
    total_valid_loss = total_valid_loss / len(valid_dataloader)
    
    
    valid_acc_list = [0,] * num_regions
    for i in range(num_regions):
        valid_acc_list[i] = 100. * valid_correct[i] / len(valid_dataloader.dataset)
        print(f"EPOCH {epoch} / {EPOCHS}, Position {i} Valid ACC : {valid_acc_list[i]:.2f}")
    
    valid_mae_list = [0,] * num_regions
    for i in range(num_regions):
        valid_mae_list[i] = valid_mae[i] / len(valid_dataloader.dataset)
        print(f"EPOCH {epoch} / {EPOCHS}, Position {i} Valid MAE : {valid_mae_list[i]:.2f}")
    
    
    #mean train acc
    print(f"EPOCH {epoch} / {EPOCHS}, Mean Valid ACC : {np.mean(valid_acc_list):.2f}")
    print(f"EPOCH {epoch} / {EPOCHS}, Mean Valid MAE : {np.mean(valid_mae_list):.2f}")
    print(f"EPOCH {epoch} / {EPOCHS}, Loss : {total_valid_loss:.2f}")
    
    return total_valid_loss, np.mean(valid_acc_list), np.mean(valid_mae_list)






