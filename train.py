





##
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import cv2
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import argparse

from skimage import exposure

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


### dataset ###
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split # train , test 분리에 사용.
from sklearn.model_selection import KFold
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


from PIL import Image
import cv2


##
import torch
import torch.nn.functional as F

import wandb
import time

### modules
from modules.model import CNN, Hybrid

#from modules.model import CNN, Hybrid
from modules.dataset import SeverityDataset

# from modules.trainer import train_latentmixup, train_mixup, train_multipatch

# # dul 
# from modules.loss import ClsLoss,get_loss

###



# dataset

def train(args, model, num_classes, num_regions, head_n, EPOCHS, epoch, train_dataloader, optimizer, classify_criterion, DEVICE):
    #여기에 num_classes_list, num_regions_list, n_head도
    model.train()
    
    train_correct = [0,] * num_regions
    train_mae = [0,] * num_regions
    
    
    total_train_loss = 0
    
    for idx, (img, label,label_float, filename,file_index) in enumerate(train_dataloader):
        img, label,label_float = img.to(DEVICE), label.to(DEVICE), label_float.to(DEVICE)
        optimizer.zero_grad()
        pred = model(img, head_n= head_n)
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
    
    total_train_loss = total_train_loss / len(train_dataloader)
    
    
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
    
    return total_train_loss, np.mean(train_acc_list), np.mean(train_mae_list)



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




def plot_tsne(args, model, num_classes, num_regions, head_n, valid_dataloader, DEVICE):
        print("Get Embedding")
        model.eval()    
        
        embedding_list = []
        label_list = []
        position_list = []
        with torch.no_grad():
            for idx, (img, label,label_float, filename,file_index) in enumerate(valid_dataloader):
                img, label, label_float = img.to(DEVICE), label.to(DEVICE),label_float.to(DEVICE)
                pred = model(img, head_n=head_n,embedding=True)
                for i in range(num_regions):
                    pos_probs = pred[:,i,:]
                    embedding_list += pos_probs.cpu().numpy().tolist()
                    label_list += label[:,i].cpu().numpy().tolist()
                    position_list += [i+1] * len(label)
        
        embedding_list = np.array(embedding_list)
        print(embedding_list.shape)
        # t-sne
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(embedding_list)
        return X_2d, label_list, position_list



def load_pretrained_model_without_omni_heads(model, checkpoint_path):
    # 체크포인트 불러오기
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    #import pdb; pdb.set_trace()
    #pretrained_dict = checkpoint['state_dict']  # 모델의 state_dict 불러오기
    
    # 현재 모델의 state_dict
    model_dict = model.state_dict()
    
    # omni_heads 제외
    filtered_dict = {
        k: v for k, v in checkpoint.items() if not (k.startswith('fc') or k.startswith('omni_heads'))
    }
    # 불러온 파라미터를 현재 모델 state_dict에 적용
    model_dict.update(filtered_dict)
    
    # 업데이트된 state_dict를 모델에 로드
    model.load_state_dict(model_dict)
    print("Loaded pretrained model without omni_heads.")
    
    return model




def main(args):
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('Using Pytorch version : ',torch.__version__,' Device : ',DEVICE)
    
    if args.linux:
        print('Using Linux')
        base_path = '/shared/home/mai/jongbub/ssl/cycle_het/'
        data_path = '/shared/home/mai/jongbub/ssl/cycle_het/dataset/'
    else:
        print('Using Windows')
        base_path = '/hdd/project/cycle_het/'
        data_path = '/hdd/project/cycle_het/dataset/'
    
    pretrained = 'scratch'
    if args.pretrained != None:
        pretrained = 'pretrained'
    
    
    model_name = 'single_{}_{}_{}_{}_{}_{}_ep{}'.format(args.arch, args.backbone, args.optimizer, args.loss,args.dataset[0],pretrained, args.epochs)    
    
    
    
    wandb.login(key='9b0830eae021991e53eaabb9bb697d9efef8fd58') # 본인 키 삽입.
    wandb.init(project="cycle")
    wandb.config.update(args)
    experiment_name = model_name # check point의 이름이 될 것.
    wandb.run.name = experiment_name
    wandb.run.save()
    
    ###
    if args.tsne:
        tsne_folder = os.path.join(base_path,'plots/{}'.format(model_name))
        os.makedirs(tsne_folder,exist_ok=True)
    
    
    
    ####
    # data loading
    
    
    # cycle에서는 여기를 for문으로 들어가야함.
    # 데이터셋에 맞게 불러오기.
    print('Data Loading...')
    
    num_classes_list = []
    num_regions_list = []
    
    print(args.dataset)
    dataset_list = args.dataset
    
    train_data_loader_list = []
    valid_data_loader_list = []
    test_data_loader_list = []
    
    consensus_data_loader_list = []
    consensus_bool_list = []
    
    
    for dataset in dataset_list:
        #dataset 이름 맞게 할당.
        if dataset =='brixia':
            df_name = 'brixia_severity.csv'
            data_folder_name = 'brixia_registered_images'
            num_class = 4
        elif dataset == 'inha':
            df_name = 'inha_severity.csv'
            data_folder_name = 'inha_registered_images'
            num_class = 5
        elif dataset == 'ralo':
            df_name = 'ralo_severity.csv'
            data_folder_name = 'ralo_registered_images'
            num_class = 5
        elif dataset == 'edema':
            df_name = 'edema_severity.csv'
            data_folder_name = 'edema_registered_images'
            num_class = 4

        label_data = pd.read_csv( os.path.join(data_path,'result_df',df_name) )
        # data 불러오고, consensus test 있으면 분리하기, train, valid 분리. subject_id 걸러내기. 
        
        ###
        num_regions = label_data['num_split'].unique()[0]
        print('Num Regions : ',num_regions, ' Num Class : ',num_class)
        
        num_classes_list.append(num_class)
        num_regions_list.append(num_regions)
        
        
        ###
        
        # consensus_csv Filename과 겹치는 데이터 제거
        consensus_label_data = label_data[label_data['consensus']== 1]
        label_data = label_data[label_data['consensus'] == 0]
        subject_array=label_data['subject_id'].unique()
        
        train_subject_data, else_subject_data = train_test_split(subject_array, test_size=0.3, random_state=42)
        valid_subject_data, test_subject_data = train_test_split(else_subject_data, test_size=0.5, random_state=42)

        train_data = label_data[label_data['subject_id'].isin(train_subject_data)]['file_id'].to_numpy().tolist()
        valid_data = label_data[label_data['subject_id'].isin(valid_subject_data)]['file_id'].to_numpy().tolist()
        test_data = label_data[label_data['subject_id'].isin(test_subject_data)]['file_id'].to_numpy().tolist()    
        
        if consensus_label_data.shape[0] > 0:
            consensus_data = consensus_label_data['file_id'].to_numpy().tolist()
            consensus_bool = True
        else:
            consensus_bool = False
            consensus_data = []
        
        #y label

        train_y_data = label_data[label_data['file_id'].isin(train_data)]['severity'].to_numpy()
        valid_y_data = label_data[label_data['file_id'].isin(valid_data)]['severity'].to_numpy()
        test_y_data = label_data[label_data['file_id'].isin(test_data)]['severity'].to_numpy()
        
        if consensus_bool:
            consensus_y_data = consensus_label_data['severity'].to_numpy()
        
        
        ####
        # 데이터셋 별 분포 확인.
        # print('Train Data Distribution')
        # print(Counter(train_y_data))
        # print('Valid Data Distribution')
        # print(Counter(valid_y_data))
        # print('Test Data Distribution')
        # print(Counter(test_y_data))
        # if consensus_bool:
        #     print('Consensus Data Distribution')
        #     print(Counter(consensus_y_data))
        
        ####
        
        
        # dataset 만들기.
        train_dataset = SeverityDataset(filename_list=train_data,
                                        label_list=train_y_data,
                                        label_df=label_data,
                                        prefix=os.path.join(data_path,data_folder_name),
                                        transform=torch.nn.Sequential(
                                                                transforms.Resize(size=[512, 512], antialias=True),
                                                            ),
                                        augmentation=None,
                                        train=True)
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=args.num_workers
                                                    )

        valid_dataset = SeverityDataset(filename_list=valid_data,
                                        label_list=valid_y_data,
                                        label_df=label_data,
                                        prefix=os.path.join(data_path,data_folder_name),
                                        transform=torch.nn.Sequential(
                                                                transforms.Resize(size=[512, 512], antialias=True),
                                                            ),
                                        augmentation=None,
                                        train=False)
        
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=args.num_workers
                                                    )
        
        test_dataset = SeverityDataset(filename_list=test_data,
                                        label_list=test_y_data,
                                        label_df=label_data,
                                        prefix=os.path.join(data_path,data_folder_name),
                                        transform=torch.nn.Sequential(
                                                                transforms.Resize(size=[512, 512], antialias=True),
                                                            ),
                                        augmentation=None,
                                        train=False)
        
        test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=args.num_workers
                                                    )
        
        if consensus_bool:
            consensus_dataset = SeverityDataset(filename_list=consensus_data,
                                        label_list=consensus_y_data,
                                        label_df=consensus_label_data,
                                        prefix=os.path.join(data_path,data_folder_name),
                                        transform=torch.nn.Sequential(
                                                                transforms.Resize(size=[512, 512], antialias=True),
                                                            ),
                                        augmentation=None,
                                        train=False)
        
            consensus_dataloader = torch.utils.data.DataLoader(consensus_dataset, 
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=args.num_workers
                                                    )
        
        train_data_loader_list.append(train_dataloader)
        valid_data_loader_list.append(valid_dataloader)
        test_data_loader_list.append(test_dataloader)
        
        
        if consensus_bool:
            consensus_data_loader_list.append(consensus_dataloader)
        else:
            consensus_data_loader_list.append(None)
        consensus_bool_list.append(consensus_bool)
    
    
    EPOCHS = args.epochs
    lr = args.lr    
        
    if args.arch == 'cnn':
        print('CNN')
        model = CNN(backbone=args.backbone, num_classes_list = num_classes_list, num_regions_list=num_regions_list).to(DEVICE)
        # class 에서 num_region을 받지 않도록하고, forward에서 받도록 수정.
    elif args.arch == 'hybrid':
        print('Hybrid')
        model = Hybrid(backbone=args.backbone, num_classes_list = num_classes_list, num_regions_list=num_regions_list).to(DEVICE)    
    
    ###
    # 학습된 모델에서 백본 레이어 불러오기.
    if args.pretrained != None:
        model = load_pretrained_model_without_omni_heads(model, args.pretrained)    
    
    
    if args.optimizer == 'SGD':        
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)    
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                    #weight_decay=args.weight_decay
                                    )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)    
    
    classify_criterion = nn.CrossEntropyLoss()
    
    
    # trainer
    
    all_train_loss_list = [ [] for _ in range(len(num_regions_list)) ]
    all_valid_loss_list = [ [] for _ in range(len(num_regions_list)) ]

    all_train_acc_list = [ [] for _ in range(len(num_regions_list)) ]
    all_valid_acc_list = [ [] for _ in range(len(num_regions_list)) ]
    
    all_train_mae_list = [ [] for _ in range(len(num_regions_list)) ]
    all_valid_mae_list = [ [] for _ in range(len(num_regions_list)) ] 
    
    
    all_consensus_test_acc_list = [ [] for _ in range(len(num_regions_list)) ]
    all_consensus_test_loss_list = [ [] for _ in range(len(num_regions_list)) ]    
    
    best_valid_acc = 0
    best_train_acc_with_val = 0
    best_consensus_test_acc_with_val = 0    
    
    for epoch in range(EPOCHS):
        epoch = epoch + 1
        print(f"--START EPOCH {epoch} / {EPOCHS} --")
        print(model_name)
        start_time = time.time()
        
        # 이게 for문으로 감싸야함.
        for head_n in range( len(num_regions_list) ):
            print("=====================================")            
            print(f"Head {dataset_list[head_n]} Training")
            train_loss, train_acc, train_mae = train(args, model, num_classes_list[head_n], num_regions_list[head_n], head_n,
                                                     EPOCHS, epoch, 
                                                     train_data_loader_list[head_n],
                                                     optimizer, classify_criterion, DEVICE)
            
            all_train_loss_list[head_n].append(train_loss)
            all_train_acc_list[head_n].append(train_acc)
            all_train_mae_list[head_n].append(train_mae)
            
            
            
            print("=====================================")
            print(f"Head {dataset_list[head_n]} Validation")
            valid_loss, valid_acc, valid_mae = evaluate(args, model, num_classes_list[head_n], num_regions_list[head_n], head_n,
                                                        EPOCHS, epoch, 
                                                        valid_data_loader_list[head_n],
                                                        optimizer, classify_criterion, DEVICE)
            
            all_valid_loss_list[head_n].append(valid_loss)
            all_valid_acc_list[head_n].append(valid_acc)
            all_valid_mae_list[head_n].append(valid_mae)
            
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_train_acc_with_val = train_acc
                # save best
                torch.save(model.state_dict(), os.path.join(base_path,'checkpoint','{}_best.pth'.format(model_name)) )            
            
            wandb.log({
                f"{dataset_list[head_n]} Train Loss": train_loss,
                f"{dataset_list[head_n]} Valid Loss": valid_loss,
                f"{dataset_list[head_n]} Train ACC": train_acc,
                f"{dataset_list[head_n]} Valid ACC": valid_acc
            })            
            
            
            
            

    
    ### 학습 종료 후 ###
    # best model 불러오기.
    model.load_state_dict( torch.load(os.path.join(base_path,'checkpoint','{}_best.pth'.format(model_name))) )
    
    ###
    
    model.eval()
    # validation
    for head_n in range( len(num_regions_list) ):
        
        pred_list = []
        label_list = []
        
        print(" ===================================== ")
        print("Dataset : ",dataset_list[head_n])
        with torch.no_grad():
            for idx, (img, label,label_float, filename, file_index) in enumerate(test_data_loader_list[head_n]):
                img, label,label_float = img.to(DEVICE), label.to(DEVICE),label_float.to(DEVICE)
                pred = model(img, head_n)
                for i in range( num_regions_list[head_n] ):
                    pos_probs = pred
                    pos_probs = pos_probs[:,i,:]
                    prediction = pos_probs.max(1,keepdim=True)[1].view(-1)
                    pred_list += prediction.cpu().numpy().tolist()
                    label_list += label[:,i].cpu().numpy().tolist()
            
            # print mean acc
            test_acc = 100. * sum([1 for i in range(len(pred_list)) if pred_list[i] == label_list[i]]) / len(pred_list)
            # print mae
            test_mae = sum([abs(pred_list[i] - label_list[i]) for i in range(len(pred_list))]) / len(pred_list)
            
            print(f"Test ACC : {test_acc:.2f}, Test MAE : {test_mae:.2f}")
            
            if args.tsne:
                X_2d, label_list, position_list = plot_tsne(args, model, num_classes_list[head_n], num_regions_list[head_n], head_n, test_data_loader_list[head_n], DEVICE)
                df = pd.DataFrame()
                df['x'] = X_2d[:,0]
                df['y'] = X_2d[:,1]
                df['label'] = label_list
                df['position'] = position_list
                
                plt.figure(figsize=(10,10))
                sns.scatterplot(x='x', y='y', hue='label',style='position',data=df,palette='Set1')
                plt.savefig(os.path.join(tsne_folder,f'{dataset_list[head_n]}_test_tsne.png'))            
        wandb.log({
            f"{dataset_list[head_n]} Test ACC": test_acc,
            f"{dataset_list[head_n]} Test MAE": test_mae
        })
    
        
        if consensus_bool_list[head_n]:

            pred_list = []
            label_list = []
            with torch.no_grad():
                for idx, (img, label,label_float, filename, file_index) in enumerate(consensus_data_loader_list[head_n]):
                    img, label,label_float = img.to(DEVICE), label.to(DEVICE),label_float.to(DEVICE)
                    pred = model(img, head_n)
                    for i in range( num_regions_list[head_n]):
                        pos_probs = pred
                        pos_probs = pos_probs[:,i,:]
                        prediction = pos_probs.max(1,keepdim=True)[1].view(-1)
                        pred_list += prediction.cpu().numpy().tolist()
                        label_list += label[:,i].cpu().numpy().tolist()
                
                # print mean acc
                consensus_acc = 100. * sum([1 for i in range(len(pred_list)) if pred_list[i] == label_list[i]]) / len(pred_list)
                consensus_mae = sum([abs(pred_list[i] - label_list[i]) for i in range(len(pred_list))]) / len(pred_list)
                print(f"Consensus Test ACC : {consensus_acc:.2f}, Consensus Test MAE : {consensus_mae:.2f}")      
                
            wandb.log({
                f"{dataset_list[head_n]} Consensus Test ACC": consensus_acc,
                f"{dataset_list[head_n]} Consensus Test MAE": consensus_mae
            })
            
            #plot tsne
            if args.tsne:
                X_2d, label_list, position_list = plot_tsne(args, model, num_classes_list[head_n], num_regions_list[head_n], head_n, consensus_data_loader_list[head_n], DEVICE)
                df = pd.DataFrame()
                df['x'] = X_2d[:,0]
                df['y'] = X_2d[:,1]
                df['label'] = label_list
                df['position'] = position_list
                
                plt.figure(figsize=(10,10))
                sns.scatterplot(x='x', y='y', hue='label',style='position',data=df,palette='Set1')
                plt.savefig(os.path.join(tsne_folder,f'{dataset_list[head_n]}_consensus_tsne.png'))
       

    
    
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    
    ###
    #parser.add_argument('--dataset', default='brixia', type=str, help='[brixia, inha, ralo, edema]')
    parser.add_argument(
        '--dataset',
        type=str,
        action='append',  # 값을 리스트에 추가
        help='[brixia, inha, ralo, edema]'
    )
    
    # pretrained model checkpoint
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained model checkpoint')
    
    ###
    
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--num_workers', default=1, type=int, help='num workers')
    parser.add_argument('--linux', action='store_true')
    
    parser.add_argument('--arch', default='cnn', type=str, help='[cnn, hybrid]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='[ resnet18,resnet34,resnet50, mobilenet_v3_small, densenet121 ]')
    parser.add_argument('--optimizer', default='SGD', type=str, help='[SGD, Adam]')
    
    parser.add_argument('--tsne', action='store_true')
    parser.add_argument('--name', default='base', type=str, help='name')
    parser.add_argument('--loss', default='ce', type=str, help='[ce,]')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='label_smoothing')
    

    
    args = parser.parse_args()
    
    main(args)