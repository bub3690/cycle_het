import torch
import torch.nn as nn
import torchvision

# 현재 폴더 .을 path로 추가

from .vit import PatchEmbedding, TransformerEncoderBlock
from .backbones import model_dict
import torch.nn.functional as F





class CNN(nn.Module):
    def __init__(self,backbone,num_classes_list=[4],num_regions_list=[6]):
        super(CNN, self).__init__()
        self.backbone, self.num_features = model_dict[backbone]
        
        # numclass 와 region을 list로 받아서 처리
        self.num_classes_list = num_classes_list
        self.num_regions_list = num_regions_list
    
        # remove GAP, FC layer
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
            self.img_size = 16
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
            self.img_size = 16
        
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        #multi-task heads
        #Partailly from ARK model
        self.omni_heads = []  
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)        
        
        
    def pool_rois(self, x, head_n, crop_size=None):
        """
        Pool the ROIs from the feature map.
        ex)
        0 51 0 64  (1)
        0 51 64 128 (2)
        38 89 0 64 (3)
        38 89 64 128 (4)
        76 128 0 64 (5)
        76 128 64 128 (6)
        """
        #print(x.shape   )
        num_regions = self.num_regions_list[head_n]
        
        if crop_size is None:
            crop_size = x.shape[2:4]

        if num_regions == 6:
            boxes = [torch.tensor([0, 0, 0.4, 0.5]), # A (1)
                    torch.tensor([0.3, 0, 0.7, 0.5]), # B (2)
                    torch.tensor([0.6, 0, 1, 0.5]), # C (3)
                    torch.tensor([0, 0.5, 0.4, 1]), # D (4)
                    torch.tensor([0.3, 0.5, 0.7, 1]), # E (5)
                    torch.tensor([0.6, 0.5, 1, 1]) # F (6)
                    ]
        elif num_regions == 4:
            boxes = [   torch.tensor([0, 0, 0.5, 0.5]), # A (1)
                        torch.tensor([0.5, 0, 1, 0.5]), # B (2)
                        torch.tensor([0, 0.5, 0.5, 1]), # C (3)
                        torch.tensor([0.5, 0.5, 1, 1]) # D (4)
                    ]
        elif num_regions == 2:
            #좌우 2개의 박스.
            # 잘못 출력됨. x,y가 바뀌어서 출력됨.
            boxes = [   torch.tensor([0, 0, 1.0, 0.5]), # A (1)
                        torch.tensor([0, 0.5, 1.0, 1.0]) # B (2)
                    ]
        elif num_regions == 1:
            #전체
            boxes = [   torch.tensor([0, 0, 1.0, 1.0]) # A (1)
                    ]

        out = []
        for b in boxes:
            # print(int(b[0]*x.shape[2]),int(b[2]*x.shape[2]), 
            #       int(b[1]*x.shape[3]),int(b[3]*x.shape[3]))
            car = F.interpolate(
                x[:, :,
                  int(b[0]*x.shape[2]):int(b[2]*x.shape[2]), # 세로
                  int(b[1]*x.shape[3]):int(b[3]*x.shape[3])  # 가로
                  ],
                size=crop_size,
                mode='bilinear',
                align_corners=False
            )
            
            out.append(car)
        return torch.stack(out,dim=1)
    
    def forward(self, x, head_n, embedding=False,projection=False):
        # head_n : 몇번째 head인지.
        
        num_regions = self.num_regions_list[head_n]
        
        layer5_out = self.backbone(x)
        
        layer5_out = self.pool_rois(layer5_out,head_n) # torch.Size([1, 6, 512, 16, 16])
        b = layer5_out
        #print(b.shape)
        retina_net_class = []
        
        projection_embeddings = []
        
        for i in range(num_regions):
            pred_i = b[:,i,:,:,:]
            #print('selection : ',pred_i.shape)
            pred_i = self.score_GAP(pred_i)
            pred_i = pred_i.view(-1,self.num_features)
            #print("gap : ",pred_i.shape)
            
            if projection:
                projection_embeddings.append(pred_i)
            
            if embedding == False:
                pred_i = self.omni_heads[head_n](pred_i)
                #print(pred_i.shape)
                pred_i = F.softmax(pred_i, dim=1)
            retina_net_class.append(pred_i)
        retina_net_class = torch.stack(retina_net_class,dim=1)

        if projection:
            projection_embeddings = torch.stack(projection_embeddings,dim=1)
            return retina_net_class, projection_embeddings
        
        return retina_net_class




class Hybrid(nn.Module):
    def __init__(self,backbone,num_classes_list=[4],num_regions_list=[6]):
        super(Hybrid, self).__init__()
        self.backbone,num_features = model_dict[backbone]
        
        # numclass 와 region을 list로 받아서 처리
        self.num_classes_list = num_classes_list
        self.num_regions_list = num_regions_list
        
        # backbone에서 GAP제거
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
            self.img_size = 16
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
            self.img_size = 16
        
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        
        ###
        # https://github.com/FrancescoSaverioZuppichini/ViT
        # Transformer hybrid network
        
        self.patch_emb = PatchEmbedding(in_channels=num_features, patch_size=1, emb_size=768, img_size=self.img_size)
        
        # 2개의 transformer block
        self.transformer  = TransformerEncoderBlock()
        
        self.num_features = 768     
        #self.fc = nn.Linear(768, self.num_classes)
        
        #multi-task heads
        #Partailly from ARK model
        self.omni_heads = []  
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)                
        
    
        
    def pool_rois(self, x, head_n, crop_size=None):
        """
        Pool the ROIs from the feature map.
        ex)
        0 51 0 64  (1)
        0 51 64 128 (2)
        38 89 0 64 (3)
        38 89 64 128 (4)
        76 128 0 64 (5)
        76 128 64 128 (6)
        """
        #print(x.shape   )
        num_regions = self.num_regions_list[head_n]
        
        if crop_size is None:
            crop_size = x.shape[2:4]

        if num_regions == 6:
            boxes = [torch.tensor([0, 0, 0.4, 0.5]), # A (1)
                    torch.tensor([0.3, 0, 0.7, 0.5]), # B (2)
                    torch.tensor([0.6, 0, 1, 0.5]), # C (3)
                    torch.tensor([0, 0.5, 0.4, 1]), # D (4)
                    torch.tensor([0.3, 0.5, 0.7, 1]), # E (5)
                    torch.tensor([0.6, 0.5, 1, 1]) # F (6)
                    ]
        elif num_regions == 4:
            boxes = [   torch.tensor([0, 0, 0.5, 0.5]), # A (1)
                        torch.tensor([0.5, 0, 1, 0.5]), # B (2)
                        torch.tensor([0, 0.5, 0.5, 1]), # C (3)
                        torch.tensor([0.5, 0.5, 1, 1]) # D (4)
                    ]
        elif num_regions == 2:
            #좌우 2개의 박스.
            # 잘못 출력됨. x,y가 바뀌어서 출력됨.
            boxes = [   torch.tensor([0, 0, 1.0, 0.5]), # A (1)
                        torch.tensor([0, 0.5, 1.0, 1.0]) # B (2)
                    ]
        elif num_regions == 1:
            #전체
            boxes = [   torch.tensor([0, 0, 1.0, 1.0]) # A (1)
                    ]

        out = []
        for b in boxes:
            # print(int(b[0]*x.shape[2]),int(b[2]*x.shape[2]), 
            #       int(b[1]*x.shape[3]),int(b[3]*x.shape[3]))
            car = F.interpolate(
                x[:, :,
                  int(b[0]*x.shape[2]):int(b[2]*x.shape[2]), # 세로
                  int(b[1]*x.shape[3]):int(b[3]*x.shape[3])  # 가로
                  ],
                size=crop_size,
                mode='bilinear',
                align_corners=False
            )
            
            out.append(car)
            
        return torch.stack(out,dim=1)

    
    def forward(self, x, head_n, embedding=False,projection=False):
        # head_n : 몇번째 head인지.
        
        num_regions = self.num_regions_list[head_n]
        layer5_out = self.backbone(x)
        
        
        #print(b.shape)
        
        ###
        #여기서 vit 작업.
        # print('before patch : ',layer5_out.shape)
        layer5_out = self.patch_emb(layer5_out)
        
        ###
        layer5_out = self.transformer(layer5_out)
        layer5_out = layer5_out.reshape(-1,16,16,768)
        layer5_out = layer5_out.permute(0,3,1,2)
        # print('after transformer : ',layer5_out.shape)
        
        
        b = self.pool_rois(layer5_out,head_n) # torch.Size([1, 6, 512, 16, 16])
        
        retina_net_class = []
        projection_embeddings = []
        for i in range(num_regions):
            pred_i = b[:,i,:,:,:]
            #print('selection : ',pred_i.shape)
            pred_i = self.score_GAP(pred_i)
            pred_i = pred_i.view(-1,768)
            #print("gap : ",pred_i.shape)
            
            if projection:
                projection_embeddings.append(pred_i)
            
            if embedding == False:                
                pred_i = self.omni_heads[head_n](pred_i)
                #print(pred_i.shape)
                pred_i = F.softmax(pred_i, dim=1)
            retina_net_class.append(pred_i)
        retina_net_class = torch.stack(retina_net_class,dim=1)
        
        if projection:
            projection_embeddings = torch.stack(projection_embeddings,dim=1)
            return retina_net_class, projection_embeddings
        
        return retina_net_class



if __name__ == "__main__":
    #model = CNN('densenet121')
    

    model = CNN('densenet121',num_classes=5,num_regions=4)
    
    sample = torch.randn(8, 3, 512, 512)
    print(model(sample).shape)
    
    #print(model)