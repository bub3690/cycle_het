import torch
import torch.nn as nn
import torchvision

# 현재 폴더 .을 path로 추가

from .vit import PatchEmbedding, TransformerEncoderBlock , PositionalEmbedding
from .backbones import model_dict
import torch.nn.functional as F





def froze_for_linear_eval(model):
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze omni_heads parameters
    for head in model.omni_heads:
        for param in head.parameters():
            param.requires_grad = True
    
    return model


class CNN(nn.Module):
    def __init__(self,backbone,num_classes_list=[4],num_regions_list=[6],  project_features=False,use_mlp=False):
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
        
        self.projector = None 
        if project_features:
            # supervised pretrain 방법
            encoder_features = self.num_features
            self.num_features = project_features
            
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)
        
        
        
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
            
            if self.projector:
                # 공통된 스페이스로 투영
                pred_i = self.projector(pred_i)
            
            if projection:
                #projection 출력용 파라미터임.
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
    def __init__(self,backbone,num_classes_list=[4],num_regions_list=[6],project_features=False,use_mlp=False):
        super(Hybrid, self).__init__()
        self.backbone,num_features = model_dict[backbone]
        
        self.embed_dim = 768
        self.num_features = 768         
        
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
        
        self.projector = None 
        if project_features:
            # supervised pretrain 방법
            encoder_features = self.num_features
            self.num_features = project_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)        
        
        
        ###
        # https://github.com/FrancescoSaverioZuppichini/ViT
        # Transformer hybrid network
        
        self.patch_emb = PatchEmbedding(in_channels=num_features, patch_size=1, emb_size=self.embed_dim, img_size=self.img_size)
        
        # 2개의 transformer block
        self.transformer  = TransformerEncoderBlock()
        
    
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
            
            if self.projector:
                # 공통된 스페이스로 투영
                pred_i = self.projector(pred_i)            
            
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiRegionHybridDecoderWithPosition(nn.Module):
    def __init__(self, backbone, num_classes_list=[4], num_regions_list=[6], embed_dim=768, project_features=False, use_mlp=False):
        super(MultiRegionHybridDecoderWithPosition, self).__init__()
        self.backbone, num_features = model_dict[backbone]
        self.embed_dim = embed_dim
        self.num_classes_list = num_classes_list
        self.num_regions_list = num_regions_list
        self.num_features = self.embed_dim  # Embed dimension + spatial info (x, y, w, h)
        # Backbone 설정
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
            self.img_size = 16
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
            self.img_size = 16

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))

        # Feature Projector
        self.projector = None 
        if project_features:
            # supervised pretrain 방법
            encoder_features = self.num_features
            self.num_features = project_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)       

        # Transformer Encoder
        self.patch_emb = PatchEmbedding(in_channels=num_features, patch_size=1, emb_size=self.embed_dim, img_size=self.img_size)
        self.transformer  = TransformerEncoderBlock(emb_size=self.embed_dim)

        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoderLayer(
            d_model=self.embed_dim + 4,  # Embed dimension + spatial info (x, y, w, h)
            nhead=4,
            dim_feedforward=512,
            dropout=0.1
        )
        #self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # 학습 가능한 쿼리
        self.queries = nn.Parameter(torch.randn(sum(self.num_regions_list), self.embed_dim))

        self.omni_heads = []  
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)                

    def pool_rois_with_position(self, x, head_n, crop_size=None):
        """
        Pool the ROIs from the feature map and compute their positional information.
        """
        num_regions = self.num_regions_list[head_n]
        if crop_size is None:
            crop_size = x.shape[2:4]

        #print(num_regions)
        
        # Define region boxes (normalized coordinates)
        if num_regions == 6:
            boxes = [torch.tensor([0, 0, 0.4, 0.5]), 
                     torch.tensor([0.3, 0, 0.7, 0.5]),
                     torch.tensor([0.6, 0, 1, 0.5]),
                     torch.tensor([0, 0.5, 0.4, 1]),
                     torch.tensor([0.3, 0.5, 0.7, 1]),
                     torch.tensor([0.6, 0.5, 1, 1])]
        elif num_regions == 4:
            boxes = [torch.tensor([0, 0, 0.5, 0.5]), 
                     torch.tensor([0.5, 0, 1, 0.5]),
                     torch.tensor([0, 0.5, 0.5, 1]),
                     torch.tensor([0.5, 0.5, 1, 1])]
        elif num_regions == 2:
            boxes = [torch.tensor([0, 0, 1.0, 0.5]),
                     torch.tensor([0, 0.5, 1.0, 1.0])]
        elif num_regions == 1:
            boxes = [torch.tensor([0, 0, 1.0, 1.0])]

        features = []
        positions = []
        for b in boxes:
            cropped = F.interpolate(
                x[:, :,
                  int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                  int(b[1] * x.shape[3]):int(b[3] * x.shape[3])],
                size=crop_size,
                mode='bilinear',
                align_corners=False
            )
            features.append(cropped)

            # Compute normalized center and size
            center_x = (b[0] + b[2]) / 2
            center_y = (b[1] + b[3]) / 2
            width = b[2] - b[0]
            height = b[3] - b[1]
            positions.append(torch.tensor([center_x, center_y, width, height]).to(x.device))

        features = torch.stack(features, dim=1)  # Shape: (batch_size, num_regions, channels, crop_h, crop_w)
        positions = torch.stack(positions, dim=0).unsqueeze(0).repeat(features.size(0), 1, 1)  # Shape: (batch_size, num_regions, 4)
        return features, positions

    def forward(self, x, head_n, embedding=False,projection=False):
        """
        Args:
            x: Input tensor (batch_size, channels, height, width)
        Returns:
            logits: Multi-region classification outputs
        """
        num_regions = self.num_regions_list[head_n]
        # Backbone과 Transformer Encoder를 통해 특징 추출
        layer5_out = self.backbone(x)
        layer5_out = self.patch_emb(layer5_out)
        encoder_output = self.transformer(layer5_out)
        encoder_output = encoder_output.reshape(-1,16,16,768)
        encoder_output = encoder_output.permute(0,3,1,2)
        
        features, positions = self.pool_rois_with_position(encoder_output, head_n)
        #print(features.shape, positions.shape)        
        
        ## 아래를 고치기
        
        # Transformer Decoder
        #print(self.queries.shape)
        
        # (batch_size, num_regions, embed_dim)
        queries = self.queries.repeat(features.size(0), 1, 1)  
        #print(queries.shape)
        queries_with_position = torch.cat([queries, positions], dim=-1)  # Append spatial info
        
        # queries_with_position: (batch_size, num_queries , embed_dim + 4)
        # features: (batch_size, num_regions, channels, crop_h, crop_w)
        
        trasnformer_embeddings = []
        for i in range(num_regions):
            pred_i = features[:,i,:,:,:]
            #print('selection : ',pred_i.shape)
            pred_i = self.score_GAP(pred_i)
            pred_i = pred_i.view(-1,768)
            pred_i = torch.cat([pred_i, positions[:, i]], dim=-1)  # Append spatial info
            trasnformer_embeddings.append(pred_i)
        
        trasnformer_embeddings = torch.stack(trasnformer_embeddings,dim=1)
        #print('transformer embeddings: ',trasnformer_embeddings.shape)
        #print('queries with position: ',queries_with_position.shape)
        decoder_output = self.transformer_decoder(queries_with_position, trasnformer_embeddings)
        #print('decoder_output: ',decoder_output.shape)
        #decoder_output = trasnformer_embeddings
        

        # 리전별 분류
        retina_net_class = []
        projection_embeddings = []
        for i in range(num_regions):
            pred_i = decoder_output[:, i, :self.embed_dim]
            #print('pred_i: ',pred_i.shape)
            if self.projector:
                pred_i = self.projector(pred_i)
                
            if projection:
                projection_embeddings.append(pred_i)                
                
            if embedding == False:
                pred_i = self.omni_heads[head_n](pred_i)
                pred_i = F.softmax(pred_i, dim=1)
            retina_net_class.append(pred_i)
        retina_net_class = torch.stack(retina_net_class, dim=1)

        if projection:
            projection_embeddings = torch.stack(projection_embeddings,dim=1)
            return retina_net_class, projection_embeddings        
        #print(retina_net_class)
        return retina_net_class


class MultiRegionDecoderWithPositionEncoding2(nn.Module):
    def __init__(self, backbone, embed_dim=768, num_classes=4, decoder_dim=512):
        super(MultiRegionDecoderWithPositionEncoding2, self).__init__()
        self.backbone, num_features = model_dict[backbone]
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Backbone 설정
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
            self.img_size = 16
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
            self.img_size = 16

        
        self.patch_emb = PatchEmbedding(in_channels=num_features, patch_size=1, emb_size=self.embed_dim, img_size=self.img_size)
        self.transformer  = TransformerEncoderBlock(emb_size=self.embed_dim)
        
        # Positional Encoding Layer
        self.position_encoding = nn.Linear(2, embed_dim)  # ROI 위치 정보를 Embedding에 매핑

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=decoder_dim,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Region별 분류기
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, positions, embedding=False):
        """
        Args:
            x: Input tensor (batch_size, channels, height, width)
            positions: Tensor of shape (batch_size, num_queries, 2) containing center coordinates of ROIs.
        Returns:
            logits: Classification outputs for each region.
        """
        batch_size = x.size(0)        
        layer5_out = self.backbone(x)
        layer5_out = self.patch_emb(layer5_out)
        encoder_output = self.transformer(layer5_out)
        encoder_output = encoder_output.reshape(-1,16*16,768).permute(0, 2, 1) 
        #num_queries = positions.size(1)  # ROI 수

        # ROI 위치 정보를 Positional Encoding으로 변환
        position_encoded = self.position_encoding(positions)  # Shape: (batch_size, num_queries, embed_dim)

        # Query 생성 (Positional Encoding만 사용)
        queries = position_encoded  # Shape: (batch_size, num_queries, embed_dim)
        
        queries = queries.permute(1, 0, 2)  # (num_queries, batch_size, embed_dim)
        encoder_output= encoder_output.permute(2, 0, 1)  # (num_patches, batch_size, embed_dim)

        #print(queries.shape, encoder_output.shape)
        # Transformer Decoder
        decoder_output = self.transformer_decoder(
            queries,  # (num_queries, batch_size, embed_dim)
            encoder_output  # (num_patches, batch_size, embed_dim)
        )

        # Region별 분류
        decoder_output = decoder_output.permute(1, 0, 2)  # Shape: (batch_size, num_queries, embed_dim)
        
        if embedding:
            return decoder_output
        
        logits = self.classifier(decoder_output)  # (batch_size, num_queries, num_classes)

        return logits, decoder_output



class MultiRegionDecoderWithFPN(nn.Module):
    def __init__(self, backbone, embed_dim=768, num_classes=4, decoder_dim=512):
        super(MultiRegionDecoderWithFPN, self).__init__()
        self.backbone, num_features = model_dict[backbone]
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        
        
        # Backbone 설정
        if backbone == 'densenet121':
            self.backbone = self.backbone()
            self.backbone_layers = [self.backbone.features[i] for i in range(len(self.backbone.features))]
            num_features_list = [ num_features//(2**i) for i in range(4)]
            # reverse
            num_features_list = num_features_list[::-1]            
        else:
            self.backbone = self.backbone()
            self.backbone_layers = list(self.backbone.children())[:-2]
            num_features_list = [ num_features//(2**i) for i in range(4)]
            # reverse
            num_features_list = num_features_list[::-1]
        
        # for i in range( len(self.backbone_layers)-4):
        #     print(i)
        #     print(self.backbone_layers[i+4])
    
        # FPN Layers
        self.fpn_lateral_convs = nn.ModuleList([
            nn.Conv2d(num_features_list[i], embed_dim, kernel_size=1) for i in range( len(self.backbone_layers)-4 )
        ])
        self.fpn_output_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1) for i in range(len(self.backbone_layers) -4  )
        ])
        

        self.transformer  = TransformerEncoderBlock(emb_size=self.embed_dim)        
        self.postional_embedding = PositionalEmbedding(1024,embed_dim)
        
        # Positional Encoding Layer
        self.position_encoding = nn.Linear(2, embed_dim)  # ROI 위치 정보를 Embedding에 매핑

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=decoder_dim,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Region별 분류기
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, positions, embedding=False):
        """
        Args:
            x: Input tensor (batch_size, channels, height, width)
            positions: Tensor of shape (batch_size, num_queries, 2) containing center coordinates of ROIs.
        Returns:
            logits: Classification outputs for each region.
        """
        batch_size = x.size(0)
        features = []
        # Backbone 각 레이어에서 feature 추출
        for i,layer in enumerate(self.backbone_layers):
            x = layer(x)
            if i >= 4:
                features.append(x)
        

        # FPN 구성
        fpn_features = []
        fpn_out = None
        for i in range(len(features) - 1, -1, -1):  # Top-down Pathway
            lateral = self.fpn_lateral_convs[i](features[i])  # Lateral Connection
            if fpn_out is not None:  # Upsample and add to lateral
                fpn_out = F.interpolate(fpn_out, size=lateral.shape[-2:], mode='nearest')
                fpn_out += lateral
            else:
                fpn_out = lateral
            fpn_features.append(self.fpn_output_convs[i](fpn_out))  # Apply 3x3 Conv

        # 모든 FPN Feature를 concatenate
        fpn_features = [F.adaptive_avg_pool2d(f, (16, 16)) for f in fpn_features]  # Align resolution to 16x16
        fpn_features = torch.cat([f.flatten(2).permute(0, 2, 1) for f in fpn_features], dim=1)  # Combine features
        
        #print(fpn_features.shape)
        # Transformer Encoder
        encoder_output = fpn_features  # 이미 정렬된 feature map
        encoder_output = self.postional_embedding(encoder_output)
        encoder_output = self.transformer(encoder_output)
        encoder_output = encoder_output.permute(1, 0, 2)  # (num_patches, batch_size, embed_dim)

        
        # ROI 위치 정보를 Positional Encoding으로 변환
        position_encoded = self.position_encoding(positions)  # Shape: (batch_size, num_queries, embed_dim)
        queries = position_encoded.permute(1, 0, 2)  # (num_queries, batch_size, embed_dim)
        
        # Transformer Decoder
        decoder_output = self.transformer_decoder(
            queries,  # (num_queries, batch_size, embed_dim)
            encoder_output  # (num_patches, batch_size, embed_dim)
        )

        # Region별 분류
        decoder_output = decoder_output.permute(1, 0, 2)  # Shape: (batch_size, num_queries, embed_dim)
        
        if embedding:
            return decoder_output
        
        logits = self.classifier(decoder_output)  # (batch_size, num_queries, num_classes)
        return logits, decoder_output



if __name__ == "__main__":
    #model = CNN('densenet121')
    

    #model = CNN('densenet121',num_classes=5,num_regions=4)
    model = MultiRegionDecoderWithFPN('resnet18')
    
    sample = torch.randn(2, 3, 512, 512)
    positions = torch.tensor([
            [[0.2, 0.3], 
             [0.4, 0.5], 
             [0.6, 0.7], 
             [0.8, 0.9]],  # ROI 좌표 for batch 1
            
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]   # ROI 좌표 for batch 2
        ])
    # (batch_size, num_queries, 2)
    print(model(sample,positions))
      # Shape: (batch_size, num_queries, 2)
        
        
    #print(model)