
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import timm

class ModifiedDensenet121(nn.Module):
    def __init__(self):
        super(ModifiedDensenet121, self).__init__()
        self.densenet = torchvision.models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        
        # Remove avgpool layer, classifier layer
        self.features = nn.Sequential(*list(self.densenet.children())[:-1])
        
        
    def forward(self, x):
        features = self.features(x)
        return features


def resnet18():
    return torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')

def resnet34():
    return torchvision.models.resnet34(weights='ResNet34_Weights.DEFAULT')

def resnet50():
    return torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')

def densenet121():
    return ModifiedDensenet121()

def pvt_v2_b0():
    transformer_encoder = timm.create_model('pvt_v2_b0', pretrained=True)
    return transformer_encoder

def mobilenet_v3_small():
    return torchvision.models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'densenet121': [densenet121, 1024],
    'pvt_v2_b0': [pvt_v2_b0, 256],
    'mobilenet_v3_small': [mobilenet_v3_small, 576],
}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



if __name__ == '__main__':
    model = vit_base_patch16_224()
    #model = mobilenet_v3_small()
    # classifier 제거
    #model.classifier = Identity()
    
    # transformer head 제거. feature 남기기.
    
    model = nn.Sequential(*list(model.children())[:-2])
    
    print(model)
    sample = torch.randn(1, 3, 512, 512)
    
    print(model(sample).shape)
