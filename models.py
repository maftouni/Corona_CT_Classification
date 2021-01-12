from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F
from torch.nn import init
from basic_layers import *
from attention_module import *
    
class DenseNet169(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,is_batchnorm=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        self.is_batchnorm=is_batchnorm
        densenet = models.densenet169(pretrained=pretrained)

        self.firstconv = densenet.features.conv0
        self.firstbn = densenet.features.norm0
        self.firstrelu = densenet.features.relu0
        self.firstmaxpool = densenet.features.pool0
        self.encoder1 = densenet.features.denseblock1
        self.transition1=densenet.features.transition1
        self.encoder2 = densenet.features.denseblock2
        self.transition2=densenet.features.transition2
        self.encoder3 = densenet.features.denseblock3
        self.transition3=densenet.features.transition3
        self.encoder4 = densenet.features.denseblock4
        self.norm5=densenet.features.norm5
        
        self.num_ftrs = densenet.classifier.in_features
        self.classifier = nn.Linear(self.num_ftrs,self.num_classes )
           
    def forward(self, x):
        
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e2 = self.encoder1(e1)
        e3 = self.transition1(e2)
        e3 = self.encoder2(e3)
        e4 = self.transition2(e3)
        e4 = self.encoder3(e4)
        e5 = self.transition3(e4)
        e5 = self.encoder4(e5)
        e5 = self.norm5(e5)
        e5= F.adaptive_avg_pool2d(e5, (1, 1)).view(e5.shape[0], -1)
        x_out = self.classifier(e5)
       
         
        return x_out
        
                
class DenseNet121(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,is_batchnorm=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        self.is_batchnorm=is_batchnorm
        filters = [64, 128, 256, 512,1024]
        densenet = models.densenet121(pretrained=pretrained)

        self.firstconv = densenet.features.conv0
        self.firstbn = densenet.features.norm0
        self.firstrelu = densenet.features.relu0
        self.firstmaxpool = densenet.features.pool0
        self.encoder1 = densenet.features.denseblock1
        self.transition1=densenet.features.transition1
        self.encoder2 = densenet.features.denseblock2
        self.transition2=densenet.features.transition2
        self.encoder3 = densenet.features.denseblock3
        self.transition3=densenet.features.transition3
        self.encoder4 = densenet.features.denseblock4
        self.norm5=densenet.features.norm5
        
        self.num_ftrs = densenet.classifier.in_features
        self.fc = nn.Linear(self.num_ftrs,self.num_classes )       
    
    def forward(self, x):
        
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e2 = self.encoder1(e1)
        e3 = self.transition1(e2)
        e3 = self.encoder2(e3)
        e4 = self.transition2(e3)
        e4 = self.encoder3(e4)
        e5 = self.transition3(e4)
        e5 = self.encoder4(e5)
        e5 = self.norm5(e5)
        e5= F.adaptive_avg_pool2d(e5, (1, 1)).view(e5.shape[0], -1)
        #print('ee5',list(e5.shape))
    
        x_out = self.fc(e5)
        #print("shape of x_out",list(x_out .shape))
         
        return x_out
   

 
class Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class ResidualAttentionModel(nn.Module):
    
    def __init__(self,n_classes):
        
        super(ResidualAttentionModel, self).__init__()
        input_size = 224
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.residual_block1 = ResidualBlock(32, 128)  # 32*32
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128, size1=(input_size, input_size), size2=(int(input_size/2), int(input_size/2)))  # 32*32
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 16*16
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256, size=(int(input_size/2), int(input_size/2)))  # 16*16
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256, size=(int(input_size/2), int(input_size/2)))  # 16*16 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 8*8
        self.residual_block5 = ResidualBlock(1024, 1024)  # 8*8
        self.residual_block6 = ResidualBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=int(input_size/4))
        )
        self.fc = nn.Linear(1024,n_classes)

    def forward(self, x):
        #print('xxxxx',list(x.shape))
        out = self.conv1(x)
        # out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        #print('outttttttttttttttt',list(out.shape))
        out = self.attention_module1(out)
        #print('outttttttttttt_atte',list(out.shape))
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out    

    
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB,a,b, nb_classes=2):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
       
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()
        
        # Create new classifier
        
        self.classifier = nn.Linear(a+b, 755)
        self.classifier2 = nn.Linear(755, nb_classes)
       
        
    def forward(self, x):
        
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x) 
        x2 = x2.view(x2.size(0), -1)  
        x = torch.cat((x1, x2), dim=1)  
        x = self.classifier(F.relu(x))
        x = self.classifier2(F.relu(x))
     
        return x    
    
    
    
