from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F
from torch.nn import init
from grid_attention_layer import GridAttentionBlock2D_TORR as AttentionBlock2D
from utils import unetConv2, unetUp, conv2DBatchNormRelu, conv2DBatchNorm
from utils import init_weights

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 6, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.batch = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)
        
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.conv(x)
        #print(list(x.type()))
        x= self.batch(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )
            
            for m in self.children():
                 init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.block(x)


class UnetUp2_CT(nn.Module):
    def __init__(self,in_size1,in_size2, out_size, is_batchnorm=True):
        super(UnetUp2_CT, self).__init__()
        self.conv = conv_block(in_size1+in_size2, out_size)
        self.up = nn.Upsample(scale_factor=2)
        
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        out = self.conv(torch.cat([inputs1, outputs2], 1))
        return out
   
    
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):

        x = self.conv(x)
        return x    
    
    
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.up(x)
        return x    
    
    
    
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
    
    
class UnetGridGatingSignal2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1,stride=1, padding=0, bias=True),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1,stride=1, padding=0, bias=True),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

    
    

        
        
class UNet11_Attention(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_batchnorm=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        self.is_batchnorm = is_batchnorm
        self.pool = nn.MaxPool2d(2, 2)

        self.num_classes = num_classes

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[3],
                                   self.relu)

        self.conv3 = nn.Sequential(
            self.encoder[6],
            self.relu,
            self.encoder[8],
            self.relu,
        )
        self.conv4 = nn.Sequential(
            self.encoder[11],
            self.relu,
            self.encoder[13],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[16],
            self.relu,
            self.encoder[18],
            self.relu,
        )
       
        filters = [32,64, 128, 256, 512]
        
        self.Up5=nn.Upsample(scale_factor=2)
        self.Att5 = Attention_block(filters[4], filters[4], filters[4])
        self.Up_conv5 = UnetUp2_CT(filters[4],filters[4], filters[3], is_batchnorm)
        
        self.Up4=nn.Upsample(scale_factor=2)
        self.Att4 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[3])
        self.Up_conv4 = UnetUp2_CT(filters[3],filters[3], filters[2], is_batchnorm)

        self.Up3=nn.Upsample(scale_factor=2)
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[2])
        self.Up_conv3 = UnetUp2_CT(filters[2],filters[2], filters[1], is_batchnorm)
        self.Up_conv2 = UnetUp2_CT(filters[1],filters[1], filters[0], is_batchnorm)
        
        #self.dec1 = ConvRelu(128, num_filters)
        
        self.center = ConvRelu(512, 512)
        self.dec5 = DecoderBlock(filters[4]+filters[4],filters[4], filters[3])
        self.dec4 = DecoderBlock(filters[4],filters[3], filters[2])
        self.dec3 = DecoderBlock(filters[3],filters[2], filters[1])
        #self.dec2 = DecoderBlock(filters[1]+filters[1],filters[1], filters[0])
        self.dec2 = ConvRelu(filters[1]+filters[1], filters[0])

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        
        
        # Decoder
        center = self.center(conv5)
        upgating=self.Up5(center)
        
        x4 = self.Att5(g= upgating, x=conv4)
        dec5 = self.dec5(torch.cat([x4, self.Up5(center)], 1))
        
        x3 = self.Att4(g=dec5, x=conv3)
        dec4 = self.dec4(torch.cat([x3, dec5], 1))
        
        x2 = self.Att3(g=dec4, x=conv2)
        dec3 = self.dec3(torch.cat([x2, dec4], 1))
        dec2 = self.dec2(torch.cat([conv1, dec3], 1))
        #dec1 = self.pool(dec2)
        
        # Decoder
        #upgating=self.Up5(conv5)
        #x4 = self.Att5(g= upgating, x=conv4)
        #d5 = self.Up_conv5(x4, conv5)
      
        #d4 = self.Up4(d5)
        #x3 = self.Att4(g=d4, x=conv3)
        #d4 = self.Up_conv4(x3,d5)
       
        
        #d3 = self.Up3(d4)
        #x2 = self.Att3(g=d3, x=conv2)
        #d3 = self.Up_conv3(x2,d4)
        #d2 = self.Up_conv2(conv1,d3)
        
        #d3=self.Up5(d3)
        
        #dec1 = self.dec1(torch.cat([d3, conv1], 1))
       

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec2), dim=1)
        else:
            x_out = self.final(dec2)

        return x_out


class UNet16_Attention(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False,is_batchnorm=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        self.is_batchnorm = is_batchnorm
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        
        filters = [32,64, 128, 256, 512]
        
        self.Up5=nn.Upsample(scale_factor=2)
        self.Att5 = Attention_block(filters[4], filters[4], filters[4])
        self.Up_conv5 = UnetUp2_CT(filters[4],filters[4], filters[3], is_batchnorm)
        
        self.Up4=nn.Upsample(scale_factor=2)
        self.Att4 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[3])
        self.Up_conv4 = UnetUp2_CT(filters[3],filters[3], filters[2], is_batchnorm)
        
        self.Up3=nn.Upsample(scale_factor=2)
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[2])
        self.Up_conv3 = UnetUp2_CT(filters[2],filters[2], filters[1], is_batchnorm)
        
        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)
        
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(256, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(128, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        #center = self.center(self.pool(conv5))
        
        # Decoder
        upgating=self.Up5(conv5)
        x4 = self.Att5(g= upgating, x=conv4)
        d5 = self.Up_conv5(x4, conv5)
      

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=conv3)
        d4 = self.Up_conv4(x3,d5)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=conv2)
        d3 = self.Up_conv3(x2,d4)

        d3=self.Up5(d3)
       
        dec1 = self.dec1(torch.cat([d3, conv1], 1))
        

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class LinkNet34_Attention(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,is_batchnorm=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        self.is_batchnorm=is_batchnorm
        
        resnet = models.resnet34(pretrained=pretrained)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        filters = [64,64, 128, 256, 512]
        
        
        self.classifier1 = nn.Linear(filters[4],filters[1])
        self.classifier2 = nn.Linear(filters[1],2 )

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e2 = self.encoder1(e1)
        e3 = self.encoder2(e2)
        e4 = self.encoder3(e3)
        e5 = self.encoder4(e4)
        e5= F.adaptive_avg_pool2d(e5, (1, 1)).view(e5.shape[0], -1)
        
        x_out = self.classifier1(e5)
        x_out = self.classifier2(x_out)
        
        return x_out


    
    
    
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
        self.classifier = nn.Linear(self.num_ftrs,2 )
        
        
        

    
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
    
        x_out = self.classifier(e5)
        #print("shape of x_out",list(x_out .shape))
         
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
        self.classifier = nn.Linear(self.num_ftrs,2 )
        
        
       
    
        
    
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
    
        x_out = self.classifier(e5)
        #print("shape of x_out",list(x_out .shape))
         
        return x_out
   

 
    
class DenseNet121_Attention(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,is_batchnorm=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        self.is_batchnorm=is_batchnorm
        nonlocal_mode='concatenation_sigmoid'
        aggregation_mode='concat'
        filters = [128, 256, 512,1024]
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
        
        #self.num_ftrs = densenet.classifier.in_features
        #self.classifier = nn.Linear(self.num_ftrs,2 ) 
        
        
        ################
        # Attention Maps
        self.compatibility_score1 = AttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        self.compatibility_score2 = AttentionBlock2D(in_channels=filters[3], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        #########################
        # Aggreagation Strategies
        self.attention_filter_sizes = [filters[2], filters[3]]

        if aggregation_mode == 'concat':
            #self.classifier = nn.Linear(filters[2]+filters[3]+filters[3], self.num_classes)
            self.aggregate = nn.Linear(filters[2]+filters[3]+filters[3], self.num_classes)

        else:
            self.classifier1 = nn.Linear(filters[2], n_classes)
            self.classifier2 = nn.Linear(filters[3], n_classes)
            self.classifier3 = nn.Linear(filters[3], n_classes)
            self.classifiers = [self.classifier1, self.classifier2, self.classifier3]

            if aggregation_mode == 'mean':
                self.aggregate = self.aggregation_sep

            elif aggregation_mode == 'deep_sup':
                self.classifier = nn.Linear(filters[2] + filters[3] + filters[3], self.num_classes)
                self.aggregate = self.aggregation_ds

            elif aggregation_mode == 'ft':
                self.classifier = nn.Linear(self.num_classes*3, self.num_classes)
                self.aggregate = self.aggregation_ft
            else:
                raise NotImplementedError

        ####################
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def aggregation_sep(self, *attended_maps):
        return [ clf(att) for clf, att in zip(self.classifiers, attended_maps) ]

    def aggregation_ft(self, *attended_maps):
        preds =  self.aggregation_sep(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1))

    def aggregation_ds(self, *attended_maps):
        preds_sep =  self.aggregation_sep(*attended_maps)
        pred = self.aggregation_concat(*attended_maps)
        return [pred] + preds_sep

    def aggregation_concat(self, *attended_maps):
        return self.classifier(torch.cat(attended_maps, dim=1))        
    
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
        pooled= F.adaptive_avg_pool2d(e5, (1, 1)).view(e5.shape[0], -1)
        #print('ee5',list(e5.shape))
        
        # Attention Mechanism
        g_conv1, att1 = self.compatibility_score1(e3, e5)
        g_conv2, att2 = self.compatibility_score2(e4, e5)
        
        # flatten to get single feature vector
        fsizes = self.attention_filter_sizes
        g1 = torch.sum(g_conv1.view(e5.shape[0], fsizes[0], -1), dim=-1)
        g2 = torch.sum(g_conv2.view(e5.shape[0], fsizes[1], -1), dim=-1)
        
         
        return self.aggregate(torch.cat((g1, g2, pooled), dim=1))
    

    
    
class DenseNet121_reduced(nn.Module):
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
        #self.encoder3 = densenet.features.denseblock3
        #self.transition3=densenet.features.transition3
        #self.encoder4 = densenet.features.denseblock4
        #self.norm5=densenet.features.norm5
        
        self.num_ftrs = densenet.classifier.in_features
        self.classifier = nn.Linear(256,2 )
        
        
       
    
        
    
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
        #e4 = self.encoder3(e4)
        #e5 = self.transition3(e4)
        #e5 = self.encoder4(e5)
        #e5 = self.norm5(e5)
        e5= F.adaptive_avg_pool2d(e4, (1, 1)).view(e4.shape[0], -1)
        #print('ee5',list(e5.shape))
        
    
        x_out = self.classifier(e5)
        #print("shape of x_out",list(x_out .shape))
         
        return x_out

class previous_model(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,is_batchnorm=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        self.is_batchnorm=is_batchnorm
        filters = [16, 32, 64, 128,256]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = ConvRelu(3, filters[0])
        self.conv2 = ConvRelu(filters[0], filters[1])
        self.conv3 = ConvRelu(filters[1], filters[2])
        self.conv4 = ConvRelu(filters[2], filters[3])
        self.conv5 = ConvRelu(filters[3], filters[4])
        
        
        self.classifier1 = nn.Linear(filters[2],filters[2] )  
        self.classifier2 = nn.Linear(filters[2],filters[0] ) 
        self.classifier3 = nn.Linear(filters[0],2 )  
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        
        # Encoder
        x = self.conv1(x)
        e1 = self.pool(x)
        e2 = self.conv2(e1)
        e2 = self.pool(e2)
        e3 = self.conv3(e2)
        #e4 = self.pool(e3)
        #e5 = self.conv4(e4)
        #e5 = self.pool(e5)
        #e5 = self.conv5(e5)
        #e5 = self.pool(e5)
        
        e5= F.adaptive_avg_pool2d(e3, (1, 1)).view(e3.shape[0], -1)
        #print('ee5',list(e5.shape))
        
    
        x_out = self.classifier1(e5)
        x_out = self.classifier2(x_out)
        x_out = self.classifier3(x_out)
        #print("shape of x_out",list(x_out .shape))
        #x_out = self.sigmoid (x_out)
        return x_out    
    
    

class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class UNet(nn.Module):
    """
    Vanilla UNet.
    Implementation from https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """
    output_downscaled = 1
    module = UNetModule

    def __init__(self,
                 input_channels: int = 3,
                 filters_base: int = 32,
                 down_filter_factors=(1, 2, 4, 8, 16),
                 up_filter_factors=(1, 2, 4, 8, 16),
                 bottom_s=4,
                 num_classes=1,
                 add_output=True):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(self.module(
                down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i]))
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final = nn.Conv2d(up_filter_sizes[0], num_classes, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))

        if self.add_output:
            x_out = self.conv_final(x_out)
            if self.num_classes > 1:
                x_out = F.log_softmax(x_out, dim=1)
        return x_out


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out