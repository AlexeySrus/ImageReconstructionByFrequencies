import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


padding_mode: str = 'zeros'

bce_loss = nn.BCELoss(size_average=True)
def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0

    for i in range(0,len(preds)):
        # print("i: ", i, preds[i].shape)
        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target)
        if(i==0):
            loss0 = loss
    return loss0, loss

fea_loss = nn.MSELoss(size_average=True)
kl_loss = nn.KLDivLoss(size_average=True)
l1_loss = nn.L1Loss(size_average=True)
smooth_l1_loss = nn.SmoothL1Loss(size_average=True)
def muti_loss_fusion_kl(preds, target, dfs, fs, mode='MSE'):
    loss0 = 0.0
    loss = 0.0

    for i in range(0,len(preds)):
        # print("i: ", i, preds[i].shape)
        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target)
        if(i==0):
            loss0 = loss

    for i in range(0,len(dfs)):
        if(mode=='MSE'):
            loss = loss + fea_loss(dfs[i],fs[i]) ### add the mse loss of features as additional constraints
            # print("fea_loss: ", fea_loss(dfs[i],fs[i]).item())
        elif(mode=='KL'):
            loss = loss + kl_loss(F.log_softmax(dfs[i],dim=1),F.softmax(fs[i],dim=1))
            # print("kl_loss: ", kl_loss(F.log_softmax(dfs[i],dim=1),F.softmax(fs[i],dim=1)).item())
        elif(mode=='MAE'):
            loss = loss + l1_loss(dfs[i],fs[i])
            # print("ls_loss: ", l1_loss(dfs[i],fs[i]))
        elif(mode=='SmoothL1'):
            loss = loss + smooth_l1_loss(dfs[i],fs[i])
            # print("SmoothL1: ", smooth_l1_loss(dfs[i],fs[i]).item())

    return loss0, loss

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1,stride=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate,stride=stride, padding_mode=padding_mode)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(RSU7,self).__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1) ## 1 -> 1/2

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        b, c, h, w = x.shape

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


class myrebnconv(nn.Module):
    def __init__(self, in_ch=3,
                       out_ch=1,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       dilation=1,
                       groups=1):
        super(myrebnconv,self).__init__()

        self.conv = nn.Conv2d(in_ch,
                              out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_ch)
        self.rl = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.rl(self.bn(self.conv(x)))


class ISNetGTEncoder(nn.Module):

    def __init__(self,in_ch=1,out_ch=1):
        super(ISNetGTEncoder,self).__init__()

        self.conv_in = myrebnconv(in_ch,16,3,stride=2,padding=1) # nn.Conv2d(in_ch,64,3,stride=2,padding=1)

        self.stage1 = RSU7(16,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,32,128)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(128,32,256)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(256,64,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,64,512)


        self.side1 = nn.Conv2d(64,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1, padding_mode=padding_mode)

    def compute_loss(self, preds, targets):

        return muti_loss_fusion(preds,targets)

    def forward(self,x):

        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        #stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)


        #side output
        d1 = self.side1(hx1)
        d1 = _upsample_like(d1,x)

        d2 = self.side2(hx2)
        d2 = _upsample_like(d2,x)

        d3 = self.side3(hx3)
        d3 = _upsample_like(d3,x)

        d4 = self.side4(hx4)
        d4 = _upsample_like(d4,x)

        d5 = self.side5(hx5)
        d5 = _upsample_like(d5,x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,x)

        # d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)], [hx1,hx2,hx3,hx4,hx5,hx6]

class ISNetDIS(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(ISNetDIS,self).__init__()

        self.enable_upscale: bool = False

        self.conv_in = nn.Conv2d(in_ch,64,3,stride=2,padding=1, padding_mode=padding_mode)
        self.pool_in = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage1 = RSU7(64,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        # self.side1 = nn.Sequential(nn.Conv2d(64,out_ch,3,padding=1), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 1, padding=0))
        # self.side2 = nn.Sequential(nn.Conv2d(64,out_ch,3,padding=1), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 1, padding=0))
        # self.side3 = nn.Sequential(nn.Conv2d(128,out_ch,3,padding=1), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 1, padding=0))
        # self.side4 = nn.Sequential(nn.Conv2d(256,out_ch,3,padding=1), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 1, padding=0))
        # self.side5 = nn.Sequential(nn.Conv2d(512,out_ch,3,padding=1), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 1, padding=0))
        # self.side6 = nn.Sequential(nn.Conv2d(512,out_ch,3,padding=1), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 1, padding=0))

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1, padding_mode=padding_mode)
        
        # self.outconv = nn.Conv2d(6*out_ch,out_ch,3)

    def compute_loss_kl(self, preds, targets, dfs, fs, mode='MSE'):

        # return muti_loss_fusion(preds,targets)
        return muti_loss_fusion_kl(preds, targets, dfs, fs, mode=mode)

    def compute_loss(self, preds, targets):

        # return muti_loss_fusion(preds,targets)
        return muti_loss_fusion(preds, targets)

    def forward(self, x):
        # Normilize from 0..1 to -1..1
        x = (x - 0.5) * 2

        hx = x

        hxin = self.conv_in(hx)
        #hx = self.pool_in(hxin)
            torch.cat((t4_wf, hf4), dim=1)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)
        if self.enable_upscale:
            d1 = _upsample_like(d1,x)

        d2 = self.side2(hx2d)
        if self.enable_upscale:
            d2 = _upsample_like(d2,x)

        d3 = self.side3(hx3d)
        if self.enable_upscale:
            d3 = _upsample_like(d3,x)

        d4 = self.side4(hx4d)
        if self.enable_upscale:
            d4 = _upsample_like(d4,x)

        d5 = self.side5(hx5d)
        if self.enable_upscale:
            d5 = _upsample_like(d5,x)

        d6 = self.side6(hx6)
        if self.enable_upscale:
            d6 = _upsample_like(d6,x)

        # d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6), 1))

        # Denormilize only RGB channels from -1..1 to 0..2
        d1[:, :3] = d1[:, :3] + 1
        d2[:, :3] = d2[:, :3] + 1
        d3[:, :3] = d3[:, :3] + 1
        d4[:, :3] = d4[:, :3] + 1
        d5[:, :3] = d5[:, :3] + 1
        d6[:, :3] = d6[:, :3] + 1

        return [d1, d2, d3, d4, d5, d6], [hx1d,hx2d,hx3d,hx4d,hx5d,hx6]


if __name__ == '__main__':
    import numpy as np
    model = ISNetDIS()
    inp = torch.rand(1, 3, 512, 512)

    with torch.no_grad():
        out = model(inp)

    for block_output, block_name in zip(out[0], ('d{}'.format(i) for i in range(1, 6 + 1))):
        print('Shape of {}: {}'.format(block_name, block_output.shape))

    print()

    for block_output, block_name in zip(out[1], ('hx{}d'.format(i) for i in range(1, 6 + 1))):
        print('Shape of {}: {}'.format(block_name, block_output.shape))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Params: {} params'.format(params))
