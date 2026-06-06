import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # Standard SiLU activation function

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ConvNeXtBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
            self.model = models.convnext_tiny(weights=weights)
        else:
            self.model = models.convnext_tiny(weights=None)
            
        self.features = self.model.features
        
        # Divide into stages to extract multi-scale feature maps
        self.stage1 = nn.Sequential(*self.features[0:2])  # Stride 4
        self.stage2 = nn.Sequential(*self.features[2:4])  # Stride 8 (P3)
        self.stage3 = nn.Sequential(*self.features[4:6])  # Stride 16 (P4)
        self.stage4 = nn.Sequential(*self.features[6:8])  # Stride 32 (P5)
        
    def forward(self, x):
        x1 = self.stage1(x)  # stride 4
        x2 = self.stage2(x1) # stride 8 (P3) - channels: 192
        x3 = self.stage3(x2) # stride 16 (P4) - channels: 384
        x4 = self.stage4(x3) # stride 32 (P5) - channels: 768
        return x2, x3, x4

class PANetNeck(nn.Module):
    def __init__(self, in_channels=[192, 384, 768], out_channels=256):
        super().__init__()
        # Projection convolutions to map backbone channel counts to target neck width
        self.proj3 = ConvBlock(in_channels[0], out_channels, 1)
        self.proj4 = ConvBlock(in_channels[1], out_channels, 1)
        self.proj5 = ConvBlock(in_channels[2], out_channels, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Top-down convolutions
        self.td_conv4 = ConvBlock(out_channels * 2, out_channels, 3, padding=1)
        self.td_conv3 = ConvBlock(out_channels * 2, out_channels, 3, padding=1)
        
        # Bottom-up downsamplers
        self.downsample3 = ConvBlock(out_channels, out_channels, 3, stride=2, padding=1)
        self.downsample4 = ConvBlock(out_channels, out_channels, 3, stride=2, padding=1)
        
        # Bottom-up convolutions
        self.bu_conv4 = ConvBlock(out_channels * 2, out_channels, 3, padding=1)
        self.bu_conv5 = ConvBlock(out_channels * 2, out_channels, 3, padding=1)
        
    def forward(self, p3, p4, p5):
        # Project channel widths
        c3 = self.proj3(p3)
        c4 = self.proj4(p4)
        c5 = self.proj5(p5)
        
        # Top-down aggregation path
        p5_td = c5
        p4_td = self.td_conv4(torch.cat([c4, self.upsample(p5_td)], dim=1))
        p3_td = self.td_conv3(torch.cat([c3, self.upsample(p4_td)], dim=1))
        
        # Bottom-up aggregation path
        n3 = p3_td
        n4 = self.bu_conv4(torch.cat([p4_td, self.downsample3(n3)], dim=1))
        n5 = self.bu_conv5(torch.cat([p5_td, self.downsample4(n4)], dim=1))
        
        return n3, n4, n5

class DecoupledHeadLevel(nn.Module):
    def __init__(self, in_channels=256, num_classes=5):
        super().__init__()
        # Classification branch
        self.cls_conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, padding=1),
            ConvBlock(in_channels, in_channels, 3, padding=1)
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        
        # Regression & Objectness branch
        self.reg_conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, padding=1),
            ConvBlock(in_channels, in_channels, 3, padding=1)
        )
        self.reg_pred = nn.Conv2d(in_channels, 4, 1)
        self.obj_pred = nn.Conv2d(in_channels, 1, 1)
        
    def forward(self, x):
        cls_feat = self.cls_conv(x)
        cls_out = self.cls_pred(cls_feat)
        
        reg_feat = self.reg_conv(x)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)
        
        return cls_out, reg_out, obj_out

class Detector(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, neck_channels=256):
        super().__init__()
        self.backbone = ConvNeXtBackbone(pretrained=pretrained)
        self.neck = PANetNeck(in_channels=[192, 384, 768], out_channels=neck_channels)
        
        self.head3 = DecoupledHeadLevel(in_channels=neck_channels, num_classes=num_classes)
        self.head4 = DecoupledHeadLevel(in_channels=neck_channels, num_classes=num_classes)
        self.head5 = DecoupledHeadLevel(in_channels=neck_channels, num_classes=num_classes)
        
    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        
        cls3, reg3, obj3 = self.head3(n3)
        cls4, reg4, obj4 = self.head4(n4)
        cls5, reg5, obj5 = self.head5(n5)
        
        return {
            'p3': (cls3, reg3, obj3),
            'p4': (cls4, reg4, obj4),
            'p5': (cls5, reg5, obj5)
        }

def decode_predictions(reg_pred, obj_pred, cls_pred, stride, device):
    """
    Decodes vectorized anchor-free predictions into Pascal VOC bbox format coordinates,
    confidence score, and class probabilities.
    
    Args:
        reg_pred (Tensor): Regression output [B, 4, H, W]
        obj_pred (Tensor): Objectness output [B, 1, H, W]
        cls_pred (Tensor): Classification output [B, num_classes, H, W]
        stride (int): Stride of the current feature level (8, 16, or 32)
        device (torch.device): Device
        
    Returns:
        bboxes (Tensor): [B, H*W, 4] Pascal VOC format coordinates [xmin, ymin, xmax, ymax]
        obj_scores (Tensor): [B, H*W, 1] Objectness scores (after Sigmoid)
        cls_probs (Tensor): [B, H*W, num_classes] Class probabilities (after Sigmoid)
    """
    B, _, H, W = reg_pred.shape
    
    # Sigmoid for objectness and classification
    obj_scores = torch.sigmoid(obj_pred.permute(0, 2, 3, 1).reshape(B, H * W, 1))
    cls_probs = torch.sigmoid(cls_pred.permute(0, 2, 3, 1).reshape(B, H * W, -1))
    
    # Bbox coordinates offset and scale
    reg = reg_pred.permute(0, 2, 3, 1).reshape(B, H * W, 4)
    
    # Meshgrid grid cell centers
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid_x = grid_x.reshape(1, H * W)
    grid_y = grid_y.reshape(1, H * W)
    
    # Sigmoid of offset predictions tx, ty
    tx = torch.sigmoid(reg[..., 0])
    ty = torch.sigmoid(reg[..., 1])
    tw = reg[..., 2]
    th = reg[..., 3]
    
    # Decode centers
    cx = (grid_x + tx) * stride
    cy = (grid_y + ty) * stride
    
    # Decode width and height (exp is scale factor times stride)
    w = torch.exp(tw) * stride
    h = torch.exp(th) * stride
    
    # Map to Pascal VOC [xmin, ymin, xmax, ymax]
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    
    bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    
    return bboxes, obj_scores, cls_probs
