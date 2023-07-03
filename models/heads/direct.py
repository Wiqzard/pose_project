
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseRegression(nn.Module):
    def __init__(self, backbone, emb_dim=768):
        super(PoseRegression, self).__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool1d(9) # 16 / 25
        #self.fc1 = nn.LazyLinear(1024)
        self.fc1 = nn.Linear(9*emb_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        # self.fc3 = nn.Linear(1024, 512)
        # self.fc4 = nn.Linear(512, 256)
        self.fc_r = nn.Linear(256, 6)
        self.fc_t = nn.Linear(256, 3)

    def forward(self, input_data):
        #assert "roi_img" in input_data

        #x = self.backbone.forward_features(input_data["roi_img"])
        x = self.backbone.forward_features(input_data[0]["roi_img"])#, return_intermediates=True)[1][-1]
                                                                 #["out"]
        #if x.ndim > 2:
        x = self.pool(x.permute(0,2,1)).flatten(1)    
        #x = self.pool(x.permute(0,3,1,2).flatten(2)    ).flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        r = self.fc_r(x)
        t = self.fc_t(x)
        return r, t
