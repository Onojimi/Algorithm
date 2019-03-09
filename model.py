import torch
import torch.nn as nn
import torch.nn.functional as F

class textCNN(nn.Module):
    def __init__(self,args):
        super(textCNN,self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V,D)
        '''
        self.conv13 = nn.Conv2d(Ci,Co,(3,D))
        self.conv14 = nn.Conv2d(Ci,Co,(4,D))
        self.conv15 = nn.Conv2d(Ci,Co,(5,D))
        '''
        self.convs = nn.ModuleList([nn.Conv2d(Ci,Co,(K,D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        
    def conv_and_pool(self,x,conv):
        x = F.relu(conv(x).squeeze(3))
        x = F.max_pool1d(x,x.size(2).squeeze(2))
        return x

    def forward(self,x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        '''
        x1 = self.conv_and_pool(x,self.conv13)
        x2 = self.conv_and_pool(x,self.conv14)
        x3 = self.conv_and_pool(x,self.conv15)
        x = torch.cat((x1,x2,x3),1)
        '''
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.maxpool_1d(line,line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x,1)
        x = self.dropout(x)
        return x
