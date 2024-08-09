#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, Dim, H_Dim1, H_Dim2):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(Dim*2, H_Dim1)
        self.layer2 = nn.Linear(H_Dim1, H_Dim2)
        self.layer3 = nn.Linear(H_Dim2, Dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)

    def forward(self, new_x, h):
        inputs = torch.cat((new_x, h), dim=1)
        D_h1 = F.relu(self.layer1(inputs))
        D_h2 = F.relu(self.layer2(D_h1))
        D_logit = self.layer3(D_h2)
        D_prob = torch.sigmoid(D_logit)
        return D_prob
    
class Generator(nn.Module):
    def __init__(self, Dim, H_Dim1, H_Dim2):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(Dim*2, H_Dim1)
        self.layer2 = nn.Linear(H_Dim1, H_Dim2)
        self.layer3 = nn.Linear(H_Dim2, Dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)

    def forward(self, new_x, m):
        inputs = torch.cat((new_x, m), dim=1)
        G_h1 = F.relu(self.layer1(inputs))
        G_h2 = F.relu(self.layer2(G_h1))
        G_prob = torch.sigmoid(self.layer3(G_h2))
        return G_prob

# %%
