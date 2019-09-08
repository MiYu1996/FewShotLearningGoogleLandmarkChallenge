import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
# from qpth.qp import QPFunction


def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)


def kronecker(A, B):
    """
    Kronecker product between matrices A and B
    """
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0),  A.size(1) * B.size(1))


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


class SiameseNetworks(nn.Module):
    def __init__(self, base_CNN, in_shape, out_shape):
        super().__init__()

        self.base_CNN = base_CNN
        self.fc1 = nn.Linear(in_shape, out_shape)
        self.fc2 = nn.Linear(out_shape, 1)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def before_dist(self, x):
        out1 = flatten(self.base_CNN(x))
        out2 = F.relu(self.fc1(out1))
        return out2

    def forward(self, x, y):
        out1 = self.before_dist(x)
        out2 = self.before_dist(y)
        p = F.sigmoid(self.fc2(torch.abs(out1 - out2)))

        return p


class PrototypicalNetworks(nn.Module):
    """
    Notes:
    - base_net needs to have flattened output
    - dist_func should be a generic distance function like PairwiseDistance in PyTorch
    """
    def __init__(self, base_net, dist_func):
        super().__init__()

        self.base_net = base_net
        self.dist_func = dist_func

    def forward(self, S_k, Q_k):
        N_c, N_S, C, W, H = S_k.shape
        N_c, N_Q, C, W, H = Q_k.shape
        S_k = S_k.view(-1, C, W, H)
        Q_k = Q_k.view(-1, C, W, H)
        c_k = torch.mean(self.base_net(S_k).view(N_c, N_S, -1), dim=1)
        f_x = self.base_net(Q_k)
        c_k = c_k.repeat(N_c * N_Q, 1)
        f_x = torch.repeat_interleave(f_x, N_c, dim=0)
        logits = -self.dist_func(f_x, c_k).view(-1, N_c)

        return logits


class MetaSVMNetworks(nn.Module):
    """
    Notes:
    - base_net needs to have flattened output
    - C is the regularization parameter for soft SVM
    """
    def __init__(self, base_net, C=50):
        super().__init__()

        self.base_net = base_net
        self.C = C
        self.gamma = torch.randn(1).cuda()
        self.gamma.requires_grad = True

    def forward(self, S_k, Q_k):
        N_c, N_S, C, W, H = S_k.shape
        N_c, N_Q, C, W, H = Q_k.shape

        S_k = S_k.view(-1, C, W, H)
        Q_k = Q_k.view(-1, C, W, H)
        f_x = self.base_net(Q_k)
        K = self.base_net(S_k)

        # G = torch.eye(N_c * N_S * N_c).cuda()
        # p = -1.0 * kronecker(torch.eye(N_c).cuda(), torch.ones(1, N_S).cuda()).view(-1)
        # h = -self.C * p
        # b = torch.zeros(N_c * N_S).cuda()
        # A = torch.eye(N_c * N_S).repeat(1, N_c).cuda()
        # aul = K.mm(torch.transpose(K, 0, 1))
        # Q = kronecker(torch.eye(N_c).cuda(), aul)
        # alpha = QPFunction(verbose=True, maxIter=30)(Q, p.detach(), G.detach(), h.detach(), A.detach(), b.detach())

        aul = K.mm(torch.transpose(K, 0, 1)) + self.C * torch.eye(N_c * N_S).cuda()
        Q = kronecker(torch.eye(N_c).cuda(), aul)
        p = 1.0 * kronecker(torch.eye(N_c).cuda(), torch.ones(1, N_S).cuda()).view(-1, 1)
        alpha = torch.solve(p.detach(), Q)[0]

        alpha = alpha.view(N_c, -1)
        w_k = alpha.mm(K)
        logits = -self.gamma * (f_x.mm(torch.transpose(w_k, 0, 1)))

        return logits

class ResBlock(nn.Module):
    def __init__(self, in_channel, D_filters, drop_prob=0.1, use_block=False, block_size=5):
        super().__init__()

        self.D = D_filters

        self.conv1 = nn.Conv2d(in_channel, self.D, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.D)
        self.conv2 = nn.Conv2d(self.D, self.D, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.D)
        self.conv3 = nn.Conv2d(self.D, self.D, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.D)
        self.jump_conv = nn.Conv2d(in_channel, self.D, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        if use_block:
            self.drop = DropBlock2D(block_size=block_size, drop_prob=drop_prob)
        else:
            self.drop = nn.Dropout(p=drop_prob)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.jump_conv.weight)

    def forward(self, x):
        resi = x

        out = self.bn1(self.conv1(x))
        out = nn.LeakyReLU(0.1)(out)

        out = self.bn2(self.conv2(out))
        out = nn.LeakyReLU(0.1)(out)

        out = self.bn3(self.conv3(out))
        out = nn.LeakyReLU(0.1)(out)

        out += self.jump_conv(resi)
        out = self.maxpool(out)
        out = self.drop(out)

        return out


class ResNet12(nn.Module):
    """
    Base net used in MetaSVM
    """
    def __init__(self, resi_filters=(64, 96, 128, 256)):
        super().__init__()

        D1, D2, D3, D4 = resi_filters
        self.block1 = ResBlock(3, D1)
        self.block2 = ResBlock(D1, D2)
        self.block3 = ResBlock(D2, D3, use_block=True)
        self.block4 = ResBlock(D3, D4, use_block=True)
        self.final_conv1 = nn.Conv2d(D4, 1024, 1)
        self.final_conv2 = nn.Conv2d(1024, 384, 1)
        self.mean_pool = nn.AvgPool2d(6)
        self.drop = nn.Dropout(p=0.1)

        nn.init.kaiming_normal_(self.final_conv1.weight)
        nn.init.kaiming_normal_(self.final_conv2.weight)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.final_conv1(out)
        out = self.mean_pool(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.final_conv2(out)
        out = flatten(out)

        return out

if __name__ == '__main__':
    a = torch.randn(2, 3, 3, 84, 84).cuda()
    b = torch.randn(2, 2, 3, 84, 84).cuda()
    base_net = ResNet12()
    base_net.to(device=torch.device('cuda'))
    meta_svm = MetaSVMNetworks(base_net)
    meta_svm.to(device=torch.device('cuda'))
    c = meta_svm(a, b)
    d = torch.sum(c)
    d.backward()
    print(c.shape)
