import torch
import torch.nn as nn


class AmpFuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)
        x = self.conv1(x)
        return x

class PhaFuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)
        x = self.conv1(x)
        return x

class IFFT(nn.Module):
    def __init__(self, out_channels=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, amp, pha):
        real = amp * torch.cos(pha) + 1e-8
        imag = amp * torch.sin(pha) + 1e-8
        x = torch.complex(real, imag)
        x = torch.abs(torch.fft.irfftn(x, dim=(-2, -1)))
        x = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x = self.conv1(x)
        return x

def fft(input):
    '''
    input: tensor of shape (batch_size, 1, height, width)
    mask: tensor of shape (height, width)
    '''
    # 执行2D FFT
    img_fft = torch.fft.rfftn(input, dim=(-2, -1))
    amp = torch.abs(img_fft)
    pha = torch.angle(img_fft)
    return amp, pha


class Fuse_block(nn.Module):
    def __init__(self, dim, channels=1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(dim, channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.down_conv = nn.Sequential(
            nn.Sequential(nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels * 4, channels * 2, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1), nn.Tanh()),
        )

    def forward(self, ir, vi, frefus):
        x = torch.cat([frefus], dim=1)  # n,c,h,w
        return x

class freFuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 8
        self.ff1 = AmpFuse()
        self.ff2 = PhaFuse()
        self.ifft = IFFT(self.channel)
        self.fus_block = Fuse_block(self.channel * 3)

    def forward(self, ir, vi): # ir, vi: 没经过处理啊
        ir_amp, ir_pha = fft(ir)
        vi_amp, vi_pha = fft(vi)# 快速傅里叶变换
        amp = self.ff1(ir_amp, vi_amp) # 不是频率啊，是幅值和相位
        pha = self.ff2(ir_pha, vi_pha)
        frefus = self.ifft(amp, pha)
        # # 这里vi和ir尺寸改一下
        # ir = torch.randn(1, 64, 128, 128)
        # vi = torch.randn(1, 64, 128, 128)
        # fus = self.fus_block(ir, vi, frefus)
        # fus = (fus - torch.min(fus)) / (torch.max(fus) - torch.min(fus))
        return frefus,amp,pha
        # return fus, fus, fus

#频域损失函数
def cal_fre_loss(amp, pha, ir, vi,mask):
    real = amp * torch.cos(pha) + 1e-8
    imag = amp * torch.sin(pha) + 1e-8
    x = torch.complex(real, imag)
    x = torch.abs(torch.fft.irfftn(x, dim=(-2, -1)))
    loss_ir = cc(x * mask, ir * mask)
    loss_vi = cc(x * (1 - mask), vi * (1 - mask))
    return loss_ir + loss_vi

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1**2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1.0, 1.0)
    return cc.mean()

import torch
from torch import nn
from einops import rearrange
class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,  attn_drop=0., proj_drop=0.):
        """
        :param dim: 输入特征的维度
        :param num_heads: 注意力头的数量，默认为 8
        :param qkv_bias: 是否在 QKV 投影中使用偏置，默认为 False
        :param attn_drop: 注意力矩阵的 dropout 概率
        :param proj_drop: 输出投影的 dropout 概率
        """
        super().__init__()  # 调用父类的初始化方法

        self.heads = num_heads  # 保存注意力头的数量

        # 定义一个 Softmax，用于计算注意力权重
        self.attend = nn.Softmax(dim=1)
        # 定义一个 Dropout，用于对注意力权重进行随机丢弃
        self.attn_drop = nn.Dropout(attn_drop)

        # 定义一个线性层，用于生成 QKV 矩阵
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        # 定义一个可学习的参数 temp，用于调整注意力计算
        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        # 定义输出投影，包括一个线性层和一个 Dropout
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),  # 线性层，用于将维度映射回原始输入维度
            nn.Dropout(proj_drop)  # Dropout，用于随机丢弃部分输出
        )

    def forward(self, x):
        # 在通道维度上分成多头
        # [batch_size, seq_length, dim] ===> [batch_size, heads, seq_length, head_dim]
        # torch.Size([1, 784, 64])      ===> torch.Size([1, 8, 784, 8])
        b, c, gao, kuan = x.shape

        # print(type(w))
        x = to_3d(x)
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads) # 第一步就错了？

        # 对 w 沿着最后一个维度进行归一化，标准化操作
        # torch.Size([1, 8, 784, 8])
        w_normed = torch.nn.functional.normalize(w, dim=-2)
        # 对归一化后的 w 进行平方
        w_sq = w_normed ** 2
        # 计算注意力权重 Pi：对 w_sq 沿着最后一个维度求和后乘以 temp，再通过 Softmax
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # 形状为 [batch_size, heads, seq_length]

        # 论文中注意力算子的相关计算步骤
        # 该算子：实现低秩投影，避免计算令牌间成对相似性，具有线性计算和内存复杂度。
        # 计算注意力得分 dots: Pi 先进行归一化，再扩展一个维度，与 w 的平方相乘。
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        # 计算注意力矩阵 attn，公式中为 1 / (1 + dots)
        attn = 1. / (1 + dots)
        # 对注意力矩阵进行 dropout 操作，防止过拟合。
        attn = self.attn_drop(attn)

        # 计算输出，公式中为 -w * Pi * attn
        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        # 将输出重新排列为 [batch_size, seq_length, dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 将输出通过输出投影层
        out = self.to_out(out)
        return to_4d(out,gao,kuan)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)




if __name__ == '__main__':
    batch_size = 1
    channel = 64
    H = 28
    W = 28
    seq_length = H * W
    input = torch.rand(1, 64, 28, 28)
    print('input_size:', input.size())
    # 1 X 64 X 28 X 28 ====> 1 X 784 X 64
    input = to_3d(input)

    # 初始化 AttentionTSSA 模型
    model = AttentionTSSA(dim=channel)
    output = model(input)
    output = to_4d(output, H, W)
    print('output_size:', output.size())

    # 计算模型的总参数量，并打印
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')

