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
def cal_fre_loss(amp, pha, ir, vi):
    real = amp * torch.cos(pha) + 1e-8
    imag = amp * torch.sin(pha) + 1e-8
    x = torch.complex(real, imag)
    x = torch.abs(torch.fft.irfftn(x, dim=(-2, -1)))
    loss_ir = cc(x, ir)
    loss_vi = cc(x, vi)
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

if __name__ == '__main__':
    ir = torch.randn(1, 1, 128, 128)
    vi = torch.randn(1, 1, 128, 128)
    fus = freFuse()
    # fus,amp,pha = fus(ir, vi)
    frefus = fus(ir, vi)


