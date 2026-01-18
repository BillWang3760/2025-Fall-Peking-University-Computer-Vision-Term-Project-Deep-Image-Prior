import os
import sys
import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from models.skip import skip
from utils.inpainting_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor

print("Using device:", device)


def load_and_crop(path, size, crop_div=32):
    img_pil, _ = get_image(path, size)

    img_pil = crop_image(img_pil, crop_div)

    img_np = pil_to_np(img_pil)
    return img_np


def create_trimap_masks(mask_tensor, gap_pixels=15):
    kernel_size = gap_pixels * 2 + 1

    dilated = F.max_pool2d(mask_tensor, kernel_size, stride=1, padding=gap_pixels)

    eroded = -F.max_pool2d(-mask_tensor, kernel_size, stride=1, padding=gap_pixels)

    mask_strict_fg = eroded

    mask_strict_bg = 1 - dilated

    mask_transition = dilated - eroded

    return mask_strict_fg, mask_strict_bg, mask_transition


def get_trimap(mask, width=15):
    """
    生成三层掩码：内部、外部、以及中间的“空白带”
    """
    kernel = torch.ones(1, 1, width, width).to(mask.device)

    # 向外扩大，得到“大圈”
    dilated = (F.conv2d(mask, kernel, padding=width // 2) > 0).float()

    # 向内缩小，得到“小圈”

    eroded = 1 - (F.conv2d(1 - mask, kernel, padding=width // 2) > 0).float()

    # 严格背景：大圈之外
    mask_bg = 1 - dilated
    # 严格前景：小圈之内
    mask_fg = eroded
    # 过渡带（Gap）：大圈减去小圈
    mask_gap = dilated - eroded

    return mask_fg, mask_bg, mask_gap


imsize = -1
dim_div_by = 32

src_path = 'newdata/kid2.jpg'
tgt_path = 'newdata/lake.jpg'
mask_path = 'newdata/mask11.jpg'

src_np = load_and_crop(src_path, imsize, dim_div_by)
tgt_np = load_and_crop(tgt_path, imsize, dim_div_by)
mask_np = load_and_crop(mask_path, imsize, dim_div_by)

if mask_np.shape[0] == 3:
    mask_np = mask_np[0:1, :, :]
mask_np = (mask_np > 0.5).astype(np.float32)

src_var = np_to_torch(src_np).type(dtype)
tgt_var = np_to_torch(tgt_np).type(dtype)
mask_var = np_to_torch(mask_np).type(dtype)
# 稍微模糊一下 Mask，消除锯齿
mask_var = F.avg_pool2d(mask_var, kernel_size=3, stride=1, padding=1)

GAP_PIXELS = 14
mask_strict_fg, mask_strict_bg, mask_gap = create_trimap_masks(mask_var, GAP_PIXELS)
mask_fg, mask_bg, mask_gap = get_trimap(mask_var, width=15)
gap = torch_to_np(mask_gap)

# 可视化一下三个区域
print("Mask Shapes Preview (FG / Gap / BG):")
plot_image_grid([
    torch_to_np(mask_strict_fg),
    torch_to_np(mask_gap),
    torch_to_np(mask_strict_bg)
], nrow=3, factor=4)


def gradient(img):
    """
        使用 Sobel 算子计算梯度（比简单差分稳定）
        img: [B, C, H, W]
        """
    sobel_x = torch.tensor(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)

    sobel_y = sobel_x.transpose(2, 3)

    # 对 RGB 每个通道独立做
    gx = F.conv2d(img, sobel_x.repeat(img.shape[1], 1, 1, 1),
                  padding=1, groups=img.shape[1])
    gy = F.conv2d(img, sobel_y.repeat(img.shape[1], 1, 1, 1),
                  padding=1, groups=img.shape[1])
    return gx, gy
def structure_tensor_orientation(img, smooth=7):
    """
    返回切向方向单位向量 (tx, ty)
    """
    gx, gy = gradient(img)

    # 平滑梯度乘积
    Jxx = F.avg_pool2d(gx * gx, smooth, 1, smooth // 2)
    Jyy = F.avg_pool2d(gy * gy, smooth, 1, smooth // 2)
    Jxy = F.avg_pool2d(gx * gy, smooth, 1, smooth // 2)

    # 主方向角（法向）
    theta = 0.5 * torch.atan2(2 * Jxy, Jxx - Jyy)

    # 切向方向 = 法向 + 90°
    tx = -torch.sin(theta)
    ty = torch.cos(theta)

    return tx, ty
def directional_gradient(img, tx, ty):
    gx, gy = gradient(img)
    return gx * tx + gy * ty
def anisotropic_texture_loss(out, tgt, alpha, tx, ty):
    """
    沿结构张量切向方向传播纹理
    """
    d_out = directional_gradient(out, tx, ty)
    d_tgt = directional_gradient(tgt, tx, ty)

    w = 4 * alpha * (1 - alpha)
    return torch.mean(torch.abs(d_out - d_tgt) * w)

def gradient_direction_loss(out, src, alpha, eps=1e-6):
    """
    梯度方向（法向）一致性损失
    """
    gx_o, gy_o = gradient(out)
    gx_s, gy_s = gradient(src)

    # 梯度向量
    dot = gx_o * gx_s + gy_o * gy_s
    mag_o = torch.sqrt(gx_o ** 2 + gy_o ** 2 + eps)
    mag_s = torch.sqrt(gx_s ** 2 + gy_s ** 2 + eps)

    cos_sim = dot / (mag_o * mag_s + eps)

    # 1 - cos(theta)
    loss_dir = (1.0 - cos_sim) * alpha

    return torch.mean(loss_dir)

def low_freq_loss(out, tgt, alpha, kernel=31):
    """
    背景低频一致性损失
    """
    pad = kernel // 2
    out_lp = F.avg_pool2d(out, kernel, stride=1, padding=pad)
    tgt_lp = F.avg_pool2d(tgt, kernel, stride=1, padding=pad)

    bg_weight = (1 - alpha)
    return torch.mean(((out_lp - tgt_lp) * bg_weight) ** 2)

def illumination(img, kernel=61):
    """
    提取光照分量（大尺度低频）
    """
    pad = kernel // 2
    # 转灰度，更符合光照定义
    gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    illum = F.avg_pool2d(gray, kernel, stride=1, padding=pad)
    return illum

input_depth = 32
pad = 'reflection'

net = skip(input_depth, src_np.shape[0],
           num_channels_down=[128] * 5,
           num_channels_up=[128] * 5,
           num_channels_skip=[128] * 5,
           filter_size_up=3, filter_size_down=3,
           upsample_mode='nearest', filter_skip_size=1,
           need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

net_input = get_noise(input_depth, 'noise', src_np.shape[1:]).type(dtype)
optimizer = torch.optim.Adam(net.parameters(), lr=0.006)

lambda_struct = 10.0
lambda_bg = 50.0
lambda_tv = 0.2
num_iter = 1001
show_every = 50

print("Starting optimization with Transition Band...")


# --- 1. 生成平滑过渡权重图 ---
def get_soft_weights(mask, blur_radius=21):
    """
    生成一个从中心向外平滑扩散的权重图
    blur_radius 越大，过渡区域越宽，纹理融合越自然
    """
    # 确保是单通道
    mask = mask[:, :1, :, :]

    # 使用高斯模糊平滑 Mask 边缘
    # sigma 越大，边缘越“肉”，融合带越长
    padding = blur_radius // 2
    soft_mask = F.avg_pool2d(mask, kernel_size=blur_radius, stride=1, padding=padding)
    # 多跑几次平滑，让权重曲线更接近 Sigmoid
    for _ in range(3):
        soft_mask = F.avg_pool2d(soft_mask, kernel_size=blur_radius, stride=1, padding=padding)

    return soft_mask

def multiscale_gradient_loss(out, src, alpha, scales=(1, 2, 4)):
    loss = 0.0
    for s in scales:
        if s > 1:
            out_s = F.avg_pool2d(out, s)
            src_s = F.avg_pool2d(src, s)
            alpha_s = F.avg_pool2d(alpha, s)
        else:
            out_s, src_s, alpha_s = out, src, alpha

        gx_o, gy_o = gradient(out_s)
        gx_s, gy_s = gradient(src_s)

        loss += torch.mean(torch.abs(gx_o - gx_s) * alpha_s)
        loss += torch.mean(torch.abs(gy_o - gy_s) * alpha_s)

    return loss / len(scales)

def alpha_annealing(alpha, iteration, max_iter):
    """
    前期过渡区宽，后期边界更硬
    """
    gamma = 1.0 + 4.0 * (1 - iteration / max_iter)
    return torch.clamp(alpha ** gamma, 0.0, 1.0)

def illumination_loss(out, tgt, alpha, kernel=61):
    """
    背景光照一致性约束
    """
    illum_out = illumination(out, kernel)
    illum_tgt = illumination(tgt, kernel)

    bg_weight = (1 - alpha)
    return torch.mean(((illum_out - illum_tgt) * bg_weight) ** 2)
def gradient_blend_consistency_loss(out, src, tgt, alpha):
    """
    强制融合区梯度连续：
    grad(out) ≈ alpha * grad(src) + (1-alpha) * grad(tgt)
    """
    gx_o, gy_o = gradient(out)
    gx_s, gy_s = gradient(src)
    gx_t, gy_t = gradient(tgt)

    gx_target = alpha * gx_s + (1 - alpha) * gx_t
    gy_target = alpha * gy_s + (1 - alpha) * gy_t

    # 只在过渡带起作用
    blend_weight = 4 * alpha * (1 - alpha)

    loss = torch.mean(torch.abs(gx_o - gx_target) * blend_weight) + \
           torch.mean(torch.abs(gy_o - gy_target) * blend_weight)

    return loss
def tangential_texture_loss(out, tgt, alpha, direction="horizontal"):
    """
    在过渡区域，强制纹理沿切向连续（模拟水面延展）
    """
    gx_o, gy_o = gradient(out)
    gx_t, gy_t = gradient(tgt)

    # 过渡区权重
    w = 4 * alpha * (1 - alpha)

    if direction == "horizontal":
        # 水面：主要约束 x 方向梯度
        loss = torch.mean(torch.abs(gx_o - gx_t) * w)
    elif direction == "vertical":
        loss = torch.mean(torch.abs(gy_o - gy_t) * w)
    else:
        loss = 0.0

    return loss

def blended_loss(out, src, tgt, alpha, iteration, max_iter):
    alpha_t = alpha_annealing(alpha, iteration, max_iter)

    # 1. 背景结构 & 光照
    loss_pixel = torch.mean(((out - tgt) * (1 - alpha_t)) ** 2)
    loss_lf = low_freq_loss(out, tgt, alpha_t)
    loss_illum = illumination_loss(out, tgt, alpha_t)

    # 2. 前景结构
    loss_grad_fg = multiscale_gradient_loss(out, src, alpha_t)
    loss_dir = gradient_direction_loss(out, src, alpha_t)

    # 3. 法向连续（你已有的）
    loss_grad_blend = gradient_blend_consistency_loss(out, src, tgt, alpha_t)

    tx, ty = structure_tensor_orientation(tgt_var)

    loss_aniso = anisotropic_texture_loss(
        out, tgt_var, alpha_t, tx, ty
    )

    return (
        50.0 * loss_pixel +
        25.0 * loss_lf +
        20.0 * loss_illum +
        6.0  * loss_grad_fg +
        4.0  * loss_dir +
        12.0 * loss_grad_blend +
        18.0 * loss_aniso
    )




# --- 3. 初始化数据 ---
# 在主流程中：
soft_alpha = get_soft_weights(mask_var, blur_radius=51)  # 增加模糊半径以扩大过渡区

# 可视化权重图预览
print("Soft Transition Alpha Map (White=FG, Black=BG):")
plot_image_grid([torch_to_np(soft_alpha)], factor=4)


# --- 4. 修改 Closure 函数 ---
def closure():
    optimizer.zero_grad()

    # DIP 常用技巧：给输入加噪
    net_input_perturbed = net_input + (torch.randn_like(net_input) * 0.02)
    out = net(net_input_perturbed)

    # 使用平滑混合 Loss
    total_loss = blended_loss(out, src_var, tgt_var, soft_alpha,iteration=i,max_iter=num_iter)

    total_loss.backward()

    if i % show_every == 0:
        # 这里为了观察，计算一下分项
        with torch.no_grad():
            l_px = torch.mean(((out - tgt_var) * (1 - soft_alpha)) ** 2)
            print(f"Iter {i} | Total Loss: {total_loss.item():.6f} | Pixel: {l_px.item():.6f}")

    return total_loss, out


# 训练循环
for i in range(num_iter):
    total_loss, out_img = closure()
    optimizer.step()

    if i % show_every == 0:
        print(f"Iter {i} | Total: {total_loss.item():.4f} ")

        # 实时显示
        out_np_cur = torch_to_np(out_img)
        plot_image_grid([out_np_cur], factor=5, nrow=1)
        plt.imsave("final3.png", out_np_cur.transpose(1, 2, 0))

# 保存最终结果
final_out = torch_to_np(net(net_input))
print("Finished. Saved to final.png")