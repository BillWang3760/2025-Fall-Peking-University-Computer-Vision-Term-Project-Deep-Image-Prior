import cv2
import numpy as np
from scipy.ndimage import sobel
from skimage.filters import sobel_h, sobel_v
from skimage.metrics import normalized_mutual_information

import os
import sys
import torch
import torch.optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from models.skip import skip
from utils.inpainting_utils import *

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

def to_gray(img):
    if img.ndim == 3:
        return 0.299 * img[...,0] + 0.587 * img[...,1] + 0.114 * img[...,2]
    return img.astype(np.float32)

def gradient(img):
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    return gx, gy

def gradient_mag_dir(img):
    gx, gy = gradient(img)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)
    return mag, ang
def masked_mean(metric_map, mask, eps=1e-8):
    """
    metric_map: [H, W]
    mask:       [H, W], 0/1 或 [0,1]
    """
    #把metric_map变为灰度图
    if metric_map.ndim == 3:
        metric_map = to_gray(metric_map)

    return np.sum(metric_map * mask) / (np.sum(mask) + eps)


def gradient_direction_consistency(out, ref, mask, eps=1e-6):
    gx_o, gy_o = np.gradient(out)
    gx_r, gy_r = np.gradient(ref)

    dot = gx_o * gx_r + gy_o * gy_r
    mag_o = np.sqrt(gx_o**2 + gy_o**2 + eps)
    mag_r = np.sqrt(gx_r**2 + gy_r**2 + eps)

    cos_sim = dot / (mag_o * mag_r + eps)
    return masked_mean(cos_sim, mask)
def boundary_smoothness(img, mask):
    gx, gy = gradient(img)
    grad_mag = np.sqrt(gx**2 + gy**2)

    grad_diff = np.abs(
        grad_mag -
        cv2.GaussianBlur(grad_mag, (7,7), 0)
    )

    return masked_mean(1.0 / (grad_diff + 1e-6), mask)
from skimage.metrics import structural_similarity as ssim

def masked_ssim(img, ref, mask, win_size=7):
    img = to_gray(img).astype(np.float32)
    ref = to_gray(ref).astype(np.float32)

    H, W = img.shape
    win = min(win_size, H, W)

    # win_size 必须是奇数且 >=3
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return np.nan  # 或者 return 0，看你论文习惯

    _, ssim_map = ssim(
        img,
        ref,
        win_size=win,
        data_range=1.0,
        full=True
    )

    return masked_mean(ssim_map, mask)

def vifp_mscale(ref, dist, sigma_nsq=2.0):
    """
    Multi-scale Pixel-domain VIF
    ref, dist: grayscale, float32, range [0,1]
    """
    ref = ref.astype(np.float32)
    dist = dist.astype(np.float32)

    eps = 1e-10
    num = 0.0
    den = 0.0

    for scale in range(4):
        if scale > 0:
            ref = cv2.pyrDown(ref)
            dist = cv2.pyrDown(dist)

        mu1 = cv2.GaussianBlur(ref, (5,5), 1.0)
        mu2 = cv2.GaussianBlur(dist, (5,5), 1.0)

        sigma1_sq = cv2.GaussianBlur(ref*ref, (5,5), 1.0) - mu1*mu1
        sigma2_sq = cv2.GaussianBlur(dist*dist, (5,5), 1.0) - mu2*mu2
        sigma12 = cv2.GaussianBlur(ref*dist, (5,5), 1.0) - mu1*mu2

        sigma1_sq = np.maximum(sigma1_sq, eps)
        sigma2_sq = np.maximum(sigma2_sq, eps)

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        sv_sq = np.maximum(sv_sq, eps)

        num += np.sum(np.log1p((g*g * sigma1_sq) / (sv_sq + sigma_nsq)))
        den += np.sum(np.log1p(sigma1_sq / sigma_nsq))

    return num / den
def VIF_Fusion(fused, src1, src2):
    f = to_gray(fused)
    a = to_gray(src1)
    b = to_gray(src2)

    vif_fa = vifp_mscale(a, f)
    vif_fb = vifp_mscale(b, f)

    return vif_fa + vif_fb

def GFF(fused, src1, src2, mask):
    f = to_gray(fused)
    a = to_gray(src1)
    b = to_gray(src2)

    def grad(img):
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(gy, gx)
        return mag, ang

    Gf, Af = grad(f)
    Ga, Aa = grad(a)
    Gb, Ab = grad(b)

    Gs = np.maximum(Ga, Gb)
    As = np.where(Ga >= Gb, Aa, Ab)

    Cg = (2 * Gf * Gs) / (Gf**2 + Gs**2 + 1e-6)
    Ca = (1 + np.cos(Af - As)) / 2

    Q = Cg * Ca
    return masked_mean(Q, mask)

def Q_ABF(fused, A, B,transition_mask):
    f = to_gray(fused)
    A = to_gray(A)
    B = to_gray(B)

    def edge_info(img):
        gx = sobel_h(img)
        gy = sobel_v(img)
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(gy, gx)
        return mag, ang

    Gf, Af = edge_info(f)
    Ga, Aa = edge_info(A)
    Gb, Ab = edge_info(B)

    Qa = (2 * Gf * Ga) / (Gf**2 + Ga**2 + 1e-8)
    Qb = (2 * Gf * Gb) / (Gf**2 + Gb**2 + 1e-8)

    Pa = (1 + np.cos(Af - Aa)) / 2
    Pb = (1 + np.cos(Af - Ab)) / 2

    Qaf = Qa * Pa
    Qbf = Qb * Pb

    Q_abf=np.maximum(Qaf, Qbf)
    return masked_mean(Q_abf, transition_mask)

def gradient_fusion_performance(fused, src, tgt):
    fused = to_gray(fused).astype(np.float32)
    src = to_gray(src).astype(np.float32)
    tgt = to_gray(tgt).astype(np.float32)

    def grad_mag(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)

    g_f = grad_mag(fused)
    g_s = grad_mag(src)
    g_t = grad_mag(tgt)

    # 融合梯度是否覆盖输入梯度
    g_max = np.maximum(g_s, g_t)
    eps = 1e-6

    Q = np.mean(g_f / (g_max + eps))
    return Q
def evaluate_all(fused, src_fg, src_bg,gap):
    return {
        "GFF": float(GFF(fused, src_fg, src_bg,gap)),
        "Q_ABF": float(Q_ABF(fused, src_fg, src_bg,gap)),
        "Gradient_Fusion_Performance": float(gradient_fusion_performance(fused, src_fg, src_bg)),
        "BSI(T)": float(boundary_smoothness(fused, gap)),
        "SSIM(T)": float(masked_ssim(fused, src_bg, gap)),
    }
fused = cv2.imread("5.png")[..., ::-1] / 255.0
alpha = cv2.imread("result_alpha.png")[..., ::-1] / 255.0
poisson = cv2.imread("result_poisson.png")[..., ::-1] / 255.0
laplacian=cv2.imread("result_laplacian.png")[..., ::-1] / 255.0
fg = cv2.imread("newdata/kid2.jpg")[..., ::-1] / 255.0
bg = cv2.imread("newdata/lake.jpg")[..., ::-1] / 255.0
mask_gap=cv2.imread("newdata/gap3.png", 0)  # 灰度 mask
for name, img in {
    "Alpha": alpha,
    "Poisson": poisson,
    "Ours": fused,
    "Laplacian": laplacian
}.items():
    scores = evaluate_all(img, fg, bg,mask_gap)
    print(name, scores)
#1,2,3的SSIM高
#5的GFF，BSI，SSIM高