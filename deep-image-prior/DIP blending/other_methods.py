import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
fg = cv2.imread("newdata/kid2.jpg")     # 前景
bg = cv2.imread("newdata/lake.jpg")
mask = cv2.imread("newdata/mask11.jpg", 0)  # 灰度 mask

# 将 mask 转为 0/1
_, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# 调整尺寸一致（简单 resize）
h, w = bg.shape[:2]
fg = cv2.resize(fg, (w, h))
mask_bin = cv2.resize(mask_bin, (w, h))

# 可视化
plt.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
plt.title("Background"); plt.axis("off")

plt.figure()
plt.imshow(cv2.cvtColor(fg, cv2.COLOR_BGR2RGB))
plt.title("Foreground"); plt.axis("off")

plt.figure()
plt.imshow(mask_bin, cmap='gray')
plt.title("Mask"); plt.axis("off")

plt.show()
def alpha_blending(fg, bg, mask_bin):
    # 归一化 mask
    alpha = mask_bin.astype(np.float32) / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    # 融合
    blended = fg * alpha + bg * (1 - alpha)
    blended = blended.astype(np.uint8)
    return blended

result_alpha = alpha_blending(fg, bg, mask_bin)
cv2.imwrite("result_alpha.png", result_alpha)

plt.imshow(cv2.cvtColor(result_alpha, cv2.COLOR_BGR2RGB))
plt.title("Alpha Blending")
plt.axis("off")
plt.show()
def poisson_blending(fg, bg, mask_bin):
    #寻找mask的中心点
    ys, xs = np.where(mask_bin == 255)
    center_y = int((ys.min() + ys.max()) / 2)
    center_x = int((xs.min() + xs.max()) / 2)
    center = (center_x, center_y)
    blended = cv2.seamlessClone(fg, bg, mask_bin, center, cv2.NORMAL_CLONE)
    return blended
result_poisson = poisson_blending(fg, bg, mask_bin)
cv2.imwrite("result_poisson.png", result_poisson)
plt.imshow(cv2.cvtColor(result_poisson, cv2.COLOR_BGR2RGB))
plt.title("Poisson Blending")
plt.axis("off")
plt.show()


# #
#
# def laplacian_pyramid_blending(fg, bg, mask, levels=4):
#     fg = fg.astype(np.float32)
#     bg = bg.astype(np.float32)
#     mask = mask.astype(np.float32) / 255.0  # 归一化到 [0, 1]
#     # ---------- Gaussian pyramid ----------
#     def gaussian_pyramid(img, levels):
#         gp = [img]
#         for _ in range(levels):
#             img = cv2.pyrDown(img)
#             gp.append(img)
#         return gp
#
#     # ---------- Laplacian pyramid ----------
#     def laplacian_pyramid(gp):
#         lp = []
#         for i in range(len(gp) - 1):
#             size = (gp[i].shape[1], gp[i].shape[0])
#             up = cv2.pyrUp(gp[i + 1], dstsize=size)
#             lp.append(gp[i] - up)
#         lp.append(gp[-1])
#         return lp
#
#     # image pyramids
#     gp_fg = gaussian_pyramid(fg, levels)
#     gp_bg = gaussian_pyramid(bg, levels)
#     lp_fg = laplacian_pyramid(gp_fg)
#     lp_bg = laplacian_pyramid(gp_bg)
#
#     # ---------- mask pyramid ----------
#     gp_mask = [mask]
#     for i in range(1, levels + 1):
#         m = cv2.pyrDown(gp_mask[-1])
#
#         m = np.clip(m * 1.2, 0, 1)
#         gp_mask.append(m)
#
#     # ---------- blend ----------
#     lp_blended = []
#     for lf, lb, gm in zip(lp_fg, lp_bg, gp_mask):
#         gm = cv2.merge([gm, gm, gm])
#         lp_blended.append(lf * gm + lb * (1 - gm))
#
#     # ---------- reconstruct ----------
#     blended = lp_blended[-1]
#     for i in range(levels - 1, -1, -1):
#         size = (lp_blended[i].shape[1], lp_blended[i].shape[0])
#         blended = cv2.pyrUp(blended, dstsize=size)
#         blended = blended + lp_blended[i]
#
#     return np.clip(blended, 0, 255).astype(np.uint8)
# result_laplacian = laplacian_pyramid_blending(fg, bg, mask_bin, levels=4)
# cv2.imwrite("result_laplacian.png", result_laplacian)
# plt.imshow(cv2.cvtColor(result_laplacian, cv2.COLOR_BGR2RGB))
# plt.title("Laplacian Pyramid Blending")
# plt.axis("off")
# plt.show()