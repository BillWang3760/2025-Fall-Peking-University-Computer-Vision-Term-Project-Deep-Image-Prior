import numpy as np
from PIL import Image
import cv2


def generate_mask_from_target(
    img_path,
    save_path='mask.jpg',
    white_thresh=30,
    morph_kernel=5
):
    """
    根据目标图自动生成 mask：
    - 主体区域：255
    - 背景（近白）：0

    参数:
    img_path: 目标图路径 (512x512)
    white_thresh: 白色判定阈值（越小越严格）
    morph_kernel: 形态学核大小（0 表示不做）
    """

    # 1. 读取图像
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img).astype(np.float32)

    # 2. 计算与白色的距离
    white = np.array([255, 255, 255], dtype=np.float32)
    dist = np.linalg.norm(img_np - white, axis=2)

    # 3. 生成 mask
    mask = np.zeros((672, 672), dtype=np.uint8)
    mask[dist >= white_thresh] = 255

    # 4. 形态学优化（可选，但强烈推荐）
    if morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 5. 保存
    Image.fromarray(mask).save(save_path)
    print(f"Mask saved to {save_path}")

    return mask
generate_mask_from_target(
    img_path='newdata/kid2.jpg',
    save_path='newdata/mask11.jpg',
    white_thresh=40,
    morph_kernel=7
)
