# Compare the high-resolution images with the original images
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import numpy as np
def eval_resolution(high_res_images, images):
    for i, (high_res_img, img) in enumerate(zip(high_res_images, images)):
        high_res_img = high_res_img.reshape(img.height, img.width, 3).astype(np.uint8)
        img = np.array(img)

        # Calculate the SSIM and PSNR
        ssim_value = ssim(high_res_img, img, win_size=3, multichannel=True)
        psnr_value = psnr(high_res_img, img)
        print(f'Image {i+1} - SSIM: {ssim_value}, PSNR: {psnr_value}')