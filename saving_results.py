import cv2
import numpy as np
import os
import csv
from skimage.metrics import peak_signal_noise_ratio

folder_path = r"C:\Users\Riya\Desktop\sem 6 study material\CVPR\CVPR lab\case study\raw-890"

# List to hold results
results = []

# going thru files in a loop 
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        file_path = os.path.join(folder_path, filename)

        img = cv2.imread(file_path)      # processing images
        if img is None:
            print(f"Failed to read {filename}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        noise = cv2.randn(np.zeros_like(img), (0), (30)) # gaussian noise
        noisy = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

        # Apply filters
        gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
        median = cv2.medianBlur(noisy, 5)
        bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)

        # evaluate PSNR
        psnr_g = peak_signal_noise_ratio(img, gaussian)
        psnr_m = peak_signal_noise_ratio(img, median)
        psnr_b = peak_signal_noise_ratio(img, bilateral)

        results.append([filename, psnr_g, psnr_m, psnr_b])


output_csv = os.path.join(folder_path, "filter_results.csv")
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "PSNR_Gaussian", "PSNR_Median", "PSNR_Bilateral"])
    writer.writerows(results)

print("✅ Results saved to", output_csv)
