
import cv2
import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim

def read_image(file_storage):
    img = Image.open(file_storage.stream).convert('RGB')
    return np.array(img)

def get_hash_similarity(img1, img2):
    hash1 = imagehash.phash(Image.fromarray(img1))
    hash2 = imagehash.phash(Image.fromarray(img2))
    return 100 - (hash1 - hash2) / len(hash1.hash) ** 2 * 100  # porcentagem

def get_structural_similarity(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sim, _ = ssim(img1_gray, img2_gray, full=True)
    return sim * 100

def detect_editing_artifacts(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 20:
        return True, 80  # Probabilidade de edição
    return False, 10
