import cv2
import numpy as np
from tqdm import tqdm
import random

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), apply_augmentation=True):
        self.target_size = target_size
        self.apply_augmentation = apply_augmentation
        
    def preprocess(self, image_path, segment=False):
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Unable to read image {image_path}")
                return np.zeros((*self.target_size, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        img = cv2.resize(img, self.target_size)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if segment:
            pass
            
        if self.apply_augmentation:
            img = self.apply_material_augmentation(img)
            
        return img
    
    def apply_material_augmentation(self, img):
        original = img.copy()
        
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            mean = np.mean(img)
            img = (img - mean) * factor + mean
            img = np.clip(img, 0, 255).astype(np.uint8)
            
        if np.random.random() < 0.3:
            for c in range(3):
                if np.random.random() < 0.5:
                    img[:,:,c] = img[:,:,c] * np.random.uniform(0.8, 1.2)
            img = np.clip(img, 0, 255).astype(np.uint8)
            
        if np.random.random() < 0.3:
            height, width = img.shape[:2]
            src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            dist = np.random.randint(-20, 20, 8).reshape(4, 2).astype(np.float32)
            dst_points = src_points + dist
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            img = cv2.warpPerspective(img, M, (width, height), 
                                     borderMode=cv2.BORDER_REFLECT)
        
        if np.random.random() < 0.4:
            angle = np.random.uniform(-15, 15)
            height, width = img.shape[:2]
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            img = cv2.warpAffine(img, M, (width, height), 
                                borderMode=cv2.BORDER_REFLECT)
        
        if np.random.random() < 0.2:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 5, img.shape).astype(np.int32)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
        if np.random.random() < 0.2:
            h, w = img.shape[:2]
            patch_h, patch_w = int(h * np.random.uniform(0.1, 0.3)), int(w * np.random.uniform(0.1, 0.3))
            x, y = np.random.randint(0, w - patch_w), np.random.randint(0, h - patch_h)
            img[y:y+patch_h, x:x+patch_w] = original[y:y+patch_h, x:x+patch_w]
            
        return img
    
    def batch_preprocess(self, image_paths, segment=False):
        processed_images = []
        
        for img_path in tqdm(image_paths, desc="Preprocessing images"):
            processed = self.preprocess(img_path, segment)
            processed_images.append(processed)
            
        return processed_images