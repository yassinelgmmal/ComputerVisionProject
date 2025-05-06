import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import exposure
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        self.model = base_model
        print(f"Feature extractor initialized with output shape: {self.model.output_shape}")
        
    def extract_deep_features(self, img):
        if img.shape[:2] != self.input_shape[:2]:
            img = cv2.resize(img, self.input_shape[:2])
            
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
            
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        
        features = self.model.predict(x, verbose=0)
        return features[0]
    
    def extract_color_features(self, img):
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        for i in range(3):
            channel = img[:, :, i]
            mean = np.mean(channel)
            std = np.std(channel)
            skewness = np.mean(((channel - mean)/std)**3) if std > 0 else 0
            hist_features.extend([mean, std, skewness])
            
        avg_colors = [np.mean(img[:,:,i]) for i in range(3)]
        if avg_colors[2] > 0:
            r_b_ratio = avg_colors[0] / avg_colors[2]
        else:
            r_b_ratio = 1
        if avg_colors[1] > 0:
            r_g_ratio = avg_colors[0] / avg_colors[1]
        else:
            r_g_ratio = 1
            
        hist_features.extend([r_b_ratio, r_g_ratio])
            
        return np.array(hist_features)
    
    def extract_texture_features(self, img):
        try:
            gray = rgb2gray(img)
            
            gray_uint8 = (gray * 255).astype(np.uint8)
            
            hog_features = hog(
                gray, 
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), 
                visualize=False,
                block_norm='L2-Hys'
            )
            
            radius = 2
            n_points = 8 * radius
            lbp = local_binary_pattern(gray_uint8, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2), density=True)
            
            rescaled_gray = (gray_uint8 // 16).astype(np.uint8)
            
            try:
                glcm = graycomatrix(rescaled_gray, [1, 3], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                  levels=16, symmetric=True, normed=True)
                
                glcm_features = []
                for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
                    glcm_features.extend(graycoprops(glcm, prop).flatten())
            except Exception as e:
                print(f"GLCM calculation failed: {e}")
                glcm_features = np.zeros(8)
                
            texture_features = np.concatenate([
                hog_features[:200],
                lbp_hist,
                np.array(glcm_features)
            ])
            
            return texture_features
            
        except Exception as e:
            print(f"Error extracting texture features: {e}")
            return np.zeros(300)
    
    def extract_material_specific_features(self, img):
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img.copy()
                
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
            shine_ratio = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
            
            rescaled_gray = (gray // 16).astype(np.uint8)
            
            try:
                glcm = graycomatrix(rescaled_gray, [1], [0], 16, symmetric=True, normed=True)
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
            except:
                homogeneity, contrast, energy = 0, 0, 0
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_mean = np.mean(edge_magnitude)
            edge_std = np.std(edge_magnitude)
            
            uniformity = 1.0 - (np.std(gray) / 128.0 if np.mean(gray) > 0 else 1)
            
            try:
                blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
                detail = gray.astype(float) - blur.astype(float)
                detail_energy = np.mean(np.abs(detail))
            except:
                detail_energy = 0
                
            try:
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                hist = hist / np.sum(hist)
                entropy = -np.sum(hist * np.log2(hist + 1e-7))
            except:
                entropy = 0
                
            material_features = np.array([
                shine_ratio,
                homogeneity,
                contrast,
                energy,
                edge_mean,
                edge_std,
                uniformity,
                detail_energy,
                entropy
            ])
            
            return material_features
            
        except Exception as e:
            print(f"Error extracting material-specific features: {e}")
            return np.zeros(9)
    
    def extract_features(self, img):
        try:
            deep_features = self.extract_deep_features(img)
            
            color_features = self.extract_color_features(img)
            
            texture_features = self.extract_texture_features(img)
            
            material_features = self.extract_material_specific_features(img)
            
            all_features = np.concatenate([
                deep_features,
                color_features,
                texture_features,
                material_features
            ])
            
            return all_features
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return np.zeros(1500)
    
    def batch_extract_features(self, images, batch_size=4):
        features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:min(i+batch_size, len(images))]
            
            batch_features = []
            for img in tqdm(batch, desc="Extracting features"):
                try:
                    features_vector = self.extract_features(img)
                    batch_features.append(features_vector)
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    placeholder = np.zeros(1500)
                    batch_features.append(placeholder)
            
            features.extend(batch_features)
            
            import gc
            gc.collect()
            
        return features