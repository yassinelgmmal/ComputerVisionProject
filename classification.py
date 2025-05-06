import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import os
import gc

class CustomClassifier:
    def __init__(self, input_shape, num_classes, learning_rate=0.001, dropout_rate=0.5, l2_reg=0.01):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = self._build_model()
        
    def _build_model(self):
        print(f"Building model for input shape: {self.input_shape} with {self.num_classes} classes")
        
        model = Sequential([
            Dense(128, kernel_regularizer=l2(self.l2_reg*2), input_shape=(self.input_shape,)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(self.dropout_rate),
            Dense(96, kernel_regularizer=l2(self.l2_reg*1.5)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(self.dropout_rate*0.9),
            Dense(64, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(), 
            Activation('relu'),
            Dropout(self.dropout_rate*0.8),
            Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(self.l2_reg/2))
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=64, epochs=30, 
              patience=8, class_weights=None, max_samples=None):
        if isinstance(X_train, list):
            X_train = np.array(X_train)
        if isinstance(y_train, list):
            y_train = np.array(y_train)
        if isinstance(X_val, list):
            X_val = np.array(X_val)
        if isinstance(y_val, list):
            y_val = np.array(y_val)
            
        print("Training shapes:", X_train.shape, y_train.shape)
        print("Validation shapes:", X_val.shape, y_val.shape)
        
        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)
            
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        print("Class distribution in training set:")
        for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
            print(f"  Class {cls}: {count} samples")
        
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        
        if max_samples and max_samples < len(X_train):
            indices = []
            for cls in range(self.num_classes):
                cls_indices = np.where(y_train == cls)[0]
                if len(cls_indices) > 0:
                    n_samples = int(max_samples * (np.sum(y_train == cls) / len(y_train)))
                    n_samples = max(n_samples, 1)
                    sampled_indices = np.random.choice(cls_indices, 
                                                      size=min(n_samples, len(cls_indices)), 
                                                      replace=False)
                    indices.extend(sampled_indices)
            
            np.random.shuffle(indices)
            
            indices = indices[:max_samples]
            
            X_train_subset = X_train[indices]
            y_train_cat_subset = y_train_cat[indices]
            
            print(f"Using {len(indices)} samples for training to reduce overfitting")
        else:
            X_train_subset = X_train
            y_train_cat_subset = y_train_cat
        
        checkpoint_dir = './model_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                min_delta=0.01
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-5
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'waste_model_best.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]
        
        history = self.model.fit(
            X_train_subset, y_train_cat_subset,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val_cat),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
            
        X = np.nan_to_num(X)
            
        if len(X) > 500:
            batch_size = 200
            all_predictions = []
            
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                batch = X[i:end_idx]
                batch_preds = self.model.predict(batch, verbose=0)
                all_predictions.append(batch_preds)
            
            probabilities = np.vstack(all_predictions)
        else:
            probabilities = self.model.predict(X, verbose=0)
        
        return np.argmax(probabilities, axis=1)
    
    def evaluate(self, X, y):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)
            
        X = np.nan_to_num(X)
            
        y_cat = to_categorical(y, self.num_classes)
        
        return self.model.evaluate(X, y_cat, verbose=0)
    
    def plot_training_history(self, history):
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Waste Classification Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Waste Classification Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()