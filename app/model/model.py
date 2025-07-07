from keras.api.models import load_model
import numpy as np
import cv2
import os
import tensorflow as tf
from keras.api.preprocessing.image import img_to_array
from keras.src.layers import Layer, GlobalAveragePooling2D, Dense, Reshape, Multiply
from keras.api.saving import register_keras_serializable

@register_keras_serializable()
def se_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation Block."""
    channel_axis = -1  # TensorFlow channels-last
    filters = input_tensor.shape[channel_axis]

    se = GlobalAveragePooling2D()(input_tensor)  # Squeeze
    se = Dense(filters // ratio, activation='relu')(se)  # Reduction
    se = Dense(filters, activation='sigmoid')(se)  # Excitation
    se = Reshape((1, 1, filters))(se)  # Reshape to match input dimensions
    se = Multiply()([input_tensor, se])  # Scale input_tensor

    return se

@register_keras_serializable()
class SoftPool(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super(SoftPool, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        exp_inputs = tf.exp(inputs)
        pooled_exp = tf.nn.avg_pool(exp_inputs, ksize=self.pool_size, strides=self.strides, padding=self.padding.upper())
        pooled_inputs = tf.nn.avg_pool(inputs * exp_inputs, ksize=self.pool_size, strides=self.strides, padding=self.padding.upper())
        return pooled_inputs / (pooled_exp + 1e-6)

    def get_config(self):
        config = super().get_config() # Change this line to call super() correctly
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
      
# Fungsi Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Menghitung Dice Loss"""
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    union = tf.reduce_sum(y_true, axis=(1, 2, 3)) + tf.reduce_sum(y_pred, axis=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice


# Fungsi Weighted Categorical Cross-Entropy
def weighted_categorical_crossentropy(y_true, y_pred, weights):
    """Menghitung Weighted Categorical Cross-Entropy"""
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    weighted_cce = cce * tf.reduce_sum(weights * y_true, axis=-1)
    return tf.reduce_mean(weighted_cce, axis=(1, 2))


# Registrasi fungsi agar bisa diserialisasi
@register_keras_serializable()
def DCCE(y_true, y_pred, weights=None):
    """Dice Loss + Weighted Categorical Cross-Entropy"""
    dice = dice_loss(y_true, y_pred)
    if weights is not None:
        cce = weighted_categorical_crossentropy(y_true, y_pred, weights)
    else:
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        cce = tf.reduce_mean(cce, axis=(1, 2))

    return tf.reduce_mean(dice + cce)


# **Subclassing tf.keras.losses.Loss untuk DCCE**
@register_keras_serializable()
class DCCELoss(tf.keras.losses.Loss):
    def __init__(self, weights, name="DCCELoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weights = weights

    def call(self, y_true, y_pred):
        return DCCE(y_true, y_pred, weights=self.weights)

    def get_config(self):
        config = super().get_config()
        config.update({'weights': self.weights.numpy().tolist()})  # Convert weights to list
        return config

    @classmethod
    def from_config(cls, config):
        config['weights'] = tf.Variable(config['weights'], dtype=tf.float32, trainable=False)  # Convert back to Variable
        return cls(**config)

def iou_per_class(y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int) -> tf.Tensor:
    """Menghitung IoU untuk setiap kelas (abaikan class 0, input one-hot)."""
    iou_scores = []
    y_pred_labels = tf.argmax(y_pred, axis=-1)  # Konversi prediksi ke label kelas
    y_true_labels = tf.argmax(y_true, axis=-1)  # Konversi mask ke label kelas

    for i in range(1, num_classes):  # Mulai dari class 1
        true_mask = tf.equal(y_true_labels, i)
        pred_mask = tf.equal(y_pred_labels, i)

        intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(true_mask, pred_mask), tf.float32))
        iou = tf.math.divide_no_nan(intersection, union)
        iou_scores.append(iou)

    return tf.concat([[0.0], iou_scores], axis=0)  # Class 0 diisi 0

def dice_per_class(y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int) -> tf.Tensor:
    """Menghitung Dice score untuk setiap kelas (abaikan class 0, input one-hot)."""
    dice_scores = []
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    y_true_labels = tf.argmax(y_true, axis=-1)

    for i in range(1, num_classes):  # Mulai dari class 1
        true_mask = tf.equal(y_true_labels, i)
        pred_mask = tf.equal(y_pred_labels, i)

        intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))
        sum_true_pred = tf.reduce_sum(tf.cast(true_mask, tf.float32)) + tf.reduce_sum(tf.cast(pred_mask, tf.float32))
        dice = tf.math.divide_no_nan(2 * intersection, sum_true_pred)
        dice_scores.append(dice)

    return tf.concat([[0.0], dice_scores], axis=0)  # Class 0 diisi 0

@register_keras_serializable()
def macro_iou(y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int = 4) -> tf.Tensor:
    """Menghitung Macro IoU (hanya class 1,2,3, input one-hot)."""
    iou_scores = iou_per_class(y_true, y_pred, num_classes)
    return tf.reduce_mean(iou_scores[1:])  # Abaikan class 0

@register_keras_serializable()
def macro_dice_score(y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int = 4) -> tf.Tensor:
    """Menghitung Macro Dice (hanya class 1,2,3, input one-hot)."""
    dice_scores = dice_per_class(y_true, y_pred, num_classes)
    return tf.reduce_mean(dice_scores[1:])  # Abaikan class 0

@register_keras_serializable()
def pixel_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Menghitung Akurasi Pixel (abaikan class 0, input one-hot)."""
    y_true_labels = tf.argmax(y_true, axis=-1)
    y_pred_labels = tf.argmax(y_pred, axis=-1)

    # Create a mask for non-zero classes in the true labels
    mask = tf.greater(y_true_labels, 0)

    # Calculate correct predictions only for the masked pixels
    # We need to ensure that both true_labels and pred_labels are within the mask
    correct_predictions_masked = tf.logical_and(tf.equal(y_true_labels, y_pred_labels), mask)
    correct = tf.reduce_sum(tf.cast(correct_predictions_masked, tf.float32))

    # Total relevant pixels are those in the true mask (non-zero classes)
    total = tf.reduce_sum(tf.cast(mask, tf.float32))

    return tf.math.divide_no_nan(correct, total) 


def load_unet_model():
  model_path = os.path.join(os.path.dirname(__file__), "unet_se_softpool_(70_20_10).keras")
  try:
      model = load_model(model_path, custom_objects={
        'SoftPool': SoftPool,
        'se_block': se_block,
        'DCCELoss': DCCELoss,
        'macro_iou': macro_iou,
        'macro_dice_score': macro_dice_score,
        'pixel_accuracy': pixel_accuracy,
      })
      return model
  except Exception as e:
      raise ValueError(f"Error loading model: {e}")

def predict(model, image: np.ndarray):
  resized = cv2.resize(image, (256, 256))
  image_array = img_to_array(resized)
  image_array = np.expand_dims(image_array, axis=0)
  image_array = image_array / 255.0
  
  prediction = model.predict(image_array)
  
  return prediction