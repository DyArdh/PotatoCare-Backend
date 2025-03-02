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
        config = super(SoftPool, self).get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
      
# Dice Loss Function
def dice_loss(y_true, y_pred, smooth=1e-6):
    # Ensure predictions are probabilities
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)

    # Compute Dice coefficient
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))  # Sum over spatial dims
    union = tf.reduce_sum(y_true, axis=(1, 2, 3)) + tf.reduce_sum(y_pred, axis=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice  # Return loss

# Categorical Cross-Entropy Function
def categorical_crossentropy(y_true, y_pred):
    # Ensure predictions are probabilities
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(cce, axis=(1, 2))  # Reduce over spatial dims

# Combined Dice + Categorical Cross-Entropy Loss
@register_keras_serializable()
def DCCE(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)  # Dice loss is already batch-reduced
    cce = categorical_crossentropy(y_true, y_pred)  # Spatially-reduced CCE
    return dice + tf.reduce_mean(cce)  # Reduce CCE over batch and combine

def load_unet_model():
  model_path = os.path.join(os.path.dirname(__file__), "unet_se_softpool_dcce_(70_20_10).keras")
  try:
      model = load_model(model_path, custom_objects={
          'SoftPool': SoftPool,
          'se_block': se_block
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