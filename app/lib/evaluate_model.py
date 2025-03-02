import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score

def iou(y_true, y_pred, num_classes, exclude_background=True):
  y_true = tf.argmax(y_true, axis=-1)
  y_pred = tf.argmax(y_pred, axis=-1)
  
  iou_scores = []
  weights = []
  start_class = 1 if exclude_background else 0
  
  for i in range(start_class, num_classes):
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.logical_or(tf.equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
    
    iou = tf.math.divide_no_nan(intersection, union)
    iou_scores.append(iou)
    
    weight = tf.reduce_sum(tf.cast(tf.equal(y_true, i), tf.float32))
    weights.append(weight)
    
  weights = tf.convert_to_tensor(weights, dtype=tf.float32)
  weights = tf.math.divide_no_nan(weights, tf.reduce_sum(weights))
  
  weighted_iou = tf.reduce_sum(tf.convert_to_tensor(iou_scores, dtype=tf.float32) * weights)
  return weighted_iou

def dice(y_true, y_pred, num_classes, exclude_background=True):
  y_true = tf.argmax(y_true, axis=-1)
  y_pred = tf.argmax(y_pred, axis=-1)

  dice_scores = []
  weights = []
  start_class = 1 if exclude_background else 0
  
  for i in range(start_class, num_classes):
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.equal(y_true, i), tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(y_pred, i), tf.float32))
    dice = tf.math.divide_no_nan(2 * intersection, union)
    dice_scores.append(dice)

    weight = tf.reduce_sum(tf.cast(tf.equal(y_true, i), tf.float32))
    weights.append(weight)
    
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    weights = tf.math.divide_no_nan(weights, tf.reduce_sum(weights))
    
    weighted_dice = tf.reduce_sum(tf.convert_to_tensor(dice_scores, dtype=tf.float32) * weights)
    return weighted_dice
  
def mean_average_precision(y_true, y_pred, num_classes=4, exclude_background=True):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    ap_scores = []
    start_class = 1 if exclude_background else 0

    for i in range(start_class, num_classes):
        y_true_class = (y_true == i).numpy().astype(int)
        y_pred_class = tf.nn.softmax(y_pred)[:, i].numpy()

        if np.sum(y_true_class) == 0:
            print(f"Warning: No positive samples for class {i}. Skipping AP calculation for this class.")
            continue

        ap_score = average_precision_score(y_true_class, y_pred_class)
        ap_scores.append(ap_score)

    mean_ap = np.mean(ap_scores) if len(ap_scores) > 0 else 0.0

    return mean_ap
  
def evaluate_segmentation_metrics(y_true, y_pred, num_classes, background=True):
  iou_score = iou(y_true, y_pred, num_classes, exclude_background=background)
  dice_score = dice(y_true, y_pred, num_classes, exclude_background=background)
  map_score = mean_average_precision(y_true, y_pred, num_classes, exclude_background=background)
  
  return {
    "iou": iou_score,
    "dice": dice_score,
    "map": map_score
  }