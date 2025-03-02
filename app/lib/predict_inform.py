from collections import Counter

def count_class_pixels(mask, exclude_background=False):
    flattened_mask = mask.flatten()
    
    # If excluding background, remove class 0 from the count
    if exclude_background:
        flattened_mask = flattened_mask[flattened_mask != 0]
        
    class_counts = Counter(flattened_mask)
    return class_counts

def get_class_probabilities(class_counts, total_pixels, exclude_background=False):
    if exclude_background:
        total_pixels -= class_counts.get(0, 0)  # Subtract background pixels from total
    
    probabilities = {class_id: count / total_pixels for class_id, count in class_counts.items()}
    return probabilities

def get_predicted_class(class_counts, exclude_background=False):
    if exclude_background:
        class_counts = {k: v for k, v in class_counts.items() if k != 0}  # Exclude background class
    
    predicted_class = max(class_counts, key=class_counts.get)
    return predicted_class
