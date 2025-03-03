import numpy as np

def count_pixels_per_class(mask, exclude_background=False):
    class_counts = {}
    unique_classes = np.unique(mask)
    for class_id in unique_classes:
        if exclude_background and class_id == 0:
            continue  # Skip background class
        class_counts[int(class_id)] = int(np.sum(mask == class_id))

    # Convert class IDs to meaningful names
    class_names = {
        0: 'background',
        1: 'early_blight',
        2: 'healthy',
        3: 'late_blight'
    }
    named_counts = {class_names[k]: v for k, v in class_counts.items() if k in class_names}

    return named_counts

def calculate_class_percentages(class_pixel_counts):
    """
    Menghitung persentase setiap kelas dari dictionary jumlah piksel.

    Args:
        class_pixel_counts (dict): Dictionary yang berisi jumlah piksel per kelas (nama kelas sebagai kunci).

    Returns:
        dict: Dictionary yang berisi nama kelas dan persentasenya.
    """
    total_pixels = sum(class_pixel_counts.values())

    if total_pixels == 0:
        return {}  # Kembalikan dictionary kosong jika tidak ada piksel selain background

    class_percentages = {}
    for class_name, count in class_pixel_counts.items():
        percentage = (count / total_pixels) * 100
        class_percentages[class_name] = round(percentage, 2)

    return class_percentages

# Bisa
def get_predicted_class(class_counts, exclude_background=False):
    if exclude_background:
        class_counts = {k: v for k, v in class_counts.items() if k != 0}  
    
    predicted_class = max(class_counts, key=class_counts.get)
    return predicted_class
