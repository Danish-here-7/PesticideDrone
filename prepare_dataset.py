import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.utils import img_to_array, load_img

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    return df

def load_image(image_path, size=(224, 224), color_mode="rgb"):
    img = load_img(image_path, target_size=size, color_mode=color_mode)
    return img_to_array(img) / 255.0

def prepare_dataset(rgb_folder, thermal_folder, metadata_path):
    metadata = load_metadata(metadata_path)

    rgb_images = []
    thermal_images = []
    metadata_inputs = []
    labels = []

    for _, row in metadata.iterrows():
        rgb = load_image(os.path.join(rgb_folder, row['filename']), color_mode='rgb')
        thermal = load_image(os.path.join(thermal_folder, row['filename']), color_mode='grayscale')

        features = [row['month'], row['region_code'], row['migration_density']]  # normalized
        label = row['label']  # pest (1) / no pest (0)

        rgb_images.append(rgb)
        thermal_images.append(thermal)
        metadata_inputs.append(features)
        labels.append(label)

    return tf.data.Dataset.from_tensor_slices(((rgb_images, thermal_images, metadata_inputs), labels))
