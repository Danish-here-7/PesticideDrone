from preprocessing.prepare_dataset import prepare_dataset
from model.pest_model import build_model

dataset = prepare_dataset(
    rgb_folder="data/rgb_images",
    thermal_folder="data/thermal_maps",
    metadata_path="data/metadata.csv"
)

dataset = dataset.batch(16).shuffle(100)

model = build_model()
model.fit(dataset, epochs=10)
