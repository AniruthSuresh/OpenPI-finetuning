import os
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image
from io import BytesIO

def save_images_from_parquet(parquet_dir: str, output_dir: str, image_column='image', id_column='id'):
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topics_printed = False

    for parquet_file in sorted(parquet_dir.glob("*.parquet")):

        if not topics_printed:
            try:
                schema = pq.read_schema(parquet_file)
                print("--- Parquet Topic Names ---")
                print(schema.names)
                print("---------------------------\n")
                topics_printed = True
            except Exception as e:
                print(f"[ERROR] Could not read schema from {parquet_file.name}: {e}")

        print(f"[INFO] Reading: {parquet_file.name}")
        table = pq.read_table(parquet_file)
        df = table.to_pandas()

        subdir = output_dir / parquet_file.stem
        subdir.mkdir(exist_ok=True)

        for idx, row in df.iterrows():
            image_data_dict = row.get(image_column)
            image_id = row.get(id_column, f"{idx:04d}")
            try:
                # The 'bytes' key likely contains a raw bytes object already.
                if isinstance(image_data_dict, dict) and "bytes" in image_data_dict:
                    byte_array = image_data_dict["bytes"]
                    image = Image.open(BytesIO(byte_array))
                else:
                    raise ValueError(f"Unsupported image format at row {idx}")

                # Save image
                image.save(subdir / f"{image_id}.png")

            except Exception as e:
                print(f"[ERROR] Failed to save image at row {idx} in {parquet_file.name}: {e}")


# Example usage (no changes needed here)
save_images_from_parquet(
    parquet_dir="/home/aniruth/Desktop/RRC/OpenPI-finetuning/src/openpi/utils/converted_libero/data/chunk-000",
    output_dir="/home/aniruth/Desktop/RRC/OpenPI-finetuning/src/openpi/utils/parquet_output",
    image_column="image",
    id_column="image_id"
)
