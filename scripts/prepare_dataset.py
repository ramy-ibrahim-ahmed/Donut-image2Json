import json
import shutil
from pathlib import Path
import argparse
from .config import KEY_DIR_NAME, IMAGE_DIR_NAME, DATA_DIR


def create_metadata_file(data_dir: str):
    """
    Combines individual JSON annotations into a single metadata.jsonl file
    and places it inside the image directory.
    """
    base_path = Path(data_dir)
    metadata_path = base_path.joinpath(KEY_DIR_NAME)
    image_path = base_path.joinpath(IMAGE_DIR_NAME)

    if not metadata_path.exists():
        print(f"Metadata directory not found: {metadata_path}")
        print("Skipping preparation.")
        return

    print("Starting dataset preparation...")
    metadata_list = []

    for file_name in metadata_path.glob("*.json"):
        with open(file_name, "r") as json_file:
            data = json.load(json_file)
            text = json.dumps(data)
            if image_path.joinpath(f"{file_name.stem}.jpg").is_file():
                metadata_list.append(
                    {"text": text, "file_name": f"{file_name.stem}.jpg"}
                )

    metadata_file = image_path.joinpath("metadata.jsonl")
    with open(metadata_file, "w") as outfile:
        for entry in metadata_list:
            json.dump(entry, outfile)
            outfile.write("\n")

    # Clean up the original key directory
    shutil.rmtree(metadata_path)
    print(f"Successfully created {metadata_file}")
    print(f"Removed original metadata directory: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare dataset for Donut model training."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help="Path to the root data directory.",
    )
    args = parser.parse_args()
    create_metadata_file(args.data_dir)
