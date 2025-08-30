import json
from datasets import load_dataset
from transformers import DonutProcessor
from .config import *


new_special_tokens = []
task_start_token = "<s>"
eos_token = "</s>"


def json2token(
    obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True
):
    """
    Convert a JSON object to a token sequence.
    """
    global new_special_tokens
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    (
                        new_special_tokens.append(f"<s_{k}>")
                        if f"<s_{k}>" not in new_special_tokens
                        else None
                    )
                    (
                        new_special_tokens.append(f"</s_{k}>")
                        if f"</s_{k}>" not in new_special_tokens
                        else None
                    )
                output += (
                    f"<s_{k}>"
                    + json2token(
                        obj[k], update_special_tokens_for_json_key, sort_json_key
                    )
                    + f"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [
                json2token(item, update_special_tokens_for_json_key, sort_json_key)
                for item in obj
            ]
        )
    else:
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"
        return obj


def preprocess_documents_for_donut(sample):
    """
    Convert the JSON string in 'text' to a tokenized format for the Donut model.
    """
    text = json.loads(sample["text"])
    d_doc = task_start_token + json2token(text) + eos_token
    image = sample["image"].convert("RGB")
    return {"image": image, "text": d_doc}


def transform_and_tokenize(
    sample, processor, split="train", max_length=512, ignore_id=-100
):
    """
    Apply the processor to tokenizing image and text.
    """
    pixel_values = processor(
        sample["image"], random_padding=split == "train", return_tensors="pt"
    ).pixel_values.squeeze()

    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "target_sequence": sample["text"],
    }


# --- Main Data Loading Function ---


def get_processed_dataset():
    """
    Loads, preprocesses, and tokenizes the dataset, returning the split dataset
    and the configured processor.
    """
    # 1. Load raw dataset from image folder
    dataset = load_dataset(
        "imagefolder",
        data_dir=f"{DATA_DIR}/{IMAGE_DIR_NAME}",
        split="train",
    )
    print(f"Dataset loaded with {len(dataset)} images.")

    # 2. Convert JSON strings to Donut's token format and collect special tokens
    proc_dataset = dataset.map(preprocess_documents_for_donut)

    # 3. Setup processor with new special tokens
    processor = DonutProcessor.from_pretrained(BASE_MODEL_NAME)
    unique_tokens = list(set(new_special_tokens))
    unique_tokens.sort()
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": unique_tokens + [task_start_token] + [eos_token]}
    )

    # Configure feature extractor
    processor.feature_extractor.size = [IMAGE_HEIGHT, IMAGE_WIDTH]
    processor.feature_extractor.do_align_long_axis = False

    # 4. Transform and tokenize the dataset
    processed_dataset = proc_dataset.map(
        lambda sample: transform_and_tokenize(sample, processor, max_length=MAX_LENGTH),
        remove_columns=["image", "text"],
    )

    # 5. Split dataset into training and testing sets
    processed_dataset = processed_dataset.train_test_split(test_size=TRAIN_TEST_SPLIT)
    print("Dataset after splitting:\n", processed_dataset)

    return processed_dataset, processor
