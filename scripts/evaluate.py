import torch
import random
from transformers import VisionEncoderDecoderModel, DonutProcessor, logging
from tqdm import tqdm
from .dataloader import get_processed_dataset
from .config import TRAINED_MODEL_DIR, DEVICE

logging.set_verbosity_error()


def run_prediction(sample, model, processor, device):
    """
    Generates a prediction for a single sample.
    """
    pixel_values = torch.tensor(sample["pixel_values"]).unsqueeze(0).to(device)
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)

    target = processor.token2json(sample["target_sequence"])
    return prediction, target


def main():
    print("Loading trained model and processor...")
    model = VisionEncoderDecoderModel.from_pretrained(TRAINED_MODEL_DIR).to(DEVICE)
    processor = DonutProcessor.from_pretrained(TRAINED_MODEL_DIR)

    print("Loading test dataset...")
    # We call this again to get the same train/test split.
    # For a more robust pipeline, you could save the test set to disk during training.
    processed_dataset, _ = get_processed_dataset()
    test_dataset = processed_dataset["test"]

    # --- Test a single random sample ---
    print("\n--- Running prediction on a random sample ---")
    random_sample = test_dataset[random.randint(0, len(test_dataset) - 1)]
    prediction, target = run_prediction(random_sample, model, processor, DEVICE)
    print(f"Reference:\n {target}")
    print(f"Prediction:\n {prediction}\n")

    # --- Evaluate on the entire test set ---
    print("--- Evaluating on the entire test set ---")
    true_counter = 0
    total_counter = 0

    for sample in tqdm(test_dataset):
        prediction, target = run_prediction(sample, model, processor, DEVICE)
        for s in zip(prediction.values(), target.values()):
            # Simple exact match accuracy
            if str(s[0]).strip() == str(s[1]).strip():
                true_counter += 1
            total_counter += 1

    if total_counter > 0:
        accuracy = (true_counter / total_counter) * 100
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
    else:
        print("No items to evaluate.")


if __name__ == "__main__":
    main()
