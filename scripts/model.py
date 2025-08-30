from transformers import VisionEncoderDecoderModel, logging
from .config import BASE_MODEL_NAME, DEVICE


def load_model(processor):
    """
    Loads and configures the VisionEncoderDecoderModel for Donut.
    """
    print("Loading and configuring the model...")
    logging.set_verbosity_error()
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_NAME)
    logging.set_verbosity_warning()

    # Resize token embeddings to match the newly added special tokens
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Configure model settings
    model.config.encoder.image_size = processor.feature_extractor.size[::-1]
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        ["<s>"]
    )[0]

    model.to(DEVICE)
    print("Model configured and moved to device:", DEVICE)
    return model
