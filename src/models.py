from transformers import CLIPProcessor, CLIPModel


def create_model(model_name: str):
    if model_name is not None:
        model = CLIPModel.from_pretrained(model_name)
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def create_processor(model_name: str):
    if model_name is not None:
        return CLIPProcessor.from_pretrained(model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")