from .bot_rgcn import BotRGCN

MODEL_REGISTRY = {
    'bot_rgcn': BotRGCN,
}

def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)
