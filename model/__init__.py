from .bot_rgcn import BotRGCN
from .bot_ergcn import BotEvolvingRGCN

MODEL_REGISTRY = {
    'bot_rgcn': BotRGCN,
    'bot_ergcn': BotEvolvingRGCN,
}

def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)
