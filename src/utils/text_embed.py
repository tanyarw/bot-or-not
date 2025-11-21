import torch
from transformers import pipeline
from tqdm import tqdm


class TextEmbedder:
    def __init__(self, model_name, batch_size, device):
        self.pipeline_device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.model = None

    def _pool(self, f):
        t = torch.tensor(f)

        if t.ndim == 3:
            t = t.squeeze(0) # (seq_len, hidden_dim)

        return t.mean(dim=0)  # (hidden_dim,)

    def embed_batch(self, texts, show_progress=False, to_device="cpu"):
        if self.model is None:
            self.model = pipeline(
                "feature-extraction",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.pipeline_device,
            )

        all_embs = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding text")

        for start in iterator:
            batch = texts[start : start + self.batch_size]
            batch = [t if isinstance(t, str) else "" for t in batch]

            feats = self.model(batch, truncation=True)

            pooled = [self._pool(f) for f in feats]

            embs = torch.stack(pooled).to(to_device)
            all_embs.append(embs)

        return torch.cat(all_embs, dim=0)
