import torch
from transformers import pipeline
from tqdm import tqdm
from transformers import AutoTokenizer


class TextEmbedder:
    def __init__(self, model_name, batch_size, device):
        self.pipeline_device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.model = None

    def embed_batch(self, texts, show_progress=False, to_device="cpu"):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        if self.model is None:
            self.model = pipeline(
                "feature-extraction",
                model=self.model_name,
                tokenizer=tokenizer,
                device=self.pipeline_device,
                torch_dtype=torch.float16
            )

        texts = [t if isinstance(t, str) else "" for t in texts]
        all_embeddings = []
        
        with torch.no_grad():
            pbar = tqdm(
                range(0, len(texts), self.batch_size),
                desc="Embedding batch",
                disable=not show_progress
            )
            
            for i in pbar:
                batch_texts = texts[i:i+self.batch_size]
                
                # Tokenize just this batch
                encodings = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.pipeline_device)
                
                # Process immediately
                outputs = self.model.model(**encodings)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0).to(to_device)