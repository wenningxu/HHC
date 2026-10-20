import torch
import clip


def encode_text_cpu_float32(model, texts):
    """Return old encode_text outputs using one explicit CPU fp32 CLIP pass."""
    if next(model.parameters()).device.type != 'cpu' or next(model.parameters()).dtype != torch.float32:
        raise ValueError('helper is CPU float32 only')
    tokens = clip.tokenize(list(texts), truncate=True)
    x = model.token_embedding(tokens).float() + model.positional_embedding.float()
    word = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    word = model.ln_final(word).float()
    eot = tokens.argmax(dim=-1)
    global_text = word[torch.arange(tokens.shape[0]), eot] @ model.text_projection.float()
    return global_text, word
