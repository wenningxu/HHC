"""Small audit helper for one-position-per-draw sampling policies."""
import torch
from models.two_person_transformer import selected_log_odds

def selection_counts(logits, valid, temperature=0.5, draws=64, generator=None):
    if logits.ndim != 4 or logits.shape[2] != 3 or valid.shape != logits.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("logits [B,T,3,V], valid bool [B,T] required")
    if temperature <= 0 or draws <= 0 or not valid.any(dim=1).all():
        raise ValueError("positive temperature/draws and nonempty valid required")
    b, t, _, v = logits.shape; probs = torch.softmax(logits / temperature, -1)
    p = torch.multinomial(probs.reshape(-1, v), draws, True, generator=generator).reshape(b, t, 3, draws)
    a = logits.argmax(-1); result = {}
    for part, label in enumerate(("body", "hand", "root")):
        q, target = p[:, :, part], a[:, :, part]; m = valid[:, :, None]
        # Match production ranking: raw-logit softmax, not sampling temperature.
        conf = logits[:, :, part].softmax(-1).gather(-1, q)
        positions = [("highest_confidence", conf.masked_fill(~m, -1).argmax(1))]
        u = torch.rand((b, t, draws), generator=generator, device=logits.device).masked_fill(~valid[:, :, None], -1).argmax(1)
        positions.append(("uniform_position", u))
        g = -torch.empty_like(conf).exponential_(generator=generator).log()
        n = (conf.clamp_min(torch.finfo(conf.dtype).tiny).log() + g).masked_fill(~m, -torch.inf).argmax(1)
        positions.append(("noisy_log_confidence", n))
        odds = selected_log_odds(logits[:, :, part], q)
        positions.append(('highest_log_odds', odds.masked_fill(~m, -torch.inf).argmax(1)))
        positions.append(('noisy_log_odds', (odds + g).masked_fill(~m, -torch.inf).argmax(1)))
        out = {"proposal_count": int(m.sum().item() * draws), "proposal_non_argmax": int(((q != target[:, :, None]) & m).sum().item())}
        for name, pos in positions:
            chosen = q.gather(1, pos[:, None]).squeeze(1)
            out[f"{name}_accepted_count"] = b * draws
            out[f"{name}_accepted_non_argmax"] = int((chosen != target.gather(1, pos)).sum().item())
            out[f'{name}_unique_positions_sum'] = sum(int(row.unique().numel()) for row in pos)
        result[label] = out
    return result
