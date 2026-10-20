import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, clip
from utils.hhc_cpu_text_cache import encode_text_cpu_float32

def old(model, texts):
    tokens=clip.tokenize(list(texts),truncate=True)
    word=model.token_embedding(tokens).type(model.dtype)+model.positional_embedding.type(model.dtype)
    word=model.transformer(word.permute(1,0,2)); word=model.ln_final(word).permute(1,0,2).float()
    return model.encode_text(tokens).float(), word

def main():
    model,_=clip.load('ViT-B/32',device='cpu',jit=False); model.float().eval()
    base=['hello two people walk together','A '*200,'two people shake hands and then turn around.','']
    rows=[]
    for n in (1,8,37,128):
        texts=[base[i%len(base)] for i in range(n)]
        a,b=old(model,texts); c,d=encode_text_cpu_float32(model,texts)
        rows.append({'batch':n,'global_equal':bool(torch.equal(a,c)),'word_equal':bool(torch.equal(b,d)),
                     'global_max_abs':float((a-c).abs().max()),'word_max_abs':float((b-d).abs().max())})
    out={'schema':'hhc_cpu_text_cache_audit_v1','model':'ViT-B/32','rows':rows,'all_equal':all(r['global_equal'] and r['word_equal'] for r in rows),'recommend_integration':all(r['global_equal'] and r['word_equal'] for r in rows),'note':'Only CPU float32 was tested; helper is intentionally rejected for GPU or other dtype.'}
    p=Path('output/hhc_sota_20261016/cpu_text_cache_audit.json'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
