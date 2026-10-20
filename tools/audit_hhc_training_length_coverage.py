import hashlib, json, random
from pathlib import Path
import h5py, numpy as np
DATA=Path('/workspace/HHC/dataset/interhuman_smpl_interx_layout_v1_20261004')
def crop(name,n,unit=4,seed=114514):
 d=hashlib.sha256(f'{seed}\0{name}'.encode()).digest(); s=int.from_bytes(d[:4],'little'); nr=np.random.RandomState(s); pr=random.Random(s); mode=nr.choice(['single','single','double']); u=n//unit-(1 if mode=='double' else 0); l=u*unit; st=pr.randint(0,n-l); return mode,u,l,st
def main():
 ids=(DATA/'split/train.txt').read_text().splitlines(); rows=[]; missing=0
 with h5py.File(DATA/'motions.h5','r') as h:
  for name in ids:
   if name not in h or 'motion30' not in h[name]: missing+=1; continue
   n=int(h[name]['motion30'].shape[0]); mode,u,l,st=crop(name,n); rows.append({'name':name,'source_frames':n,'crop_mode':str(mode),'crop_units':u,'crop_frames':l,'crop_start':st,'first75_end_frame_exclusive':st+min(u,75)*4,'dropfirst_max_original_token':75 if u>=76 else (u-1 if u else None)})
 def c(f): return sum(bool(f(r)) for r in rows)
 sm={'count':len(rows),'source_min':min((r['source_frames'] for r in rows),default=None),'source_max':max((r['source_frames'] for r in rows),default=None),'source_lt30':c(lambda r:r['source_frames']<30),'source_ge1000':c(lambda r:r['source_frames']>=1000),'crop_units_min':min((r['crop_units'] for r in rows),default=None),'crop_units_max':max((r['crop_units'] for r in rows),default=None),'crop_units_gt75':c(lambda r:r['crop_units']>75),'visible_original_token75':c(lambda r:r['crop_units']>=76),'tail_beyond_first75_tokens':c(lambda r:r['crop_units']>75)}
 out={'schema':'hhc_training_length_coverage_v2','data':str(DATA),'split':'train','cache_seed':114514,'cache_rule':'build_tokens keeps 30<=source motion30<1000; crop units=floor(n/4), minus 1 for double; TokenData tokens[:75], optional dropfirst exposes original token 75','summary':sm,'missing_groups':missing,'rows':rows}; print(json.dumps(out,separators=(',',':')))
if __name__=='__main__': main()
