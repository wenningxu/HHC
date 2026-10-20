"""Prepare 64MiB parts for the connector transport's effective 100MiB limit."""
import argparse,hashlib,json,pathlib,os
def main():
 p=argparse.ArgumentParser();p.add_argument('file');args=p.parse_args()
 source=pathlib.Path(args.file).resolve();dest=source.with_name(source.name+'.parts');dest.mkdir(exist_ok=True)
 whole=hashlib.sha256();parts=[]
 with source.open('rb') as f:
  i=0
  while True:
   block=f.read(64*1024*1024)
   if not block:break
   i+=1;whole.update(block);part=dest/f'{source.name}.part{i:03d}'
   sha=hashlib.sha256(block).hexdigest()
   if not part.exists() or part.stat().st_size!=len(block) or hashlib.file_digest(part.open('rb'),'sha256').hexdigest()!=sha:
    tmp=part.with_suffix('.tmp');tmp.write_bytes(block);os.replace(tmp,part)
   parts.append({'index':i,'file':str(part),'bytes':len(block),'sha256':sha,'cloud_status':'pending'})
 manifest={'schema':'hhc_drive_split_v1','source_name':source.name,'bytes':source.stat().st_size,'sha256':whole.hexdigest(),'parts':parts,'reassembly':'Concatenate parts in index order as binary, then verify full bytes/SHA256 before torch.load.'}
 receipt=source.with_suffix(source.suffix+'.receipt.json')
 if receipt.exists():
  expected=json.loads(receipt.read_text());assert manifest['bytes']==expected['bytes'] and manifest['sha256']==expected['sha256']
 target=dest/'manifest.json'
 if target.exists():
  old=json.loads(target.read_text())
  for row,previous in zip(parts,old['parts']):
   if row['sha256']==previous['sha256']:row.update({k:v for k,v in previous.items() if k.startswith('cloud_')})
 target.write_text(json.dumps(manifest,indent=2)+'\n');print(str(target),flush=True)
if __name__=='__main__':main()
