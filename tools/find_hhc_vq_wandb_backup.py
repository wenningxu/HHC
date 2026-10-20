"""Read public W&B run file metadata only; never download weights locally."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    query='query { project(name:"NewInter",entityName:"edmond_gla") { run(name:"knk4gll9") { files(first:100) { edges { node { name url } } pageInfo { endCursor hasNextPage } } } } }'
    request=urllib.request.Request('https://api.wandb.ai/graphql',data=json.dumps({'query':query}).encode(),
                                   headers={'Content-Type':'application/json','User-Agent':'HHC-backup-metadata-audit'})
    try:
        with urllib.request.urlopen(request,timeout=30) as response:
            payload=json.load(response)
        value={'http_status':200,'public_unauthenticated_request':True,'payload':payload,
               'checkpoint_downloaded':False,'gpu_work':False}
    except urllib.error.HTTPError as error:
        value={'http_status':error.code,'error':error.read().decode('utf-8',errors='replace')[:2000],
               'checkpoint_downloaded':False,'gpu_work':False}
    except Exception as error:
        value={'error':repr(error),'checkpoint_downloaded':False,'gpu_work':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
