"""Verify regenerated MM requests against immutable main requests and source IDs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    validation=ROOT/'output/interhuman_validation_requests/validation.json'
    assert sha(validation)=='2053a04a5f6169ef9ae52ecb286c1d4dcc0df1de3a21aaa01024a801c0c2308c'
    assert sha(ROOT/'dataset/interhuman_official/split/test.txt')=='c34e91a66e51ec1c9283b026e632ae330431e568ece0821858c5b2fb212f1dd1'
    v=json.loads(validation.read_text())
    vm_path=ROOT/'output/interhuman_validation_mm_requests/validation_mm.json'
    vm=json.loads(vm_path.read_text())
    assert vm['split']=='val' and vm['prompts']==100 and vm['repeats']==11 and vm['pairs']==10
    assert vm['source_validation_request_sha256']==sha(validation)
    assert vm['request_sha256']==canonical(vm['requests'])
    assert [{k:value for k,value in row.items() if k!='prompt_index'} for row in vm['requests']]==v['requests'][:100]
    main_manifest=json.loads((ROOT/'output/interhuman_fair_requests_v3/manifest.json').read_text())
    mm_manifest=json.loads((ROOT/'output/interhuman_fair_mm_requests/manifest.json').read_text())
    assert mm_manifest['status']=='complete' and len(mm_manifest['replications'])==20
    entries=[]
    for rep,entry in enumerate(mm_manifest['replications']):
        path=Path(entry['path']);assert sha(path)==entry['sha256']
        mm=json.loads(path.read_text())
        assert mm['request_sha256']==entry['request_sha256']==canonical(mm['requests'])
        assert mm['full_effective_entries']==1098 and len(mm['requests'])==100
        assert len({r['identifier'] for r in mm['requests']})==100
        main_entry=main_manifest['replications'][rep]
        main_path=Path(main_entry['path']);assert sha(main_path)==main_entry['sha256']
        ordinary=json.loads(main_path.read_text())['requests']
        assert len(ordinary)==1056
        matched=0
        for row in mm['requests']:
            position=row['source_position_in_full_1098_order']
            assert 0<=position<1098
            if position<len(ordinary):
                assert all(row[key]==value for key,value in ordinary[position].items() if key!='position')
                matched+=1
        entries.append({'replication':rep,'overlap_with_frozen_main':matched,'mm_request_sha256':entry['request_sha256']})
    report={'status':'passed','validation_prompts':100,'validation_request_sha256':vm['request_sha256'],
            'formal_repetitions':20,'formal_prompts':2000,'matching_frozen_main_requests':sum(e['overlap_with_frozen_main'] for e in entries),
            'entries':entries,'all_main_request_hashes_unchanged':True,
            'original_mm_file_byte_identity_proven':False,
            'limits':'Original MM files were not recovered. Deterministic generators re-executed; content hashes now frozen. Main-overlap checks prove agreement on shared requests; they do not prove missing original file hashes.'}
    destination=ROOT/'output/interhuman_fair_mm_requests/restoration_audit.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
