"""Cloud CPU check: exact pinned weights, finite encodings, fail-before-unpickle."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from models import interhuman_evaluator_wrapper as evaluator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2)
    model,config=evaluator.load_official_interclip('cpu')
    checkpoint=torch.load(evaluator.CHECKPOINT_PATH,map_location='cpu',weights_only=False,mmap=True)
    expected={key.removeprefix('model.'):value for key,value in checkpoint['state_dict'].items()}
    actual=model.state_dict()
    assert set(expected)==set(actual)
    assert all(torch.equal(value,actual[key]) for key,value in expected.items())
    with torch.inference_mode():
        z=model.encode_text({'text':['Two people shake hands.','Two people bow to each other.']})['text_emb']
        assert z.shape==(2,512) and torch.isfinite(z).all()
    original=evaluator._require_hash
    def reject(path,expected_hash):
        if path==evaluator.CHECKPOINT_PATH:raise ValueError('forged checkpoint rejected')
        return original(path,expected_hash)
    with patch.object(evaluator,'_require_hash',side_effect=reject),patch.object(evaluator.torch,'load') as unpickle:
        try:evaluator.load_official_interclip('cpu')
        except ValueError as error:assert str(error)=='forged checkpoint rejected'
        else:raise AssertionError('Tampered checkpoint guard did not reject')
        unpickle.assert_not_called()
    tests=runpy.run_path(str(ROOT/'tests/test_interhuman_evaluator_wrapper.py'))
    names=[key for key,value in tests.items() if key.startswith('test_') and callable(value)]
    for name in names:tests[name]()
    result={'status':'passed','device':'cpu','torch_version':torch.__version__,
            'state_tensors_exactly_equal_to_pinned_checkpoint':len(actual),
            'text_embedding_shape':[2,512],'tampered_hash_rejected_before_unpickling':True,
            'existing_wrapper_regressions':names,'checkpoint_sha256':evaluator.CHECKPOINT_SHA256,
            'wrapper_sha256':evaluator.sha256_file(Path(evaluator.__file__)),
            'evaluator_math_changed':False,'is_quality_evaluation':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
