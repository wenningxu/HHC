"""Stdlib AST audit that recovery infrastructure retains original VQ math."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=[ROOT/'train/train_vq_interhuman_smpl_raw.py',ROOT/'tools/train_hhc_native_vq_cloud.py']
    trees=[ast.parse(path.read_text(encoding='utf-8')) for path in paths]
    def function(tree,name):return next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
    def dump(node):return ast.dump(node,include_attributes=False)
    assert dump(function(trees[0],'validate'))==dump(function(trees[1],'validate'))
    originals=[node for node in ast.walk(function(trees[0],'main')) if isinstance(node,ast.Assign)]
    rebuilt=[node for node in ast.walk(function(trees[1],'main')) if isinstance(node,ast.Assign)]
    for name in ('components','loss','optimizer'):
        expected=next(node for node in originals if any(isinstance(t,ast.Name) and t.id==name for t in node.targets))
        actual=next(node for node in rebuilt if any(isinstance(t,ast.Name) and t.id==name for t in node.targets))
        assert dump(expected)==dump(actual),name
    report={'status':'passed','gpu_work':False,'checkpoint_loaded':False,
        'validation_function_ast_identical':True,'loss_and_optimizer_assignment_ast_identical':True,
        'sources':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'limits':'Checks copied validation, loss expression and optimizer, not complete runtime equivalence. New independent resumable batch order and CPU validation are explicitly recorded.'}
    output=ROOT/'output/hhc_sota_20261016/vq_rebuild_source_audit.json'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
