"""Cluster bootstrap by motion; perturbation directions are repeated measures."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(report):
    assert report['status'] == 'complete'
    sample_count = len(report['requests'])
    draw = np.random.default_rng(20261016).integers(0, sample_count, (10000, sample_count))
    results = []
    for epsilon in (.02, .1):
        rows = [r for r in report['rows'] if r['epsilon'] == epsilon]
        groups = sorted({r['group'] for r in rows})
        vectors = {g: np.array([r['unit_embedding_l2'] for r in rows if r['group']==g]).mean(0) for g in groups}
        measures = {}
        for group, vector in vectors.items():
            means = vector[draw].mean(1)
            measures[group] = {'mean':float(vector.mean()),'ci95_percentile':np.quantile(means,[.025,.975]).tolist()}
        ratios = {}
        for numerator, denominator in [('rotation_so3','position'),('rotation_6d_gaussian','rotation_so3')]:
            ratios[numerator+'/'+denominator] = {
                'ratio_of_means':float(vectors[numerator].mean()/vectors[denominator].mean()),
                'ci95_percentile':np.quantile(vectors[numerator][draw].mean(1)/vectors[denominator][draw].mean(1),[.025,.975]).tolist()}
        results.append({'epsilon':epsilon,'unit_embedding_l2':measures,'ratios':ratios})
    return {'status':'complete','sample_count':sample_count,'directions':report['directions'],
            'length_min':min(report['lengths']),'length_max':max(report['lengths']),
            'bootstrap':'10000 paired draws, cluster=motion, average directions within motion first; seed20261016',
            'results':results,'is_fid':False,'is_model_improvement':False,
            'interpretation':'Sensitivity depends on perturbation geometry. Do not infer loss weights, model-error contributions, general feature importance, or method rankings from these probes.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(json.loads(args.report.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
