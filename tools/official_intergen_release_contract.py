"""Pure CPU evidence contract for releasing saved HHC after official reproduction."""
import math
import re
import statistics

FLOAT=r'(?:-?(?:[0-9]+(?:[.][0-9]*)?|[.][0-9]+)(?:[eE][+-]?[0-9]+)?|nan|inf|-inf)'
SCALARS={'MM Distance':'Matching','FID':'FID','Diversity':'Diversity','Multimodality':'Multimodality'}
SUMMARY={'MM Distance':'Matching','R_precision':'R_precision','FID':'FID','Diversity':'Diversity','MultiModality':'Multimodality'}
WAITING={'waiting_for_official_training','waiting_for_official_final_checkpoint','waiting_for_official_reproduction_review','waiting_for_gpu_headroom'}


def waiting_recovery_allowed(previous,attributed_waiter_live,has_dispatch_intents):
    """A durable intent fences replay even if a crash left the status unchanged."""
    return bool(previous and previous.get('status') in WAITING
                and not attributed_waiter_live and not has_dispatch_intents
                and not previous.get('phase') and not previous.get('phase_exit_codes'))


def require(condition,message):
    if not condition:raise ValueError(message)


def moments(values):
    return {'mean':statistics.mean(values),'ci95':1.96*statistics.pstdev(values)/math.sqrt(len(values))}


def parse_official_reproduction_log(text):
    """Require actual per-rep metrics and consistent original 20-rep summaries."""
    require('Traceback' not in text,'Evaluation log contains a traceback')
    starts=list(re.finditer(r'^=+\s*Replication\s+(\d+)\s*=+\s*$',text,re.MULTILINE))
    require([int(x[1]) for x in starts]==list(range(20)),'Need all20 distinct ordered replications')
    rows=[]
    for i,start in enumerate(starts):
        block=text[start.end():starts[i+1].start() if i<19 else len(text)]
        row={}
        for label,key in SCALARS.items():
            found=re.findall(r'^---> \[InterGen\] '+re.escape(label)+r':\s*('+FLOAT+r')\s*$',block,re.MULTILINE)
            require(len(found)==1,'Missing/duplicate '+label+' in rep'+str(i))
            value=float(found[0]);require(math.isfinite(value) and value>=(-1e-5 if key=='FID' else 0),'Nonfinite/negative '+key)
            row[key]=value
        line=re.findall(r'^---> \[InterGen\] R_precision:(.*)$',block,re.MULTILINE)
        require(len(line)==1,'Missing/duplicate retrieval metrics')
        found=re.findall(r'\(top ([123])\):\s*('+FLOAT+r')',line[0])
        require([int(k) for k,v in found]==[1,2,3],'Need all3 retrieval ranks')
        values=[float(v) for k,v in found]
        require(all(math.isfinite(v) and 0<=v<=1 for v in values) and values==sorted(values),'Invalid retrieval values')
        row['R_precision']=values;rows.append(row)
    computed={key:moments([row[key] for row in rows]) for key in SCALARS.values()}
    computed['R_precision']=[moments([row['R_precision'][j] for row in rows]) for j in range(3)]
    for label,key in SUMMARY.items():
        section=re.search(r'^=+ '+re.escape(label)+r' Summary =+\s*$',text,re.MULTILINE)
        require(section is not None,'Missing '+label+' summary')
        tail=text[section.end():];next_header=re.search(r'^=+ .* Summary =+\s*$',tail,re.MULTILINE)
        block=tail[:next_header.start()] if next_header else tail
        line=re.findall(r'^---> \[InterGen\](.*)$',block,re.MULTILINE)
        require(len(line)==1,'Missing/duplicate InterGen summary')
        if key=='R_precision':
            parsed=re.findall(r'\(top ([123])\) Mean:\s*('+FLOAT+r') CInt:\s*('+FLOAT+r')',line[0])
            require([int(k) for k,a,b in parsed]==[1,2,3],'Incomplete retrieval summary')
            for j,(_,mean,ci) in enumerate(parsed):
                require(abs(float(mean)-computed[key][j]['mean'])<=.00011 and abs(float(ci)-computed[key][j]['ci95'])<=.00011,'Retrieval summary inconsistent with replications')
        else:
            found=re.findall(r'Mean:\s*('+FLOAT+r') CInterval:\s*('+FLOAT+r')',line[0])
            require(len(found)==1,'Invalid scalar summary')
            mean,ci=map(float,found[0]);require(abs(mean-computed[key]['mean'])<=.00011 and abs(ci-computed[key]['ci95'])<=.00011,'Summary inconsistent with replications')
    return {'replications':20,'rows':rows,'metrics':computed}


def validate_release_evidence(final,review,report,log_text,report_sha,log_sha,config):
    """Positive result requires final weights AND independently reviewed reproduction."""
    require(final.get('status')=='verified_final' and final.get('step')==704000,'Not final704k')
    require(final.get('input_dim')==262 and final.get('source_commit')==config['source_commit'],'Wrong baseline model identity')
    require(final.get('runtime_wrapper_sha256')==config['runtime_wrapper_sha256'],'Wrong runtime wrapper')
    require(final.get('attributed_trainer_live') is False,'Training still live')
    require(final.get('cpu_only') is True and final.get('cpu_cuda_initialized') is False,'Final acceptance was not CPU-only')
    require(review.get('status')=='official_intergen_reproduction_accepted','Reproduction has not been reviewed')
    require(review.get('checkpoint_sha256')==report.get('checkpoint_sha256')==final['sha256'],'Checkpoint identity differs across review/report/final')
    require(review.get('report_sha256')==report_sha and report.get('raw_log_sha256')==log_sha,'Report/log changed after acceptance')
    require(report.get('status')=='complete' and report.get('protocol')=='official_intergen_eval20','Missing completed official reproduction')
    for key,value in {'replication_times':20,'batch_size':96,'main_samples_per_rep':1056,'mm_num_samples':100,'mm_num_repeats':30}.items():
        require(report.get(key)==value,'Wrong evaluation protocol '+key)
    require(report.get('source_commit')==config['source_commit'],'Wrong evaluation source revision')
    require(report.get('eval_source_sha256')==config['eval_source_sha256'],'Wrong evaluation script')
    require(report.get('evaluator_checkpoint_sha256')==config['evaluator_checkpoint_sha256'],'Wrong evaluator weights')
    parsed=parse_official_reproduction_log(log_text)
    for key,value in parsed['metrics'].items():
        require(key in report.get('metrics',{}),'Missing report metric '+key)
        expected=value if isinstance(value,list) else [value]
        actual=report['metrics'][key] if isinstance(value,list) else [report['metrics'][key]]
        require(len(expected)==len(actual),'Wrong metric rank count')
        for a,b in zip(actual,expected):
            require(all(math.isfinite(float(a[field])) and abs(float(a[field])-b[field])<1e-8 for field in ('mean','ci95')),'Parsed report metrics differ from raw log')
    return parsed
