#!/usr/bin/env python3
"""Audit visual sidecars without decoding every image payload."""
from __future__ import annotations
import argparse,json,re,struct,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
PAT=re.compile(r"task(?P<task>\d+)_episode(?P<episode>\d+)_.*_(?P<outcome>success|failure)\.npz$")
def npy_header(zf,name):
 with zf.open(name) as f:
  version=np.lib.format.read_magic(f)
  shape,fortran,dtype=np.lib.format._read_array_header(f,version)
  return {'shape':list(shape),'dtype':str(dtype),'fortran_order':bool(fortran)}
def suite(path):
 n=path.name
 return 'libero_spatial' if 'libero_spatial' in n else 'libero_90' if 'libero_90' in n else 'unknown'
def main():
 p=argparse.ArgumentParser();p.add_argument('--sidecar-directory',type=Path,required=True);p.add_argument('--trace',type=Path,action='append',default=[]);p.add_argument('--trace-directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();possible=defaultdict(list)
 trace_paths=list(a.trace)
 if a.trace_directory is not None:trace_paths.extend(sorted(a.trace_directory.glob('*.jsonl')))
 if not trace_paths:raise ValueError('at least one trace is required')
 for path in trace_paths:
  for line in path.open(encoding='utf-8'):
   r=json.loads(line)
   if r.get('event')=='episode_end':possible[(int(r['task_id']),int(r['episode_idx']),'success' if r['success'] else 'failure')].append({'suite':suite(path),'trace':str(path),'steps':int(r['executed_actions'])})
 files=[]
 for path in sorted(a.sidecar_directory.glob('*.npz')):
  m=PAT.search(path.name)
  if not m:continue
  key=(int(m['task']),int(m['episode']),m['outcome']);candidates=possible.get(key,[])
  with zipfile.ZipFile(path) as zf:
   headers={name[:-4]:npy_header(zf,name) for name in ('action_indices.npy','agent_images.npy','wrist_images.npy')}
   with zf.open('action_indices.npy') as f:indices=np.load(f,allow_pickle=False)
  shape_ok=(headers['agent_images']['shape']==headers['wrist_images']['shape'] and headers['agent_images']['shape'][0]==len(indices) and headers['agent_images']['shape'][1:]==[224,224,3])
  contiguous=bool(np.array_equal(indices,np.arange(len(indices))))
  unique=len(candidates)==1 and candidates[0]['steps']==len(indices)
  files.append({'path':str(path),'key':{'task_id':key[0],'episode_idx':key[1],'outcome':key[2]},'headers':headers,'frames':len(indices),'shape_ok':shape_ok,'indices_contiguous':contiguous,'candidate_traces':candidates,'mapping_status':'unique' if unique else 'ambiguous_or_unmatched'})
 result={'files':len(files),'unique_mappings':sum(x['mapping_status']=='unique' for x in files),'ambiguous_or_unmatched':sum(x['mapping_status']!='unique' for x in files),'shape_failures':sum(not x['shape_ok'] for x in files),'index_failures':sum(not x['indices_contiguous'] for x in files),'records':files}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
