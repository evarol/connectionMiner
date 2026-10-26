"""Immutable eight-view inputs plus resident alternatives and leakage-aware splits."""
from pathlib import Path
import hashlib,json
from functools import lru_cache
import numpy as np,pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'outputs/model_validation_study';PRE=ROOT/'outputs/gene_panels_motor_expansion_pilot';EIGHT=ROOT/'outputs/eight_panel_morphology'
STAGES=['joint_P20MN_P24preMN','joint_P45MN_P48preMN','joint_Adult_local_export']
KEYS=['A1','B1','C1','A2','B2','C2','D','E','A3','B3','C3']
PANELS=['strict_hvg2000_all25','strict_hvg161','local_curated_synaptic','consensus_hvg2000']
@lru_cache(None)
def gene_indices(panel):
 manifest=json.loads((PRE/'genes/panels_manifest.json').read_text());p=next(x for x in manifest['panels'] if x['key']==panel)
 genes=pd.read_csv(PRE/'genes/full_common_current.csv',keep_default_na=False);pub=pd.read_csv(PRE/'pilot_constraints/joint_Adult_local_export/gene_expression_GEO/genes.csv')
 keep=set(pub.full_common_column);ix=np.array([i for i in p['indices'] if i in keep],dtype=int)
 return ix,genes.iloc[ix].FBgn.to_list()
def paths(key):
 if key in ['D','E']:
  family,ds=('visual','Adult') if key=='D' else ('central_complex','CX_Adult');b=PRE/'bundles'/family/ds;g=PRE/'gene_expression'/(family+'__'+ds)/'G.npz';indices=np.arange(741 if key=='D' else 274);cw=b/'C.npy';ww=b/'W_measured.npy'
 else:
  ds=STAGES['ABC'.index(key[0])];family='motor';version=int(key[1]);b=PRE/'bundles/motor'/ds if version==1 else PRE/'pilot_constraints'/ds
  if version==3:b=b/'resident'
  g=PRE/'gene_expression'/('motor__'+ds)/'G.npz';indices=np.arange(730 if version==1 else 1617)
  if key[0]=='C' and version!=1:g=PRE/'pilot_constraints/joint_Adult_local_export/gene_expression_GEO/G.npz'
  if version==3:
   indices=np.load(PRE/'motor_connectome/resident_pilot/all_class_indices.npy');cw=PRE/'motor_connectome/resident_pilot/C.npy';ww=PRE/'motor_connectome/resident_pilot/W_measured.npy'
  else:cw=b/'C.npy';ww=b/'W_measured.npy'
 return dict(key=key,family=family,dataset=ds,C=cw,W=ww,D=b/'P_constraint.npz',G=g,canonical_indices=indices,membership=b/'single_cell_to_metacell.csv.gz')
@lru_cache(None)
def entity_groups(family):
 m=json.loads((EIGHT/'morphology'/family/'geometry_index.json').read_text());n=len(m['targets']);parent=np.arange(n)
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return int(i)
 seen={}
 for t in m['targets']:
  i=t['canonical_index']
  # Sharing any contributor/geometry joins split groups conservatively, including fractional MN contributors.
  ids=list(t.get('target_neuron_ids',[]))+list(t['neuron_ids'])
  for token in ids:
   if token in seen:
    a,b=find(i),find(seen[token]);parent[max(a,b)]=min(a,b)
   else:seen[token]=i
 return np.array([find(i) for i in range(n)],dtype=np.uint64)
def fold_ids(key,scheme='dyad',seed=20260907):
 p=paths(key);g=entity_groups(p['family'])[p['canonical_indices']];a=np.minimum(g[:,None],g[None,:]);b=np.maximum(g[:,None],g[None,:]);salt={'motor':17,'visual':31,'central_complex':43}[p['family']]
 if scheme=='block':
  blocks=(g*np.uint64(11400714819323198485)+np.uint64(seed+salt))%np.uint64(5)
  return ((blocks[:,None]+blocks[None,:])%5).astype('uint8')
 h=(a*np.uint64(11400714819323198485))^(b*np.uint64(14029467366897019727))^np.uint64(seed+salt)
 h^=h>>np.uint64(30);h*=np.uint64(13787848793156543929);h^=h>>np.uint64(27)
 return (h%5).astype('uint8')
def load_dataset(key,panel='strict_hvg161',scheme='dyad',fold=0,split_seed=20260907):
 p=paths(key);ix,genes=gene_indices(panel);G=sparse.load_npz(p['G']).tocsr()
 if p['G'].parent.name=='gene_expression_GEO':
  pub=pd.read_csv(p['G'].parent/'genes.csv');look=dict(zip(pub.full_common_column,range(len(pub))));ix=np.array([look[i] for i in ix])
 G=G[:,ix].toarray().astype('float32');D=sparse.load_npz(p['D']).tocsr().astype('float32');C=np.load(p['C']).astype('float32');W=np.load(p['W']).astype(bool)
 support=np.asarray(D.sum(1)).ravel()>0;W&=support[:,None]&support[None,:]
 # Self-connections are not used to evaluate pairwise generalization.
 np.fill_diagonal(W,False)
 f=fold_ids(key,scheme,split_seed);train=W&(f!=fold)&(f!=((fold+1)%5));val=W&(f==((fold+1)%5));test=W&(f==fold)
 return dict(key=key,family=p['family'],dataset=p['dataset'],canonical_indices=p['canonical_indices'],genes=genes,G=G,D=D,C=C,measured_mask=np.load(p['W']).astype(bool),eligible_mask=W,train_mask=train,val_mask=val,test_mask=test,fold_ids=f,weight=1.0,scheme=scheme,fold=fold,source_paths={k:str(v) for k,v in p.items() if isinstance(v,Path)})
def set_family_weights(data):
 for d in data:d['weight']=1.0/(sum(x['family']==d['family'] for x in data)*max(1,int(d['train_mask'].sum())))
 return data
def cross_family_purge(data,heldout_family):
 path=OUT/'checks/shared_visual_cx_roots.json'
 if heldout_family=='motor' or not path.exists():return data,[]
 # The overlap contains identical canonical names in both families; derive them from exact primary root membership.
 visual=pd.read_csv(EIGHT/'morphology/visual/type_neuron_morphology_crosswalk.tsv.gz',sep='\t',dtype=str);cx=pd.read_csv(EIGHT/'morphology/central_complex/target_neuron_swc_crosswalk.tsv',sep='\t',dtype=str)
 cx=cx[cx.is_default_geometry.str.lower().eq('true')]
 vroot=next(c for c in ['root_id','neuron_id'] if c in visual);overlap=set(visual[vroot])&set(cx.root_id)
 excluded=[]
 for d in data:
  if d['family'] not in ['visual','central_complex']:continue
  meta=json.loads((EIGHT/'morphology'/d['family']/'geometry_index.json').read_text());bad={t['canonical_index'] for t in meta['targets'] if overlap&set(t['neuron_ids'])};q=np.isin(d['canonical_indices'],list(bad));mask=~(q[:,None]|q[None,:]);d['train_mask']&=mask;d['val_mask']&=mask;excluded.append(dict(key=d['key'],excluded_target_indices=np.flatnonzero(q).tolist(),overlap_roots=len(overlap)))
 return data,excluded

def metrics(C,pred,mask):
 from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
 y=np.asarray(C[mask],dtype=float);z=np.asarray(pred[mask],dtype=float);good=np.isfinite(z);y=y[good];z=z[good];q=np.clip(z,1e-6,1-1e-6)
 return {'n':len(y),'positive':int(y.sum()),'prevalence':float(y.mean()) if len(y) else None,'average_precision':float(average_precision_score(y,z)) if len(np.unique(y))==2 else None,'roc_auc':float(roc_auc_score(y,z)) if len(np.unique(y))==2 else None,'brier':float(np.mean((np.clip(z,0,1)-y)**2)) if len(y) else None,'raw_mse':float(np.mean((z-y)**2)) if len(y) else None,'log_loss':float(log_loss(y,q,labels=[0,1])) if len(y) else None,'excluded_nonfinite':int((~good).sum())}
if __name__=='__main__':
 rows=[]
 for key in KEYS:
  d=load_dataset(key);out=OUT/'splits'/key;out.mkdir(exist_ok=True,parents=True)
  for scheme in ['dyad','block']:np.save(out/(scheme+'_fold_ids.npy'),fold_ids(key,scheme))
  rows.append(dict(key=key,family=d['family'],G=list(d['G'].shape),P=list(d['D'].shape),C=list(d['C'].shape),eligible=int(d['eligible_mask'].sum()),train=int(d['train_mask'].sum()),val=int(d['val_mask'].sum()),test=int(d['test_mask'].sum()),positive=int(d['C'][d['eligible_mask']].sum())))
 (OUT/'dataset_inventory.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
