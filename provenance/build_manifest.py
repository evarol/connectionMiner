"""Hash existing matrix/crosswalk artifacts without modifying their contents."""
from pathlib import Path
import sys,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/model_validation_study/scripts'))
from study_data import paths,KEYS,PRE,EIGHT,gene_indices,PANELS
import numpy as np
from scipy import sparse
cache={}
def artifact(p):
 p=Path(p);r=p.resolve()
 if str(r) not in cache:
  h=hashlib.sha256()
  with r.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  cache[str(r)]={'resolved_path':str(r),'bytes':r.stat().st_size,'sha256':h.hexdigest()}
 return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),**cache[str(r)]}
rows=[]
for key in KEYS:
 p=paths(key);row={'key':key,'family':p['family'],'dataset':p['dataset'],'canonical_indices':p['canonical_indices'].tolist(),'artifacts':{}}
 for name in ['C','W','D','G','membership']:
  a=artifact(p[name])
  if name in ['D','G']:a['shape']=list(sparse.load_npz(p[name]).shape)
  if name in ['C','W']:a['shape']=list(np.load(p[name],mmap_mode='r').shape)
  row['artifacts'][name]=a
 rows.append(row)
crosswalks=[]
for family,names in {
 'motor':['target_neuron_swc_crosswalk.tsv','geometry_inventory.tsv','source_file_versions.tsv','geometry_index.json'],
 'visual':['type_neuron_morphology_crosswalk.tsv.gz','geometry_index.json'],
 'central_complex':['target_neuron_swc_crosswalk.tsv','neuron_manifest.tsv','geometry_index.json']}.items():
 for name in names:crosswalks.append({'family':family,**artifact(EIGHT/'morphology'/family/name)})
out={'schema_version':1,'date':'2026-09-07','source_code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'scope':'Immutable curated inputs; hashes identify bytes, not independent biological validation. External source paths require the original local data.',
 'axes':{'C':'presynaptic target entry × postsynaptic target entry','W':'measured C indicator','D':'target entry × metacell; binary permission, not fitted P','G':'metacell × gene; fraction of constituent cells positive in original expression','membership':'single cell → exact metacell'},
 'datasets':rows,'morphology_crosswalks':crosswalks,
 'gene_definitions':[artifact(PRE/'genes/full_common_current.csv'),artifact(PRE/'genes/panels_manifest.json'),artifact(PRE/'pilot_constraints/joint_Adult_local_export/gene_expression_GEO/genes.csv')],
 'experiment_gene_panels':{k:{'n_genes':len(gene_indices(k)[1]),'FBgn':gene_indices(k)[1]} for k in PANELS}}
(ROOT/'provenance/curated_inputs_2026-09-07.json').write_text(json.dumps(out,indent=2)+'\n')
print(f'Hashed {len(cache)} distinct files across {len(rows)} views.')
