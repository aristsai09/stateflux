from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import urllib.parse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def safe(x):
    if x is None:
        return ''
    try:
        if pd.isna(x):
            return ''
    except Exception:
        pass
    return str(x).strip()


def norm_name(x):
    s = safe(x).lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def find_root() -> Path:
    here = Path(__file__).resolve()
    cands = [Path.cwd(), here.parents[1], here.parents[2] if len(here.parents) > 2 else here.parent]
    for p0 in list(cands):
        cands += list(p0.parents[:4])
    seen = set()
    for p in cands:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if (p / 'output_round03' / 'tables' / 'ROUND03_SUMMARY.json').exists() and (p / 'scripts' / 'p05_round03.py').exists():
            return p
    raise FileNotFoundError('Round03 project root not found. Merge this rescue package INTO the existing P05 project root.')


ROOT = find_root()
OUT = ROOT / 'output_round03b'
TABLES = OUT / 'tables'
MANIFESTS = OUT / 'manifests'
LOGS = OUT / 'logs'
CACHE = OUT / 'cache'
METAB_NEW = OUT / 'metabolism_new'
XTB_NEW = OUT / 'xtb_dili_new'
STATES = OUT / 'states'
MODELS = OUT / 'models'
for d in [OUT, TABLES, MANIFESTS, LOGS, CACHE, METAB_NEW, XTB_NEW, STATES, MODELS]:
    d.mkdir(parents=True, exist_ok=True)
LOGFILE = LOGS / 'pipeline.log'


def log(msg):
    s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(s, flush=True)
    with LOGFILE.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


@contextlib.contextmanager
def stage(name):
    t = time.time()
    log(f'========== START {name} ==========')
    try:
        yield
    except Exception as e:
        log(f'FATAL {name}: {type(e).__name__}: {e}')
        log(traceback.format_exc())
        raise
    finally:
        log(f'========== END {name} ({(time.time()-t)/60:.1f} min) ==========')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return {} if default is None else default


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    tmp.replace(p)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_round03_module():
    src = ROOT / 'scripts' / 'p05_round03.py'
    spec = importlib.util.spec_from_file_location('p05_round03_base', src)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    # Redirect helper outputs so Round03 itself is never overwritten.
    mod.ROOT = ROOT
    mod.OUT = OUT
    mod.TABLES = TABLES
    mod.MANIFESTS = MANIFESTS
    mod.LOGS = LOGS
    mod.CACHE = CACHE
    mod.METAB_NEW = METAB_NEW
    mod.XTB_DILI = XTB_NEW
    mod.STATES = STATES
    mod.MODELS = MODELS
    mod.LOGFILE = LOGFILE
    mod.log = log
    return mod


R3 = load_round03_module()

# Predeclared, chemistry-motivated features. No outcome-guided feature selection is performed.
STATE_CORE = [
    'parent_electrophilicity_eV',
    'parent_vip_eV',
    'parent_vea_eV',
    'parent_homo_lumo_gap_eV',
    'parent_fukui_plus_absmax',
    'parent_local_electrophilicity_proxy_eV',
    'microstate_range_electrophilicity_eV',
    'met_max_electrophilicity_eV',
]

FLUX_CORE = [
    'max_reactive_alert_gain',
    'phase1_delta_max_electrophilicity_eV',
    'phase1_positive_flux_electrophilicity_eV',
    'phase2_negative_flux_electrophilicity_eV',
    'path_peak_gain_max_electrophilicity_eV',
    'path_cumulative_abs_max_electrophilicity_eV',
    'path_net_min_vip_eV',
    'path_peak_gain_max_vea_eV',
    'step_delta_absmax_fukui_plus_absmax',
    'path_peak_gain_max_local_electrophilicity_proxy_eV',
    'path_cumulative_abs_max_gsolv_Eh',
    'metabolic_reactive_product_fraction',
    'metabolic_phase2_fraction',
    'max_quantum_path_length',
    'reachable_quantum_paths',
    'edge_count_computed',
]

SALT_SUFFIXES = [
    ' hydrochloride',' dihydrochloride',' hydrobromide',' sulfate',' bisulfate',' hemisulfate',
    ' mesylate',' dimesylate',' esylate',' tosylate',' sodium',' disodium',' potassium',' calcium',
    ' magnesium',' acetate',' succinate',' fumarate',' maleate',' tartrate',' citrate',' phosphate',
    ' monohydrate',' dihydrate',' trihydrate',' hydrate',' lactate',' besylate',' camsylate',' napsylate',
    ' oxalate',' gluconate',' palmitate',' tromethamine',' meglumine',' benzoate',' dimaleate'
]


def name_variants(name):
    vals = [safe(name)]
    cur = safe(name)
    changed = True
    while changed and cur:
        changed = False
        lo = cur.lower()
        for suf in SALT_SUFFIXES:
            if lo.endswith(suf):
                cur = cur[:-len(suf)].strip(' ,;-')
                vals.append(cur)
                changed = True
                break
    return list(dict.fromkeys(v for v in vals if v))


def offline_structure_map():
    p = ROOT / 'data' / 'DILImap_metadata.xlsx'
    mp = {}
    if not p.exists():
        return mp
    try:
        d = pd.read_excel(p, sheet_name=0)
        if {'compound_name', 'smiles'}.issubset(d.columns):
            for _, r in d[d.smiles.notna()].iterrows():
                smi = safe(r.smiles)
                if not smi:
                    continue
                for v in name_variants(r.compound_name):
                    mp.setdefault(norm_name(v), smi)
    except Exception as e:
        log(f'Offline DILImap map unavailable: {e}')
    return mp


def request_json(session, url, timeout, retries):
    last = {'ok': False, 'url': url}
    for i in range(int(retries)):
        try:
            r = session.get(url, timeout=float(timeout), headers={'User-Agent': 'P05-StateFlux-DILI-R03B/1.0'})
            last = {'ok': r.status_code == 200, 'status': int(r.status_code), 'url': url}
            if r.status_code == 200:
                try:
                    last['json'] = r.json()
                except Exception as e:
                    last['error'] = f'json_decode:{type(e).__name__}'
                return last
            if r.status_code in (400, 404):
                return last
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(30, 2 ** (i + 1)))
                continue
            return last
        except Exception as e:
            last = {'ok': False, 'url': url, 'error': repr(e)}
            time.sleep(min(10, 2 ** i))
    return last


def pubchem_lookup(query, cfg, session):
    """Current PUG-REST lookup.

    Round03 accidentally requested CID as a *property tag*. CID is returned with the
    property record but is not itself a property selector. This function removes that
    invalid tag and also uses a CID-first fallback.
    """
    q = safe(query)
    if not q:
        return {'ok': False, 'query': q, 'error': 'empty_query'}
    props = 'SMILES,ConnectivitySMILES,InChIKey,MolecularFormula,MolecularWeight,Title'
    enc = urllib.parse.quote(q, safe='')
    base = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
    timeout = cfg['structure_repair']['timeout_seconds']
    retries = cfg['structure_repair']['max_retries']

    direct = request_json(session, f'{base}/compound/name/{enc}/property/{props}/JSON', timeout, retries)
    if direct.get('ok') and direct.get('json'):
        try:
            o = direct['json']['PropertyTable']['Properties'][0]
            smi = safe(o.get('SMILES')) or safe(o.get('ConnectivitySMILES'))
            if smi:
                return {'ok': True, 'query': q, 'source': 'PubChem_name_property', 'resolved_smiles': smi, **o}
        except Exception as e:
            direct['parse_error'] = repr(e)

    # Fallback: resolve CID first, then request properties by CID.
    cids = request_json(session, f'{base}/compound/name/{enc}/cids/JSON', timeout, retries)
    if cids.get('ok') and cids.get('json'):
        try:
            cid = cids['json']['IdentifierList']['CID'][0]
            hit = request_json(session, f'{base}/compound/cid/{cid}/property/{props}/JSON', timeout, retries)
            if hit.get('ok') and hit.get('json'):
                o = hit['json']['PropertyTable']['Properties'][0]
                smi = safe(o.get('SMILES')) or safe(o.get('ConnectivitySMILES'))
                if smi:
                    return {'ok': True, 'query': q, 'source': 'PubChem_CID_fallback', 'resolved_smiles': smi, **o}
        except Exception as e:
            cids['parse_error'] = repr(e)
    return {'ok': False, 'query': q, 'direct': {k:v for k,v in direct.items() if k != 'json'}, 'cid_lookup': {k:v for k,v in cids.items() if k != 'json'}}


def chembl_lookup(chembl_id, cfg, session):
    cid = safe(chembl_id)
    if not cid:
        return {'ok': False, 'query': cid, 'error': 'empty_chembl'}
    url = f'https://www.ebi.ac.uk/chembl/api/data/molecule/{urllib.parse.quote(cid, safe="")}.json'
    hit = request_json(session, url, cfg['structure_repair']['timeout_seconds'], cfg['structure_repair']['max_retries'])
    if hit.get('ok') and hit.get('json'):
        try:
            obj = hit['json']
            ms = obj.get('molecule_structures') or {}
            smi = safe(ms.get('canonical_smiles'))
            if smi:
                return {'ok': True, 'query': cid, 'source': 'ChEMBL_API', 'resolved_smiles': smi}
        except Exception as e:
            return {'ok': False, 'query': cid, 'error': repr(e)}
    return {'ok': False, 'query': cid, 'status': hit.get('status'), 'error': hit.get('error', '')}


def repair_frame(df, id_col, name_col, cfg, cache_name):
    out = df.copy()
    # Pandas 3.x forbids assigning strings into an all-NaN float column.
    # Round03 often carries pubchem_inchikey as float64 because every value was NaN.
    # Force string-bearing metadata columns to object dtype before live repair writes them.
    if 'pubchem_inchikey' in out.columns:
        out['pubchem_inchikey'] = out['pubchem_inchikey'].astype('object')
    if 'structure_source_round03b' in out.columns:
        out['structure_source_round03b'] = out['structure_source_round03b'].astype('object')
    if 'resolved_smiles' not in out.columns:
        out['resolved_smiles'] = ''
    out['resolved_smiles'] = out['resolved_smiles'].fillna('').astype(str)
    if 'structure_source_round03b' not in out.columns:
        out['structure_source_round03b'] = ''

    offline = offline_structure_map()
    # Deterministic local recovery first.
    for i, row in out.iterrows():
        if safe(row.get('resolved_smiles')):
            continue
        smi = safe(row.get('smiles'))
        source = ''
        if smi:
            source = 'existing_input_smiles'
        if not smi:
            for v in name_variants(row.get(name_col, '')):
                smi = safe(offline.get(norm_name(v)))
                if smi:
                    source = 'DILImap_metadata_offline'
                    break
        if smi and R3.standardize_smiles(smi).get('parse_ok'):
            out.at[i, 'resolved_smiles'] = smi
            out.at[i, 'structure_source_round03b'] = source

    cachep = CACHE / cache_name
    cache = read_json(cachep, {})
    session = requests.Session()
    attempts = []
    missing = [i for i, s in out['resolved_smiles'].items() if not safe(s)]
    log(f'{cache_name}: {len(missing)} unresolved structures before live repair')

    for i in tqdm(missing, desc='R03B structure repair'):
        row = out.loc[i]
        key = safe(row.get(id_col)) or norm_name(row.get(name_col))
        if key in cache and cache[key].get('ok'):
            hit = cache[key]
        else:
            hit = None
            # ChEMBL identifier first when available.
            if cfg['structure_repair'].get('use_chembl_fallback', True):
                chembl_id = safe(row.get('molecule_chembl_id'))
                if chembl_id:
                    hh = chembl_lookup(chembl_id, cfg, session)
                    attempts.append({'row_id': key, 'compound': safe(row.get(name_col)), 'query_type': 'chembl_id', **{k:v for k,v in hh.items() if k != 'resolved_smiles'}})
                    if hh.get('ok'):
                        hit = hh
            # PubChem: CAS, full name/base names, ChEMBL ID as synonym.
            if hit is None:
                qs = []
                cas = safe(row.get('CAS_number'))
                if cas:
                    qs.append(('CAS', cas))
                for v in name_variants(row.get(name_col, '')):
                    qs.append(('name', v))
                chembl_id = safe(row.get('molecule_chembl_id'))
                if chembl_id:
                    qs.append(('ChEMBL_synonym', chembl_id))
                seen = set()
                for qtype, q in qs:
                    if not q or q.lower() in seen:
                        continue
                    seen.add(q.lower())
                    hh = pubchem_lookup(q, cfg, session)
                    attempts.append({'row_id': key, 'compound': safe(row.get(name_col)), 'query_type': qtype, **{k:v for k,v in hh.items() if k not in {'resolved_smiles','direct','cid_lookup'}}})
                    if hh.get('ok'):
                        hit = hh
                        break
                    time.sleep(float(cfg['structure_repair']['delay_seconds']))
            if hit is None:
                hit = {'ok': False, 'query': safe(row.get(name_col)), 'error': 'all_sources_failed'}
            cache[key] = hit
            write_json(cachep, cache)

        smi = safe(hit.get('resolved_smiles'))
        if hit.get('ok') and smi and R3.standardize_smiles(smi).get('parse_ok'):
            out.at[i, 'resolved_smiles'] = smi
            out.at[i, 'structure_source_round03b'] = safe(hit.get('source')) or 'live_repair'
            if 'pubchem_cid' in out.columns and hit.get('CID') is not None:
                out.at[i, 'pubchem_cid'] = hit.get('CID')
            if 'pubchem_inchikey' in out.columns and hit.get('InChIKey') is not None:
                out.at[i, 'pubchem_inchikey'] = hit.get('InChIKey')

    # Apply the exact same chemical-standardization/exclusion policy used in Round03.
    std = pd.DataFrame([R3.standardize_smiles(x) for x in tqdm(out['resolved_smiles'], desc='R03B standardize')], index=out.index)
    for c in std.columns:
        out[c] = std[c]
    if attempts:
        ap = TABLES / 'structure_repair_attempts.csv'
        old = pd.read_csv(ap) if ap.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(attempts)], ignore_index=True, sort=False).drop_duplicates().to_csv(ap, index=False)
    return out


def parse_new_metabolism(master):
    names = dict(zip(master.LTKBID.astype(str), master.CompoundName.astype(str)))
    rows = []
    for mode in ['sequence', 'phaseII']:
        d = METAB_NEW / mode
        if not d.exists():
            continue
        files = list(d.glob('*.csv'))
        for p in tqdm(files, desc=f'Parse R03B metabolism/{mode}'):
            cid = p.stem
            parent = ''
            m = master[master.LTKBID.astype(str) == cid]
            if len(m):
                parent = safe(m.iloc[0].standardized_smiles)
            rows.extend(R3.parse_bt_csv(p, cid, names.get(cid, cid), mode, parent))
    e = pd.DataFrame(rows)
    if e.empty:
        return e
    e = e.drop_duplicates(['LTKBID','precursor_smiles','product_smiles','mode','mechanism'])
    parent_map = dict(zip(master.LTKBID.astype(str), master.standardized_smiles.astype(str)))
    return R3.reconstruct_generations(e, parent_map)


def morgan_matrix(smiles, nbits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
    out = np.zeros((len(smiles), nbits), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(safe(s))
        if mol is None:
            continue
        fp = gen.GetFingerprintAsNumPy(mol)
        out[i, :] = np.asarray(fp, dtype=np.float32)
    return out


def rdkit_fp(smi, nbits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
    m = Chem.MolFromSmiles(safe(smi))
    return gen.GetFingerprint(m) if m is not None else None


def scaffold_groups(df):
    vals = []
    for i, r in df.reset_index(drop=True).iterrows():
        sc = safe(r.get('scaffold_smiles'))
        vals.append(sc if sc else f'acyclic_{safe(r.get("LTKBID"))}_{i}')
    return np.asarray(vals, object)


def metric_dict(y, p, thr=0.5):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel() if len(y) else (0,0,0,0)
    return {
        'n': int(len(y)), 'positives': int(y.sum()), 'negatives': int((y==0).sum()),
        'auroc': float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else np.nan,
        'auprc': float(average_precision_score(y,p)) if len(np.unique(y)) == 2 else np.nan,
        'balanced_accuracy': float(balanced_accuracy_score(y,pred)) if len(np.unique(y)) == 2 else np.nan,
        'accuracy': float(accuracy_score(y,pred)), 'precision': float(precision_score(y,pred,zero_division=0)),
        'sensitivity': float(recall_score(y,pred,zero_division=0)),
        'specificity': float(tn/(tn+fp)) if tn+fp else np.nan,
        'f1': float(f1_score(y,pred,zero_division=0)), 'mcc': float(matthews_corrcoef(y,pred)) if len(np.unique(pred)) > 1 else 0.0,
        'brier': float(brier_score_loss(y,p)), 'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def base_model(seed, n_trees=550):
    return ExtraTreesClassifier(
        n_estimators=int(n_trees), max_depth=24, min_samples_leaf=1,
        max_features='sqrt', class_weight='balanced', random_state=int(seed), n_jobs=-1
    )


def meta_model(seed, C, l1_ratio):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
        ('lr', LogisticRegression(
            penalty='elasticnet', solver='saga', C=float(C), l1_ratio=float(l1_ratio),
            class_weight='balanced', max_iter=6000, random_state=int(seed)
        ))
    ])


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-5, 1-1e-5)
    return np.log(p/(1-p))


def feature_matrix(df, columns):
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return np.zeros((len(df), 0), np.float32), []
    return df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32), cols


def valid_sgkf(y, groups, n_splits, seed):
    # Reduce fold count only if the requested grouping is impossible.
    ns = int(n_splits)
    while ns >= 2:
        try:
            sp = StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=int(seed))
            idx = list(sp.split(np.zeros(len(y)), y, groups))
            if idx and all(len(np.unique(y[tr])) >= 2 and len(np.unique(y[te])) >= 2 for tr, te in idx):
                return idx
        except Exception:
            pass
        ns -= 1
    raise RuntimeError('Unable to create grouped stratified CV splits.')


def inner_base_oof(X, y, groups, n_splits, seed, n_trees):
    p = np.full(len(y), np.nan, float)
    for k, (tr, va) in enumerate(valid_sgkf(y, groups, n_splits, seed)):
        m = base_model(seed + 101*k, n_trees)
        m.fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:,1]
    if np.isnan(p).any():
        raise RuntimeError('Inner base OOF contains missing predictions.')
    return p


def residual_cv(df, state_cols, flux_cols, cfg, repeats, folds, inner_folds, label='DILI', n_trees=550):
    data = df.reset_index(drop=True).copy()
    y = data.primary_label.astype(int).to_numpy()
    groups = scaffold_groups(data)
    Xfp = morgan_matrix(data.standardized_smiles.tolist())
    Xstate, state_used = feature_matrix(data, state_cols)
    Xflux, flux_used = feature_matrix(data, flux_cols)
    panels = {
        'ECFP': None,
        'ECFP_StateCore': Xstate,
        'ECFP_FluxCore': Xflux,
        'ECFP_StateFluxCore': np.hstack([Xstate, Xflux]) if Xstate.shape[1] + Xflux.shape[1] else np.zeros((len(data),0)),
    }
    pred_cube = {k: np.zeros((len(data), int(repeats)), float) for k in panels}
    fold_rows = []
    C = cfg['validation']['primary_meta_C']
    l1 = cfg['validation']['primary_meta_l1_ratio']

    for rep in range(int(repeats)):
        splits = valid_sgkf(y, groups, int(folds), int(cfg['random_seed']) + rep*1009)
        for fold, (tr, te) in enumerate(splits):
            base = base_model(int(cfg['random_seed']) + rep*1009 + fold, n_trees)
            base.fit(Xfp[tr], y[tr])
            pte = base.predict_proba(Xfp[te])[:,1]
            pin = inner_base_oof(Xfp[tr], y[tr], groups[tr], int(inner_folds), int(cfg['random_seed']) + rep*1009 + fold*37, max(250, n_trees//2))
            pred_cube['ECFP'][te, rep] = pte
            fm = metric_dict(y[te], pte); fm.update(dataset=label, model='ECFP', repeat=rep, fold=fold); fold_rows.append(fm)

            for pname, Xextra in panels.items():
                if pname == 'ECFP' or Xextra is None or Xextra.shape[1] == 0:
                    continue
                Xin = np.column_stack([logit(pin), Xextra[tr]])
                Xte = np.column_stack([logit(pte), Xextra[te]])
                mm = meta_model(int(cfg['random_seed']) + rep*1009 + fold*53, C, l1)
                mm.fit(Xin, y[tr])
                pp = mm.predict_proba(Xte)[:,1]
                pred_cube[pname][te, rep] = pp
                fm = metric_dict(y[te], pp); fm.update(dataset=label, model=pname, repeat=rep, fold=fold); fold_rows.append(fm)

    preds = {k: v.mean(axis=1) for k,v in pred_cube.items()}
    metrics = []
    for name, p in preds.items():
        md = metric_dict(y,p); md.update(dataset=label, model=name, repeats=int(repeats), folds=int(folds)); metrics.append(md)
    predrows = []
    for i, r in data.iterrows():
        rec = {'dataset':label, 'LTKBID':safe(r.get('LTKBID')), 'compound':safe(r.get('CompoundName')), 'y_true':int(y[i])}
        for name,p in preds.items(): rec['p_'+name] = float(p[i])
        predrows.append(rec)
    return pd.DataFrame(metrics), pd.DataFrame(predrows), pd.DataFrame(fold_rows), preds, {'state_used':state_used,'flux_used':flux_used}


def paired_bootstrap(y, p0, p1, n, seed):
    y=np.asarray(y,int); p0=np.asarray(p0,float); p1=np.asarray(p1,float)
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; rng=np.random.default_rng(int(seed)); da=[]; dp=[]
    if len(pos)==0 or len(neg)==0:
        return {}
    for _ in range(int(n)):
        idx=np.concatenate([rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)])
        yy=y[idx]
        da.append(roc_auc_score(yy,p1[idx])-roc_auc_score(yy,p0[idx]))
        dp.append(average_precision_score(yy,p1[idx])-average_precision_score(yy,p0[idx]))
    def s(a):
        a=np.asarray(a,float)
        return {'median':float(np.median(a)),'ci_low':float(np.quantile(a,.025)),'ci_high':float(np.quantile(a,.975)),'p_positive':float(np.mean(a>0))}
    return {'delta_auroc':s(da),'delta_auprc':s(dp),'n_bootstrap':int(n)}


def max_train_similarity(train, ext):
    tfps = [rdkit_fp(s) for s in train.standardized_smiles]
    out=[]
    for s in ext.standardized_smiles:
        f=rdkit_fp(s)
        sims=DataStructs.BulkTanimotoSimilarity(f, tfps) if f is not None and tfps else []
        out.append(max(sims) if sims else np.nan)
    return np.asarray(out,float)


def overlap_audit(train, ext, setname, cfg):
    if ext.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train_keys=set(train.inchikey.dropna().astype(str)) if 'inchikey' in train else set()
    train_scaf=set(safe(x) for x in train.scaffold_smiles if safe(x))
    sim=max_train_similarity(train,ext)
    a=ext.copy().reset_index(drop=True)
    a['max_train_tanimoto']=sim
    a['exact_inchikey_overlap']=a.inchikey.astype(str).isin(train_keys) if 'inchikey' in a else False
    a['scaffold_seen_in_train']=a.scaffold_smiles.map(lambda x: safe(x) in train_scaf and bool(safe(x)))
    a['strict_nonoverlap']=(~a.exact_inchikey_overlap)&(a.max_train_tanimoto<float(cfg['validation']['strict_exact_tanimoto']))
    a['scaffold_novel'] = a.strict_nonoverlap & (~a.scaffold_seen_in_train) & (a.max_train_tanimoto<float(cfg['validation']['scaffold_novel_tanimoto']))
    a['validation_set']=setname
    return a, a[a.strict_nonoverlap].copy(), a[a.scaffold_novel].copy()


def fit_full_and_external(dev, external_sets, state_cols, flux_cols, cfg, n_trees=650):
    dev=dev.reset_index(drop=True).copy(); y=dev.primary_label.astype(int).to_numpy(); groups=scaffold_groups(dev)
    Xfp=morgan_matrix(dev.standardized_smiles.tolist()); Xstate,state_used=feature_matrix(dev,state_cols); Xflux,flux_used=feature_matrix(dev,flux_cols)
    panels={'ECFP_StateCore':Xstate,'ECFP_FluxCore':Xflux,'ECFP_StateFluxCore':np.hstack([Xstate,Xflux])}
    base=base_model(int(cfg['random_seed'])+777,n_trees); base.fit(Xfp,y)
    # Fully out-of-fold base probabilities are the only base scores used to train the meta layer.
    pin=inner_base_oof(Xfp,y,groups,int(cfg['validation']['outer_folds']),int(cfg['random_seed'])+991,max(300,n_trees//2))
    metas={}
    for name,Xe in panels.items():
        mm=meta_model(int(cfg['random_seed'])+1234,float(cfg['validation']['primary_meta_C']),float(cfg['validation']['primary_meta_l1_ratio']))
        mm.fit(np.column_stack([logit(pin),Xe]),y); metas[name]=(mm, name)
    rows=[]; preds=[]
    for setname,ds in external_sets.items():
        if ds is None or ds.empty: continue
        d=ds.reset_index(drop=True).copy(); Xf=morgan_matrix(d.standardized_smiles.tolist()); pb=base.predict_proba(Xf)[:,1]
        Xs,_=feature_matrix(d,state_used); Xq,_=feature_matrix(d,flux_used); mats={'ECFP_StateCore':Xs,'ECFP_FluxCore':Xq,'ECFP_StateFluxCore':np.hstack([Xs,Xq])}
        yy=d.primary_label.astype(int).to_numpy(); md=metric_dict(yy,pb); md.update(dataset=setname,model='ECFP'); rows.append(md)
        local={'ECFP':pb}
        for name,(mm,_) in metas.items():
            pp=mm.predict_proba(np.column_stack([logit(pb),mats[name]]))[:,1]; local[name]=pp; md=metric_dict(yy,pp); md.update(dataset=setname,model=name); rows.append(md)
        for i,r in d.iterrows():
            z={'dataset':setname,'LTKBID':safe(r.get('LTKBID')),'compound':safe(r.get('CompoundName')),'y_true':int(yy[i])}
            if 'max_train_tanimoto' in d: z['max_train_tanimoto']=float(d.iloc[i].max_train_tanimoto)
            for name,p in local.items(): z['p_'+name]=float(p[i])
            preds.append(z)
    return pd.DataFrame(rows),pd.DataFrame(preds)


def analog_pairs(dev, pred_df, flux_cols, cfg):
    d=dev.reset_index(drop=True).copy()
    pp=pred_df.set_index('LTKBID').to_dict('index')
    fps=[rdkit_fp(s) for s in d.standardized_smiles]
    ids=d.LTKBID.astype(str).tolist(); names=d.CompoundName.astype(str).tolist(); yy=d.primary_label.astype(int).to_numpy(); scafs=d.scaffold_smiles.fillna('').astype(str).tolist()
    pairs=[]
    for i in range(len(d)):
        sims=DataStructs.BulkTanimotoSimilarity(fps[i],fps[i+1:]) if fps[i] is not None else []
        for off,sim in enumerate(sims):
            j=i+1+off
            if yy[i]==yy[j]: continue
            id1,id2=ids[i],ids[j]
            if id1 not in pp or id2 not in pp: continue
            r1,r2=pp[id1],pp[id2]
            base_sep=abs(float(r1['p_ECFP'])-float(r2['p_ECFP'])); flux_sep=abs(float(r1['p_ECFP_StateFluxCore'])-float(r2['p_ECFP_StateFluxCore']))
            same_scaf=bool(scafs[i] and scafs[i]==scafs[j])
            pairs.append({'id1':id1,'compound1':names[i],'y1':int(yy[i]),'id2':id2,'compound2':names[j],'y2':int(yy[j]),'tanimoto':float(sim),'same_murcko_scaffold':same_scaf,'strict_cliff':bool(sim>=float(cfg['validation']['cliff_strict_tanimoto'])),'delta_prob_ecfp':base_sep,'delta_prob_stateflux_core':flux_sep,'stateflux_minus_ecfp_separation':flux_sep-base_sep})
    if not pairs:
        return pd.DataFrame(),pd.DataFrame()
    p=pd.DataFrame(pairs).sort_values(['tanimoto','same_murcko_scaffold'],ascending=[False,False]).reset_index(drop=True)
    top=p.head(int(cfg['validation']['analog_top_n'])).copy()
    # Add interpretable feature differences for the top analog pairs.
    frows=[]; byid=d.set_index('LTKBID')
    for _,r in top.iterrows():
        a=byid.loc[r.id1]; b=byid.loc[r.id2]
        for c in flux_cols:
            if c not in d.columns: continue
            av=pd.to_numeric(a.get(c),errors='coerce'); bv=pd.to_numeric(b.get(c),errors='coerce')
            frows.append({'id1':r.id1,'compound1':r.compound1,'id2':r.id2,'compound2':r.compound2,'tanimoto':r.tanimoto,'feature':c,'value1':av,'value2':bv,'delta_2_minus_1':float(bv-av) if pd.notna(av) and pd.notna(bv) else np.nan,'abs_delta':float(abs(bv-av)) if pd.notna(av) and pd.notna(bv) else np.nan})
    return top,pd.DataFrame(frows)


def rank_qchem_candidates(preds, analogs):
    p0=ROOT/'output_round03'/'manifests'/'qchem_candidate_edges_round03.csv'
    if not p0.exists(): return pd.DataFrame()
    e=pd.read_csv(p0); score={}; analog_bonus={}
    if not preds.empty:
        for _,r in preds.iterrows():
            score[safe(r.LTKBID)]=abs(float(r.p_ECFP_StateFluxCore)-float(r.p_ECFP))
    if not analogs.empty:
        for _,r in analogs.iterrows():
            analog_bonus[r.id1]=max(analog_bonus.get(r.id1,0),float(r.tanimoto)); analog_bonus[r.id2]=max(analog_bonus.get(r.id2,0),float(r.tanimoto))
    e['residual_model_disagreement']=e.LTKBID.astype(str).map(score).fillna(0)
    e['discordant_analog_similarity']=e.LTKBID.astype(str).map(analog_bonus).fillna(0)
    base=pd.to_numeric(e['qchem_edge_score'],errors='coerce').fillna(0) if 'qchem_edge_score' in e.columns else pd.Series(np.zeros(len(e)),index=e.index)
    e['round03b_priority_score']=base+2.0*e.residual_model_disagreement+1.5*e.discordant_analog_similarity
    e=e.sort_values('round03b_priority_score',ascending=False)
    e.to_csv(MANIFESTS/'qchem_candidate_edges_round03b_priority.csv',index=False)
    return e


def run_hagan_lowdim(cfg):
    p=ROOT/'output_round03'/'tables'/'hagan_stateflux_features.csv'
    if not p.exists():
        log('Hagan low-dimensional transfer skipped: feature table missing')
        return pd.DataFrame(),pd.DataFrame()
    d=pd.read_csv(p)
    if 'primary_label' not in d or 'standardized_smiles' not in d:
        log('Hagan low-dimensional transfer skipped: metadata columns missing')
        return pd.DataFrame(),pd.DataFrame()
    d=d[d.primary_label.notna()&d.standardized_smiles.notna()].copy(); d.primary_label=d.primary_label.astype(int)
    metrics,preds,folds,pp,used=residual_cv(d,STATE_CORE,FLUX_CORE,cfg,int(cfg['hagan_lowdim']['outer_repeats']),int(cfg['hagan_lowdim']['outer_folds']),int(cfg['hagan_lowdim']['inner_folds']),'Hagan_genotox',n_trees=325)
    metrics.to_csv(TABLES/'hagan_lowdim_metrics.csv',index=False); preds.to_csv(TABLES/'hagan_lowdim_predictions.csv',index=False); folds.to_csv(TABLES/'hagan_lowdim_fold_metrics.csv',index=False)
    y=d.primary_label.astype(int).to_numpy(); write_json(TABLES/'hagan_lowdim_bootstrap_stateflux_vs_ecfp.json',paired_bootstrap(y,pp['ECFP'],pp['ECFP_StateFluxCore'],int(cfg['validation']['paired_bootstrap']),int(cfg['random_seed'])+5000))
    write_json(TABLES/'hagan_lowdim_feature_manifest.json',used)
    return metrics,preds


def package_return(cfg):
    outzip=ROOT/'P05_R03B_RETURN_TO_CHATGPT.zip'
    with zipfile.ZipFile(outzip,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if not p.is_file(): continue
            # Raw xTB/BioTransformer artifacts stay in the project folder; only compact summaries return.
            if any(part in {'xtb_dili_new','metabolism_new'} for part in p.relative_to(OUT).parts): continue
            z.write(p,p.relative_to(ROOT))
        for rel in ['p05_round03b_config.json','scripts/p05_round03b_rescue.py']:
            p=ROOT/rel
            if p.exists(): z.write(p,p.relative_to(ROOT))
    return outzip


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='p05_round03b_config.json'); args=ap.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    log('P05 StateFlux-DILI Round03B rescue launched'); log(f'Root: {ROOT}')

    with stage('A repair missing structures with corrected PubChem PUG-REST plus ChEMBL/offline fallbacks'):
        mp=TABLES/'master_compound_audit_round03b.csv'; cp=MANIFESTS/'dilimap_validation_round03b.csv'
        if mp.exists() and cp.exists():
            log('Stage A cached outputs found; reusing repaired structure tables.')
            master=pd.read_csv(mp); challenge=pd.read_csv(cp)
        else:
            master=pd.read_csv(ROOT/'output_round03'/'tables'/'master_compound_audit_round03.csv'); master['name_norm']=master.CompoundName.map(norm_name)
            challenge=pd.read_csv(ROOT/'output_round03'/'manifests'/'dilimap_blind_validation_round03.csv'); challenge['name_norm']=challenge.compound_name.map(norm_name)
            master=repair_frame(master,'LTKBID','CompoundName',cfg,'pubchem_master_round03b.json'); master['primary_label']=pd.to_numeric(master.primary_label,errors='coerce'); master.to_csv(mp,index=False)
            challenge=repair_frame(challenge,'challenge_id','compound_name',cfg,'pubchem_dilimap_round03b.json'); challenge.to_csv(cp,index=False)
        miss_before=pd.read_csv(ROOT/'output_round03'/'tables'/'master_compound_audit_round03.csv')
        n0=int((miss_before.resolved_smiles.fillna('').astype(str).str.len()==0).sum()); n1=int((master.resolved_smiles.fillna('').astype(str).str.len()==0).sum())
        summary={'master_n':len(master),'eligible_n':int(master.eligible.fillna(False).sum()),'missing_structure_before':n0,'missing_structure_after':n1,'structures_recovered':n0-n1,'extreme_eligible_n':int((master.eligible.fillna(False)&master.primary_label.notna()).sum()),'new_extreme_eligible_n':int((master.eligible.fillna(False)&master.primary_label.notna()&(master.Comment.astype(str).str.lower()=='new')).sum())}
        write_json(TABLES/'structure_repair_summary.json',summary); log(f"Structure repair recovered {summary['structures_recovered']} master structures; temporal extreme eligible n={summary['new_extreme_eligible_n']}")

    with stage('B incremental BioTransformer/xTB only for newly recovered labeled or challenge compounds'):
        old_feat=pd.read_csv(ROOT/'output_round03'/'tables'/'stateflux_v3_features.csv'); oldids=set(old_feat.LTKBID.astype(str))
        extras=[]
        for _,r in master[master.eligible.fillna(False)&master.primary_label.notna()&(~master.LTKBID.astype(str).isin(oldids))].iterrows(): extras.append(r.to_dict())
        for _,r in challenge[challenge.eligible.fillna(False)].iterrows():
            cid=safe(r.challenge_id)
            if cid and cid not in oldids:
                extras.append({'LTKBID':cid,'CompoundName':r.compound_name,'standardized_smiles':r.standardized_smiles,'eligible':True,'formal_charge':r.formal_charge,'primary_label':np.nan,'v2_class':'dilimap','Comment':'DILImap_blind','scaffold_smiles':r.scaffold_smiles,'inchikey':r.inchikey})
        new=pd.DataFrame(extras).drop_duplicates('LTKBID') if extras else pd.DataFrame(columns=['LTKBID','CompoundName','standardized_smiles','eligible'])
        new.to_csv(MANIFESTS/'new_compute_compounds_round03b.csv',index=False); log(f'Round03B new compute universe: {len(new)} compounds')
        if len(new): R3.run_new_metabolism(new,cfg)

        # Recompute StateFlux descriptors ONLY for genuinely new compound IDs. Existing
        # Round03 feature rows are immutable and are reused exactly, avoiding another
        # expensive pass over the 279k-edge Round03 metabolic graph.
        old_xtb=pd.read_csv(ROOT/'output_round03'/'tables'/'dili_xtb_round03_complete.csv')
        if len(new):
            new_edges=parse_new_metabolism(new)
            if not new_edges.empty: new_edges.to_csv(TABLES/'new_metabolic_edges_round03b.csv',index=False)
            newnodes=R3.build_dili_nodes(new,new_edges,cfg)
            reused,todo=R3.merge_reused_xtb(newnodes,old_xtb); log(f'New xTB nodes needing computation: {len(todo)} (reused={len(reused)})')
            newxtb=R3.run_xtb(todo,XTB_NEW,cfg,'Round03B incremental GFN2-xTB') if len(todo) else pd.DataFrame()
            if len(newxtb): newxtb.to_csv(TABLES/'new_xtb_round03b.csv',index=False)
            xtb_new=pd.concat([reused,newxtb],ignore_index=True,sort=False).drop_duplicates(['LTKBID','smiles','node_type'],keep='last')
            newfeat=R3.graph_features(new,new_edges,newnodes,xtb_new,None)
            newfeat.to_csv(TABLES/'new_stateflux_features_round03b.csv',index=False)
            feat=pd.concat([old_feat,newfeat],ignore_index=True,sort=False).drop_duplicates('LTKBID',keep='last')
        else:
            feat=old_feat.copy()
        feat.to_csv(TABLES/'stateflux_round03b_features.csv',index=False)
        log(f'Round03B feature table: {len(feat)} compounds, {len(feat.columns)} columns')

    with stage('C freeze strict temporal/DILImap nonoverlap validation sets and applicability tiers'):
        chnames=set(challenge.name_norm.astype(str)); dev0=master[master.eligible.fillna(False)&(master.Comment.astype(str).str.lower()=='unchanged')&master.primary_label.notna()].copy(); dev0=dev0[~dev0.CompoundName.map(norm_name).isin(chnames)].copy()
        feat=pd.read_csv(TABLES/'stateflux_round03b_features.csv'); fmeta=feat.drop(columns=[c for c in ['CompoundName','primary_label','v2_class','Comment','scaffold_smiles','standardized_smiles'] if c in feat.columns])
        dev=dev0.merge(fmeta,on='LTKBID',how='inner')
        temporal0=master[master.eligible.fillna(False)&(master.Comment.astype(str).str.lower()=='new')&master.primary_label.notna()].copy(); temporal0=temporal0.merge(fmeta,on='LTKBID',how='inner')
        ch=challenge[challenge.eligible.fillna(False)&challenge.True_label.isin(['+','-'])].copy(); ch['primary_label']=ch.True_label.map({'+':1,'-':0}); ch['LTKBID']=ch.challenge_id.astype(str); ch['CompoundName']=ch.compound_name.astype(str); ch=ch.merge(fmeta,on='LTKBID',how='inner')
        ta,ts,tn=overlap_audit(dev,temporal0,'temporal_v2_new',cfg); da,ds,dn=overlap_audit(dev,ch,'dilimap_blind',cfg)
        ta.to_csv(TABLES/'overlap_audit_temporal.csv',index=False); da.to_csv(TABLES/'overlap_audit_dilimap.csv',index=False)
        ts.to_csv(MANIFESTS/'temporal_strict_nonoverlap.csv',index=False); tn.to_csv(MANIFESTS/'temporal_scaffold_novel.csv',index=False); ds.to_csv(MANIFESTS/'dilimap_strict_nonoverlap.csv',index=False); dn.to_csv(MANIFESTS/'dilimap_scaffold_novel.csv',index=False)
        dev.to_csv(MANIFESTS/'development_round03b.csv',index=False)
        log(f'Development n={len(dev)}; temporal resolved={len(temporal0)}, strict={len(ts)}, scaffold-novel={len(tn)}; DILImap resolved={len(ch)}, strict={len(ds)}, scaffold-novel={len(dn)}')

    with stage('D nested scaffold residual stacking with predeclared low-dimensional StateFlux features'):
        dev=pd.read_csv(MANIFESTS/'development_round03b.csv'); metrics,preds,folds,pp,used=residual_cv(dev,STATE_CORE,FLUX_CORE,cfg,int(cfg['validation']['outer_repeats']),int(cfg['validation']['outer_folds']),int(cfg['validation']['inner_folds']),'DILI',n_trees=550)
        metrics.to_csv(TABLES/'dili_lowdim_metrics.csv',index=False); preds.to_csv(TABLES/'dili_lowdim_predictions.csv',index=False); folds.to_csv(TABLES/'dili_lowdim_fold_metrics.csv',index=False); write_json(TABLES/'lowdim_feature_manifest.json',used)
        y=dev.primary_label.astype(int).to_numpy(); write_json(TABLES/'paired_bootstrap_lowdim_flux_vs_ecfp.json',paired_bootstrap(y,pp['ECFP'],pp['ECFP_StateFluxCore'],int(cfg['validation']['paired_bootstrap']),int(cfg['random_seed']))); write_json(TABLES/'paired_bootstrap_flux_increment_over_statecore.json',paired_bootstrap(y,pp['ECFP_StateCore'],pp['ECFP_StateFluxCore'],int(cfg['validation']['paired_bootstrap']),int(cfg['random_seed'])+1))
        exsets={}
        for nm,rel in [('temporal_strict','temporal_strict_nonoverlap.csv'),('temporal_scaffold_novel','temporal_scaffold_novel.csv'),('dilimap_strict','dilimap_strict_nonoverlap.csv'),('dilimap_scaffold_novel','dilimap_scaffold_novel.csv')]:
            p=MANIFESTS/rel
            if p.exists():
                q=pd.read_csv(p)
                if len(q): exsets[nm]=q
        em,ep=fit_full_and_external(dev,exsets,used['state_used'],used['flux_used'],cfg); em.to_csv(TABLES/'external_strict_metrics.csv',index=False); ep.to_csv(TABLES/'external_strict_predictions.csv',index=False)
        log('DILI low-dimensional nested validation complete.')

    with stage('E structural analog / toxicity-cliff rescue analysis'):
        dev=pd.read_csv(MANIFESTS/'development_round03b.csv'); preds=pd.read_csv(TABLES/'dili_lowdim_predictions.csv'); used=read_json(TABLES/'lowdim_feature_manifest.json',{}); top,fd=analog_pairs(dev,preds,used.get('flux_used',[]),cfg); top.to_csv(TABLES/'discordant_analog_pairs.csv',index=False); fd.to_csv(TABLES/'discordant_analog_feature_deltas.csv',index=False); rank_qchem_candidates(preds,top)
        if len(top): log(f'Discordant analog pairs saved: {len(top)}; strict cliff count={(top.strict_cliff==True).sum()}; top similarity={top.tanimoto.max():.3f}')
        else: log('No opposite-label analog pairs could be generated.')

    with stage('F independent Hagan low-dimensional residual transfer test'):
        if cfg.get('hagan_lowdim',{}).get('enabled',True): run_hagan_lowdim(cfg)
        else: log('Hagan low-dimensional stage disabled in config.')

    with stage('G final audit, Q-Chem gate diagnostics, and return package'):
        sm=read_json(TABLES/'structure_repair_summary.json',{}); dm=pd.read_csv(TABLES/'dili_lowdim_metrics.csv'); boot=read_json(TABLES/'paired_bootstrap_lowdim_flux_vs_ecfp.json',{}); em=pd.read_csv(TABLES/'external_strict_metrics.csv') if (TABLES/'external_strict_metrics.csv').exists() else pd.DataFrame(); hm=pd.read_csv(TABLES/'hagan_lowdim_metrics.csv') if (TABLES/'hagan_lowdim_metrics.csv').exists() else pd.DataFrame(); analog=pd.read_csv(TABLES/'discordant_analog_pairs.csv') if (TABLES/'discordant_analog_pairs.csv').exists() and (TABLES/'discordant_analog_pairs.csv').stat().st_size>2 else pd.DataFrame()
        by={r.model:r for _,r in dm.iterrows()}; base=by.get('ECFP'); full=by.get('ECFP_StateFluxCore'); state=by.get('ECFP_StateCore')
        gate={'status':'REVIEW','reason':'Point estimate/CI require scientific review before Q-Chem.'}
        try:
            dlo=float(boot['delta_auroc']['ci_low']); dmed=float(boot['delta_auroc']['median'])
            if dlo>0 and dmed>0.005: gate={'status':'GO','reason':'Low-dimensional StateFlux adds positive AUROC with paired 95% CI above zero.'}
            elif float(boot['delta_auroc']['ci_high'])<=0 or dmed<-0.01: gate={'status':'PIVOT','reason':'Incremental StateFlux remains non-positive after rescue; do not launch a broad Q-Chem campaign solely to force predictive gain.'}
        except Exception: pass
        summary={'completed_at':time.strftime('%Y-%m-%d %H:%M:%S'),'structure_repair':sm,'development_n':int(base.n) if base is not None else None,'dili_metrics':dm.to_dict('records'),'paired_bootstrap_flux_vs_ecfp':boot,'external_metrics':em.to_dict('records'),'hagan_lowdim_metrics':hm.to_dict('records'),'discordant_analog_pairs_n':len(analog),'strict_cliffs_n':int(analog.strict_cliff.sum()) if len(analog) and 'strict_cliff' in analog else 0,'qchem_gate':gate}
        write_json(TABLES/'ROUND03B_SUMMARY.json',summary); write_json(TABLES/'QCHEM_GATE.json',gate)
        prov={}
        for rel in ['scripts/p05_round03b_rescue.py','p05_round03b_config.json','output_round03/tables/ROUND03_SUMMARY.json','output_round03/tables/stateflux_v3_features.csv','output_round03/tables/dili_xtb_round03_complete.csv','data/DILIrank2_FDA.xlsx','data/DILImap_metadata.xlsx']:
            p=ROOT/rel
            if p.exists(): prov[rel]={'sha256':sha256_file(p),'bytes':p.stat().st_size}
        write_json(TABLES/'round03b_provenance_sha256.json',prov)
        audit=(
            'Round03B is a rescue/validation round. It corrects the PubChem property request, reuses all successful Round03 metabolism/xTB calculations, computes only newly recovered labeled/challenge chemistry, removes exact structural overlap from external evaluations, and tests a predeclared low-dimensional StateFlux panel using nested scaffold residual stacking. It also creates discordant structural-analog pairs and reruns the independent Hagan transfer test with the same low-dimensional representation.\n'
        )
        (OUT/'ROUND03B_AUDIT.txt').write_text(audit,encoding='utf-8')
        outzip=package_return(cfg); log(f'Return ZIP created: {outzip}')


if __name__=='__main__':
    main()
