#!/usr/bin/env python3
"""Build/query evidence-based indexes without importing training or environment code."""
import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from sanitize_experiment_metadata import redact

UNKNOWN = 'unknown'
RUN_FIELDS = ['run_id','experiment_id','experiment_ids','experiment_group_id','environment_id','benchmark_id','protocol_id','protocol_fingerprint','method','method_id','algorithm','method_variant_id','method_parameters','seed','status','result_status','git_commit','config_path','parent_run_id','role','attempt','attempt_source','logical_run_id','retry_of','run_path','result_paths','source_records','method_aliases','conflicts','execution_evidence','trained_steps','replay_on_switch','epsilon_on_switch','completed_phases','scheduler_task_ids','scheduler_dependencies','observed_at']
ANALYSIS_FIELDS = ['analysis_id','experiment_id','activity_kind','status','input_run_ids','unresolved_inputs','environment_ids','benchmark_ids','result_paths','source_records','input_sources']
RUN_FIELDS += ['algorithm_config','architecture_id','agent_type','hidden_dim','mixer_type','architecture_source']
VDN_CONFIGS = ('vdn', 'vdn_1ff', 'vdn_1ff_128')
SUITE_ENVS = {'smac':'smac', 'continual':'smacv2','hetero':'smacv2','history':'smacv2','vdn_action_set':'smacv2', **{x:'smacv2' for x in ('ts3','ts4','ts5','ts6','ts7','ts8')}}
ENV_SELECTORS = {'sc2':'smac','sc2wrapped':'smacv2','action_set_protoss':'smacv2','history_protoss':'smacv2','sequential_protoss':'smacv2','exact_protoss':'smacv2','hetero_protoss':'smacv2','ts4_protoss':'smacv2','particle':'mpe','stag_hunt':'stag-hunt'}

def dumps(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
def digest(x): return hashlib.sha256(dumps(x).encode()).hexdigest()[:16]
def load_json(text):
    try: return json.loads(text)
    except (ValueError, TypeError): return {}
def present(x): return x is not None and x != '' and x != UNKNOWN

def normalize_status(value):
    return {'complete':'completed','completed':'completed','failed':'failed','error':'failed','interrupted':'interrupted','stopped':'interrupted','running':'running_reported','started':'running_reported'}.get(str(value).lower(), UNKNOWN)

def literal_assignment(text, name):
    try:
        for node in ast.parse(text).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
                return ast.literal_eval(node.value)
    except (SyntaxError,ValueError,TypeError):pass
    return None

def arguments(argv):
    result={}
    for i,arg in enumerate(argv[:-1]):
        if arg.startswith('--') and not argv[i+1].startswith('--'):
            result[arg[2:].replace('-','_')]=argv[i+1]
    return result

def declared_inputs(value):
    ids=set();paths=set()
    def visit(x,key=''):
        if isinstance(x,dict):
            for k,v in x.items():visit(v,k)
        elif isinstance(x,list):
            for v in x:visit(v,key)
        elif isinstance(x,str):
            if key in ('run_id','parent_run_id','input_run_id','source_run_id','input_run_ids','run_ids','source_run_ids'):ids.add(x)
            if key in ('parent','baseline'):
                ids.add(x.rstrip('/').split('/')[-1])
                if '/' in x:paths.add(x)
            if any(k in key for k in ('path','input','source','checkpoint','pool','root')) and ('/' in x or x.endswith('.th')):paths.add(x)
    visit(value)
    return ids,paths

def analysis_payload(record):
    """Separate consumed inputs from output identity and unused protocol references."""
    spec=record.get('spec',{})
    if spec:
        args=arguments(spec.get('argv',[]))
        args.pop('run_id',None)
        metadata={k:v for k,v in spec.get('metadata',{}).items()
                  if k in ('parent','baseline','input_run_ids','source_run_ids','input','inputs','source','checkpoint')}
        return dict(arguments=args,metadata=metadata)
    launch=record.get('launch',{})
    request={k:v for k,v in launch.get('request',{}).items()
             if k in ('parent','checkpoint_path','input_run_ids','input','inputs','source')}
    config={k:v for k,v in record.get('config',{}).items()
            if k in ('parent','checkpoint_path','input_run_ids','input','inputs','source')}
    payload=dict(location=record.get('run_dir'),request=request,config=config)
    if launch.get('request',{}).get('suite')=='network_similarity':
        payload['protocol']=record.get('protocol',{})
    return payload

class Source:
    """Filesystem or immutable Git tree. Blob caching avoids repeated branch reads."""
    blobs = {}
    def __init__(self, root, commit=None):
        self.root, self.commit = Path(root), commit
        self.tree = {}
        self.cache = {}
        if commit:
            data = subprocess.check_output(['git','ls-tree','-rz',commit,'--','docs/results','configs','src','vendor/smacv2/smacv2/env/starcraft2/maps'],cwd=root)
            for item in data.split(b'\0'):
                if item:
                    meta, path = item.split(b'\t',1)
                    self.tree[path.decode()] = meta.decode().split()[2]
    def paths(self):
        if self.commit: return sorted(self.tree)
        return sorted(str(p.relative_to(self.root)) for folder in ('configs','docs/results','src','vendor/smacv2/smacv2/env/starcraft2/maps') for p in (self.root/folder).rglob('*') if p.is_file() and '.git' not in p.parts)
    def exists(self, path): return path in self.tree if self.commit else (self.root/path).is_file()
    def text(self, path):
        if not self.exists(path): return ''
        if not self.commit: return (self.root/path).read_text(errors='replace')
        blob = self.tree[path]
        if blob not in self.blobs:
            self.blobs[blob] = subprocess.check_output(['git','cat-file','blob',blob],cwd=self.root).decode(errors='replace')
        return self.blobs[blob]
    def json(self,path):
        key=('json',path)
        if key not in self.cache:self.cache[key]=load_json(self.text(path))
        return self.cache[key]
    def locator(self,path): return 'git:'+self.commit+':'+path if self.commit else path
    def csv(self,path):
        key=('csv',path)
        if key not in self.cache:self.cache[key]=list(csv.DictReader(io.StringIO(self.text(path))))
        return self.cache[key]

def environment(config, protocol, request, binding=UNKNOWN):
    explicit = config.get('environment_id') or protocol.get('environment_id') or binding
    if explicit != UNKNOWN: return explicit
    return ENV_SELECTORS.get(config.get('env'), SUITE_ENVS.get(request.get('suite'), UNKNOWN))

def task_spec(config, protocol, env, request=None):
    """Only structured task definitions. No run/file-name classification."""
    request = request or {}
    ea = config.get('env_args') or {}
    tasks = protocol.get('tasks') or ea.get('rosters') or {}
    order = protocol.get('order') or protocol.get('task_order') or config.get('task_schedule') or protocol.get('task_schedule')
    stages = protocol.get('stages')
    if isinstance(stages,list) and stages:
        order = [s.get('task',UNKNOWN) if isinstance(s,dict) else s for s in stages]
    if not order and isinstance(config.get('tasks'),list): order = config['tasks']
    scenario = request.get('scenario') or config.get('scenario')
    if not order and scenario in protocol.get('scenarios', {}):
        order = [stage[1] for stage in protocol['scenarios'][scenario]]
    if not order and config.get('env') == 'hetero_protoss' and config.get('branch_task') in tasks and 'A' in tasks:
        order = ['A', config['branch_task'], 'A']
    unordered = not order and isinstance(tasks,dict) and bool(tasks)
    if unordered: order = sorted(tasks)
    map_name = ea.get('map_name') or config.get('map_name')
    if not order and map_name: order = [map_name]
    if not order: return None
    if not isinstance(order,list): return None
    caps = ea.get('capability_config') or {}
    shape = {k:caps[k] for k in ('n_units','n_enemies') if k in caps}
    settings = {k:ea[k] for k in ('difficulty','step_mul','game_version','obs_all_health','use_unit_ranges','action_mask') if k in ea}
    # Budget, method, seed and data-source paths are protocol attributes, not task identity.
    definitions=[]
    for label in order:
        definition = tasks.get(label,label) if isinstance(tasks,dict) and isinstance(label,str) else label
        if isinstance(definition,list): definition={'homogeneous_team_counts':definition}
        definitions.append({'environment_id':env,'map':map_name or UNKNOWN,'definition':definition,'shape':shape,'environment_settings':settings})
    return dict(labels=order,definitions=definitions,kind='task_collection' if unordered else 'task_sequence' if len(definitions)>1 else 'task')

def started(config, statuses, launch):
    if launch.get('request',{}).get('dry_run'): return False
    if any(float(s.get(k,0) or 0)>0 for s in statuses.values() if isinstance(s,dict) for k in ('t_env','trained_steps','updates','episode')): return True
    # A failed process attempt is retained, without claiming any training progress.
    return bool(launch.get('started') and statuses.get('launcher_status.json',{}).get('state') in ('failed','interrupted'))

def status_from_runtime(statuses):
    if (statuses.get('training_complete.json',{}).get('t_env',0) or 0)>0:return 'completed'
    s=statuses.get('status.json',{})
    result=normalize_status(s.get('status',s.get('state')))
    if result != UNKNOWN: return result
    launcher=statuses.get('launcher_status.json',{})
    if launcher.get('exit_code') not in (None,0): return 'failed'
    return UNKNOWN  # exited/exit_code=0 is process termination, not training completion.

def refs_in(value, known):
    text = dumps(value)
    # Match complete run tokens in explicit metadata paths/fields, not seed/name guesses.
    return sorted(set(re.findall(r'[A-Za-z0-9_.-]+', text)) & known.keys())


def canonical_ast(node):
    """Version 3: normalize Python 3.8 Index wrappers and optional AST metadata."""
    if isinstance(node,ast.Index):
        return canonical_ast(node.value)
    if isinstance(node,ast.AST):
        return dict(node=type(node).__name__,fields={
            key:canonical_ast(value) for key,value in ast.iter_fields(node)
            if value is not None and not (key=='type_params' and value==[])})
    if isinstance(node,list):
        return [canonical_ast(value) for value in node]
    return node


def registry_key(target):
    """Read the same literal key from pre-3.9 and modern Subscript nodes."""
    node=target.slice
    if isinstance(node,ast.Index):
        node=node.value
    return ast.literal_eval(node)


def backbone_signature(text, class_name, agent=False, full_class=False):
    """Version 3 retains version 2 coverage with interpreter-independent encoding."""
    try:
        tree=ast.parse(text)
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == class_name)
        if full_class:
            return hashlib.sha256(dumps(canonical_ast(cls)).encode()).hexdigest()
        nodes = []
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            if fn.name == 'forward':
                nodes.append(canonical_ast(fn))
            if agent and fn.name == '__init__':
                nodes.extend(canonical_ast(n) for n in fn.body
                             if isinstance(n, ast.Assign) and any(
                                 isinstance(t, ast.Attribute) and t.attr in ('net', 'rnn', 'fc2', 'hidden_dim')
                                 for t in n.targets))
        return hashlib.sha256(dumps(nodes).encode()).hexdigest() if nodes else UNKNOWN
    except (SyntaxError, StopIteration):
        return UNKNOWN


class ArchitectureResolver:
    """Join resolved config fields to reviewed backbones in the run's Git version."""
    def __init__(self, root, registry):
        self.root, self.registry = Path(root), registry
        self.cache = {}
        if registry.get('schema_version') != 1 or registry.get('resolver_version') != 3:
            raise ValueError('Unsupported architecture registry/resolver version')

    def code(self, commit, path):
        key = commit, path
        if key not in self.cache:
            result = subprocess.run(['git', 'show', commit + ':' + path], cwd=self.root,
                                    capture_output=True, text=True)
            self.cache[key] = result.stdout if result.returncode == 0 else ''
        return self.cache[key]

    def resolve(self, config, commit, config_path, source_hashes=None):
        agent = config.get('agent', UNKNOWN)
        width = config.get('rnn_hidden_dim', UNKNOWN)
        mixer = config.get('mixer', UNKNOWN)
        result = dict(architecture_id=UNKNOWN, agent_type=agent, hidden_dim=width, mixer_type=mixer,
                      architecture_source=dict(config=config_path, code_commit=commit,
                                               basis='resolved_config_and_git_backbone',
                                               source_verification='not_checked', reason=UNKNOWN))
        source = result['architecture_source']
        matches = [a for a in self.registry['architectures']
                   if (agent, width, mixer) == (a['agent_type'], a['hidden_dim'], a['mixer_type'])]
        if len(matches) != 1:
            source['reason'] = 'missing_or_unregistered_structure'
            return result
        if config.get('mac') != 'basic_mac':
            source['reason'] = 'missing_or_unregistered_controller'
            return result
        if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', str(commit)):
            source['reason'] = 'missing_code_version'
            return result
        architecture = matches[0]
        path = architecture['implementation_paths'][0]
        registry_path = 'src/modules/agents/__init__.py'
        mixer_path = 'src/modules/mixers/vdn.py'
        gate_path = 'src/modules/layers/act_layer.py'
        texts = {p: self.code(commit, p) for p in architecture['implementation_paths']}
        source['implementations'] = [dict(path='git:' + commit + ':' + p,
                                         sha256=hashlib.sha256(text.encode()).hexdigest())
                                     for p, text in texts.items() if text]
        if not all(texts.values()):
            source['reason'] = 'unavailable_code_version'
            return result
        source['source_verification']='git_commit_only'
        hashes=source_hashes or {}
        if not isinstance(hashes,dict):
            source['reason']='unsupported_source_hash_format'
            return result
        mismatched=[p for p,text in texts.items() if p in hashes and
                    hashlib.sha256(text.encode()).hexdigest()!=hashes[p]]
        if mismatched:
            source.update(reason='launch_source_hash_mismatch',mismatched_paths=mismatched)
            return result
        if all(p in hashes for p in texts):
            source['source_verification']='launch_hashes_match_git'
        try:
            registry_nodes=ast.parse(texts[registry_path]).body
            gate_imported=any(isinstance(n,ast.ImportFrom) and n.module=='modules.layers.act_layer'
                              and any(a.name=='ActivateLayer' and a.asname in (None,a.name) for a in n.names)
                              for n in ast.parse(texts[path]).body)
            imported=any(isinstance(n,ast.ImportFrom) and n.level==1 and
                         n.module==Path(path).stem and any(
                             a.name==architecture['agent_class'] and a.asname in (None,a.name)
                             for a in n.names) for n in registry_nodes)
            registered = imported and any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Name)
                             and n.value.id == architecture['agent_class'] and any(
                                 isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                                 and t.value.id == 'REGISTRY' and registry_key(t) == agent
                                 for t in n.targets)
                             for n in registry_nodes)
        except (SyntaxError, ValueError, TypeError):
            registered = False
        if not registered:
            source['reason'] = 'agent_registry_mismatch'
        elif (not gate_imported
              or backbone_signature(texts[path], architecture['agent_class'], True) != architecture['backbone_signature']
              or backbone_signature(texts[mixer_path], 'VDNMixer') != architecture['mixer_signature']
              or backbone_signature(texts[gate_path], 'ActivateLayer', full_class=True) != architecture['gate_signature']):
            source['reason'] = 'unrecognized_backbone_implementation'
        else:
            result['architecture_id'] = architecture['id']
            source['reason'] = 'matched_registered_backbone'
        return result

class Builder:
    def __init__(self, root, include_local=False):
        self.root=Path(root);self.catalog=load_json((self.root/'docs/results/catalog.json').read_text())
        self.include_local=include_local
        spec=self.catalog.get('global_index',{})
        self.sources=[Source(root)] + [Source(root,b['commit']) for b in spec.get('branch_sources',[])]
        self.runs=defaultdict(list);self.analyses=[];self.issues=[];self.tasks={};self.benchmarks={};self.sequences={};self.protocols=[];self.catalogs=[];self.runtime_analyses=[];self.scheduler_rows=[];self.collection=[]
        self.bindings=spec.get('experiment_bindings',{})
        self.raw=spec.get('runtime_metadata','docs/results/runtime_records.jsonl')
        self.envs=load_json((self.root/'docs/environments/environments.json').read_text())
        self.architectures=load_json((self.root/'docs/architectures/architectures.json').read_text())
        self.architecture_resolver=ArchitectureResolver(root,self.architectures)
    def benchmark(self,config,protocol,request,binding,source):
        env=environment(config,protocol,request,binding)
        defs=task_spec(config,protocol,env,request)
        if env==UNKNOWN or not defs: return env,UNKNOWN
        kind=defs['kind'];labels=defs['labels'];defs=defs['definitions']
        tids=[]
        for label,definition in zip(labels,defs):
            tid=env+'-task-'+digest(definition)
            self.tasks.setdefault(tid,dict(id=tid,**definition,labels=[],sources=[]))
            if label not in self.tasks[tid]['labels']:self.tasks[tid]['labels'].append(label)
            if source not in self.tasks[tid]['sources']:self.tasks[tid]['sources'].append(source)
            tids.append(tid)
        if kind=='task_collection':tids=sorted(set(tids))
        bid=env+'-'+kind+'-'+digest(tids)
        b=self.benchmarks.setdefault(bid,dict(id=bid,environment_id=env,kind=kind,task_ids=tids,name=(' + ' if kind=='task_collection' else ' → ').join(map(str,labels)),protocol_ids=[],sources=[]))
        if kind=='task_sequence':
            sequence_id=env+'-sequence-'+digest(tids)
            self.sequences.setdefault(sequence_id,dict(id=sequence_id,environment_id=env,task_ids=tids))
            b['task_sequence_id']=sequence_id
        pid=protocol.get('id') or config.get('protocol_id') or UNKNOWN
        if pid!=UNKNOWN and pid not in b['protocol_ids']:b['protocol_ids'].append(pid)
        if source not in b['sources']:b['sources'].append(source)
        return env,bid
    def candidate(self,run_id,row,config,protocol,request,source,experiment,results,priority=1,status=None,evidence='published_run_record'):
        row,config,protocol,request,source,results=redact([row,config,protocol,request,source,results])
        binding=self.bindings.get(experiment,{}).get('environment_id',UNKNOWN)
        # Native single-agent snapshots are flat config objects.
        env,bid=self.benchmark(config,protocol,request,binding,source)
        raw_method=next((v for v in (config.get('method'),request.get('method'),protocol.get('method'),row.get('method')) if isinstance(v,str) and v),UNKNOWN)
        algorithm=config.get('algorithm') or protocol.get('algorithm') or request.get('algorithm') or row.get('algorithm') or self.bindings.get(experiment,{}).get('algorithm',UNKNOWN)
        params=config.get(raw_method) or protocol.get(raw_method) or (protocol.get('method') if isinstance(protocol.get('method'),dict) else {}) or {}
        algorithm_config=algorithm
        algorithm='vdn' if algorithm in VDN_CONFIGS else algorithm
        variant=dict(identity_version=2,algorithm=algorithm,method=raw_method,parameters=params)
        for k in ('optimizer','lr','replay_on_switch','epsilon_on_switch'):
            v=config.get(k,protocol.get(k,row.get(k)))
            if present(v):variant[k]=v
        for k in ('group','ts3_group','ts4_group','condition'):
            if present(request.get(k)) and request[k]!='N':variant[k]=request[k]
        role=row.get('role') or config.get('role') or config.get('preference_role') or request.get('preference_role') or request.get('ts3_role') or request.get('hetero_role') or 'independent'
        metadata={**load_json(request.get('experiment_metadata') or '{}'), **row.get('_metadata',{})}
        attempt=row.get('attempt') or metadata.get('attempt')
        attempt_source='metadata' if attempt else UNKNOWN
        if not attempt:
            m=re.search(r'(?:^|_)attempt(\d+)(?:$|_)',run_id)
            if m:attempt=int(m.group(1));attempt_source='run_id_label'
        parent=row.get('parent_run_id') or request.get('parent') or config.get('parent')
        if parent:parent=str(parent).replace('\\','/').rstrip('/').split('/')[-1]
        seed=config.get('seed',request.get('seed',row.get('seed',UNKNOWN)))
        commit=row.get('commit') or row.get('source_commit') or config.get('launch_git_commit') or UNKNOWN
        result=dict(run_id=run_id,experiment_id=experiment,experiment_ids=[experiment],experiment_group_id=metadata.get('experiment_group_id',row.get('experiment_group_id',UNKNOWN)),environment_id=env,benchmark_id=bid,protocol_id=protocol.get('id') or config.get('protocol_id') or UNKNOWN,protocol_fingerprint=digest(protocol or config) if protocol or config else UNKNOWN,method=row.get('method') or raw_method,method_id=raw_method,algorithm=algorithm,method_variant_id='method-'+digest(variant),method_parameters=variant,seed=str(seed),status=status or normalize_status(row.get('status')),result_status=UNKNOWN,git_commit=commit,config_path=row.get('resolved_config',UNKNOWN),parent_run_id=str(parent) if parent else ('none' if 'parent_run_id' in row or 'parent' in request or 'parent' in config else UNKNOWN),role=role,attempt=str(attempt) if attempt else UNKNOWN,attempt_source=attempt_source,logical_run_id=digest([metadata.get('experiment_group_id'),metadata.get('branch_id'),str(seed)]) if metadata.get('experiment_group_id') and metadata.get('branch_id') else UNKNOWN,retry_of=row.get('retry_of') or metadata.get('retry_of') or UNKNOWN,run_path=row.get('run_dir') or UNKNOWN,result_paths=results,source_records=[source],method_aliases=[row.get('method') or raw_method],conflicts=[],execution_evidence=evidence,trained_steps=row.get('trained_steps') or row.get('new_steps') or row.get('completed_global_steps') or UNKNOWN,
                    replay_on_switch=config.get('replay_on_switch',protocol.get('replay_on_switch',row.get('replay_on_switch',UNKNOWN))),
                    epsilon_on_switch=config.get('epsilon_on_switch',protocol.get('epsilon_on_switch',row.get('epsilon_on_switch',UNKNOWN))),
                    completed_phases=row.get('_completed_phases',[]),scheduler_task_ids=[],scheduler_dependencies=[],
                    observed_at=row.get('_observed_at',UNKNOWN),_priority=priority)
        result['algorithm_config']=algorithm_config
        result.update(self.architecture_resolver.resolve(config,commit,result['config_path'],row.get('_source_sha256')))
        self.runs[run_id].append(result)
    def published(self):
        seen=set()
        for si,src in enumerate(self.sources):
            catalog=src.json('docs/results/catalog.json')
            experiments=catalog.get('experiments',[])
            covered=set()
            for exp in experiments:
                eid=exp['id'];files=exp.get('files',{});self.catalogs.append((src,exp))
                for rel in [exp.get('report')]+list(files.values()):
                    if isinstance(rel,str) and not src.exists('docs/results/'+rel):
                        self.issues.append(dict(kind='missing_catalog_artifact',experiment_id=eid,source=src.locator('docs/results/'+rel)))
                run_paths=[v for k,v in files.items() if k=='runs' or k.endswith('_runs')]
                for rel in run_paths:
                    path='docs/results/'+rel;covered.add(path)
                    self.published_csv(src,path,exp,si,seen)
            # Include unregistered nested runs.csv, using the nearest catalog report directory.
            for path in src.paths():
                if path.startswith('docs/results/') and path.endswith('/runs.csv') and path not in covered:
                    parents=[e for e in experiments if path.startswith('docs/results/'+str(PurePosixPath(e.get('report','')).parent)+'/')]
                    exp=max(parents,key=lambda e:len(e.get('report','')),default={'id':UNKNOWN,'files':{}})
                    self.published_csv(src,path,exp,si,seen)
            for path in src.paths():
                if not path.startswith('configs/') or not path.endswith('.json'):continue
                p=src.json(path)
                if not isinstance(p,dict):continue
                req={'suite':'continual'} if path.startswith(('configs/continual/','configs/archive/continual/')) else {}
                # A protocol's code-backed suite declaration, not its filename.
                binding=self.bindings.get('protocol:'+p.get('id',''),{}).get('environment_id',UNKNOWN)
                env,bid=self.benchmark({},p,req,binding,src.locator(path))
                self.protocols.append(dict(path=src.locator(path),protocol_id=p.get('id',UNKNOWN),environment_id=env,benchmark_id=bid))
    def published_csv(self,src,path,exp,si,seen):
        key=(src.tree.get(path) if src.commit else hashlib.sha256(src.text(path).encode()).hexdigest(),exp['id'])
        if key in seen:return
        seen.add(key)
        if not src.exists(path):self.issues.append(dict(kind='missing_source',source=src.locator(path)));return
        for number,row in enumerate(src.csv(path),2):
            rid=row.get('run_id')
            if not rid:self.issues.append(dict(kind='missing_run_id',source=src.locator(path),row=number));continue
            steps=row.get('trained_steps') or row.get('new_steps') or row.get('new_training_steps') or row.get('completed_global_steps') or row.get('cumulative_steps')
            try: positive=float(steps or 0)>0
            except ValueError:positive=False
            # runs.csv denotes actual runs under the existing DATA.md contract. Explicit planned rows are excluded.
            if row.get('status') in ('planned','pending','queued') and not positive:
                self.issues.append(dict(kind='unexecuted_record',run_id=rid,source=src.locator(path)));continue
            if row.get('activity_kind') in ('evaluation','offline_analysis','summary'):
                continue
            config_path=str(PurePosixPath(path).parent/(row.get('config') or 'missing'))
            snapshot=src.json(config_path)
            config=snapshot.get('config',snapshot) if isinstance(snapshot,dict) else {}
            protocol=snapshot.get('protocol',{}) if isinstance(snapshot,dict) else {}
            row=dict(row,resolved_config=src.locator(config_path) if src.exists(config_path) else UNKNOWN)
            if not protocol and isinstance(config,dict) and config.get('task_schedule'):protocol=dict(task_schedule=config['task_schedule'])
            row['commit']=row.get('commit') or row.get('source_commit') or snapshot.get('launch_git_commit')
            row['_source_sha256']=snapshot.get('source_sha256')
            stage_path=exp.get('files',{}).get('stage_metrics')
            stage_data=[r for r in src.csv('docs/results/'+stage_path) if r.get('run_id')==rid] if stage_path else []
            row['_completed_phases']=sorted({int(r.get('phase',r.get('stage'))) for r in stage_data if str(r.get('phase',r.get('stage',''))).isdigit()})
            # Missing protocol order can be recovered from explicit endpoint training-task columns.
            if not task_spec(config,protocol,environment(config,protocol,{},self.bindings.get(exp['id'],{}).get('environment_id',UNKNOWN))):
                observed={int(r.get('phase',r.get('stage'))):r.get('training_task',r.get('trained_game')) for r in stage_data
                          if str(r.get('phase',r.get('stage',''))).isdigit() and r.get('training_task',r.get('trained_game'))}
                if observed:
                    protocol=dict(protocol,order=[observed[k] for k in sorted(observed)],sequence_source='observed_stage_metrics')
            if self.bindings.get(exp['id'],{}).get('environment_id')=='gym-classic-control':
                goals=literal_assignment(src.text('src/single_agent/mountaincar.py'),'GOALS')
                if goals:protocol=dict(protocol,tasks=goals)
            results=[src.locator('docs/results/'+v) for v in exp.get('files',{}).values() if isinstance(v,str) and src.exists('docs/results/'+v)]
            if exp.get('report') and src.exists('docs/results/'+exp['report']):results.append(src.locator('docs/results/'+exp['report']))
            self.candidate(rid,row,config,protocol,{},src.locator(path)+'#row='+str(number),exp['id'],results,priority=100-si)
    def runtime(self):
        path=self.root/self.raw
        if not path.exists():return
        self.runtime_analyses=[]
        for number,line in enumerate(path.read_text().splitlines(),1):
            x=json.loads(line);kind=x['kind'];source=self.raw+'#line='+str(number)
            if kind=='collection':self.collection.append(x);continue
            if kind=='scheduler_analysis':self.runtime_analyses.append((x,source));continue
            if kind=='scheduler_training':self.scheduler_rows.append((x,source));continue
            launch=x.get('launch',{});req=launch.get('request',{});config=x.get('config',{});protocol=x.get('protocol') or launch.get('details',{}).get('protocol') or {}
            statuses=x.get('status_documents',{})
            if '/evaluation/' in x['run_dir'] or req.get('suite')=='network_similarity' or req.get('ts3_role')=='eval_fwt' or config.get('evaluate'):
                aid=x['host']+':'+x['run_dir'] if '/evaluation/' in x['run_dir'] else req.get('run_id') or x['run_id']
                self.runtime_analyses.append((dict(x,kind='runtime_analysis',analysis_id=aid),source));continue
            rid=req.get('run_id') or x['run_id']
            if not started(config,statuses,launch):
                self.issues.append(dict(kind='unconfirmed_launch',run_id=rid,source=source));continue
            row={'seed':req.get('seed',config.get('seed',UNKNOWN)),'commit':launch.get('git_commit'), 'run_dir':x['host']+':'+x['run_dir'],'resolved_config':x['host']+':'+x['run_dir']+'/config.json','experiment_group_id':launch.get('metadata',{}).get('experiment_group_id',UNKNOWN),'completed_global_steps':statuses.get('status.json',{}).get('t_env') or statuses.get('training_complete.json',{}).get('t_env')}
            metadata=launch.get('metadata',{})
            row['_metadata']=metadata
            row['_source_sha256']=launch.get('source_sha256')
            row['_observed_at']=self.collection[-1].get('captured_at',UNKNOWN) if self.collection else UNKNOWN
            eid=UNKNOWN  # Conversation/group identity is not a Catalog experiment ID.
            self.candidate(rid,row,config,protocol,req,source,eid,[],priority=200,status=status_from_runtime(statuses),evidence='runtime_progress' if any((v.get('t_env',0) or 0)>0 for v in statuses.values()) else 'failed_process_attempt')
    def merge(self):
        final=[]
        for rid,rows in sorted(self.runs.items()):
            rows.sort(key=lambda x:-x['_priority']);r={k:v for k,v in rows[0].items() if not k.startswith('_')}
            for field in ('experiment_id','result_paths','source_records','method_aliases','completed_phases'):
                vals=[]
                for x in rows:
                    value=x[field] if isinstance(x[field],list) else [x[field]]
                    vals.extend(v for v in value if present(v))
                r[field]=sorted(set(vals)) or ([] if field in ('result_paths','completed_phases') else [UNKNOWN])
            # Public experiment memberships take precedence over a scheduler conversation UUID for navigation.
            r['experiment_id']=sorted(set(x['experiment_id'] for x in rows if x['_priority']<200 and x['experiment_id']!=UNKNOWN)) or r['experiment_id']
            r['experiment_ids']=r['experiment_id']
            r['experiment_id']=next((x['experiment_id'] for x in rows if x['_priority']<200 and x['experiment_id']!=UNKNOWN),r['experiment_ids'][0])
            for field in ('experiment_group_id','environment_id','benchmark_id','protocol_id','git_commit','parent_run_id','seed','config_path','method_id','algorithm','attempt','attempt_source','logical_run_id','retry_of','trained_steps','replay_on_switch','epsilon_on_switch','observed_at'):
                candidates=[x[field] for x in rows if present(x[field])]
                if candidates:r[field]=candidates[0]
                # Runtime/protocol identities retain conflicting historical evidence instead of silently replacing it.
                if field in ('environment_id','benchmark_id','git_commit','parent_run_id','seed') and len(set(map(str,candidates)))>1:
                    conflict={'field':field,'values':sorted(set(map(str,candidates)))};r['conflicts'].append(conflict)
                    if field in ('environment_id','benchmark_id','seed'):r[field]=UNKNOWN
            statuses=[x['status'] for x in rows if x['status']!=UNKNOWN]
            terminal=set(statuses)&{'completed','failed','interrupted'}
            if len(terminal)>1:r['status']=UNKNOWN;r['conflicts'].append({'field':'status','values':sorted(terminal)})
            elif terminal:r['status']=next(iter(terminal))
            elif statuses:r['status']=statuses[0]
            # Prefer published method labels; preserve variants/configuration separately.
            published=[x for x in rows if x['_priority']<200]
            if published:r['method']=published[0]['method']
            # Use one internally consistent variant object, rather than mixing snapshots.
            variant=next((x for x in rows if x['method_id']==r['method_id'] and x['method_id']!=UNKNOWN),rows[0])
            r['method_parameters']=variant['method_parameters'];r['method_variant_id']=variant['method_variant_id']
            r['algorithm_config']=variant['algorithm_config']
            # Architecture evidence is atomic: never mix width from one snapshot with another agent.
            architecture=next((x for x in rows if x['architecture_id']!=UNKNOWN),rows[0])
            for field in ('architecture_id','agent_type','hidden_dim','mixer_type','architecture_source'):
                r[field]=architecture[field]
            disagreements=[]
            for field in ('architecture_id','agent_type','hidden_dim','mixer_type'):
                values=sorted({str(x[field]) for x in rows if present(x[field])})
                if len(values)>1:
                    disagreements.append(dict(field=field,values=values))
                    r[field]=UNKNOWN
            if disagreements:
                r['architecture_id']=UNKNOWN
                r['conflicts'].extend(disagreements)
                r['architecture_source']=dict(reason='conflicting_architecture_records',records=[x['architecture_source'] for x in rows])
            unsafe=[x['architecture_source'] for x in rows if x['architecture_source'].get('reason') in
                    ('launch_source_hash_mismatch','unrecognized_backbone_implementation','agent_registry_mismatch',
                     'unsupported_source_hash_format')]
            if unsafe and not disagreements:
                r['architecture_id']=UNKNOWN
                r['architecture_source']=dict(reason='unverified_implementation_evidence',records=unsafe)
            if r['conflicts']:self.issues.append(dict(kind='conflicting_run_records',run_id=rid,conflicts=r['conflicts']))
            final.append(r)
        self.final=final;return final
    def complete_results(self):
        for r in self.final:
            n=len(self.benchmarks.get(r['benchmark_id'],{}).get('task_ids',[]))
            expected={0} if r['role'] in ('prefix','shared-prefix') else set(range(n))
            available=set(r['completed_phases'])
            parent=next((p for p in self.final if p['run_id']==r['parent_run_id']),None)
            if parent:
                own_tasks=self.benchmarks.get(r['benchmark_id'],{}).get('task_ids',[])
                parent_tasks=self.benchmarks.get(parent['benchmark_id'],{}).get('task_ids',[])
                available.update(i for i in parent['completed_phases']
                                 if i < min(len(own_tasks),len(parent_tasks)) and own_tasks[i]==parent_tasks[i])
            if r['status']=='completed' and n and expected <= available and r['result_paths']:
                r['result_status']='reported_prefix_result' if r['role'] in ('prefix','shared-prefix') else 'reported_complete_result'
            elif r['result_paths']:r['result_status']='published_partial_or_unverified'
            else:r['result_status']='runtime_only'

    def analyses_index(self):
        known={r['run_id']:r for r in self.final};result={}
        seen=set()
        for src,exp in self.catalogs:
            # Offline datasets also exist beside training comparisons; index explicit analysis files only.
            if exp.get('category') not in ('behavioral-analysis','diagnostics'):continue
            key=(exp['id'],dumps(exp.get('files',{})))
            if key in seen:continue
            seen.add(key);payload=[];paths=[];has_data=False
            for value in exp.get('files',{}).values():
                p='docs/results/'+value
                if not src.exists(p):continue
                paths.append(src.locator(p))
                if p.endswith('.json'):payload.append(src.json(p))
                elif p.endswith('.csv'):
                    rows=src.csv(p);has_data=has_data or bool(rows);payload.extend(rows)
            explicit,input_paths=declared_inputs(payload)
            inputs=sorted(set(refs_in(payload,known)) | (explicit & known.keys()))
            self.add_analysis(result,exp['id'],exp['id'],'offline_analysis','published_data' if has_data else UNKNOWN,inputs,paths,[src.locator('docs/results/'+exp.get('report',''))],known,explicit,input_paths)
        for x,source in self.runtime_analyses:
            spec=x.get('spec',{});aid=x['analysis_id']
            payload=analysis_payload(x)
            explicit,input_paths=declared_inputs(payload)
            explicit.update(self.analysis_dependencies(aid))
            inputs=sorted(set(refs_in(payload,known)) | (explicit & known.keys()))
            self.add_analysis(result,aid,UNKNOWN,spec.get('type','evaluation'),x.get('state',UNKNOWN),inputs,[x.get('host','')+':'+(spec.get('run_dir') or x.get('run_dir',''))],[source],known,explicit,input_paths)
        self.analyses=sorted(result.values(),key=lambda r:r['analysis_id'])
    def add_analysis(self,result,aid,eid,kind,status,inputs,paths,sources,known,explicit=(),input_paths=()):
        r=result.setdefault(aid,dict(analysis_id=aid,experiment_id=eid,activity_kind=kind,status=status,input_run_ids=[],unresolved_inputs=[],input_sources=[],environment_ids=[],benchmark_ids=[],result_paths=[],source_records=[]))
        if r['status']==UNKNOWN and status!=UNKNOWN:r['status']=status
        for key,values in [('input_run_ids',inputs),('result_paths',paths),('source_records',sources)]:r[key]=sorted(set(r[key]+values))
        r['environment_ids']=sorted({known[i]['environment_id'] for i in r['input_run_ids']}) or [UNKNOWN]
        r['benchmark_ids']=sorted({known[i]['benchmark_id'] for i in r['input_run_ids']}) or [UNKNOWN]
        r['unresolved_inputs']=sorted(set(r['unresolved_inputs']) | (set(explicit)-known.keys()))
        if r['input_run_ids']:r['unresolved_inputs']=[i for i in r['unresolved_inputs'] if i!=UNKNOWN]
        if not r['input_run_ids'] and not r['unresolved_inputs']:r['unresolved_inputs']=[UNKNOWN]
        r['input_sources']=sorted(set(r['input_sources']) | set(input_paths))
    def code_definitions(self):
        """Read literal registries and configuration declarations, never import environments."""
        seen=set()
        for src in self.sources:
            for path,name,env in (
                ('vendor/smacv2/smacv2/env/starcraft2/maps/smac_maps.py','map_param_registry','smacv2'),
                ('src/single_agent/minatar_env.py','GAMES','minatar'),
                ('src/single_agent/mountaincar.py','GOALS','gym-classic-control')):
                text=src.text(path)
                if not text or (path,text) in seen:continue
                seen.add((path,text))
                values=literal_assignment(text,name)
                if isinstance(values,dict):
                    for label,definition in values.items():
                        config={'env_args':{'map_name':label}} if env=='smacv2' else {}
                        self.benchmark(config,dict(tasks={label:definition},order=[label]),{},env,src.locator(path)+'#'+name)
                elif isinstance(values,(list,tuple)):
                    for label in values:self.benchmark({},dict(order=[label]),{},env,src.locator(path)+'#'+name)
            # MPE's scenario loader explicitly resolves this module and Scenario.make_world.
            path='src/envs/particle/scenarios/simple_tag_coop.py'
            if src.exists(path) and (path,src.text(path)) not in seen:
                seen.add((path,src.text(path)))
                definition={'scenario':'simple_tag_coop','parameters':'code_defaults; runtime unverified'}
                self.benchmark({},dict(order=['simple_tag_coop'],tasks={'simple_tag_coop':definition}),{},'mpe',src.locator(path)+'#Scenario.make_world')
            path='src/envs/stag_hunt/stag_hunt.py'
            if src.exists(path) and (path,src.text(path)) not in seen:
                seen.add((path,src.text(path)))
                self.benchmark({},dict(order=['StagHunt parameterized template'],tasks={'StagHunt parameterized template':{'parameters':UNKNOWN,'implementation':'StagHunt'}}),{},'stag-hunt',src.locator(path)+'#StagHunt')
            # Explicit YAML map declarations are code/config evidence only, never run evidence.
            for path in ('src/config/envs/sc2.yaml','src/config/envs/sc2_beta.yaml'):
                text=src.text(path)
                match=re.search(r'^\s*map_name:\s*[\"\']?([^\s\"\']+)',text,re.M)
                if match:self.benchmark({'env_args':{'map_name':match.group(1)}},{},{},'smac',src.locator(path))
            for path in src.paths():
                if path.startswith('configs/') and path.endswith('.json'):
                    protocol=src.json(path)
                    if isinstance(protocol,dict):
                        for scenario,stages in protocol.get('scenarios',{}).items():
                            if isinstance(stages,list) and all(isinstance(x,list) and len(x)==2 for x in stages):
                                self.benchmark({},dict(protocol,order=[x[1] for x in stages]),{},'smacv2',src.locator(path)+'#scenario='+scenario)

    def local_runtime(self):
        """Read accessible metadata in registered worktrees; no scheduler connection."""
        try:
            data=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=self.root,text=True)
            roots=[Path(line[9:]) for line in data.splitlines() if line.startswith('worktree ')]
        except subprocess.CalledProcessError:
            roots=[self.root]
        existing=set()
        for root in roots:
            directory=root/'runs'
            if not directory.is_dir():continue
            for path in sorted(directory.rglob('status.json')):
                run=path.parent.resolve()
                if run in existing or 'evaluation' in run.parts:continue
                existing.add(run)
                launch=load_json((run/'launch.json').read_text()) if (run/'launch.json').exists() else {}
                cfg=load_json((run/'config.json').read_text()) if (run/'config.json').exists() else {}
                statuses={name:load_json((run/name).read_text()) for name in ('status.json','launcher_status.json','training_complete.json') if (run/name).exists()}
                req=launch.get('request',{})
                if not started(cfg,statuses,launch):continue
                if req.get('suite')=='network_similarity' or cfg.get('evaluate'):continue
                protocol=load_json((run/'protocol.json').read_text()) if (run/'protocol.json').exists() else launch.get('details',{}).get('protocol',{})
                row=dict(seed=cfg.get('seed',req.get('seed',UNKNOWN)),commit=launch.get('git_commit'),run_dir=str(run),resolved_config=str(run/'config.json'),_metadata=launch.get('metadata',{}),_source_sha256=launch.get('source_sha256'))
                self.candidate(req.get('run_id') or run.name,row,cfg,protocol,req,str(run/'status.json'),UNKNOWN,[],priority=300,status=status_from_runtime(statuses),evidence='local_runtime_metadata')

    def scheduler_links(self):
        known={r['run_id']:r for r in self.final}
        self.scheduler_graph={}
        for x,source in self.scheduler_rows+self.runtime_analyses:
            if x.get('kind','').startswith('scheduler_'):
                self.scheduler_graph[x['analysis_id']]=x.get('spec',{})
        for x,source in self.scheduler_rows:
            spec=x.get('spec',{});argv=spec.get('argv',[]);rid=arguments(argv).get('run_id') or (Path(spec['run_dir']).name if spec.get('run_dir') else None)
            if argv and argv[0] in ('/usr/bin/true','tools/true'):
                self.issues.append(dict(kind='scheduler_import_placeholder',task_id=x['analysis_id'],source=source));continue
            if rid in known:
                row=known[rid]
                row['scheduler_task_ids'].append(x['analysis_id'])
                row['scheduler_dependencies']=sorted(set(row['scheduler_dependencies']+spec.get('dependencies',[])))
                row['source_records']=sorted(set(row['source_records']+[source]))
            else:self.issues.append(dict(kind='scheduler_without_runtime_evidence',task_id=x['analysis_id'],run_id=rid or UNKNOWN,state=x['state'],source=source))
        for row in self.final:
            row['scheduler_task_ids']=sorted(set(row['scheduler_task_ids']))

    def analysis_dependencies(self, task, seen=None):
        seen=set(seen or ())
        if task in seen:return []
        seen.add(task);spec=self.scheduler_graph.get(task,{})
        req=arguments(spec.get('argv',[]))
        if spec.get('type')=='training':
            rid=req.get('run_id') or (Path(spec['run_dir']).name if spec.get('run_dir') else None)
            return [rid] if rid else []
        result=[]
        for parent in spec.get('dependencies',[]):result.extend(self.analysis_dependencies(parent,seen))
        return result

    def build(self):
        self.published();self.code_definitions()
        if self.include_local:self.local_runtime()
        self.runtime();self.merge();self.scheduler_links();self.complete_results();self.analyses_index()
        self.issues=[x for x in self.issues if x['kind']!='scheduler_training_observation' or x['run_id'] not in self.runs]
        for r in self.final:
            if r['parent_run_id'] not in ('none',UNKNOWN) and r['parent_run_id'] not in self.runs:self.issues.append(dict(kind='missing_parent',run_id=r['run_id'],parent=r['parent_run_id']))
            missing=[k for k in ('experiment_id','environment_id','benchmark_id','protocol_id','seed','git_commit','parent_run_id') if r[k]==UNKNOWN]
            if missing:self.issues.append(dict(kind='unresolved_run_fields',run_id=r['run_id'],fields=missing))
            if r['algorithm']=='vdn' and r['architecture_id']==UNKNOWN:
                self.issues.append(dict(kind='unresolved_vdn_architecture',run_id=r['run_id'],source=r['architecture_source']))
        for a in self.analyses:
            if a['unresolved_inputs']:self.issues.append(dict(kind='unresolved_analysis_inputs',analysis_id=a['analysis_id'],inputs=a['unresolved_inputs']))
        for e in self.envs['environments']:
            rr=[r for r in self.final if r['environment_id']==e['id']]
            e['actual_run_count']=len(rr);e['reported_complete_result_count']=sum(r['result_status']=='reported_complete_result' for r in rr)
            e['evidence_level']='complete_results' if e['reported_complete_result_count'] else 'actual_runs' if rr else 'code_only'
        for b in self.benchmarks.values():
            rr=[r for r in self.final if r['benchmark_id']==b['id']]
            b['actual_run_count']=len(rr);b['reported_complete_result_count']=sum(r['result_status']=='reported_complete_result' for r in rr)
            b['evidence_level']='complete_results' if b['reported_complete_result_count'] else 'actual_runs' if rr else 'code_only'
        for task in self.tasks.values():
            task['associated_run_count']=sum(task['id'] in self.benchmarks.get(r['benchmark_id'],{}).get('task_ids',[]) for r in self.final)
        for sequence in self.sequences.values():
            sequence['actual_run_count']=sum(self.benchmarks.get(r['benchmark_id'],{}).get('task_sequence_id')==sequence['id'] for r in self.final)
        for architecture in self.architectures['architectures']:
            rr=[r for r in self.final if r['architecture_id']==architecture['id']]
            architecture['actual_run_count']=len(rr)
            architecture['seeds']=sorted({r['seed'] for r in rr if r['seed']!=UNKNOWN})
        return self


def csv_text(fields,rows):
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=fields,lineterminator='\n');w.writeheader()
    for r in redact(rows):w.writerow({k:dumps(r.get(k)) if isinstance(r.get(k),(list,dict)) else r.get(k,UNKNOWN) for k in fields})
    return out.getvalue()

def artifacts(b):
    benches=dict(schema_version=1,task_sequences=sorted(b.sequences.values(),key=lambda x:x['id']),generated_by='scripts/experiment_index.py build',tasks=sorted(b.tasks.values(),key=lambda x:x['id']),benchmarks=sorted(b.benchmarks.values(),key=lambda x:x['id']),protocol_sources=sorted(b.protocols,key=lambda x:x['path']))
    summary=dict(run_count=len(b.final),analysis_count=len(b.analyses),architecture_counts=dict(Counter(r['architecture_id'] for r in b.final)),environment_counts=dict(Counter(r['environment_id'] for r in b.final)),status_counts=dict(Counter(r['status'] for r in b.final)),result_counts=dict(Counter(r['result_status'] for r in b.final)),issue_counts=dict(Counter(x['kind'] for x in b.issues)),issues=b.issues)
    summary['collection']=b.collection
    summary['local_runtime_included']=b.include_local
    summary['branch_sources']=b.catalog.get('global_index',{}).get('branch_sources',[])
    raw=b.root/b.raw
    summary['runtime_snapshot_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else UNKNOWN
    summary=redact(summary)
    return {'docs/architectures/architectures.json':json.dumps(b.architectures,ensure_ascii=False,indent=2)+'\n','docs/environments/environments.json':json.dumps(b.envs,ensure_ascii=False,indent=2)+'\n','docs/environments/benchmarks.json':json.dumps(benches,ensure_ascii=False,indent=2)+'\n','docs/results/run_index.csv':csv_text(RUN_FIELDS,b.final),'docs/results/analysis_index.csv':csv_text(ANALYSIS_FIELDS,b.analyses),'docs/results/index_audit.json':json.dumps(summary,ensure_ascii=False,indent=2)+'\n'}

def read_runs(root):
    rows=list(csv.DictReader(io.StringIO((root/'docs/results/run_index.csv').read_text())))
    for row in rows:
        for k in ('experiment_ids','result_paths','source_records','method_aliases','conflicts','method_parameters','completed_phases','scheduler_task_ids','scheduler_dependencies','architecture_source'):row[k]=json.loads(row[k])
    return rows

def filter_runs(rows,args):
    for r in rows:
        if getattr(args,'environment',None) and r['environment_id']!=args.environment:continue
        if getattr(args,'architecture',None) and r['architecture_id']!=args.architecture:continue
        if getattr(args,'benchmark',None) and r['benchmark_id']!=args.benchmark:continue
        if getattr(args,'experiment',None) and args.experiment not in r['experiment_ids']:continue
        if getattr(args,'seed',None) and r['seed']!=args.seed:continue
        if getattr(args,'run_status',None) and r['status']!=args.run_status:continue
        if getattr(args,'method',None) and not any(args.method.lower() in str(x).lower() for x in [r['method'],r['method_id'],r['method_variant_id']]+r['method_aliases']):continue
        yield r

def matrix(rows,axis,include_prefix=False,coverage='reported_completed',environment_ids=(),architecture_ids=()):
    grouped=defaultdict(set);eligible=[]
    for r in rows:
        if not include_prefix and r['role'] in ('prefix','shared-prefix','pair-reference'):continue
        eligible.append(r)
    labels={}
    for r in eligible:
        variant=r['method_variant_id']
        if variant not in labels or r['method']!=r['method_id']:labels[variant]=r['method']
    def method(r):
        settings='; replay='+r['replay_on_switch']+'; epsilon='+r['epsilon_on_switch']
        return r['algorithm']+'/'+labels[r['method_variant_id']]+settings+' ['+r['method_variant_id']+']'
    def key(r):
        if axis=='environment-method':return r['environment_id'],method(r)
        if axis=='architecture-method':return r['architecture_id'],method(r)
        if axis=='architecture-environment':return r['architecture_id'],r['environment_id']
        if axis=='benchmark-seed':return r['benchmark_id'],r['seed']
        return method(r),r['seed']
    observed=defaultdict(list)
    for r in eligible:
        observed[key(r)].append(r)
        if r['status']=='completed' and (coverage!='complete_result' or r['result_status']=='reported_complete_result'):grouped[key(r)].add(r['run_id'])
    row_keys={key(r)[0] for r in eligible}
    if axis=='environment-method':row_keys.update(environment_ids)
    if axis.startswith('architecture-'):row_keys.update(architecture_ids)
    row_keys=sorted(row_keys);cols=sorted({key(r)[1] for r in eligible})
    if axis=='architecture-environment':cols=sorted(set(cols)|set(environment_ids))
    return [dict(row=a,column=c,completed_runs=len(grouped[a,c]),run_ids=sorted(grouped[a,c]),observed_runs=len(observed[a,c]),failed_runs=sum(r['status']=='failed' for r in observed[a,c]),coverage=coverage) for a in row_keys for c in cols]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);sub=p.add_subparsers(dest='command',required=True)
    for command in ('build','check'):
        q=sub.add_parser(command);q.add_argument('--include-local',action='store_true',help='Explicitly scan accessible local Git worktree runs')
    sub.add_parser('summary')
    for name in ('environments','benchmarks','tasks','task_sequences'):
        q=sub.add_parser(name);q.add_argument('--environment');q.add_argument('--observed',action='store_true')
    q=sub.add_parser('architectures');q.add_argument('--architecture');q.add_argument('--observed',action='store_true')
    q=sub.add_parser('query')
    for f in ('environment','benchmark','architecture','method','seed','experiment','run-status'):q.add_argument('--'+f)
    q=sub.add_parser('show');q.add_argument('run_id')
    q=sub.add_parser('matrix');q.add_argument('axis',choices=['environment-method','benchmark-seed','seed-coverage','architecture-method','architecture-environment']);q.add_argument('--experiment');q.add_argument('--environment');q.add_argument('--include-prefix',action='store_true');q.add_argument('--benchmark');q.add_argument('--architecture');q.add_argument('--method');q.add_argument('--seed');q.add_argument('--coverage',choices=['reported_completed','complete_result'],default='reported_completed')
    args=p.parse_args();root=args.root
    if args.command in ('build','check'):
        b=Builder(root,include_local=args.include_local).build();files=artifacts(b)
        if args.command=='build':
            for name,text in files.items():(root/name).parent.mkdir(parents=True,exist_ok=True);(root/name).write_text(text)
        else:
            stale=[name for name,text in files.items() if not (root/name).exists() or (root/name).read_text()!=text]
            if stale:p.exit(1,'Stale or missing generated files: '+', '.join(stale)+'\n')
        print(dumps({k:v for k,v in json.loads(files['docs/results/index_audit.json']).items() if k!='issues'}));return
    if args.command=='architectures':
        data=json.loads((root/'docs/architectures/architectures.json').read_text())['architectures']
        rows=read_runs(root)
        data=[a for a in data if not args.architecture or a['id']==args.architecture]
        if args.observed:data=[a for a in data if a['actual_run_count']>0]
        for a in data:a['run_ids']=[r['run_id'] for r in rows if r['architecture_id']==a['id']]
        print(json.dumps(data,ensure_ascii=False,indent=2));return
    if args.command in ('environments','benchmarks','tasks','task_sequences'):
        path='environments.json' if args.command=='environments' else 'benchmarks.json'
        data=json.loads((root/'docs/environments'/path).read_text())[args.command]
        data=[r for r in data if not args.environment or r.get('environment_id',r['id'])==args.environment]
        if args.observed:data=[r for r in data if r.get('actual_run_count',r.get('associated_run_count',0))>0]
        print(json.dumps(data,ensure_ascii=False,indent=2));return
    rows=list(filter_runs(read_runs(root),args))
    if args.command=='show':
        found=[r for r in rows if r['run_id']==args.run_id]
        if not found:p.exit(1,'Unknown run_id\n')
        analyses=list(csv.DictReader(io.StringIO((root/'docs/results/analysis_index.csv').read_text())))
        found[0]['children']=[r['run_id'] for r in read_runs(root) if r['parent_run_id']==args.run_id]
        registry=json.loads((root/'docs/environments/benchmarks.json').read_text())
        found[0]['benchmark']=next((b for b in registry['benchmarks'] if b['id']==found[0]['benchmark_id']),UNKNOWN)
        found[0]['analyses']=[a for a in analyses if args.run_id in json.loads(a['input_run_ids'])]
        print(json.dumps(found[0],ensure_ascii=False,indent=2))
    elif args.command=='matrix':
        environments=json.loads((root/'docs/environments/environments.json').read_text())['environments']
        ids=[e['id'] for e in environments if not args.environment or e['id']==args.environment]
        registry=json.loads((root/'docs/architectures/architectures.json').read_text())['architectures']
        architecture_ids=[a['id'] for a in registry if not args.architecture or a['id']==args.architecture]
        print(json.dumps(matrix(rows,args.axis,args.include_prefix,args.coverage,ids,architecture_ids),ensure_ascii=False,indent=2))
    elif args.command=='summary':print((root/'docs/results/index_audit.json').read_text())
    else: print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
