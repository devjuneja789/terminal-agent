"""Inspect complete-file predictions without executing model-selected commands."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from collections import defaultdict

from training.agenttune.action_diagnostic import parse_generated
from training.agenttune.weighted_loss import old_file_content


def module_nodes(content: str) -> dict[str, list[str]]:
    """Compare top-level bindings/definitions structurally, ignoring formatting."""
    nodes: dict[str, list[str]] = defaultdict(list)
    for node in ast.parse(content).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = f'{type(node).__name__}:{node.name}'
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            key = 'binding:'+','.join(ast.dump(t,include_attributes=False) for t in targets)
        else:
            key = ast.dump(node,include_attributes=False)
        nodes[key].append(ast.dump(node,include_attributes=False))
    return dict(nodes)


def inspect_write_prediction(text: str, row: dict) -> dict:
    gold = row['completion'][0]['content']
    expected = json.loads(gold)[0]
    if expected['tool_name'] != 'write_file':
        raise ValueError('preservation inspection requires an approved write target')
    path, target = expected['parameters']['path'], expected['parameters']['content']
    original = old_file_content(row,path)
    before, after = module_nodes(original), module_nodes(target)
    unchanged = {key:value for key,value in before.items() if after.get(key)==value}
    changed = {key:value for key,value in after.items() if before.get(key)!=value}
    score = parse_generated(text,gold)
    result = {'action_schema':score, 'expected_path':path,
        'original_content':original, 'gold_content':target,
        'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'gold_sha256':hashlib.sha256(target.encode()).hexdigest(),
        'eligible_write':False, 'full_file_preserved':False,
        'gold_content_exact':False, 'gold_ast_equal':False}
    if not score['schema_valid'] or score.get('selected_tool')!='write_file':
        result['not_inspected_reason']='invalid action or different tool selected'
        return result
    action = json.loads(text)[0]
    if action['parameters']['path'] != path:
        result['not_inspected_reason']='write targets a different path'
        return result
    content = action['parameters']['content']
    result.update(eligible_write=True,generated_content=content,
        generated_sha256=hashlib.sha256(content.encode()).hexdigest(),
        gold_content_exact=content==target,
        original_lines=len(original.splitlines()),generated_lines=len(content.splitlines()))
    result['diff_from_original']=''.join(difflib.unified_diff(original.splitlines(True),content.splitlines(True),fromfile='original/'+path,tofile='predicted/'+path))
    try:
        predicted = module_nodes(content)
    except SyntaxError as exc:
        result.update(python_parse_valid=False,syntax_error=str(exc))
        return result
    damaged = [key for key,value in unchanged.items() if predicted.get(key)!=value]
    missing = [key for key in before if key not in predicted]
    result.update(python_parse_valid=True,gold_ast_equal=ast.dump(ast.parse(content),include_attributes=False)==ast.dump(ast.parse(target),include_attributes=False),
        unchanged_original_nodes=len(unchanged),damaged_unchanged_nodes=damaged,
        missing_original_nodes=missing,
        changed_nodes_matching_gold=[key for key,value in changed.items() if predicted.get(key)==value],
        expected_changed_nodes=list(changed),full_file_preserved=not damaged and not missing)
    return result
