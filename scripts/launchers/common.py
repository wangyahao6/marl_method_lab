"""Paths and data shared by launch plan builders."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = 'configs/continual/task_transfer/knife_fig6.json'
LEGACY_DEFAULT_PROTOCOL = 'configs/continual/knife_fig6.json'


@dataclass
class LaunchSpec:
    command: list[str]
    env: dict[str, str]
    details: dict


def resolve_protocol(path):
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    try:
        key = candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return candidate
    if key.startswith('configs/continual/'):
        aliases = json.loads((ROOT / 'configs/continual/path_aliases.json').read_text())
        if key in aliases:
            return ROOT / aliases[key]
    return candidate


def load_protocol(path):
    return json.loads(resolve_protocol(path).read_text())


def prepare_smacv2(env, out):
    env['MARL_PROTOCOL'] = str(out / 'protocol.json')
    env['SC2PATH'] = str(ROOT / 'runtime/SMACv2StarCraftII')


def merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict):
            base[key] = merge(base.get(key, {}), value)
        else:
            base[key] = value
    return base


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
