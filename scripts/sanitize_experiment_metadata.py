"""Deterministic publication aliases for metadata, including nested command/error strings."""
import argparse
import hashlib
import json
import re
from pathlib import Path


VERSION = 1
ABSOLUTE = re.compile(r'^(?:/|[A-Za-z]:[\\/]|\\\\)')
EMBEDDED = re.compile(r'''(?<![\w./\\-])(?:[A-Za-z]:[\\/]|/)[^\s"':,;()<>\[\]{}]+''')
QUOTED = re.compile(r'''(["'])((?:/|[A-Za-z]:[\\/])[^"'\n]+)\1''')
REMOTE = re.compile(r'(?<![\w./\\-])(?:[\w.-]+@)?([\w.-]{2,}):(?=/|[A-Za-z]:[\\/])')
ANCHORS = {'runs', 'results', 'configs', 'src', 'vendor', 'runtime', '.venv', 'drivers', 'scripts'}


def alias(kind, value):
    return kind + '-' + hashlib.sha256(value.encode()).hexdigest()[:16]


class Redactor:
    def __init__(self, records):
        self.roots = set()
        self.hosts = set()
        self.run_ids = set()
        self._discover(records)

    def _discover(self, value, key=''):
        if isinstance(value, dict):
            for k, v in value.items():
                self._discover(v, k)
        elif isinstance(value, list):
            for v in value:
                self._discover(v, key)
        elif isinstance(value, str):
            if key in ('host', 'hostname') and value and not re.fullmatch(r'host-[0-9a-f]{16}', value):
                self.hosts.add(value)
            if key in ('root', 'roots', 'cwd') and ABSOLUTE.match(value):
                self.roots.add(value.replace('\\', '/').rstrip('/'))
            if key in ('run_id','parent_run_id','input_run_id','source_run_id','parent','run_dir'):
                self.run_ids.add(value.replace('\\','/').rstrip('/').split('/')[-1])

    def path(self, value):
        normalized = value.replace('\\', '/')
        for root in sorted(self.roots, key=lambda x: (-len(x), x)):
            if normalized == root or normalized.startswith(root + '/'):
                return 'repos/' + alias('repo', root) + normalized[len(root):]
        parts = normalized.split('/')
        for i, part in enumerate(parts):
            if part in ANCHORS and i > 1:
                root = '/'.join(parts[:i])
                return 'repos/' + alias('repo', root) + '/' + '/'.join(parts[i:])
        for i, part in enumerate(parts):
            if part and part in self.run_ids and i > 1:
                return 'locations/' + alias('root','/'.join(parts[:i])) + '/' + '/'.join(parts[i:])
        if normalized.startswith(('/usr/bin/', '/usr/local/bin/', '/bin/')):
            return 'tools/' + normalized.rsplit('/', 1)[-1]
        return 'external/' + alias('path', normalized)

    def string(self, value):
        if value.startswith('file://'):
            return 'file-ref:' + self.path(value[7:])
        for host in sorted(self.hosts, key=lambda x: (-len(x), x)):
            value = value.replace(host, alias('host', host))
        value = REMOTE.sub(lambda m: m[0] if value[m.end():].startswith('//') else
                           (m[1] if re.fullmatch(r'host-[0-9a-f]{16}', m[1])
                            else alias('host', m[1])) + ':', value)
        if (ABSOLUTE.match(value) and '\n' not in value and not value.startswith('//')
                and not re.search(r'\s+(?:/|[A-Za-z]:[\\/])|[;=]', value)):
            # Whole path fields can contain spaces; command lists are handled element by element.
            return self.path(value)
        value = QUOTED.sub(lambda m: m[1] + self.path(m[2]) + m[1], value)
        return EMBEDDED.sub(lambda m: m[0] if m[0].startswith('//') and m.start()>0
                            and value[m.start()-1]==':' else self.path(m[0]), value)

    def apply(self, value):
        if isinstance(value, dict):
            result = {self.string(k): self.apply(v) for k, v in value.items()}
            if result.get('kind') == 'collection':
                result['redaction'] = dict(version=VERSION, policy='stable_host_and_location_aliases')
            return result
        if isinstance(value, list):
            return [self.apply(v) for v in value]
        if isinstance(value, str):
            return self.string(value)
        return value


def redact(value):
    return Redactor(value).apply(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    args.output.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in redact(records)))


if __name__ == '__main__':
    main()
