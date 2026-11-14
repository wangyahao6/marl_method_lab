"""Legacy JSON writers and learner/runner statistics logging."""
import json


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def attach_legacy_metrics(logger, out):
    original_log = logger.log_stat

    def log(key, value, t, **kwargs):
        append(out / 'metrics.jsonl', dict(key=key, value=float(value), t_env=int(t)))
        original_log(key, value, t, **kwargs)

    logger.log_stat = log
