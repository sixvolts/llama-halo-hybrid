#!/usr/bin/env python3
"""Run results: verdict.json + summary.md per run, one line per run in the ledger.

  ledger.py finish <run_dir> <ledger.jsonl> key=value ...   (called by halo_finish)
  ledger.py last [<ledger.jsonl>] [n]                       the last n runs, one line each
"""
import json
import os
import sys
import time


def finish(run_dir, ledger, kv):
    steps = []
    with open(os.path.join(run_dir, 'steps.tsv')) as f:
        for line in f:
            name, verdict, secs, detail = (line.rstrip('\n').split('\t') + ['', '', '', ''])[:4]
            steps.append({'name': name, 'verdict': verdict, 'seconds': int(secs or 0), 'detail': detail})
    rec = dict(kv)
    rec['dirty'] = rec.get('dirty') == '1'
    rec['seconds'] = int(rec.get('seconds', 0))
    rec['ts'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    rec['run_dir'] = run_dir
    rec['steps'] = steps
    with open(os.path.join(run_dir, 'verdict.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    with open(ledger, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    with open(os.path.join(run_dir, 'summary.md'), 'w') as f:
        f.write('# %s %s on %s: %s\n\n' % (rec.get('suite'), rec.get('level'), rec.get('host'), rec.get('verdict')))
        f.write('commit %s%s, %s, %d min\n\n' % (rec.get('commit'), ' (dirty)' if rec['dirty'] else '', rec['ts'],
                                                rec['seconds'] // 60))
        f.write('| Step | Verdict | Time | Detail |\n|---|---|---|---|\n')
        for s in steps:
            f.write('| %s | %s | %ds | %s |\n' % (s['name'], s['verdict'], s['seconds'], s['detail'].replace('|', '/')))
    print(open(os.path.join(run_dir, 'summary.md')).read())


def last(ledger, n):
    if not os.path.exists(ledger):
        return
    lines = open(ledger).read().splitlines()[-n:]
    for line in lines:
        r = json.loads(line)
        bad = [s['name'] for s in r['steps'] if s['verdict'] not in ('PASS', 'SKIP')]
        print('%s  %-6s %-6s %-9s %-10s %s%s' % (r['ts'][:16], r['host'], r['suite'], r['level'], r['verdict'],
                                                r['commit'], ('  failing: ' + ', '.join(bad)) if bad else ''))


if __name__ == '__main__':
    if sys.argv[1] == 'finish':
        finish(sys.argv[2], sys.argv[3], dict(a.split('=', 1) for a in sys.argv[4:]))
    elif sys.argv[1] == 'last':
        last(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get('HALO_RESULTS') or os.path.join(os.environ.get('XDG_STATE_HOME') or os.path.expanduser('~/.local/state'), 'halo-test'), 'ledger.jsonl'),
             int(sys.argv[3]) if len(sys.argv) > 3 else 10)
