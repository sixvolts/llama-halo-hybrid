#!/usr/bin/env python3
"""Recurrent-state rollback and state save/load over the generated random-weight models, one model per process
(an abort in one architecture must not hide the others), checked against known-failures.txt.

  state_check.py <bin dir> <models dir> <out dir> <known-failures.txt>

A failure listed in known-failures.txt is reported as KNOWN; any other failure fails the step.
"""
import os
import re
import subprocess
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')


def known_list(path):
    res = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        test, model_re, sub, reason = (line.split(None, 3) + ['', '', '', ''])[:4]
        res.append((test, re.compile(model_re), sub, reason))
    return res


def is_known(known, test, model, sub):
    for t, m, s, reason in known:
        if t == test and m.search(model) and s in ('*', sub):
            return reason
    return None


def run_one(binary, model_path, out):
    d = os.path.join(out, 'one')
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.unlink(os.path.join(d, f))
    os.symlink(os.path.abspath(model_path), os.path.join(d, os.path.basename(model_path)))
    r = subprocess.run([binary, '--models', d], capture_output=True, text=True, timeout=900)
    text = ANSI.sub('', r.stdout + r.stderr)
    with open(os.path.join(out, '%s.%s.log' % (os.path.basename(binary), os.path.basename(model_path))), 'w') as f:
        f.write(text)
    return text, r.returncode


def parse(text, model):
    head = [ln for ln in text.splitlines() if ln.startswith('Model ')]
    row = [ln for ln in text.splitlines() if ln.startswith(model)]
    if not head or not row:
        return None
    cols = re.split(r'\s{2,}', head[-1].strip())[1:]   # column names can hold one space ("split replay")
    vals = row[-1][len(model):].split()
    if len(vals) < len(cols) or any(v not in ('PASS', 'FAIL', 'SKIP') for v in vals[:len(cols)]):
        return None   # the row was cut short by an abort
    return dict(zip(cols, vals))


def main():
    bindir, mdir, out, kpath = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    known = known_list(kpath)
    new, kn, n = [], [], 0
    models = sorted(f for f in os.listdir(mdir) if f.endswith('.gguf'))
    for test in ('test-save-load-state', 'test-recurrent-state-rollback'):
        for model in models:
            n += 1
            text, rc = run_one(os.path.join(bindir, test), os.path.join(mdir, model), out)
            res = parse(text, model)
            if res is None:
                fails = [('abort', (re.findall(r'GGML_ASSERT\([^\n]*|error: [^\n]*', text) or ['exit %d' % rc])[0][:90])]
            else:
                fails = [(k, v) for k, v in res.items() if v not in ('PASS', 'SKIP')]
            for sub, what in fails:
                reason = is_known(known, test, model, sub)
                rec = '%s %s %s: %s' % (test, model, sub, what)
                (kn if reason else new).append(rec + ('  [known: %s]' % reason if reason else ''))
            print('%-32s %-28s %s' % (test, model, 'ok' if not fails else ' '.join(s for s, _ in fails)))
    for r in kn:
        print('KNOWN ' + r)
    for r in new[:40]:
        print('NEW FAILURE ' + r)
    if len(new) > 40:
        print('... %d more new failures' % (len(new) - 40))
    detail = '%d model runs, %d new failures, %d known' % (n, len(new), len(kn))
    if new:
        detail += ': ' + '; '.join(r.split(':')[0] for r in new)[:200]
    print('detail: ' + detail)
    sys.exit(1 if new else 0)


if __name__ == '__main__':
    main()
