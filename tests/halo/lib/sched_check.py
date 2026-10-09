#!/usr/bin/env python3
"""Scheduler / multi-device exactness on tiny random-weight models (test-llama-archs -o).

Greedy output must not depend on where the layers run, as long as every device runs the same kernels:
one device vs layers split over two, one prefill lane vs two, local vs RPC loopback. Random weights make every
logit matter, so a stale read or a reordered write shows up as a different token stream.

  sched_check.py base  --bin <dir> --models <dir> --out <dir>    APU + RPC loopback (ROCR_VISIBLE_DEVICES = the APU)
  sched_check.py card  --bin <dir> --models <dir> --out <dir>    card + APU (ROCm0 = card, ROCm1 = APU)

Every group of cases must produce identical output; the card mode also checks the lane-pair hazard model:
every admitted pair has 0 cache hazards and the self-test (pair a delayed one split) finds some.
"""
import argparse
import hashlib
import os
import random
import re
import socket
import subprocess
import sys
import time

MODELS = ['qwen4exp-moe', 'qwen35moe-moe', 'qwen35-dense', 'glm5next-moe']
COMMON = ['-n', '24', '--temp', '0', '--seed', '1', '-no-cnv', '-c', '8192', '-b', '1024', '-ub', '256',
          '-ngl', '999', '-fa', 'on']


def prompt_file(out):
    p = os.path.join(out, 'prompt.txt')
    rnd = random.Random(7)
    words = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau'.split()
    with open(p, 'w') as f:
        f.write(' '.join(rnd.choice(words) for _ in range(1500)))
    return p


def run(a, tag, model, extra, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    log = os.path.join(a.out, '%s.%s.log' % (model, tag))
    cmd = [os.path.join(a.bin, 'llama-completion'), '-m', os.path.join(a.models, model + '.gguf'),
           '-f', a.prompt] + COMMON + extra
    with open(log, 'w') as f:
        f.write(' '.join(cmd) + '\n' + ' '.join('%s=%s' % kv for kv in (env_extra or {}).items()) + '\n')
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    with open(log, 'a') as f:
        f.write(r.stderr)
    if r.returncode != 0:
        return None, log, r.stderr
    h = hashlib.md5(r.stdout.encode()).hexdigest()[:12]
    return h, log, r.stderr


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


KNOWN_HITS = []
KNOWN = []   # (test, model regex, case, reason) from known-failures.txt, test = sched-<mode>


def known(mode, name, case):
    for t, m, c, reason in KNOWN:
        if t == 'sched-' + mode and m.search(name) and c in ('*', case):
            return reason
    return None


def check_group(name, results, problems):
    hashes = {tag: h for tag, (h, _, _) in results.items()}
    line = '%-16s %s' % (name, '  '.join('%s=%s' % (t, h or 'CRASH') for t, h in hashes.items()))
    print(line)
    crashed = [t for t, h in hashes.items() if h is None]
    new = [t for t in crashed if not known(MODE, name, t)]
    for t in crashed:
        if t not in new:
            print('KNOWN %s %s crashed: %s' % (name, t, known(MODE, name, t)))
            KNOWN_HITS.append('%s/%s' % (name, t))
    if new:
        problems.append('%s: %s crashed' % (name, ','.join(new)))
    ok = {t: h for t, h in hashes.items() if h is not None}
    if len(set(ok.values())) > 1:
        problems.append('%s: outputs differ (%s)' % (name, ' '.join('%s=%s' % kv for kv in ok.items())))


def mode_base(a):
    problems = []
    port = free_port()
    srv_log = open(os.path.join(a.out, 'rpc-server.log'), 'w')
    srv = subprocess.Popen([os.path.join(a.bin, 'ggml-rpc-server'), '-H', '127.0.0.1', '-p', str(port), '-d', 'ROCm0'],
                           stdout=srv_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                socket.create_connection(('127.0.0.1', port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        rpc = ['--rpc', '127.0.0.1:%d' % port]
        lst = subprocess.run([os.path.join(a.bin, 'llama-completion')] + rpc + ['--list-devices'],
                             capture_output=True, text=True, timeout=120)
        m = re.search(r'^\s*(RPC\S*?):', lst.stdout + lst.stderr, re.M)
        if not m:
            print('detail: rpc device not listed')
            print(lst.stdout + lst.stderr)
            return 1
        rdev = m.group(1)
        for model in MODELS:
            res = {
                'local':       run(a, 'local', model, ['-dev', 'ROCm0']),
                'rpc':         run(a, 'rpc', model, rpc + ['-dev', rdev]),
                'split':       run(a, 'split', model, rpc + ['-dev', 'ROCm0,' + rdev, '-ts', '1,1']),
                'split-lanes': run(a, 'split-lanes', model, rpc + ['-dev', 'ROCm0,' + rdev, '-ts', '1,1'],
                                   {'LLAMA_PREFILL_LANES': '2'}),
            }
            check_group(model, res, problems)
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()
    return report(problems, len(MODELS) * 4)


HZ = re.compile(r'names (match|differ), (\d+) vs (\d+) splits, (\d+) cache tensors shared, (\d+) cache hazards')
SELF = re.compile(r'self-test, a delayed one split: (\d+) cache hazards')


def mode_card(a):
    problems = []
    n_pairs = n_self = 0
    for model in MODELS:
        exps = ['-ot', r'ffn_(gate|up|down)_exps=ROCm1'] if 'moe' in model else []
        prod = ['-dev', 'ROCm0,ROCm1', '-ts', '1,0'] + exps   # production layout: experts on the APU
        layer = ['-dev', 'ROCm0,ROCm1', '-ts', '1,1']
        lanes = {'LLAMA_PREFILL_LANES': '2', 'GGML_SCHED_PAIR_DEBUG': '2'}
        res_p = {'prod-1lane': run(a, 'prod-1lane', model, prod),
                 'prod-2lanes': run(a, 'prod-2lanes', model, prod, lanes)}
        res_l = {'layer-1lane': run(a, 'layer-1lane', model, layer),
                 'layer-2lanes': run(a, 'layer-2lanes', model, layer, lanes)}
        check_group(model + '/prod', res_p, problems)
        check_group(model + '/layer', res_l, problems)
        for tag, (_, log, err) in (('prod-2lanes', res_p['prod-2lanes']), ('layer-2lanes', res_l['layer-2lanes'])):
            if 'two-lane prefill enabled' not in err:
                problems.append('%s/%s: lanes not enabled' % (model, tag))
            hz = HZ.findall(err)
            bad = [h for h in hz if int(h[4]) != 0]
            st = [int(x) for x in SELF.findall(err)]
            n_pairs += len(hz)
            n_self += sum(1 for x in st if x > 0)
            if bad:
                problems.append('%s/%s: %d pairs with cache hazards' % (model, tag, len(bad)))
            if hz and not any(x > 0 for x in st):
                problems.append('%s/%s: hazard self-test found nothing' % (model, tag))
        res_c = {'card': run(a, 'card', model, ['-dev', 'ROCm0'])}
        check_group(model + '/card', res_c, problems)
    if n_pairs == 0:
        problems.append('no lane pairs were checked (hazard log missing)')
    print('lane pairs checked: %d, self-tests that found hazards: %d' % (n_pairs, n_self))
    return report(problems, len(MODELS) * 5)


def report(problems, n):
    for p in problems:
        print('PROBLEM ' + p)
    kn = ', %d known crashes (%s)' % (len(KNOWN_HITS), ' '.join(KNOWN_HITS)) if KNOWN_HITS else ''
    print('detail: %s' % ('%d runs, all groups identical%s' % (n, kn) if not problems else '; '.join(problems)[:300]))
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['base', 'card'])
    ap.add_argument('--bin', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--known')
    a = ap.parse_args()
    global MODE
    MODE = 'rpc' if a.mode == 'base' else 'card'
    if a.known:
        sys.path.insert(0, os.path.dirname(__file__))
        from state_check import known_list
        KNOWN.extend(known_list(a.known))
    os.makedirs(a.out, exist_ok=True)
    a.prompt = prompt_file(a.out)
    sys.exit(mode_base(a) if a.mode == 'base' else mode_card(a))


if __name__ == '__main__':
    main()
