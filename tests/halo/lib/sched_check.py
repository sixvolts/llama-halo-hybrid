#!/usr/bin/env python3
"""Scheduler / multi-device exactness on tiny random-weight models (test-llama-archs -o).

Greedy output must not depend on where the layers run, as long as every device runs the same kernels:
one device vs layers split over two, one prefill lane vs two, local vs RPC loopback. Random weights make every
logit matter, so a stale read or a reordered write shows up as a different token stream.

  sched_check.py base  --bin <dir> --models <dir> --out <dir>    APU + RPC loopback, plain and composite
                                                                 (ROCR_VISIBLE_DEVICES = the APU)
  sched_check.py card  --bin <dir> --models <dir> --out <dir>    card + APU (ROCm0 = card, ROCm1 = APU), locally and
                                                                 behind a composite rpc-server that owns both
  sched_check.py fuse  --bin <dir> --models <dir> --out <dir>    CPU, every model: graph fast paths on vs off
                                                                 (LLAMA_NO_GRAPH_FUSE=1)

Every group of cases must produce identical output; the card mode also checks the lane-pair hazard model:
every admitted pair has 0 cache hazards and the self-test (pair a delayed one split) finds some.
"""
import argparse
import concurrent.futures
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


def prompt_file(out, n_words=1500):
    p = os.path.join(out, 'prompt.txt')
    rnd = random.Random(7)
    words = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau'.split()
    with open(p, 'w') as f:
        f.write(' '.join(rnd.choice(words) for _ in range(n_words)))
    return p


def run(a, tag, model, extra, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    log = os.path.join(a.out, '%s.%s.log' % (model, tag))
    cmd = [os.path.join(a.bin, 'llama-completion'), '-m', os.path.join(a.models, model + '.gguf'),
           '-f', a.prompt] + COMMON + extra
    with open(log, 'w') as f:
        f.write(' '.join(cmd) + '\n' + ' '.join('%s=%s' % kv for kv in (env_extra or {}).items()) + '\n')
    # no stdin: a model whose template turns on interactive mode would wait on it forever
    r = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
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


def start_server(a, devices, tag, env_extra=None):
    """an rpc-server on a free loopback port; returns (process, port), the port accepting connections"""
    port = free_port()
    log = open(os.path.join(a.out, 'rpc-server-%s.log' % tag), 'w')
    srv = subprocess.Popen([os.path.join(a.bin, 'ggml-rpc-server'), '-H', '127.0.0.1', '-p', str(port), '-d', devices],
                           stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, **(env_extra or {})))
    for _ in range(60):
        try:
            socket.create_connection(('127.0.0.1', port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.5)
    return srv, port


def stop_server(srv):
    srv.terminate()
    try:
        srv.wait(10)
    except subprocess.TimeoutExpired:
        srv.kill()


def rpc_device(a, rpc, env_extra=None):
    """the client's name for the server's (first) device, e.g. RPC0[127.0.0.1:port], or None"""
    lst = subprocess.run([os.path.join(a.bin, 'llama-completion')] + rpc + ['--list-devices'], capture_output=True,
                         text=True, timeout=120, env=dict(os.environ, **(env_extra or {})))
    m = re.search(r'^\s*(RPC\S*?):', lst.stdout + lst.stderr, re.M)
    if not m:
        print(lst.stdout + lst.stderr)
    return m.group(1) if m else None


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
    # a case listed as known (by tag) whose output differs from the group's first case is reported, not failed
    ref = next(iter(ok.values()), None)
    for t in [t for t, h in ok.items() if h != ref and known(MODE, name, t)]:
        print('KNOWN %s %s differs: %s' % (name, t, known(MODE, name, t)))
        KNOWN_HITS.append('%s/%s' % (name, t))
        del ok[t]
    if len(set(ok.values())) > 1:
        problems.append('%s: outputs differ (%s)' % (name, ' '.join('%s=%s' % kv for kv in ok.items())))


def mode_base(a):
    # plain rpc-server and a composite one (GGML_RPC_COMPOSITE=1 on the client: the server schedules its part of the
    # graph itself; the server's environment gets it too, harmlessly), each on the APU behind a loopback port; the
    # composite cases also run the paired four-lane pipeline
    problems = []
    composite = {'GGML_RPC_COMPOSITE': '1'}
    srv, port = start_server(a, 'ROCm0', 'plain')
    srv_c, port_c = start_server(a, 'ROCm0', 'composite', composite)
    try:
        rpc = ['--rpc', '127.0.0.1:%d' % port]
        rpc_c = ['--rpc', '127.0.0.1:%d' % port_c]
        rdev, rdev_c = rpc_device(a, rpc), rpc_device(a, rpc_c, composite)
        if not rdev or not rdev_c:
            print('detail: rpc device not listed')
            return 1
        for model in MODELS:
            res = {
                'local':       run(a, 'local', model, ['-dev', 'ROCm0']),
                'rpc':         run(a, 'rpc', model, rpc + ['-dev', rdev]),
                'split':       run(a, 'split', model, rpc + ['-dev', 'ROCm0,' + rdev, '-ts', '1,1']),
                'split-lanes': run(a, 'split-lanes', model, rpc + ['-dev', 'ROCm0,' + rdev, '-ts', '1,1'],
                                   {'LLAMA_PREFILL_LANES': '2'}),
                'rpc-composite': run(a, 'rpc-composite', model, rpc_c + ['-dev', rdev_c], composite),
                'split-composite-lanes': run(a, 'split-composite-lanes', model,
                                             rpc_c + ['-dev', 'ROCm0,' + rdev_c, '-ts', '1,1'],
                                             dict(composite, LLAMA_PREFILL_LANES='4')),
            }
            check_group(model, res, problems)
    finally:
        stop_server(srv)
        stop_server(srv_c)
    return report(problems, len(MODELS) * 6)


HZ = re.compile(r'names (match|differ), (\d+) vs (\d+) splits, (\d+) cache tensors shared, (\d+) cache hazards')
REFUSED = re.compile(r'cache hazards( \(\d+ across backends\))?: running them one after the other')  # names match, cross-backend hazards: lanes serialised
REORDER = 'same-stream reorders and no cross-backend hazard: interleaving'  # names match, same-stream reorders only: admitted (name guard)
DIFFERENT = 'different split structures'                          # names differ, hazards: lanes serialised


def admitted_hazards(err):
    """(pairs checked, admitted pairs with cache hazards, admitted same-stream reorders): a pair with hazards is fine when
    the scheduler ran its lanes one after the other, or when every hazard is a reorder on one stream, which the name guard
    has always admitted (5a8325082); it logs either right after the pair's hazard line (uncapped under
    GGML_SCHED_PAIR_DEBUG). The 1-lane/2-lane exact-output comparison is what checks an admitted reorder."""
    lines = [ln for ln in err.splitlines() if HZ.search(ln) or REFUSED.search(ln) or REORDER in ln or DIFFERENT in ln]
    n, bad, reorder = 0, 0, 0
    for i, l in enumerate(lines):
        m = HZ.search(l)
        if not m:
            continue
        n += 1
        nxt = lines[i + 1] if i + 1 < len(lines) else ''
        # with hazard checking on (the harness never turns it off) a names-differ pair is admitted only without
        # hazards, so one with hazards was serialised even when its DIFFERENT line fell past that log's cap
        refused = m.group(1) == 'differ' or bool(REFUSED.search(nxt)) or DIFFERENT in nxt
        if int(m.group(5)) != 0 and not refused:
            if REORDER in nxt:
                reorder += 1
            else:
                bad += 1
    return n, bad, reorder


SELF = re.compile(r'self-test, a delayed one split: (\d+) cache hazards')


def mode_card(a):
    problems = []
    n_pairs = n_self = 0
    composite = {'GGML_RPC_COMPOSITE': '1'}
    srv_c, port_c = start_server(a, 'ROCm0,ROCm1', 'composite', composite)
    rpc_c = ['--rpc', '127.0.0.1:%d' % port_c]
    rdev_c = rpc_device(a, rpc_c, composite)
    if not rdev_c:
        problems.append('composite rpc-server device not listed')
    try:
        return mode_card_models(a, problems, n_pairs, n_self, composite, port_c, rpc_c, rdev_c)
    finally:
        stop_server(srv_c)


def mode_card_models(a, problems, n_pairs, n_self, composite, port_c, rpc_c, rdev_c):
    for model in MODELS:
        # an override that matches no tensor turns pipeline parallelism off, so the lanes run (a dense model's
        # production layout and the layer split); layer-pp is the layer split with pipeline parallelism
        noop = ['-ot', 'no_such_tensor=ROCm0']
        exps = ['-ot', r'ffn_(gate|up|down)_exps=ROCm1'] if 'moe' in model else noop
        prod = ['-dev', 'ROCm0,ROCm1', '-ts', '1,0'] + exps   # production layout: experts on the APU
        layer_pp = ['-dev', 'ROCm0,ROCm1', '-ts', '1,1']
        layer = layer_pp + noop
        lanes = {'LLAMA_PREFILL_LANES': '2', 'GGML_SCHED_PAIR_DEBUG': '2'}
        res_p = {'prod-1lane': run(a, 'prod-1lane', model, prod),
                 'prod-2lanes': run(a, 'prod-2lanes', model, prod, lanes)}
        res_l = {'layer-1lane': run(a, 'layer-1lane', model, layer),
                 'layer-2lanes': run(a, 'layer-2lanes', model, layer, lanes),
                 'layer-pp': run(a, 'layer-pp', model, layer_pp)}
        check_group(model + '/prod', res_p, problems)
        check_group(model + '/layer', res_l, problems)
        for tag, (_, log, err) in (('prod-2lanes', res_p['prod-2lanes']), ('layer-2lanes', res_l['layer-2lanes'])):
            # (the INFO "two-lane prefill enabled" line is below llama-completion's log threshold)
            if 'LLAMA_PREFILL_LANES ignored' in err:
                problems.append('%s/%s: lanes not enabled' % (model, tag))
                continue
            n_hz, bad, n_reorder = admitted_hazards(err)
            if n_reorder:
                print('%s/%s: %d matching pairs interleaved with same-stream reorders only' % (model, tag, n_reorder))
            st = [int(x) for x in SELF.findall(err)]
            n_pairs += n_hz
            n_self += sum(1 for x in st if x > 0)
            n_refused = len(REFUSED.findall(err))
            if n_refused:
                print('%s/%s: %d matching pairs run one after the other (cache hazards)' % (model, tag, n_refused))
            if bad:
                reason = known(MODE, model, 'hazards')
                if reason:
                    print('KNOWN %s/%s: %d admitted pairs with cache hazards: %s' % (model, tag, bad, reason))
                    KNOWN_HITS.append('%s/%s hazards' % (model, tag))
                else:
                    problems.append('%s/%s: %d admitted pairs with cache hazards' % (model, tag, bad))
            if n_hz and not any(x > 0 for x in st):
                problems.append('%s/%s: hazard self-test found nothing' % (model, tag))
        res_c = {'card': run(a, 'card', model, ['-dev', 'ROCm0'])}
        check_group(model + '/card', res_c, problems)
        if rdev_c:
            # the production layout behind a composite rpc-server that owns both devices: its own scheduler splits
            # the graph between them (the experts go to its second device, RPC1)
            exps_c = ['-ot', r'ffn_(gate|up|down)_exps=RPC1[127.0.0.1:%d]' % port_c] if 'moe' in model else noop
            res_r = {'prod-1lane': res_p['prod-1lane'],
                     'rpc-composite': run(a, 'rpc-composite', model, rpc_c + ['-dev', rdev_c] + exps_c, composite)}
            check_group(model + '/composite', res_r, problems)
    if n_pairs == 0:
        problems.append('no lane pairs were checked (hazard log missing)')
    print('lane pairs checked: %d, self-tests that found hazards: %d' % (n_pairs, n_self))
    return report(problems, len(MODELS) * 7)


def mode_fuse(a):
    # the fork's graph fast paths (a view instead of a state gather, fused qwen4exp blocks) must not change a token:
    # the recurrent-state fast path skipped mamba's ssm scan at batch 1 (2026-10-10). A model that cannot generate
    # (encoder-only, crashes both ways) is skipped; a crash on one side only is a problem.
    problems = []
    models = sorted(f[:-5] for f in os.listdir(a.models) if f.endswith('.gguf'))
    # the tiny models are trained at 256 tokens and some have learned position embeddings: stay inside that,
    # with several ubatches of prefill before the batch-1 decode
    cpu = ['-dev', 'none', '-ngl', '0', '-t', '4', '-c', '256', '-b', '256', '-ub', '64']

    def one(model):
        return model, {'fast': run(a, 'fast', model, cpu), 'nofuse': run(a, 'nofuse', model, cpu, {'LLAMA_NO_GRAPH_FUSE': '1'})}

    n_run = n_skip = 0
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        for model, res in ex.map(one, models):
            if all(h is None for h, _, _ in res.values()):
                n_skip += 1
                continue
            n_run += 1
            check_group(model, res, problems)
    print('models compared: %d, skipped (no output either way): %d' % (n_run, n_skip))
    return report(problems, 2 * n_run)


def report(problems, n):
    for p in problems:
        print('PROBLEM ' + p)
    kn = ', %d known (%s)' % (len(KNOWN_HITS), ', '.join(KNOWN_HITS)) if KNOWN_HITS else ''
    print('detail: %s' % ('%d runs, all groups identical%s' % (n, kn) if not problems else '; '.join(problems)[:300]))
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['base', 'card', 'fuse'])
    ap.add_argument('--bin', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--known')
    a = ap.parse_args()
    global MODE
    MODE = {'base': 'rpc', 'card': 'card', 'fuse': 'fuse'}[a.mode]
    if a.known:
        sys.path.insert(0, os.path.dirname(__file__))
        from state_check import known_list
        KNOWN.extend(known_list(a.known))
    os.makedirs(a.out, exist_ok=True)
    a.prompt = prompt_file(a.out, 180 if a.mode == 'fuse' else 1500)
    sys.exit({'base': mode_base, 'card': mode_card, 'fuse': mode_fuse}[a.mode](a))


if __name__ == '__main__':
    main()
