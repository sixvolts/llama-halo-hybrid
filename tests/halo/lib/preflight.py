#!/usr/bin/env python3
"""Host-state checks before a GPU suite. A failure here means the machine is not in its reference state, so any
number measured after it would be read as a code regression by mistake (2026-09-23: a pinned perf level cost 20-27%).

  preflight.py host <gen> <machine.json> [--bless]       perf level, BAR, VRAM (GECC), the card's link, dmesg, cmdline
  preflight.py gemv <gen> <machine.json> <bin> [--bless]  six-shape q8_0 GEMV probe vs the machine's references
  preflight.py p2p  <gen> <machine.json> <p2p-bin> [--bless]  card <-> APU peer copies vs references

References live in machine.json under the generation's key; --bless records the current values there.
Exit codes: 0 ok, 1 the machine is not in its reference state, 2 an error of this tool (missing file, tool or argument).
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import devices  # noqa: E402

GEMV_SHAPES = '24:16384:3,2048:4096:3,4096:2048:3,2048:1024:3,12288:4096:3,32:32768:3'
GEMV_SLACK = 1.15     # a shape more than 15% slower than its reference fails
P2P_SLACK = 0.85      # bandwidth below 85% of the reference fails
# kernel flags each generation needs, and the ones particular cards need (by marketing name: the Radeon Pro V620
# re-enables GECC without ras_enable=0, and RAS slows peer copies; consumer RDNA2 cards have no ECC)
NEED_FLAGS = {'rdna3': ['amdgpu.runpm=0']}
CARD_FLAGS = {'V620': ['amdgpu.ras_enable=0']}
BAD_FLAGS = ['amdgpu.pcie_gen_cap']


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write('\n')


def card(gen):
    cards = [d for d in devices.gpus() if d['gen'] == gen]
    if not cards:
        print('detail: no %s card' % gen)
        sys.exit(1)
    return cards[0]


def sysfs(pci, name):
    try:
        return open('/sys/bus/pci/devices/%s/%s' % (pci, name)).read().strip()
    except OSError:
        return None


def card_link(pci):
    """The card's own PCIe link, between its topmost AMD port (the card's internal switch) and the port above it: a
    root port in a slot, the dock's bridge behind USB4 (whose root port only shows the tunnel, 2.5 GT/s x1).
    Negotiated = the slower end's max speed and the narrower end's max width; the trained width comes from the
    card's port (its speed is not checked: link power management drops an idle card to 2.5 GT/s)."""
    path = [p for p in os.path.realpath('/sys/bus/pci/devices/' + pci).split('/') if re.match(r'[0-9a-f]{4}:', p)]
    i = path.index(pci)
    while i > 0 and sysfs(path[i - 1], 'vendor') == '0x1002':
        i -= 1
    ends = path[max(i - 1, 0):i + 1]
    speed = min(((sysfs(p, 'max_link_speed') or '0').split(' ')[0] for p in ends), key=float)
    width = min(int(sysfs(p, 'max_link_width') or 0) for p in ends)
    return ends[0], ends[-1], '%s GT/s x%d' % (speed, width), sysfs(path[i], 'current_link_width')


def host(gen, mpath, bless):
    data = load(mpath)
    m = data.get(gen, {})
    c = card(gen)
    pci = c['pci']
    problems, notes = [], []

    lvl = sysfs(pci, 'power_dpm_force_performance_level')
    if lvl != 'auto':
        problems.append('perf level %s (must be auto: high is 20-27%% slower on launch-bound kernels)' % lvl)

    vram = int(sysfs(pci, 'mem_info_vram_total') or 0) >> 20
    ref_vram = m.get('vram_mib')
    if ref_vram and abs(vram - ref_vram) > ref_vram * 0.01:
        problems.append('VRAM %d MiB vs reference %d (GECC state changed?)' % (vram, ref_vram))
    notes.append('VRAM %d MiB' % vram)

    res = sysfs(pci, 'resource').splitlines()[0].split()
    bar = (int(res[1], 16) - int(res[0], 16) + 1) >> 20
    if bar < vram:
        problems.append('BAR0 %d MiB < VRAM %d MiB (small BAR: no full peer access)' % (bar, vram))

    up, down, link, trained = card_link(pci)
    ref_link = m.get('link')
    if ref_link and link != ref_link:
        problems.append('link %s-%s %s vs reference %s' % (up, down, link, ref_link))
    if trained != link.split(' x')[-1]:
        problems.append('link %s-%s trained x%s of x%s' % (up, down, trained, link.split(' x')[-1]))
    dock = sysfs(pci, 'removable') == 'removable'
    notes.append('%s %s-%s %s' % ('dock' if dock else 'slot', up, down, link))

    cmd = open('/proc/cmdline').read()
    # a dock card keeps the BIOS-built tunnel and its full BAR only without the thunderbolt host reset
    need = NEED_FLAGS.get(gen, []) + [f for name, fl in CARD_FLAGS.items() if name in c['name'] for f in fl]
    for f in need + (['thunderbolt.host_reset=false'] if dock else []):
        if f not in cmd:
            problems.append('kernel cmdline lacks %s' % f)
    for f in BAD_FLAGS:
        if f in cmd:
            problems.append('kernel cmdline has %s' % f)

    dm = subprocess.run(['sudo', '-n', 'dmesg'], capture_output=True, text=True).stdout
    mine = [ln for ln in dm.splitlines() if pci in ln or pci[5:] in ln]
    for pat in (r'power state from D3', r'device inaccessible', r'device lost', r'GPU reset begin', r'reset failed', r'GPU recovery', r'SMU is in hanged state'):
        hits = [ln for ln in mine if re.search(pat, ln)]
        if hits:
            problems.append('dmesg since boot: %s (%d lines; reboot to clear)' % (pat, len(hits)))
    faults = [ln for ln in mine if 'page fault' in ln.lower()]
    if faults:
        notes.append('%d page-fault lines in dmesg' % len(faults))
    if not dm:
        notes.append('dmesg not readable')

    if (bless or not ref_vram or not ref_link) and not problems:
        data.setdefault(gen, {}).update(vram_mib=vram, link=link)
        save(mpath, data)
        notes.append('recorded VRAM and link references')
    for p in problems:
        print('PROBLEM ' + p)
    print('detail: %s' % ('; '.join(problems) if problems else ', '.join(notes)))
    return 1 if problems else 0


def gemv(gen, mpath, bindir, bless):
    env = dict(os.environ, GGML_CUDA_NO_MMVQ_GROUP='1', GGML_CUDA_NO_GEMV_HOIST='1', TBO_MMV_SHAPES=GEMV_SHAPES)
    us = []
    for _ in range(2):   # best of two passes: the first can catch clocks still ramping
        r = subprocess.run([os.path.join(bindir, 'test-backend-ops'), 'perf', '-b', 'ROCm0', '-o', 'MUL_MAT',
                            '-p', 'q8_0.*n=3'], env=env, capture_output=True, text=True, timeout=600)
        print(r.stdout[-4000:])
        vals = {}
        for line in r.stdout.splitlines():
            mm = re.search(r'type_a=q8_0,type_b=f32,m=(\d+),n=3,k=(\d+).*?([\d.]+) us/run', line)
            if mm:
                vals['%s:%s' % (mm.group(1), mm.group(2))] = float(mm.group(3))
        us.append(vals)
    best = {k: min(u.get(k, 1e9) for u in us) for k in us[0]}
    if len(best) != 6:
        print('detail: probe printed %d of 6 shapes' % len(best))
        return 1
    data = load(mpath)
    ref = data.get(gen, {}).get('gemv_us')
    if bless or not ref:
        data.setdefault(gen, {})['gemv_us'] = best
        save(mpath, data)
        print('detail: recorded GEMV references %s' % ' '.join('%s=%.2f' % kv for kv in best.items()))
        return 0
    slow = ['%s %.2f vs %.2f us' % (k, v, ref[k]) for k, v in best.items() if k in ref and v > ref[k] * GEMV_SLACK]
    print('detail: %s' % ('slow: ' + '; '.join(slow) if slow else
                          'all 6 shapes within %d%% of reference' % round((GEMV_SLACK - 1) * 100)))
    return 1 if slow else 0


def p2p(gen, mpath, exe, bless):
    r = subprocess.run([exe], capture_output=True, text=True, timeout=300)
    print(r.stdout)
    if r.returncode != 0:
        print('detail: p2p probe failed: ' + r.stderr[-200:])
        return 1
    can = re.search(r'canAccessPeer 0->1 (\d), 1->0 (\d)', r.stdout)
    got = {}
    for mm in re.finditer(r'(card->APU|APU->card)\s+(\d+) B:\s+[\d.]+ us\s+([\d.]+) GB/s', r.stdout):
        if int(mm.group(2)) >= 8 << 20:
            got['%s %s' % (mm.group(1), mm.group(2))] = float(mm.group(3))
    problems = []
    if not can or can.group(1) != '1' or can.group(2) != '1':
        problems.append('peer access not available both ways (%s)' % (can.group(0) if can else 'no line'))
    data = load(mpath)
    ref = data.get(gen, {}).get('p2p_gbs')
    if (bless or not ref) and not problems:
        data.setdefault(gen, {})['p2p_gbs'] = got
        save(mpath, data)
        print('detail: recorded peer-copy references %s' % ' '.join('%s=%.2f' % kv for kv in got.items()))
        return 0
    for k, v in got.items():
        if ref and k in ref and v < ref[k] * P2P_SLACK:
            problems.append('%s %.2f vs %.2f GB/s' % (k, v, ref[k]))
    print('detail: %s' % ('; '.join(problems) if problems else
                          'peer copies %s GB/s' % '/'.join('%.1f' % v for v in got.values())))
    return 1 if problems else 0


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    what, gen, mpath = sys.argv[1], sys.argv[2], sys.argv[3]
    bless = '--bless' in sys.argv
    if what == 'host':
        sys.exit(host(gen, mpath, bless))
    if what == 'gemv':
        sys.exit(gemv(gen, mpath, sys.argv[4], bless))
    if what == 'p2p':
        sys.exit(p2p(gen, mpath, sys.argv[4], bless))
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    try:
        main()
    except (OSError, IndexError, KeyError, ValueError, subprocess.CalledProcessError) as e:
        print('detail: preflight.py: %s: %s' % (type(e).__name__, e))   # a checker error: SETUP, not HOST-STATE
        sys.exit(2)
