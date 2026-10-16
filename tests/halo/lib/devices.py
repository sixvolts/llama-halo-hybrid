#!/usr/bin/env python3
"""Find the GPUs on this machine and the ROCR_VISIBLE_DEVICES value for a test setup.

Devices are identified by gfx name, never by cardN or by a fixed ROCm index: a docked card
renumbers the others. The index is the agent's position among GPU agents in rocminfo order,
which is what ROCR_VISIBLE_DEVICES takes (the APU has no UUID, so indices are used for all).

  devices.py list                  JSON list of GPUs
  devices.py visible <setup>       ROCR_VISIBLE_DEVICES for apu | rdna2 | rdna3 | rdna4
                                   (card first, then the APU: ROCm0 = card, ROCm1 = APU)
  devices.py check <setup> [--build <dir>]
                                   verify the mapping under that ROCR_VISIBLE_DEVICES and, with --build, that the
                                   build's GPU_TARGETS (from its CMakeCache.txt) include every device of the setup

Exit codes: 0 ok, 1 the devices are not as expected, 2 an error of this tool (rocminfo missing, bad arguments).
"""
import json
import os
import re
import subprocess
import sys

GEN = {  # gfx target -> generation
    'gfx1151': 'apu', 'gfx1150': 'apu',
    'gfx1200': 'rdna4', 'gfx1201': 'rdna4',
    'gfx1100': 'rdna3', 'gfx1101': 'rdna3', 'gfx1102': 'rdna3',
    'gfx1030': 'rdna2', 'gfx1031': 'rdna2', 'gfx1032': 'rdna2',
}


def gpus(env=None):
    if env is None:   # enumerate every GPU: an inherited ROCR_VISIBLE_DEVICES would renumber them
        env = {k: v for k, v in os.environ.items() if k != 'ROCR_VISIBLE_DEVICES'}
    out = subprocess.run(['rocminfo'], capture_output=True, text=True, env=env, check=True).stdout
    agents, cur = [], None
    for line in out.splitlines():
        if re.match(r'^Agent \d+', line.strip()):
            cur = {}
            agents.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r'^\s{2}(Name|Uuid|Marketing Name|Device Type|BDFID|Node):\s*(.*?)\s*$', line)
        if m and m.group(1) not in cur:   # first occurrence (the agent's own, not an ISA's)
            cur[m.group(1)] = m.group(2)
    res = []
    for a in agents:
        if a.get('Device Type') != 'GPU':
            continue
        gfx = a.get('Name', '')
        bdf = int(a.get('BDFID', '0'))
        res.append({
            'index': len(res), 'gfx': gfx, 'gen': GEN.get(gfx, 'unknown'), 'name': a.get('Marketing Name', ''),
            'uuid': a.get('Uuid', ''),
            'pci': '%04x:%02x:%02x.%x' % (0, bdf >> 8, (bdf >> 3) & 0x1f, bdf & 0x7),
        })
    return res


def visible(setup, devs):
    apu = [d for d in devs if d['gen'] == 'apu']
    if not apu:
        raise SystemExit('no APU (gfx1151) found')
    if setup == 'apu':
        return str(apu[0]['index'])
    cards = [d for d in devs if d['gen'] == setup]
    if not cards:
        raise SystemExit('no %s card found (have: %s)' % (setup, ', '.join(d['gfx'] for d in devs)))
    return '%d,%d' % (cards[0]['index'], apu[0]['index'])


def build_targets(build):
    """GPU_TARGETS of a HIP build, from its CMakeCache.txt"""
    path = os.path.join(build, 'CMakeCache.txt')
    for line in open(path):
        m = re.match(r'^(?:GPU_TARGETS|AMDGPU_TARGETS):[A-Z]*=(.*)$', line.strip())
        if m and m.group(1):
            return m.group(1).split(';')
    raise SystemExit('detail: %s names no GPU_TARGETS' % path)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'list'
    devs = gpus()
    if cmd == 'list':
        print(json.dumps(devs, indent=1))
    elif cmd == 'visible':
        print(visible(sys.argv[2], devs))
    elif cmd == 'check':
        setup = sys.argv[2]
        env = dict(os.environ, ROCR_VISIBLE_DEVICES=visible(setup, devs))
        seen = [d['gen'] for d in gpus(env)]
        want = ['apu'] if setup == 'apu' else [setup, 'apu']
        if seen != want:
            raise SystemExit('detail: ROCR_VISIBLE_DEVICES=%s shows %s, expected %s' % (env['ROCR_VISIBLE_DEVICES'], seen, want))
        msg = 'ok %s: ROCR_VISIBLE_DEVICES=%s -> %s' % (setup, env['ROCR_VISIBLE_DEVICES'], ' '.join(seen))
        if '--build' in sys.argv:
            build = sys.argv[sys.argv.index('--build') + 1]
            targets = build_targets(build)
            missing = sorted({d['gfx'] for d in gpus(env)} - set(targets))
            if missing:
                raise SystemExit('detail: %s is not built for %s (GPU_TARGETS=%s)' % (build, ' '.join(missing), ';'.join(targets)))
            msg += ', built for %s' % ';'.join(targets)
        print(msg)
        print('detail: ' + msg)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == '__main__':
    try:
        main()
    except (OSError, IndexError, ValueError, subprocess.CalledProcessError) as e:
        print('detail: devices.py: %s' % e)
        sys.exit(2)
