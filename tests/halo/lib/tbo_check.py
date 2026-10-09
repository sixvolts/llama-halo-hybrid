#!/usr/bin/env python3
"""Read test-backend-ops (test mode) output on stdin; exit 0 only if every case on the tested backend passed.

A crash or abort leaves no "N/M tests passed" line, so a missing line is a failure too.
Prints "detail: <passed>/<total> passed[, failing case names]" for the step table.
"""
import re
import sys

ansi = re.compile(r'\x1b\[[0-9;]*m')
passed = total = 0
seen = False
fails = []
backend_fail = False
for line in sys.stdin:
    line = ansi.sub('', line.rstrip('\n'))
    m = re.search(r'(\d+)/(\d+) tests passed', line)
    if m:
        seen = True
        passed += int(m.group(1))
        total += int(m.group(2))
    if re.search(r'\]: FAIL\b|: FAIL$', line) and '(' in line:
        fails.append(line.strip().split('(')[0] + '(' + line.strip().split('(', 1)[1][:80])
    if re.match(r'\s*Backend \S+: FAIL', line):
        backend_fail = True

ok = seen and passed == total and not fails and not backend_fail
if not seen:
    print('detail: no "tests passed" line (crash or abort?)')
elif ok:
    print('detail: %d/%d passed' % (passed, total))
else:
    print('detail: %d/%d passed; %d failing, first: %s' % (passed, total, len(fails), fails[0] if fails else 'backend FAIL'))
for f in fails[:50]:
    print('FAIL ' + f)
sys.exit(0 if ok else 1)
