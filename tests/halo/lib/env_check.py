#!/usr/bin/env python3
"""The configuration page lists every environment switch this tree adds, and nothing it no longer reads.

  env_check.py <repo>

Names are collected from the code (getenv, ggml_env_*, the arg parser's set_env and mmq.cuh's on() reads of
GGML_* / LLAMA_* literals) minus the names the upstream base already read, and compared with the backticked names in
halo-cookbook/configuration.md. Exit 0 when they match, 1 with the differences, 2 on an error of this tool.
"""
import re
import subprocess
import sys

UPSTREAM_BASE = 'ec7630a64'   # the last upstream merge (update at every sync, see HALO-HYBRID.md)
DIRS = ['ggml/src', 'ggml/include', 'src', 'include', 'common', 'tools']
READ = r'(?<![A-Za-z0-9_])(?:getenv|ggml_env_flag_unsafe|ggml_env_flag|ggml_env_int|ggml_env_float|set_env|on)\s*\(\s*"((?:GGML|LLAMA)_[A-Z0-9_]+)"'
PAGE = 'halo-cookbook/configuration.md'
UPSTREAM_ON_PAGE = {'GGML_CUDA_DISABLE_FUSION', 'GGML_CUDA_DISABLE_GRAPHS'}   # upstream's, named for context


def code_names(repo, rev=None):
    cmd = ['git', '-C', repo, 'grep', '-hoP', READ]
    if rev:
        cmd.append(rev)
    out = subprocess.run(cmd + ['--'] + DIRS, capture_output=True, text=True)
    if out.returncode not in (0, 1):
        raise OSError(out.stderr.strip())
    return {m for m in re.findall(READ, out.stdout) if not m.startswith('LLAMA_ARG_')}


def page_names(repo):
    """every switch the page names in backticks (`NAME`, `NAME=value`); patterns such as `GGML_CUDA_*` or
    `GGML_<BACKEND>_NO_<FEATURE>` are not names"""
    text = open('%s/%s' % (repo, PAGE)).read()
    names = set()
    for token in re.findall(r'`([^`]+)`', text):
        m = re.match(r'((?:GGML|LLAMA)_[A-Z0-9_]*[A-Z0-9])(?:=|$)', token)
        if m and '*' not in token and '<' not in token.split('=')[0]:
            names.add(m.group(1))
    return {n for n in names if not n.startswith('LLAMA_ARG_')} - UPSTREAM_ON_PAGE


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    repo = sys.argv[1]
    fork = code_names(repo) - code_names(repo, UPSTREAM_BASE)
    page = page_names(repo)
    missing, stale = sorted(fork - page), sorted(page - fork)
    for n in missing:
        print('not on %s: %s' % (PAGE, n))
    for n in stale:
        print('on %s but not read by the code: %s' % (PAGE, n))
    print('detail: %d switches, %d undocumented, %d stale' % (len(fork), len(missing), len(stale)))
    sys.exit(1 if missing or stale else 0)


if __name__ == '__main__':
    try:
        main()
    except (OSError, IndexError) as e:
        print('detail: env_check.py: %s' % e)
        sys.exit(2)
