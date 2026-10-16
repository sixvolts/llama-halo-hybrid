#!/usr/bin/env python3
r"""Fetch only some tensors (default mtp.*) of a sharded HF safetensors checkpoint via HTTP range requests and write them
as one safetensors file + index, next to the config/tokenizer files, so convert_hf_to_gguf.py can run on it.
usage: fetch_hf_tensors.py <repo> <outdir> [include-regex] [exclude-regex]   (default include ^mtp\.)
e.g. the Swift-1.5 MTP head:  fetch_hf_tensors.py ukisai/Swift1.5-Qwen3.8-Flash-Next mtp-src
     its dense trunk:         fetch_hf_tensors.py ukisai/Swift1.5-Qwen3.8-Flash-Next trunk-src . 'mlp\.experts\.|ngram_embedding\.shard_|^model\.visual\.|^mtp\.'
then convert_hf_to_gguf.py <outdir> --mtp --mtp-shared-embd (head) or --no-mtp --outtype f32 (trunk, for merge_trunk.py).
HF_TOKEN in the environment is sent for gated or private repositories."""
import argparse
import collections
import json
import os
import re
import struct
import time
import urllib.error
import urllib.request

CONFIG_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="Hugging Face repository, e.g. ukisai/Swift1.5-Qwen3.8-Flash-Next")
    ap.add_argument("out", help="output directory")
    ap.add_argument("include", nargs="?", default=r"^mtp\.", help=r"regex of tensor names to fetch (default ^mtp\.)")
    ap.add_argument("exclude", nargs="?", default=None, help="regex of tensor names to leave out")
    a = ap.parse_args()
    inc = re.compile(a.include)
    exc = re.compile(a.exclude) if a.exclude else None
    out = a.out
    os.makedirs(out, exist_ok=True)
    base = f"https://huggingface.co/{a.repo}/resolve/main/"
    token = os.environ.get("HF_TOKEN")

    def get(path, rng=None, required=True):
        req = urllib.request.Request(base + path)
        if token:
            req.add_header("Authorization", "Bearer " + token)
        if rng:
            req.add_header("Range", "bytes=%d-%d" % rng)
        for t in range(5):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404):   # not there or not allowed: retrying does not help
                    if not required:
                        return None
                    hint = " (gated or private: set HF_TOKEN)" if e.code in (401, 403) else ""
                    raise SystemExit(f"{base + path}: HTTP {e.code}{hint}")
                print("retry", path, rng, e, flush=True)  # noqa: NP100
            except Exception as e:
                print("retry", path, rng, e, flush=True)  # noqa: NP100
            time.sleep(2 ** t)
        if not required:
            return None
        raise SystemExit("failed " + path)

    for f in CONFIG_FILES:
        data = get(f, required=False)
        if data is None:
            print("skip", f)  # noqa: NP100
            continue
        with open(os.path.join(out, f), "wb") as fo:
            fo.write(data)
    idx = json.loads(get("model.safetensors.index.json"))
    by_shard = collections.defaultdict(list)
    for k, v in idx["weight_map"].items():
        if inc.search(k) and not (exc and exc.search(k)):
            by_shard[v].append(k)
    if not by_shard:
        raise SystemExit(f"{a.repo}: no tensor matches {a.include}")
    hdr_out, off = {}, 0
    tmp = open(os.path.join(out, 'data.tmp'), 'wb')
    for shard, names in sorted(by_shard.items()):
        n = struct.unpack("<Q", get(shard, (0, 7)))[0]
        h = json.loads(get(shard, (8, 8 + n - 1)))
        for k in sorted(names):
            m = h[k]
            b, e = m["data_offsets"]
            data = get(shard, (8 + n + b, 8 + n + e - 1))
            if len(data) != e - b:
                raise SystemExit(f"{shard}: {k}: got {len(data)} bytes, expected {e - b}")
            hdr_out[k] = {"dtype": m["dtype"], "shape": m["shape"], "data_offsets": [off, off + len(data)]}
            tmp.write(data)
            off += len(data)
            print(f"{k} {m['dtype']} {m['shape']} {len(data) >> 20} MiB", flush=True)  # noqa: NP100
    hdr_out["__metadata__"] = {"format": "pt"}
    hj = json.dumps(hdr_out).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(os.path.join(out, "model-subset.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        tmp.close()
        with open(os.path.join(out, 'data.tmp'), 'rb') as t:
            while True:
                b = t.read(1 << 26)
                if not b:
                    break
                f.write(b)
    os.remove(os.path.join(out, 'data.tmp'))
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as fo:
        json.dump({"metadata": {"total_size": off}, "weight_map": {k: "model-subset.safetensors" for k in hdr_out if k != "__metadata__"}},
                  fo, indent=1)
    print("total", off >> 20, "MiB")  # noqa: NP100


if __name__ == "__main__":
    main()
