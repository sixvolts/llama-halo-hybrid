#!/usr/bin/env python3
# Fetch only the mtp.* tensors of a sharded HF safetensors checkpoint via HTTP range requests and write them as
# one safetensors file + index, next to the config/tokenizer files, so convert_hf_to_gguf.py --mtp can run on it.
# usage: fetch_hf_tensors.py <repo> <outdir> [include-regex] [exclude-regex]   (default include ^mtp\.)
# e.g. the Swift-1.5 MTP head:  fetch_hf_tensors.py ukisai/Swift1.5-Qwen3.8-Flash-Next mtp-src
#      its dense trunk:         fetch_hf_tensors.py ukisai/Swift1.5-Qwen3.8-Flash-Next trunk-src . 'mlp\.experts\.|ngram_embedding\.shard_|^model\.visual\.|^mtp\.'
# then convert_hf_to_gguf.py <outdir> --mtp --mtp-shared-embd (head) or --no-mtp --outtype f32 (trunk, for merge_trunk.py)
import json, struct, sys, os, urllib.request, collections, re
repo, out = sys.argv[1], sys.argv[2]
inc = re.compile(sys.argv[3] if len(sys.argv) > 3 else r'^mtp\.')
exc = re.compile(sys.argv[4]) if len(sys.argv) > 4 else None
os.makedirs(out, exist_ok=True)
base = f"https://huggingface.co/{repo}/resolve/main/"
def get(path, rng=None):
    req = urllib.request.Request(base + path)
    if rng: req.add_header("Range", "bytes=%d-%d" % rng)
    for t in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as r: return r.read()
        except Exception as e: print("retry", path, rng, e, flush=True)
    raise SystemExit("failed " + path)
for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"]:
    try: open(os.path.join(out, f), "wb").write(get(f))
    except SystemExit: print("skip", f)
idx = json.loads(get("model.safetensors.index.json"))
by_shard = collections.defaultdict(list)
for k, v in idx["weight_map"].items():
    if inc.search(k) and not (exc and exc.search(k)): by_shard[v].append(k)
hdr_out, off = {}, 0
tmp = open(os.path.join(out, 'data.tmp'), 'wb')
for shard, names in sorted(by_shard.items()):
    n = struct.unpack("<Q", get(shard, (0, 7)))[0]
    h = json.loads(get(shard, (8, 8 + n - 1)))
    for k in sorted(names):
        m = h[k]; b, e = m["data_offsets"]
        data = get(shard, (8 + n + b, 8 + n + e - 1)); assert len(data) == e - b
        hdr_out[k] = {"dtype": m["dtype"], "shape": m["shape"], "data_offsets": [off, off + len(data)]}
        tmp.write(data); off += len(data)
        print(f"{k} {m['dtype']} {m['shape']} {len(data)>>20} MiB", flush=True)
hdr_out["__metadata__"] = {"format": "pt"}
hj = json.dumps(hdr_out).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
with open(os.path.join(out, "model-subset.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hj))); f.write(hj)
    tmp.close()
    with open(os.path.join(out, 'data.tmp'), 'rb') as t:
        while True:
            b = t.read(1 << 26)
            if not b: break
            f.write(b)
os.remove(os.path.join(out, 'data.tmp'))
json.dump({"metadata": {"total_size": off}, "weight_map": {k: "model-subset.safetensors" for k in hdr_out if k != "__metadata__"}},
          open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
print("total", off >> 20, "MiB")
