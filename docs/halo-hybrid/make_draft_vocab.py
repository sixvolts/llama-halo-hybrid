#!/usr/bin/env python3
"""Build a reduced-vocabulary MTP draft head (nextn.shared_head_head over a token subset + d2t), as EAGLE-3 does.
usage: make_draft_vocab.py --mtp in-mtp.gguf --target target-shard1.gguf --out out-mtp.gguf --n 32768
                           --ids gen.jsonl [--ids-weight 10] [--corpus-ids corpus.txt ...] [--always tokens.txt]
The head rows are copied byte for byte from the target's output.weight, so each drafted logit is exactly what the full
head would give for that token; tokens outside the subset can never be drafted. Verification is unchanged."""
import sys, os, glob, re, json, argparse, collections, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q

ap = argparse.ArgumentParser()
ap.add_argument("--mtp", required=True); ap.add_argument("--target", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=32768)
ap.add_argument("--ids", action="append", default=[], help="jsonl with an 'ids' list per line (self-generated text)")
ap.add_argument("--ids-weight", type=float, default=10.0)
ap.add_argument("--corpus-ids", action="append", default=[], help="whitespace-separated token ids")
ap.add_argument("--stats", action="store_true", help="only print coverage statistics")
ap.add_argument("--eval", action="append", default=[], help="held-out jsonl for --stats coverage")
a = ap.parse_args()

def shards(p):
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", p)
    return sorted(glob.glob(m.group(1) + "-*-of-" + m.group(2) + ".gguf")) if m else [p]

cnt = collections.Counter()
held = []
for f in a.ids:
    for line in open(f):
        ids = json.loads(line)["ids"]
        for t in ids: cnt[t] += a.ids_weight
for f in a.corpus_ids:
    for t in open(f).read().split(): cnt[int(t)] += 1.0

tgt = [GGUFReader(p) for p in shards(a.target)]
out_w = next(t for r in tgt for t in r.tensors if t.name == "output.weight")
n_vocab = int(out_w.shape[1])
# control / user-defined tokens (chat template, think tags, tool calls) are always drafted
tt = tgt[0].fields.get("tokenizer.ggml.token_type")
always = set()
if tt is not None:
    types = [int(tt.parts[i][0]) for i in tt.data]
    always = {i for i, ty in enumerate(types) if ty in (3, 4)}   # CONTROL, USER_DEFINED
order = [t for t, _ in cnt.most_common() if t < n_vocab and t not in always]
sel = set(order[: max(0, a.n - len(always))]) | always
# fewer distinct tokens seen than asked for: pad with the lowest ids (BPE ids roughly follow merge frequency)
for t in range(n_vocab):
    if len(sel) >= a.n: break
    sel.add(t)
keep = sorted(sel)
print(f"vocab {n_vocab}, seen {len(cnt)} distinct, always {len(always)}, keep {len(keep)}")
if a.stats:
    s = set(keep)
    for f in a.ids + a.eval:
        tot = hit = 0
        for line in open(f):
            ids = json.loads(line)["ids"]; tot += len(ids); hit += sum(1 for t in ids if t in s)
        print(f"coverage of {f}: {hit/tot*100:.3f}% of {tot} tokens")
    sys.exit(0)

rows = np.asarray(out_w.data)                     # [n_vocab, bytes per row] for quantized types
head = np.ascontiguousarray(rows[keep])
d2t = np.asarray(keep, dtype=np.int64)

src = GGUFReader(a.mtp)
arch = bytes(src.fields["general.architecture"].parts[src.fields["general.architecture"].data[0]]).decode()
w = GGUFWriter(a.out, arch)
for f in src.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture":
        continue
    vt = f.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        sub = f.types[-1]
        val = [bytes(f.parts[i]).decode("utf-8", "replace") if sub == gguf.GGUFValueType.STRING else f.parts[i][0].item() for i in f.data]
        w.add_key_value(f.name, val, vt, sub_type=sub)
    elif vt == gguf.GGUFValueType.STRING:
        w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode("utf-8", "replace"), vt)
    else:
        w.add_key_value(f.name, f.parts[f.data[0]][0].item(), vt)
blk = None
for t in src.tensors:
    if t.name in ("d2t",) or t.name.endswith("nextn.shared_head_head.weight"):
        continue
    m = re.match(r"blk\.(\d+)\.nextn\.", t.name)
    if m: blk = m.group(1)
    w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
w.add_tensor(f"blk.{blk}.nextn.shared_head_head.weight", head, raw_shape=head.shape, raw_dtype=out_w.tensor_type)
w.add_tensor("d2t", d2t)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {a.out}: head {out_w.tensor_type.name} {head.shape}, d2t {d2t.shape}")
