#!/usr/bin/env python3
"""Rebuild a GGUF with a different dense trunk.
usage: merge_trunk.py <base.gguf shard1> <trunk.gguf (f32/bf16 conversion)> <typeref.gguf shard1> <out.gguf> [--dry] [--name=...]
Every tensor comes from <base> (all shards, its order and KV metadata) except those present in <trunk> whose type in
<typeref> is Q8_0 / F32 / BF16: those are re-encoded from the trunk data at the typeref type."""
import sys, glob, re, numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q

def shards(p):
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", p)
    return sorted(glob.glob(m.group(1) + "-*-of-" + m.group(2) + ".gguf")) if m else [p]

base_p, trunk_p, ref_p, out_p = sys.argv[1:5]
dry = "--dry" in sys.argv
base = [GGUFReader(p) for p in shards(base_p)]
trunk = {t.name: t for r in [GGUFReader(p) for p in shards(trunk_p)] for t in r.tensors}
reftype = {t.name: t.tensor_type for r in [GGUFReader(p) for p in shards(ref_p)] for t in r.tensors}
arch = base[0].fields["general.architecture"]
arch = bytes(arch.parts[arch.data[0]]).decode()
w = GGUFWriter(out_p, arch, split_max_size=45 * 1024**3, dry_run=dry)

# KV metadata from the base's first shard (split keys and the GGUF header fields are the writer's own)
for f in base[0].fields.values():
    if f.name.startswith("GGUF.") or f.name.startswith("split.") or f.name == "general.architecture":
        continue
    vtype = f.types[0]
    if vtype == gguf.GGUFValueType.ARRAY:
        sub = f.types[-1]
        val = [bytes(f.parts[i]).decode("utf-8", "replace") if sub == gguf.GGUFValueType.STRING else f.parts[i][0].item() for i in f.data]
        w.add_key_value(f.name, val, vtype, sub_type=sub)
    elif vtype == gguf.GGUFValueType.STRING:
        w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode("utf-8", "replace"), vtype)
    else:
        w.add_key_value(f.name, f.parts[f.data[0]][0].item(), vtype)
name = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--name=")), None)
if name:
    w.add_string("general.name", name)

n_rep = 0; bytes_old = bytes_new = 0; n_f32_same = n_f32_diff = 0
for r in base:
    for t in r.tensors:
        tt = trunk.get(t.name); rt = reftype.get(t.name)
        if tt is not None and rt in (Q.Q8_0, Q.F32, Q.BF16) and rt != t.tensor_type:
            sq = lambda sh: [int(x) for x in sh if int(x) != 1]
            if sq(tt.shape) != sq(t.shape):
                raise SystemExit(f"shape mismatch {t.name}: trunk {list(tt.shape)} base {list(t.shape)}")
            src = np.asarray(tt.data)
            if tt.tensor_type == Q.BF16:
                src = gguf.quants.dequantize(src, Q.BF16)
            elif tt.tensor_type != Q.F32:
                raise SystemExit(f"{t.name}: trunk type {tt.tensor_type.name}, expected F32/BF16")
            base_np_shape = [int(x) for x in reversed(t.shape)]
            src = src.astype(np.float32).reshape(base_np_shape) if len(base_np_shape) > 1 or src.ndim == 1 else src.astype(np.float32).reshape(-1)
            if t.tensor_type == Q.F32 and rt == Q.F32:
                same = np.array_equal(np.asarray(t.data, dtype=np.float32).reshape(-1), src.reshape(-1))
                n_f32_same += same; n_f32_diff += not same
                if not same:
                    print(f"  WARN f32 differs from base: {t.name}", flush=True)
            data = src if rt == Q.F32 else gguf.quants.quantize(src, rt)
            w.add_tensor(t.name, data, raw_shape=data.shape, raw_dtype=rt)
            n_rep += 1; bytes_old += t.n_bytes; bytes_new += data.nbytes
            print(f"replace {t.name:44s} {t.tensor_type.name:5s} -> {rt.name}", flush=True)
        else:
            w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
print(f"f32 cross-check: {n_f32_same} identical, {n_f32_diff} differ", flush=True)
print(f"replaced {n_rep} tensors: {bytes_old/2**30:.2f} GiB -> {bytes_new/2**30:.2f} GiB", flush=True)
if dry:
    raise SystemExit(0)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=True)
w.close()
