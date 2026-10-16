#!/usr/bin/env python3
"""Rebuild a GGUF with a different dense trunk.
usage: merge_trunk.py <base.gguf shard1> <trunk.gguf (f32/bf16 conversion)> <typeref.gguf shard1> <out.gguf> [--dry] [--name=...]
Every tensor comes from <base> (all shards, its order and KV metadata) except those present in <trunk> whose type in
<typeref> is Q8_0 / F32 / BF16: those are re-encoded from the trunk data at the typeref type."""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q  # noqa: E402


def shards(p):
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", p)
    return sorted(glob.glob(m.group(1) + "-*-of-" + m.group(2) + ".gguf")) if m else [p]


def squeeze(sh):
    return [int(x) for x in sh if int(x) != 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="the model to rebuild (any shard of it)")
    ap.add_argument("trunk", help="the f32/bf16 conversion holding the new trunk tensors (any shard of it)")
    ap.add_argument("typeref", help="the model whose tensor types the new trunk tensors take (any shard of it)")
    ap.add_argument("out", help="the GGUF to write; split into -0000N-of-0000M shards at --split-max-size")
    ap.add_argument("--dry", action="store_true", help="list the replacements, write nothing")
    ap.add_argument("--name", help="general.name of the result")
    ap.add_argument("--split-max-size", type=float, default=45, metavar="GIB", help="shard size in GiB (default 45)")
    a = ap.parse_args()

    base = [GGUFReader(p) for p in shards(a.base)]
    trunk = {t.name: t for r in [GGUFReader(p) for p in shards(a.trunk)] for t in r.tensors}
    reftype = {t.name: t.tensor_type for r in [GGUFReader(p) for p in shards(a.typeref)] for t in r.tensors}
    arch = base[0].fields["general.architecture"]
    arch = bytes(arch.parts[arch.data[0]]).decode()
    if not a.dry:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    w = GGUFWriter(a.out, arch, split_max_size=int(a.split_max_size * 1024**3), dry_run=a.dry)

    # KV metadata from the base's first shard (split keys and the GGUF header fields are the writer's own)
    for f in base[0].fields.values():
        if f.name.startswith("GGUF.") or f.name.startswith("split.") or f.name == "general.architecture":
            continue
        vtype = f.types[0]
        if vtype == gguf.GGUFValueType.ARRAY and not f.data:
            print(f"skip empty array {f.name} (gguf-py cannot write one)")  # noqa: NP100
            continue
        if vtype == gguf.GGUFValueType.ARRAY:
            sub = f.types[-1]
            val = [bytes(f.parts[i]).decode("utf-8", "replace") if sub == gguf.GGUFValueType.STRING else f.parts[i][0].item() for i in f.data]
            w.add_key_value(f.name, val, vtype, sub_type=sub)
        elif vtype == gguf.GGUFValueType.STRING:
            w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode("utf-8", "replace"), vtype)
        else:
            w.add_key_value(f.name, f.parts[f.data[0]][0].item(), vtype)
    if a.name:
        w.add_string("general.name", a.name)

    n_rep = 0
    bytes_old = bytes_new = 0
    n_f32_same = n_f32_diff = 0
    for r in base:
        for t in r.tensors:
            tt = trunk.get(t.name)
            rt = reftype.get(t.name)
            if tt is not None and rt in (Q.Q8_0, Q.F32, Q.BF16) and rt != t.tensor_type:
                if squeeze(tt.shape) != squeeze(t.shape):
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
                    n_f32_same += same
                    n_f32_diff += not same
                    if not same:
                        print(f"  WARN f32 differs from base: {t.name}", flush=True)  # noqa: NP100
                data = src if rt == Q.F32 else gguf.quants.quantize(src, rt)
                w.add_tensor(t.name, data, raw_shape=data.shape, raw_dtype=rt)
                n_rep += 1
                bytes_old += t.n_bytes
                bytes_new += data.nbytes
                print(f"replace {t.name:44s} {t.tensor_type.name:5s} -> {rt.name}", flush=True)  # noqa: NP100
            else:
                w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
    print(f"f32 cross-check: {n_f32_same} identical, {n_f32_diff} differ", flush=True)  # noqa: NP100
    print(f"replaced {n_rep} tensors: {bytes_old / 2**30:.2f} GiB -> {bytes_new / 2**30:.2f} GiB", flush=True)  # noqa: NP100
    if a.dry:
        return
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print("wrote", " ".join(sorted(glob.glob(os.path.splitext(a.out)[0] + "-*-of-*.gguf"))) or a.out, flush=True)  # noqa: NP100


if __name__ == "__main__":
    main()
