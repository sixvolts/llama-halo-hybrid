#!/usr/bin/env python3
"""Give a shared-embedding MTP draft its own LM head (nextn.shared_head_head), full vocabulary, taken from a source GGUF
tensor (e.g. output.weight of an f32 trunk conversion). Quantize the result with llama-quantize --tensor-type
shared_head_head=<type> so drafting reads a lower-bit copy while verification keeps the target's own head.
usage: add_draft_head.py <in-mtp.gguf> <source.gguf> <source tensor name> <out.gguf>
(source.gguf may be any shard of a split model; every shard is searched)"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402


def shards(p):
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", p)
    return sorted(glob.glob(m.group(1) + "-*-of-" + m.group(2) + ".gguf")) if m else [p]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mtp", help="the shared-embedding MTP draft GGUF")
    ap.add_argument("source", help="the GGUF (or one shard of it) holding the head tensor")
    ap.add_argument("tensor", help="the source tensor name, e.g. output.weight")
    ap.add_argument("out", help="the GGUF to write")
    a = ap.parse_args()

    src = GGUFReader(a.mtp)
    head = next((t for p in shards(a.source) for t in GGUFReader(p).tensors if t.name == a.tensor), None)
    if head is None:
        raise SystemExit(f"{a.source}: no tensor named {a.tensor} in any shard")
    blk = None
    for t in src.tensors:
        m = re.match(r"blk\.(\d+)\.nextn\.", t.name)
        if m:
            blk = m.group(1)
    if blk is None:
        raise SystemExit(f"{a.mtp}: no blk.N.nextn.* tensor; this is not an MTP draft")

    arch = bytes(src.fields["general.architecture"].parts[src.fields["general.architecture"].data[0]]).decode()
    w = GGUFWriter(a.out, arch)
    for f in src.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY and not f.data:
            print(f"skip empty array {f.name} (gguf-py cannot write one)")  # noqa: NP100
            continue
        if vt == gguf.GGUFValueType.ARRAY:
            sub = f.types[-1]
            val = [bytes(f.parts[i]).decode("utf-8", "replace") if sub == gguf.GGUFValueType.STRING else f.parts[i][0].item() for i in f.data]
            w.add_key_value(f.name, val, vt, sub_type=sub)
        elif vt == gguf.GGUFValueType.STRING:
            w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode("utf-8", "replace"), vt)
        else:
            w.add_key_value(f.name, f.parts[f.data[0]][0].item(), vt)
    for t in src.tensors:
        if t.name.endswith("nextn.shared_head_head.weight"):
            continue
        w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
    w.add_tensor(f"blk.{blk}.nextn.shared_head_head.weight", head.data, raw_shape=head.data.shape, raw_dtype=head.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: blk.{blk}.nextn.shared_head_head.weight {head.tensor_type.name} {[int(x) for x in head.shape]}")  # noqa: NP100


if __name__ == "__main__":
    main()
