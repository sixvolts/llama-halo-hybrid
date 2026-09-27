#!/usr/bin/env python3
"""Give a shared-embedding MTP draft its own LM head (nextn.shared_head_head), full vocabulary, taken from a source GGUF
tensor (e.g. output.weight of an f32 trunk conversion). Quantize the result with llama-quantize --tensor-type
shared_head_head=<type> so drafting reads a lower-bit copy while verification keeps the target's own head.
usage: add_draft_head.py <in-mtp.gguf> <source.gguf> <source tensor name> <out.gguf>"""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter
mtp_p, src_p, src_name, out_p = sys.argv[1:5]
src = GGUFReader(mtp_p)
head = next(t for t in GGUFReader(src_p).tensors if t.name == src_name)
arch = bytes(src.fields["general.architecture"].parts[src.fields["general.architecture"].data[0]]).decode()
w = GGUFWriter(out_p, arch)
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
    m = re.match(r"blk\.(\d+)\.nextn\.", t.name)
    if m: blk = m.group(1)
    if t.name.endswith("nextn.shared_head_head.weight"):
        continue
    w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
w.add_tensor(f"blk.{blk}.nextn.shared_head_head.weight", head.data, raw_shape=head.data.shape, raw_dtype=head.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {out_p}: blk.{blk}.nextn.shared_head_head.weight {head.tensor_type.name} {list(head.shape)}")
