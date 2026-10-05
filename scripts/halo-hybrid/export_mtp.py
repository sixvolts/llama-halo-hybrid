# Export the NextN/MTP block (blk.<n_layer>) of a split GGUF into a draft-only GGUF that borrows the target's
# token_embd/output/output_norm (halo loader: <arch>.nextn_shared_target_tensors = true). Pure Python, no numpy.
import struct, glob, sys, os
src_dir, out_path = sys.argv[1], sys.argv[2]
GGUF_MAGIC=b'GGUF'
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)
FMT={T_U8:'B',T_I8:'b',T_U16:'H',T_I16:'h',T_U32:'I',T_I32:'i',T_F32:'f',T_BOOL:'?',T_U64:'Q',T_I64:'q',T_F64:'d'}
def read_gguf(path):
    f=open(path,'rb'); assert f.read(4)==GGUF_MAGIC
    def rd(fmt): return struct.unpack('<'+fmt, f.read(struct.calcsize(fmt)))[0]
    def rstr(): n=rd('Q'); return f.read(n)
    def rval(t):
        if t in FMT: return rd(FMT[t])
        if t==T_STR: return rstr()
        if t==T_ARR:
            et=rd('I'); n=rd('Q'); return (et,[rval(et) for _ in range(n)])
        raise ValueError(t)
    ver=rd('I'); nt=rd('Q'); nkv=rd('Q')
    kv=[]
    for _ in range(nkv):
        k=rstr().decode(); t=rd('I'); v=rval(t); kv.append((k,t,v))
    tensors=[]
    for _ in range(nt):
        name=rstr().decode(); nd=rd('I'); dims=[rd('Q') for _ in range(nd)]; ty=rd('I'); off=rd('Q'); tensors.append((name,dims,ty,off))
    align=32
    for k,t,v in kv:
        if k=='general.alignment': align=v
    data_start=(f.tell()+align-1)//align*align
    return kv,tensors,data_start,align,f
# type sizes: (block size, type size in bytes)
TS={0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),21:(256,110),22:(256,82),23:(256,136),29:(256,56),30:(1,2),39:(32,17)}
def nbytes(dims,ty):
    bs,ts=TS[ty]; n=1
    for d in dims: n*=d
    assert n%bs==0; return n//bs*ts
shards=sorted(glob.glob(os.path.join(src_dir,'*.gguf')))
kv0,_,_,align,_=read_gguf(shards[0])
n_layer_all=[v for k,t,v in kv0 if k.endswith('.block_count')][0]
n_nextn=[v for k,t,v in kv0 if k.endswith('.nextn_predict_layers')][0]
arch=[v for k,t,v in kv0 if k=='general.architecture'][0].decode()
il=n_layer_all-n_nextn; prefix=f'blk.{il}.'
print(f"arch {arch}, block_count {n_layer_all}, nextn {n_nextn}: exporting {prefix}*")
# collect tensors from all shards
picked=[]
for path in shards:
    kv,tensors,data_start,al,f=read_gguf(path)
    for name,dims,ty,off in tensors:
        if name.startswith(prefix): picked.append((name,dims,ty,path,data_start+off))
print("tensors:", len(picked), "bytes:", sum(nbytes(d,t) for _,d,t,_,_ in picked)//1000000, "MB")
# metadata: copy, drop split.*, add the shared flag
out_kv=[(k,t,v) for k,t,v in kv0 if not k.startswith('split.')]
out_kv.append((f'{arch}.nextn_shared_target_tensors', T_BOOL, True))
def wstr(b): return struct.pack('<Q',len(b))+b
def wval(t,v):
    if t in FMT: return struct.pack('<'+FMT[t], v)
    if t==T_STR: return wstr(v)
    if t==T_ARR:
        et,items=v; return struct.pack('<IQ',et,len(items))+b''.join(wval(et,x) for x in items)
hdr=GGUF_MAGIC+struct.pack('<IQQ',3,len(picked),len(out_kv))
for k,t,v in out_kv: hdr+=wstr(k.encode())+struct.pack('<I',t)+wval(t,v)
off=0; infos=b''; layout=[]
for name,dims,ty,path,src_off in picked:
    infos+=wstr(name.encode())+struct.pack('<I',len(dims))+b''.join(struct.pack('<Q',d) for d in dims)+struct.pack('<IQ',ty,off)
    layout.append((path,src_off,nbytes(dims,ty),off)); off=(off+nbytes(dims,ty)+align-1)//align*align
hdr+=infos
data_start=(len(hdr)+align-1)//align*align
with open(out_path,'wb') as o:
    o.write(hdr); o.write(b'\0'*(data_start-len(hdr)))
    for path,src_off,n,dst in layout:
        o.seek(data_start+dst); f=open(path,'rb'); f.seek(src_off)
        left=n
        while left>0:
            chunk=f.read(min(left,64<<20)); o.write(chunk); left-=len(chunk)
    o.truncate(data_start+off)
print("wrote", out_path, os.path.getsize(out_path)//1000000, "MB")
