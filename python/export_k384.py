import os, sys, struct, numpy as np, torch
WORK=os.path.expanduser("~/supertonic-experiment")
sys.path.insert(0, os.path.join(WORK,"supertonic-voice-cloning/src"))
sys.path.insert(0, os.path.join(WORK,"tooling"))
from train_encoder import StyleBasis
O=os.path.join(WORK,"clone_out/overnight")
bp=np.load(os.path.join(WORK,"clone_out/pairs_p1.npz"),allow_pickle=True)
inv=np.load(os.path.join(O,"inversions/aux_pairs.npz"),allow_pickle=True)
rep=max(1,len(bp["ttl"])//max(len(inv["ttl"]),1))
FIT_TTL=np.concatenate([bp["ttl"],np.repeat(inv["ttl"],rep,0)])
FIT_DP =np.concatenate([bp["dp"], np.repeat(inv["dp"], rep,0)])
b=StyleBasis(FIT_TTL,FIT_DP,384,"cpu")
basis=b.basis.numpy().astype(np.float32); mean=b.mean.numpy().astype(np.float32); scale=b.scale.numpy().astype(np.float32)
ttl_r,ttl_c=(int(v) for v in b.ttl_shape); dp_r,dp_c=(int(v) for v in b.dp_shape)
k,dTot=basis.shape; split=b.split
out=os.path.join(O,"style_basis_k384.bin")
with open(out,"wb") as f:
    f.write(struct.pack("<7i",k,dTot,split,ttl_r,ttl_c,dp_r,dp_c))
    f.write(scale.tobytes()); f.write(mean.tobytes()); f.write(basis.tobytes())
print(f"wrote {out}: k={k} D={dTot} split={split} ttl={ttl_r}x{ttl_c} dp={dp_r}x{dp_c} ({os.path.getsize(out)} bytes)")
# round-trip check
rng=np.random.default_rng(0); c=rng.standard_normal(k).astype(np.float32)
flat=mean+(c*scale)@basis; back=((flat-mean)@basis.T)/scale
print("encode(decode(c)) max abs err", float(np.abs(back-c).max()))
