"""Does a bigger / refit basis raise the cloning ceiling past the ~0.77 plateau?
For each k, fit the basis (preset styles + inverted real speakers), project each
eval speaker's desktop inversion (its ~0.82 style) through it, decode, synthesise
held-out sentences, and score ECAPA cosine to the reference. If the ceiling
climbs toward 0.8+, a refit basis is the fix; if it stays flat, it isn't."""
import os, sys, json, numpy as np, torch
os.environ["TORCHDYNAMO_DISABLE"]="1"; os.environ["TORCHINDUCTOR_DISABLE"]="1"
WORK=os.path.expanduser("~/supertonic-experiment")
os.environ.setdefault("SB_CACHE", os.path.join(WORK,".sb_cache"))
sys.path.insert(0, os.path.join(WORK,"supertonic-voice-cloning/src"))
sys.path.insert(0, os.path.join(WORK,"tooling"))
from train_encoder import StyleBasis
from pipeline import DiffSynth
from eval_style import HELD_OUT, load16k
from speechbrain.inference import EncoderClassifier
import torchaudio
dev="cuda:0"; O=os.path.join(WORK,"clone_out/overnight")

bp=np.load(os.path.join(WORK,"clone_out/pairs_p1.npz"),allow_pickle=True)
inv=np.load(os.path.join(O,"inversions/aux_pairs.npz"),allow_pickle=True)
rep=max(1,len(bp["ttl"])//max(len(inv["ttl"]),1))
FIT_TTL=np.concatenate([bp["ttl"], np.repeat(inv["ttl"],rep,0)])
FIT_DP =np.concatenate([bp["dp"],  np.repeat(inv["dp"], rep,0)])

synth=DiffSynth(os.path.join(WORK,"assets/onnx"),device=str(dev))
enc=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
      savedir=os.environ["SB_CACHE"], run_opts={"device":dev})
resample=torchaudio.transforms.Resample(synth.sr,16000).to(dev)
def emb_wav(w):
    if w.dim()==1: w=w.unsqueeze(0)
    return torch.nn.functional.normalize(enc.encode_batch(resample(w)).squeeze(),dim=-1)
def emb_file(p):
    return torch.nn.functional.normalize(enc.encode_batch(load16k(p,dev)).squeeze(),dim=-1)

targets=["Dale","Fireside Narrator","Soothing female british voice","Stephen Fry"]
styles={}; refs={}
for t in targets:
    o=json.load(open(os.path.join(WORK,"clone_out/library/styles",t+".json")))
    d1,d2=o["style_ttl"]["dims"],o["style_dp"]["dims"]
    styles[t]=(np.array(o["style_ttl"]["data"],np.float32).reshape(d1[1],d1[2]),
               np.array(o["style_dp"]["data"],np.float32).reshape(d2[1],d2[2]))
    rp=os.path.join(O,"eval_refs",t+".wav")
    if not os.path.exists(rp): rp=os.path.join(O,"eval_refs",t+".flac")
    refs[t]=emb_file(rp)

dp0=bp["dp"][:1].astype(np.float32)
pt,pl=synth.pad_bounds(HELD_OUT,"en",dp0,1.05)
preps={t:synth.prepare(t,"en",pad_text=pt,pad_latent=pl) for t in HELD_OUT}

def score(ttl,dp):
    with torch.no_grad():
        per=[]
        for t in HELD_OUT:
            w,_,_=synth.synth(ttl,dp,preps[t]); per.append(float(torch.dot(refs_cur,emb_wav(w))))
    return float(np.mean(per))

print(f"fit set: {len(FIT_TTL)} styles (presets + {len(inv['ttl'])} inverted)")
results={}
for k in [128,256,384,512]:
    basis=StyleBasis(FIT_TTL,FIT_DP,k,dev)
    row={}
    for t in targets:
        refs_cur=refs[t]
        tt=torch.tensor(styles[t][0][None],device=dev); dd=torch.tensor(styles[t][1][None],device=dev)
        c=basis.encode(tt,dd)
        rttl,rdp=basis.decode(c)
        row[t]=round(score(rttl,rdp),3)
    results[str(k)]=row
    m=np.mean(list(row.values()))
    print(f"k={k}: mean ceiling {m:.3f}  " + "  ".join(f"{t.split()[0]}={v}" for t,v in row.items()))
json.dump(results,open(os.path.join(O,"basis_k_sweep.json"),"w"),indent=1)
open(os.path.join(O,"DONE_KSWEEP"),"w").write("done")
