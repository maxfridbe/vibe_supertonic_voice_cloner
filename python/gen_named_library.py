"""Generate a named voice library: 20 male + 20 female, 5 of each seeded from
British reference styles (perturbed enough to be distinct new voices), the rest
sampled around the male/female presets. Self-contained style JSONs + audio."""
import os, sys, json, numpy as np, torch
os.environ["TORCHDYNAMO_DISABLE"]="1"; os.environ["TORCHINDUCTOR_DISABLE"]="1"
WORK=os.path.expanduser("~/supertonic-experiment")
os.environ.setdefault("SB_CACHE", os.path.join(WORK,".sb_cache"))
sys.path.insert(0, os.path.join(WORK,"supertonic-voice-cloning/src"))
sys.path.insert(0, os.path.join(WORK,"tooling"))
from train_encoder import StyleBasis
from pipeline import DiffSynth
dev="cuda:0"; O=os.path.join(WORK,"clone_out/overnight")
OUT=os.path.join(O,"named_library"); os.makedirs(OUT,exist_ok=True)
SENT="Hello there. This is a generated voice you can try out in the app."
LIB=os.path.join(WORK,"clone_out/library/styles")
PRE=os.path.join(WORK,"assets/voice_styles")

bp=np.load(os.path.join(WORK,"clone_out/pairs_p1.npz"),allow_pickle=True)
inv=np.load(os.path.join(O,"inversions/aux_pairs.npz"),allow_pickle=True)
rep=max(1,len(bp["ttl"])//max(len(inv["ttl"]),1))
basis=StyleBasis(np.concatenate([bp["ttl"],np.repeat(inv["ttl"],rep,0)]),
                 np.concatenate([bp["dp"],np.repeat(inv["dp"],rep,0)]),128,dev)
synth=DiffSynth(os.path.join(WORK,"assets/onnx"),device=str(dev))
dp0=bp["dp"][:1].astype(np.float32)
pt,pl=synth.pad_bounds([SENT],"en",dp0,1.05); prep=synth.prepare(SENT,"en",pad_text=pt,pad_latent=pl)

def load_style(path):
    o=json.load(open(path)); d1=o["style_ttl"]["dims"]; d2=o["style_dp"]["dims"]
    return (np.array(o["style_ttl"]["data"],np.float32).reshape(d1[1],d1[2]),
            np.array(o["style_dp"]["data"],np.float32).reshape(d2[1],d2[2]))
def coeff_of(path):
    ttl,dp=load_style(path)
    return basis.encode(torch.tensor(ttl[None],device=dev),torch.tensor(dp[None],device=dev)).squeeze(0).detach().cpu().numpy()

Mp=[coeff_of(os.path.join(PRE,f"M{i}.json")) for i in range(1,6)]
Fp=[coeff_of(os.path.join(PRE,f"F{i}.json")) for i in range(1,6)]
Mb=[coeff_of(os.path.join(LIB,"Stephen Fry.json")), coeff_of(os.path.join(LIB,"rigckman my mistress.json"))]
Fb=[coeff_of(os.path.join(LIB,"Soothing female british voice.json")), coeff_of(os.path.join(LIB,"bubbly british girl.json"))]

# (name, vibe, gender, british)
M_rand=["Gravel Pete|deep, gravelly","Smooth Cassius|smooth, easy","Old Salt Silas|weathered narrator",
"Thunder Boone|booming","Mellow Marlon|warm, mellow","Grizzled Gus|rough, rugged","Midnight Cole|smooth, low",
"Rusty Buck|country twang","Velvet Vincent|smooth, deep","Captain Crag|commanding","Jazzy Julian|cool, laid-back",
"Boomer Hank|big, hearty","Whispering Wade|soft, low","Sly Reggie|sly, playful","Rumble McCoy|rumbly bass"]
M_brit=["Sir Duncan|refined British","Baron von Bass|deep aristocratic British","Professor Fig|bookish British",
"Radio Ray|BBC broadcaster","Deacon Gray|solemn British"]
F_rand=["Bright Birdie|bright, cheery","Silky Seraphina|smooth, silky","Bubbly Bex|bubbly, upbeat","Sunny Sadie|sunny, warm",
"Frosty Fern|cool, crisp","Cocoa Coretta|rich, warm","Sparkle Sky|sparkly, high","Marmalade Maisie|sweet","Lark Lucia|light, airy",
"Honey Harlow|warm, smooth","Misty Marlowe|breathy","Ruby Rae|bold","Clementine June|cheerful","Sassy Simza|sassy","Pixie Wren|tiny, bright"]
F_brit=["Duchess Delphine|regal British","Nightingale Nova|melodic British","Willow Whisper|soft British",
"Foxglove Faye|mysterious British","Aurora Belle|ethereal British"]

rng=np.random.default_rng(24)
specs=[]
for i,s in enumerate(M_rand): specs.append((s,"male",False,Mp[i%5],0.95))
for i,s in enumerate(M_brit): specs.append((s,"male",True,Mb[i%2],0.5))
for i,s in enumerate(F_rand): specs.append((s,"female",False,Fp[i%5],0.95))
for i,s in enumerate(F_brit): specs.append((s,"female",True,Fb[i%2],0.5))

def synth_np(c):
    with torch.no_grad():
        ttl,dp=basis.decode(torch.tensor(c[None],dtype=torch.float32,device=dev))
        w,_,_=synth.synth(ttl,dp,prep)
    return ttl,dp,w.squeeze().detach().cpu().numpy()

import soundfile as sf
rows=[]
for s,gender,british,center,sigma in specs:
    name,vibe=s.split("|")
    for _ in range(8):
        c=np.clip(center+sigma*rng.standard_normal(128).astype(np.float32),-3,3)
        ttl,dp,wav=synth_np(c)
        if np.abs(wav).max()>0.03 and len(wav)>int(0.6*synth.sr): break
    slug=name.lower().replace(" ","-")
    tarr=ttl.squeeze(0).detach().cpu().numpy(); darr=dp.squeeze(0).detach().cpu().numpy()
    js={"style_ttl":{"dims":[1,*tarr.shape],"data":tarr.reshape(-1).tolist()},
        "style_dp":{"dims":[1,*darr.shape],"data":darr.reshape(-1).tolist()},
        "metadata":{"name":name,"source":"generated","gender":gender,"british":british,
                    "description":vibe+(" · British" if british else ""),"sample_text":SENT}}
    json.dump(js,open(os.path.join(OUT,slug+".json"),"w"))
    sf.write(os.path.join(OUT,slug+".wav"),wav,synth.sr)
    rows.append({"slug":slug,"name":name,"gender":gender,"british":british,"vibe":vibe})
    print(f"{'GB' if british else '  '} {gender[0].upper()} {name}: {vibe}")
json.dump(rows,open(os.path.join(OUT,"index.json"),"w"),indent=1)
print(f"\n{len(rows)} voices -> {OUT}")
