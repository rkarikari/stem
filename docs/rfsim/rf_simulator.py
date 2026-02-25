"""
RF S-Meter Simulator — Streamlit App
=====================================
Full interactive UI with sidebar controls, dynamic graphs, and audio generation.
"""

import streamlit as st
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, butter, lfilter, welch
import math
import io
import tempfile
import os
import sys
import multiprocessing as mp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
def initialize_ui():
    st.set_page_config(
        page_title="RF S-Meter Simulator",
        page_icon="🧟",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Report a Bug': "https://github.com/rkarikari/stem",
            'About': "Copyright \u00a9 RNK, 2026 RadioSport. All rights reserved."
        }
    )

initialize_ui()

# ── CSS theme ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Exo 2',sans-serif;}
.stApp{background:#0a0e14;color:#c8d6e5;}
section[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid #1e2d3d;}
section[data-testid="stSidebar"] *{color:#8bafc7!important;}
section[data-testid="stSidebar"] label{color:#4fc3f7!important;font-family:'Share Tech Mono',monospace!important;font-size:0.72rem!important;letter-spacing:0.08em;text-transform:uppercase;}
h1{font-family:'Share Tech Mono',monospace!important;color:#4fc3f7!important;letter-spacing:0.12em;text-transform:uppercase;font-size:1.5rem!important;}
h2{font-family:'Share Tech Mono',monospace!important;color:#29b6f6!important;letter-spacing:0.08em;font-size:1.05rem!important;}
h3{font-family:'Share Tech Mono',monospace!important;color:#4dd0e1!important;font-size:0.9rem!important;}
[data-testid="metric-container"]{background:#0d1117;border:1px solid #1e2d3d;border-radius:4px;padding:0.5rem 0.8rem;}
[data-testid="metric-container"] label{color:#4fc3f7!important;font-family:'Share Tech Mono',monospace!important;font-size:0.68rem!important;letter-spacing:0.1em;text-transform:uppercase;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#e0f7fa!important;font-family:'Share Tech Mono',monospace!important;font-size:1.3rem!important;}
.stButton>button{background:linear-gradient(135deg,#0d47a1,#01579b);color:#e3f2fd!important;border:1px solid #1565c0;border-radius:3px;font-family:'Share Tech Mono',monospace;font-size:0.82rem;letter-spacing:0.1em;text-transform:uppercase;padding:0.5rem 1.2rem;transition:all 0.2s;}
.stButton>button:hover{background:linear-gradient(135deg,#1565c0,#0277bd);border-color:#4fc3f7;box-shadow:0 0 12px rgba(79,195,247,0.3);}
.stTabs [data-baseweb="tab-list"]{background:#0d1117;border-bottom:1px solid #1e2d3d;}
.stTabs [data-baseweb="tab"]{font-family:'Share Tech Mono',monospace;font-size:0.72rem;letter-spacing:0.1em;color:#546e7a!important;background:transparent;border:none;padding:0.55rem 1.2rem;text-transform:uppercase;}
.stTabs [aria-selected="true"]{color:#4fc3f7!important;border-bottom:2px solid #4fc3f7!important;background:transparent!important;}
.badge-ok{display:inline-block;padding:2px 7px;border-radius:2px;font-family:'Share Tech Mono',monospace;font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;background:#0a2a1a;border:1px solid #2e7d32;color:#66bb6a;}
.badge-warn{display:inline-block;padding:2px 7px;border-radius:2px;font-family:'Share Tech Mono',monospace;font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;background:#2a1f0a;border:1px solid #f57f17;color:#ffca28;}
.smh{font-family:'Share Tech Mono',monospace;color:#4fc3f7;font-size:0.68rem;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.3rem;margin-bottom:0.7rem;}
hr{border-color:#1e2d3d!important;margin:0.8rem 0;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#0a0e14;}
::-webkit-scrollbar-thumb{background:#1e2d3d;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ── DSP helpers ───────────────────────────────────────────────────────────────

def apply_bandpass(sig, sr, low=300, high=3000):
    nyq = 0.5 * sr
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)
    if lo >= hi:
        return sig
    b, a = butter(5, [lo, hi], btype='band')
    return lfilter(b, a, sig)

# ── Formant synthesis ─────────────────────────────────────────────────────────

def _formant_vowel(f1, f2, f3, dur, sr, f0=115, amp=1.0, f2b=1.0):
    n = int(sr * dur)
    if n == 0:
        return np.zeros(1)
    t = np.arange(n) / sr
    sig = np.zeros(n)
    formants = [(f1,80,1.0),(f2,130,0.8*f2b),(f3,160,0.45)]
    for k in range(1, int(4500/f0)+1):
        fq = k*f0
        ha = sum(fa*math.exp(-0.5*((fq-fc)/bw)**2) for fc,bw,fa in formants)
        ha *= 1.0/math.sqrt(k)
        sig += ha*np.sin(2*np.pi*fq*t)
    b,a = butter(1, 60/(sr/2), btype='high')
    sig = lfilter(b,a,sig)
    ramp = min(int(0.015*sr), n//4)
    env = np.ones(n); env[:ramp]=np.linspace(0,1,ramp); env[-ramp:]=np.linspace(1,0,ramp)
    sig *= env; sig /= np.max(np.abs(sig))+1e-9
    return sig*amp

def _fricative(dur, sr, cf=5500, bw=1500, amp=0.7):
    n = int(sr*dur)
    if n == 0: return np.zeros(1)
    lo=max((cf-bw//2)/(sr/2),0.001); hi=min((cf+bw//2)/(sr/2),0.999)
    b,a=butter(4,[lo,hi],btype='band')
    s=lfilter(b,a,np.random.randn(n)); s/=np.max(np.abs(s))+1e-9
    ramp=min(int(0.01*sr),n//4); env=np.ones(n)
    env[:ramp]=np.linspace(0,1,ramp); env[-ramp:]=np.linspace(1,0,ramp)
    return s*env*amp

def _plosive(dur, sr, amp=0.6):
    n=int(sr*dur)
    if n==0: return np.zeros(1)
    b,a=butter(3,[1500/(sr/2),min(7000/(sr/2),0.99)],btype='band')
    s=lfilter(b,a,np.random.randn(n))*np.exp(-np.linspace(0,12,n))
    s/=np.max(np.abs(s))+1e-9; return s*amp

IY=(270,2290,3010); IH=(390,1990,2550); EH=(530,1840,2480)
AH=(730,1090,2440); AO=(570,840,2410);  OW=(490,740,2400)
UW=(300,870,2240);  ER=(490,1350,1690)
AY1=(660,1720,2410); AY2=(270,2290,2890)
EY1=(460,1990,2850); EY2=(270,2290,2890)
N_=(400,900,2200);   W_=(290,610,2150)

def V(ph,d,sr,f0=115,a=1.0,b=1.0): return _formant_vowel(*ph,d,sr,f0=f0,amp=a,f2b=b)
def F(cf,bw,d,sr,a=0.7):           return _fricative(d,sr,cf,bw,a)
def P(d,sr,a=0.6):                 return _plosive(d,sr,a)
def Sil(s,sr):                     return np.zeros(max(1,int(sr*s)))
def C(*a):                         return np.concatenate(a)

_WORDS=[
    lambda sr: C(F(3200,1200,.05,.55,sr),V(IY,.08,sr,115,b=1.5),V(ER,.05,sr,110),V(OW,.12,sr,100)),
    lambda sr: C(V(W_,.04,sr,115,.5),V(AH,.11,sr,112),V(N_,.08,sr,105,.55)),
    lambda sr: C(P(.025,.5,sr),V(UW,.14,sr,110)),
    lambda sr: C(F(3500,1500,.06,.55,sr),V(ER,.05,sr,115),V(IY,.12,sr,108,b=1.5)),
    lambda sr: C(F(6000,2000,.06,.65,sr),V(AO,.11,sr,112),V(ER,.07,sr,105)),
    lambda sr: C(F(6000,2000,.06,.65,sr),V(AY1,.08,sr,112,b=1.3),V(AY2,.07,sr,108,b=1.5),F(2500,1000,.04,.40,sr)),
    lambda sr: C(F(5500,1500,.06,.70,sr),V(IH,.09,sr,112,b=1.4),F(3000,1500,.03,.45,sr),F(5500,1500,.03,.55,sr)),
    lambda sr: C(F(5500,1500,.06,.70,sr),V(EH,.08,sr,115,b=1.3),F(2500,1000,.03,.40,sr),V(EH,.06,sr,108,b=1.3),V(N_,.07,sr,100,.55)),
    lambda sr: C(V(EY1,.10,sr,115,b=1.4),V(EY2,.07,sr,110,b=1.5),P(.025,.55,sr)),
    lambda sr: C(V(N_,.05,sr,115,.55),V(AY1,.09,sr,112,b=1.3),V(AY2,.07,sr,108,b=1.5),V(N_,.07,sr,100,.55)),
]

_FK={
    'hello':  lambda sr: C(V(EH,.07,sr,115,b=1.3),V(N_,.03,sr,110,.5),V(OW,.10,sr,105)),
    'this':   lambda sr: C(F(3500,1500,.05,.55,sr),V(IH,.07,sr,112,b=1.4),F(5500,1500,.03,.60,sr)),
    'is':     lambda sr: C(V(IH,.06,sr,115,b=1.4),F(5000,1500,.03,.55,sr)),
    'a':      lambda sr: V(AH,.06,sr,112),
    'radio':  lambda sr: C(V(ER,.05,sr,115),V(EY1,.06,sr,112,b=1.3),V(EY2,.04,sr,108,b=1.5),V(OW,.07,sr,105)),
    'check':  lambda sr: C(F(5500,1500,.04,.60,sr),V(EH,.07,sr,112,b=1.3),F(3000,1500,.04,.55,sr)),
    'how':    lambda sr: C(V(AH,.05,sr,115),V(OW,.07,sr,110)),
    'do':     lambda sr: C(P(.02,.5,sr),V(UW,.08,sr,112)),
    'you':    lambda sr: C(V(IY,.04,sr,115,b=1.5),V(UW,.07,sr,110)),
    'copy':   lambda sr: C(F(3000,1500,.04,.50,sr),V(AO,.07,sr,112),P(.02,.5,sr),V(IY,.06,sr,108,b=1.5)),
    'over':   lambda sr: C(V(OW,.07,sr,112),V(ER,.08,sr,105)),
}

def _formant_sentence(text, sr):
    words=text.lower().replace('.','').replace(',','').replace('?','').split()
    parts=[]
    for w in words:
        if w in _FK: parts.append(_FK[w](sr))
        parts.append(Sil(0.06,sr))
    if not parts: return Sil(0.5,sr)
    r=C(*parts); r/=np.max(np.abs(r))+1e-9; return r

def _make_announcement_formant(n, sr):
    s=F(5500,1500,0.12,sr,0.80); d=_WORDS[n](sr)
    w=C(s,Sil(0.07,sr),d); w/=np.max(np.abs(w))+1e-9
    return w.astype(np.float64)

# ── TTS detection ─────────────────────────────────────────────────────────────

def _normalize_audio(data, fs, sr):
    from scipy.signal import resample as rs
    if fs != sr: data = rs(data, int(len(data) * sr / fs))
    if data.dtype == np.int16:   data = data.astype(np.float64) / 32767.0
    elif data.dtype == np.int32: data = data.astype(np.float64) / 2147483647.0
    else:                        data = data.astype(np.float64)
    if data.ndim == 2: data = data.mean(axis=1)
    return data

def _espeak_to_array(text, sr):
    """Render via espeak CLI directly — no pyttsx3 needed."""
    import subprocess, shutil
    from scipy.io import wavfile
    esp = shutil.which("espeak-ng") or shutil.which("espeak")
    if not esp: raise RuntimeError("espeak/espeak-ng not found")
    with tempfile.TemporaryDirectory() as d:
        wp = os.path.join(d, "tts.wav")
        subprocess.run([esp, "-s", "155", "-v", "en-gb", "-w", wp, text],
                       check=True, capture_output=True)
        fs, data = wavfile.read(wp)
    return _normalize_audio(data, fs, sr)

@st.cache_resource
def detect_tts():
    py3_ok=False; py3_msg=""
    gt_ok=False;  gt_msg=""
    # ── espeak / pyttsx3 ──────────────────────────────────────────────
    if sys.platform != 'darwin':
        import shutil
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            try:
                _espeak_to_array("test", 22050)
                py3_ok=True; py3_msg="espeak-ng (direct CLI)"
            except Exception as ex: py3_msg=f"espeak error: {ex}"
        else:
            try:
                import pyttsx3
                e=pyttsx3.init(); vs=e.getProperty('voices')
                mv=next((v for v in vs if 'male' in v.name.lower() and 'female' not in v.name.lower()),next(iter(vs),None))
                py3_ok=True; py3_msg=mv.name if mv else "default"
            except Exception as ex: py3_msg=str(ex)
    else:
        py3_msg="Disabled on macOS"
    # ── gTTS — ffmpeg-free MP3 decode via miniaudio ───────────────────
    try:
        from gtts import gTTS
        import miniaudio  # noqa — pure-Python MP3 decoder, no ffmpeg needed
        try:
            import requests; requests.head('https://www.google.com', timeout=4)
            gt_ok=True; gt_msg="Online + miniaudio (no ffmpeg needed)"
        except: gt_msg="No internet"
    except ImportError as ex: gt_msg=f"Missing: {ex}"
    return py3_ok, gt_ok, py3_msg, gt_msg

def _pyttsx3_proc(text, sr, q):
    try:
        import pyttsx3
        from scipy.io import wavfile
        e=pyttsx3.init(); vs=e.getProperty('voices')
        mv=next((v for v in vs if 'male' in v.name.lower() and 'female' not in v.name.lower()),next(iter(vs),None))
        if mv: e.setProperty('voice',mv.id)
        e.setProperty('rate',165)
        with tempfile.TemporaryDirectory() as d:
            wp=os.path.join(d,"tts.wav"); e.save_to_file(text,wp); e.runAndWait()
            fs,data=wavfile.read(wp)
        q.put(_normalize_audio(data, fs, sr))
    except Exception as ex: q.put(ex)

def _pyttsx3_to_array(text, sr, timeout=10):
    q=mp.Queue(); p=mp.Process(target=_pyttsx3_proc,args=(text,sr,q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(2); raise TimeoutError("pyttsx3 timed out")
    if not q.empty():
        r=q.get_nowait()
        if isinstance(r,Exception): raise r
        return r
    raise RuntimeError("pyttsx3 returned no audio")

def _gtts_to_array(text, sr, timeout=10):
    """Download MP3 from gTTS and decode with miniaudio — pure Python, no ffmpeg required."""
    from gtts import gTTS
    import miniaudio
    with tempfile.TemporaryDirectory() as d:
        mp3 = os.path.join(d, "tts.mp3")
        gTTS(text, lang='en', tld='co.uk', timeout=timeout).save(mp3)
        decoded = miniaudio.mp3_read_file_f32(mp3)
    data = np.array(decoded.samples, dtype=np.float64)
    if decoded.nchannels > 1:
        data = data.reshape(-1, decoded.nchannels).mean(axis=1)
    return _normalize_audio(data, decoded.sample_rate, sr)

def tts_to_array(text, sr, py3_ok, gt_ok):
    if py3_ok:
        # Try espeak CLI first (most reliable on Linux/cloud)
        try: return _espeak_to_array(text, sr)
        except: pass
        # Fallback to pyttsx3
        try: return _pyttsx3_to_array(text, sr)
        except: pass
    if gt_ok:
        try: return _gtts_to_array(text, sr)
        except: pass
    return _formant_sentence(text, sr)

# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(params, cb=None):
    sr=params['sr']; duration=params['duration']; text=params['text']
    band_low=params['band_low']; band_high=params['band_high']
    noise_rms=params['noise_rms']; target_rms=params['target_rms']
    agc_on=params['agc']; levels=params['levels']; seed=params['seed']
    py3_ok=params['py3_ok']; gt_ok=params['gt_ok']

    np.random.seed(seed)
    if cb: cb(0.05,"Synthesising signal voice…")
    raw=tts_to_array(text,sr,py3_ok,gt_ok)
    pk=np.max(np.abs(raw))
    if pk>0: raw/=pk
    # Use the actual TTS duration — never truncate the speech
    silence=np.zeros(int(sr*0.4))
    signal_voice=np.concatenate([raw,silence]).astype(np.float64)
    duration=len(signal_voice)/sr  # override duration to match full speech length

    rf_segs={}; agc_scales={}; table=[]
    spectra={}

    for idx,n in enumerate(levels):
        if cb: cb(0.1+0.6*idx/len(levels), f"Processing S{n}…")
        nsamp=len(signal_voice)  # match noise length to full TTS length
        noise=np.random.normal(0,noise_rms,nsamp)
        noise=apply_bandpass(noise,sr,band_low,band_high)

        if n==0:
            sig_amp=0.0; signal=np.zeros(nsamp); snr_db=-np.inf
        else:
            snr_db=n*6.0; sig_amp=noise_rms*10**(snr_db/20.0)
            signal=signal_voice*sig_amp
            signal=apply_bandpass(signal,sr,band_low,band_high)

        mix=signal+noise
        mix_rms_th=math.sqrt(sig_amp**2+noise_rms**2)
        agc=(target_rms/mix_rms_th) if (agc_on and mix_rms_th>0) else 1.0
        out_noise=noise_rms*agc; out_sig=sig_amp*agc

        table.append({'Level':f'S{n}','SNR (dB)':(f'{snr_db:+.0f}' if not math.isinf(snr_db) else '-∞'),
                      'Pre-AGC Sig':f'{sig_amp:.4f}','AGC Scale':f'{agc:.5f}',
                      'Out Noise':f'{out_noise:.5f}','Out Signal':f'{out_sig:.5f}',
                      '_snr':snr_db if not math.isinf(snr_db) else -60,
                      '_sig':sig_amp,'_agc':agc,'_on':out_noise,'_os':out_sig})

        final=(mix*agc).astype(np.float64)
        rf_segs[n]=final; agc_scales[n]=agc

        ds=max(1,nsamp//4096); seg_ds=final[::ds]; sr_ds=sr//ds
        f_s,pxx=welch(seg_ds,sr_ds,nperseg=min(512,len(seg_ds)))
        spectra[n]=(f_s,10*np.log10(pxx+1e-12))

    if cb: cb(0.75,"Building announcements…")
    audio_parts=[]
    for n in levels:
        ann=_make_announcement_formant(n,sr)
        ann=apply_bandpass(ann,sr,band_low,band_high)
        pk=np.max(np.abs(ann))
        if pk>0: ann/=pk
        ann_samples=len(ann)
        if n==0:
            an=np.random.normal(0,noise_rms,ann_samples)
            an=apply_bandpass(an,sr,band_low,band_high); am=an
        else:
            sa=noise_rms*10**((n*6.0)/20.0)
            an=np.random.normal(0,noise_rms,ann_samples)
            an=apply_bandpass(an,sr,band_low,band_high); am=ann*sa+an
        am*=agc_scales[n]
        audio_parts.append(am.astype(np.float64))
        audio_parts.append(rf_segs[n])

    if cb: cb(0.92,"Encoding WAV…")
    full=np.concatenate(audio_parts)
    full=np.clip(full,-0.98,0.98)
    ai=(full*32767).astype(np.int16)
    buf=io.BytesIO(); wavfile.write(buf,sr,ai); buf.seek(0)
    # Build per-segment WAV bytes for individual playback
    seg_wavs={}
    for n in levels:
        seg=rf_segs[n]
        seg_clipped=np.clip(seg,-0.98,0.98)
        seg_int=(seg_clipped*32767).astype(np.int16)
        sbuf=io.BytesIO(); wavfile.write(sbuf,sr,seg_int); sbuf.seek(0)
        seg_wavs[n]=sbuf.read()

    # ── Digital versions — cliff effect + AMBE codec character ─────────────────
    #
    # Research basis (DMR/P25 AMBE+2 vocoder):
    #   - Codec operates at 2–9.6 kbps, 8 kHz sample rate → 4 kHz audio bandwidth
    #   - 20 ms frames; lost frames concealed with prior frame or muted
    #   - Cliff effect threshold ≈ S4 (~18–20 dB SNR)
    #   - BELOW cliff (S0–S3): link failure — silence, garbled frames, no noise
    #   - ABOVE cliff (S4–S7): digital WINS — clean decode vs noisy analogue FM
    #   - S8–S9: analogue WINS — strong FM has wider dynamic range, warmer timbre,
    #     no codec compression. Digital sounds flat/constrained by comparison.
    #
    if cb: cb(0.96,"Encoding digital variants…")

    voice_clean = signal_voice.copy()
    pk = np.max(np.abs(voice_clean))
    if pk > 0: voice_clean /= pk

    def ambe_codec(sig, sr_in):
        """Simulate AMBE+2 vocoder character: 8 kHz resample → 4 kHz bandwidth cap,
        mild frame-level smoothing (vocoder averaging), slight dynamic compression."""
        from scipy.signal import resample_poly, butter, lfilter
        from math import gcd
        # Resample to 8 kHz (AMBE sample rate) and back — kills everything above 4 kHz
        g = gcd(sr_in, 8000)
        down, up = sr_in // g, 8000 // g
        sig_8k = resample_poly(sig, up, down)
        # Frame-level RMS normalisation (vocoder AGC per 20 ms frame)
        frame_8k = int(8000 * 0.020)
        out_8k = sig_8k.copy()
        for i in range(0, len(sig_8k) - frame_8k, frame_8k):
            chunk = sig_8k[i:i+frame_8k]
            rms = np.sqrt(np.mean(chunk**2)) + 1e-9
            # Vocoder compresses dynamic range — target normalised RMS ~0.35
            out_8k[i:i+frame_8k] = chunk * min(0.35 / rms, 3.0)
        # Mild bandpass matching AMBE: 300–3400 Hz (narrowband telephone quality)
        b, a = butter(4, [300/(4000), 3400/(4000)], btype='band')
        out_8k = lfilter(b, a, out_8k)
        # Resample back to original sr
        sig_out = resample_poly(out_8k, down, up)
        # Match length
        if len(sig_out) > len(sig): sig_out = sig_out[:len(sig)]
        elif len(sig_out) < len(sig): sig_out = np.pad(sig_out, (0, len(sig)-len(sig_out)))
        pk2 = np.max(np.abs(sig_out))
        if pk2 > 0: sig_out /= pk2
        return sig_out

    def codec_degraded_strong(sig, sr_in):
        """S8–S9: AMBE codec applied to a strong signal — analogue FM wins here.
        The codec's dynamic compression flattens the natural warmth and transients
        that wideband FM preserves. More constrained, slightly artificial timbre."""
        from scipy.signal import resample_poly, butter, lfilter
        from math import gcd
        # Same codec path but also apply slight pre-emphasis reduction
        # (analogue FM uses pre-emphasis 75µs which gives natural treble lift;
        # digital codec strips this and applies its own EQ — sounds less natural)
        g = gcd(sr_in, 8000)
        down, up = sr_in // g, 8000 // g
        sig_8k = resample_poly(sig, up, down)
        # Harder dynamic compression at strong levels — codec is more aggressive
        frame_8k = int(8000 * 0.020)
        out_8k = sig_8k.copy()
        for i in range(0, len(sig_8k) - frame_8k, frame_8k):
            chunk = sig_8k[i:i+frame_8k]
            rms = np.sqrt(np.mean(chunk**2)) + 1e-9
            # Tighter target — less dynamic range preserved
            out_8k[i:i+frame_8k] = chunk * min(0.28 / rms, 2.5)
        # Narrower bandwidth — codec strips low frequencies and some highs
        b, a = butter(4, [350/(4000), 3000/(4000)], btype='band')
        out_8k = lfilter(b, a, out_8k)
        sig_out = resample_poly(out_8k, down, up)
        if len(sig_out) > len(sig): sig_out = sig_out[:len(sig)]
        elif len(sig_out) < len(sig): sig_out = np.pad(sig_out, (0, len(sig)-len(sig_out)))
        pk2 = np.max(np.abs(sig_out))
        if pk2 > 0: sig_out /= pk2
        return sig_out

    dig_wavs = {}
    FRAME = int(sr * 0.020)   # 20 ms AMBE frame

    for n in levels:
        nsamp = len(voice_clean)

        # ── S0: silence — nothing transmitted ──────────────────────────────────
        if n == 0:
            seg_out = np.zeros(nsamp)

        # ── S1–S2: below cliff — link failed ────────────────────────────────────
        # Decoder receives bad IMBE frames: mostly silence, rare garbled bursts
        elif n <= 2:
            seg_out = np.zeros(nsamp)
            ber = {1: 0.88, 2: 0.65}[n]
            nframes = nsamp // FRAME
            for f in range(nframes):
                if np.random.rand() > ber:
                    garble = np.random.choice([-1.0, 0.0, 1.0], size=FRAME)
                    seg_out[f*FRAME:(f+1)*FRAME] = garble * 0.25

        # ── S3: cliff edge — marginal, choppy ───────────────────────────────────
        elif n == 3:
            seg_out = np.zeros(nsamp)
            ber = 0.38
            nframes = nsamp // FRAME
            for f in range(nframes):
                s = f * FRAME; e = min(s + FRAME, nsamp)
                if np.random.rand() > ber:
                    seg_out[s:e] = voice_clean[s:e]  # clean frame decode

        # ── S4: just above cliff — stable link, rare dropout ────────────────────
        elif n == 4:
            coded = ambe_codec(voice_clean, sr)
            seg_out = coded.copy()
            nframes = nsamp // FRAME
            for f in range(nframes):
                if np.random.rand() < 0.04:
                    s = f * FRAME; e = min(s + FRAME, nsamp)
                    seg_out[s:e] = 0.0

        # ── S5–S7: digital clearly wins — clean codec, analogue is noisy ────────
        # AMBE codec gives clear, noise-free decode. Analogue FM is still hissy.
        elif n <= 7:
            seg_out = ambe_codec(voice_clean, sr)

        # ── S8–S9: analogue wins — strong FM has more warmth and dynamic range ──
        # Digital codec compresses dynamics and narrows bandwidth. At this SNR
        # the analogue signal is virtually noise-free AND more natural-sounding.
        else:
            seg_out = codec_degraded_strong(voice_clean, sr)

        seg_out = np.clip(seg_out, -0.98, 0.98)
        seg_int = (seg_out * 32767).astype(np.int16)
        dbuf = io.BytesIO(); wavfile.write(dbuf, sr, seg_int); dbuf.seek(0)
        dig_wavs[n] = dbuf.read()

    if cb: cb(1.0,"Done")
    return {'wav_bytes':buf.read(),'full_audio':full,'sr':sr,
            'rf_segs':rf_segs,'spectra':spectra,'table':table,'levels':levels,
            'seg_wavs':seg_wavs,'dig_wavs':dig_wavs}

# ── Plot helpers ──────────────────────────────────────────────────────────────

PL=dict(paper_bgcolor='#0a0e14',plot_bgcolor='#0d1117',
        font=dict(family='Share Tech Mono, monospace',color='#8bafc7',size=11),
        margin=dict(l=50,r=20,t=40,b=40),
        xaxis=dict(gridcolor='#1e2d3d',linecolor='#1e2d3d',zerolinecolor='#1e2d3d'),
        yaxis=dict(gridcolor='#1e2d3d',linecolor='#1e2d3d',zerolinecolor='#1e2d3d'))
# PL without margin — use when caller needs to specify a custom margin
PL_NOM={k:v for k,v in PL.items() if k!='margin'}

SC={0:'#546e7a',1:'#26c6da',2:'#29b6f6',3:'#42a5f5',4:'#5c6bc0',
    5:'#7e57c2',6:'#ab47bc',7:'#ec407a',8:'#ef5350',9:'#ff7043'}

def hex_rgba(h, a):
    h=h.lstrip('#'); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f'rgba({r},{g},{b},{a})'

def chart_snr(rows):
    lbls=[r['Level'] for r in rows]; snrs=[r['_snr'] for r in rows]
    cols=[SC.get(int(r['Level'][1]),'#546e7a') for r in rows]
    fig=go.Figure(go.Bar(x=lbls,y=snrs,marker_color=cols,marker_line_width=0,
        text=[r['SNR (dB)']+' dB' for r in rows],textposition='outside',
        textfont=dict(family='Share Tech Mono',size=10,color='#8bafc7')))
    fig.update_layout(**PL,title=dict(text='S-LEVEL ▸ SNR LADDER',font=dict(size=12,color='#4fc3f7')),
        xaxis_title='S-Level',yaxis_title='SNR (dB)',height=300)
    return fig

def chart_agc(rows):
    lbls=[r['Level'] for r in rows]
    noise=[r['_on'] for r in rows]; sig=[r['_os'] for r in rows]
    cols=[SC.get(int(r['Level'][1]),'#546e7a') for r in rows]
    fig=go.Figure()
    fig.add_trace(go.Bar(name='Out Noise',x=lbls,y=noise,marker_color='#1e2d3d',marker_line_width=0))
    fig.add_trace(go.Bar(name='Out Signal',x=lbls,y=sig,marker_color=cols,marker_line_width=0))
    fig.update_layout(**PL,barmode='stack',
        title=dict(text='AGC OUTPUT — NOISE vs SIGNAL',font=dict(size=12,color='#4fc3f7')),
        xaxis_title='S-Level',yaxis_title='RMS',height=300,
        legend=dict(bgcolor='#0a0e14',bordercolor='#1e2d3d',borderwidth=1,font=dict(size=10)))
    return fig

def chart_flow(rows):
    lbls=[r['Level'] for r in rows]
    sig=[r['_sig'] for r in rows]; agc=[r['_agc'] for r in rows]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=lbls,y=sig,name='Pre-AGC Signal',mode='lines+markers',
        line=dict(color='#ff7043',width=2),marker=dict(size=6,color='#ff7043')))
    fig.add_trace(go.Scatter(x=lbls,y=agc,name='AGC Scale',mode='lines+markers',
        line=dict(color='#29b6f6',width=2,dash='dot'),marker=dict(size=6,color='#29b6f6')))
    yax={**PL['yaxis'],'type':'log'}
    fig.update_layout(**{**PL,'yaxis':yax},
        title=dict(text='SIGNAL AMPLITUDE vs AGC SCALE',font=dict(size=12,color='#4fc3f7')),
        xaxis_title='S-Level',yaxis_title='Value (log)',height=300,
        legend=dict(bgcolor='#0a0e14',bordercolor='#1e2d3d',borderwidth=1,font=dict(size=10)))
    return fig

def chart_spectrum_all(spectra, levels, bl, bh):
    fig=go.Figure()
    for n in levels:
        f,pxx=spectra[n]; mask=(f>=max(10,bl*0.5))&(f<=bh*2.2)
        fig.add_trace(go.Scatter(x=f[mask],y=pxx[mask],mode='lines',name=f'S{n}',
            line=dict(color=SC.get(n,'#546e7a'),width=1.5)))
    fig.add_vrect(x0=bl,x1=bh,fillcolor='rgba(79,195,247,0.05)',
        line_color='rgba(79,195,247,0.3)',line_width=1,
        annotation_text=f'PASSBAND<br>{bl}–{bh} Hz',
        annotation_font=dict(size=9,color='#4fc3f7'))
    fig.update_layout(**PL,
        title=dict(text='POWER SPECTRAL DENSITY — ALL LEVELS',font=dict(size=12,color='#4fc3f7')),
        xaxis_title='Frequency (Hz)',yaxis_title='Power (dBW/Hz)',height=360,
        legend=dict(bgcolor='#0a0e14',bordercolor='#1e2d3d',borderwidth=1,font=dict(size=10)))
    return fig

def chart_spectrum_single(spectra, n, bl, bh):
    f,pxx=spectra[n]; mask=(f>=max(10,bl*0.5))&(f<=bh*2.2)
    col=SC.get(n,'#546e7a')
    fig=go.Figure(go.Scatter(x=f[mask],y=pxx[mask],mode='lines',fill='tozeroy',
        fillcolor=hex_rgba(col,0.12),line=dict(color=col,width=2),name=f'S{n}'))
    fig.add_vrect(x0=bl,x1=bh,fillcolor='rgba(79,195,247,0.05)',
        line_color='rgba(79,195,247,0.25)',line_width=1)
    fig.update_layout(**PL_NOM,title=dict(text=f'S{n} — SPECTRUM',font=dict(size=11,color='#4fc3f7')),
        xaxis_title='Hz',yaxis_title='dB',height=220,margin=dict(l=40,r=10,t=30,b=30))
    return fig

def chart_waveform(seg, sr, n, mx=3000):
    ds=max(1,len(seg)//mx); y=seg[::ds]; t=np.arange(len(y))*ds/sr
    col=SC.get(n,'#546e7a')
    fig=go.Figure(go.Scatter(x=t,y=y,mode='lines',line=dict(color=col,width=1),name='Waveform'))
    yax={**PL['yaxis'],'range':[-1.05,1.05]}
    fig.update_layout(**{**PL_NOM,'yaxis':yax},
        title=dict(text=f'S{n} — WAVEFORM',font=dict(size=11,color='#4fc3f7')),
        xaxis_title='Time (s)',yaxis_title='Amp',height=200,margin=dict(l=40,r=10,t=30,b=30))
    return fig

def chart_gauge(n):
    fig=go.Figure(go.Indicator(
        mode="gauge+number",value=n,
        number=dict(suffix=f"  (S{n})",font=dict(family='Share Tech Mono',size=26,color='#e0f7fa')),
        gauge=dict(axis=dict(range=[0,9],tickwidth=1,tickcolor='#4fc3f7',
                             tickfont=dict(family='Share Tech Mono',size=9),nticks=10),
                   bar=dict(color=SC.get(n,'#546e7a'),thickness=0.25),
                   bgcolor='#0d1117',borderwidth=1,bordercolor='#1e2d3d',
                   steps=[dict(range=[0,1],color='#0a1a1a'),dict(range=[1,3],color='#0a1520'),
                          dict(range=[3,6],color='#0e1525'),dict(range=[6,9],color='#12102a')],
                   threshold=dict(line=dict(color='#ef5350',width=2),thickness=0.8,value=n)),
        title=dict(text='S-METER',font=dict(family='Share Tech Mono',size=11,color='#4fc3f7'))))
    fig.update_layout(paper_bgcolor='#0a0e14',font=dict(family='Share Tech Mono',color='#8bafc7'),
                      height=220,margin=dict(l=20,r=20,t=30,b=20))
    return fig

def chart_rms_timeline(rf_segs, levels, sr):
    """RMS energy over time for each S-level."""
    fig=go.Figure()
    for n in levels:
        seg=rf_segs[n]; wsize=int(sr*0.1)
        if len(seg)<wsize: continue
        steps=len(seg)//wsize
        t=[]; rms=[]
        for i in range(steps):
            chunk=seg[i*wsize:(i+1)*wsize]
            t.append(i*0.1); rms.append(float(np.sqrt(np.mean(chunk**2))))
        fig.add_trace(go.Scatter(x=t,y=rms,mode='lines',name=f'S{n}',
            line=dict(color=SC.get(n,'#546e7a'),width=1.5)))
    fig.update_layout(**PL,
        title=dict(text='RMS ENERGY TIMELINE (100ms WINDOWS)',font=dict(size=12,color='#4fc3f7')),
        xaxis_title='Time (s)',yaxis_title='RMS',height=280,
        legend=dict(bgcolor='#0a0e14',bordercolor='#1e2d3d',borderwidth=1,font=dict(size=10)))
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="smh">📡 RF S-METER SIMULATOR</div>', unsafe_allow_html=True)

    py3_ok,gt_ok,py3_msg,gt_msg=detect_tts()
    p_b="badge-ok" if py3_ok else "badge-warn"
    g_b="badge-ok" if gt_ok  else "badge-warn"
    st.markdown(f'<div style="font-size:0.72rem;line-height:2.2">'
                f'<span class="{p_b}">pyttsx3</span> {"Active" if py3_ok else "Unavailable"}<br>'
                f'<span class="{g_b}">gTTS</span> {"Active" if gt_ok else "Unavailable"}<br>'
                f'<span class="badge-ok">formant</span> Fallback</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("🎙 SIGNAL TEXT", expanded=False):
        signal_text=st.text_area("Signal text","Hello, this is a radio check. How do you copy? Over.",
                                  height=75,label_visibility="collapsed")

    with st.expander("📻 S-LEVELS", expanded=False):
        all_levels=st.checkbox("Include all S0–S9",value=True)
        if not all_levels:
            levels_sel=sorted(st.multiselect("Select levels",list(range(10)),default=list(range(10)),
                                              format_func=lambda x:f"S{x}") or [0])
        else:
            levels_sel=list(range(10))

    with st.expander("⚙ AUDIO", expanded=False):
        sr=st.select_slider("Sample Rate",options=[8000,16000,22050,44100,48000],value=44100)
        duration=st.slider("Segment Duration (s)",1.0,10.0,3.0,0.5)
        seed=st.number_input("Random Seed",value=42,min_value=0,max_value=9999,step=1)

    with st.expander("📊 RF PHYSICS", expanded=False):
        noise_rms=st.slider("Noise Floor RMS",0.1,2.0,1.0,0.05)
        target_rms=st.slider("AGC Target RMS",0.1,1.0,0.7,0.05)
        agc_on=st.checkbox("AGC Enabled",value=True)

    with st.expander("🎚 BANDPASS FILTER", expanded=False):
        band_low=st.slider("Low Cutoff (Hz)",50,1000,300,25)
        band_high=st.slider("High Cutoff (Hz)",1000,8000,3000,100)
        if band_low>=band_high:
            st.error("Low cutoff must be < high cutoff"); band_low=band_high-100

    st.markdown("---")
    run_btn=st.button("▶  GENERATE SIMULATION",use_container_width=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────

st.markdown("# 📡 RF S-Meter Simulator")
st.markdown("*IARU R.1 — S0 noise floor · S1–S9 at +6 dB steps · AGC normalisation · Formant TTS*")
st.markdown("---")

if 'result' not in st.session_state:
    st.session_state.result=None

if run_btn:
    if band_low>=band_high:
        st.error("Bandpass: Low cutoff must be less than High cutoff.")
    elif not levels_sel:
        st.error("Select at least one S-level.")
    else:
        params=dict(sr=sr,duration=duration,text=signal_text,band_low=band_low,band_high=band_high,
                    noise_rms=noise_rms,target_rms=target_rms,agc=agc_on,levels=levels_sel,
                    seed=int(seed),py3_ok=py3_ok,gt_ok=gt_ok)
        pb=st.progress(0); st_txt=st.empty()
        def cb(f,m):
            pb.progress(f)
            st_txt.markdown(f'<span style="font-family:Share Tech Mono;color:#4fc3f7;font-size:0.78rem">► {m}</span>',unsafe_allow_html=True)
        with st.spinner(""):
            result=simulate(params,cb=cb)
        pb.empty(); st_txt.empty()
        st.session_state.result=result
        st.success("✔ Simulation complete — download WAV below or explore charts.")

res=st.session_state.result

if res is None:
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:2.6rem;letter-spacing:0.2em;color:#1e2d3d;">▶ AWAITING TRANSMISSION</div>
      <div style="margin-top:0.8rem;font-size:0.8rem;letter-spacing:0.1em;color:#1a3040;">Configure parameters in the sidebar · Press GENERATE SIMULATION</div>
    </div>""",unsafe_allow_html=True)

    # Theory preview
    st.markdown("## Signal Theory Preview")
    prev=[]
    for n in range(10):
        sd=n*6.0 if n>0 else -np.inf; sa=1.0*10**(sd/20) if n>0 else 0.0
        ag=0.7/math.sqrt(sa**2+1.0)
        prev.append({'Level':f'S{n}','SNR (dB)':(f'{n*6:+d}' if n>0 else '-∞'),
                     'Pre-AGC Sig':f'{sa:.4f}','AGC Scale':f'{ag:.5f}',
                     'Out Noise':f'{1.0*ag:.5f}','Out Signal':f'{sa*ag:.5f}',
                     '_snr':n*6 if n>0 else -60,'_sig':sa,'_agc':ag,'_on':1.0*ag,'_os':sa*ag})
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(chart_snr(prev),width='stretch',key='pc1')
    with c2: st.plotly_chart(chart_flow(prev),width='stretch',key='pc2')
else:
    levels=res['levels']
    dur_s=len(res['full_audio'])/res['sr']

    # Metrics
    mc=st.columns(5)
    mc[0].metric("Sample Rate",f"{res['sr']} Hz")
    mc[1].metric("Levels",str(len(levels)))
    mc[2].metric("Duration",f"{dur_s:.1f} s")
    mc[3].metric("AGC","ON" if agc_on else "OFF")
    mc[4].metric("TTS","pyttsx3" if py3_ok else ("gTTS" if gt_ok else "Formant"))

    # ── Master playback + download bar ───────────────────────────────────────
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;letter-spacing:0.12em;
    text-transform:uppercase;color:#4fc3f7;margin-bottom:0.4rem;">
    📻 FULL TRANSMISSION PLAYBACK — S0 → S9 SEQUENCE
    </div>""", unsafe_allow_html=True)
    st.audio(res['wav_bytes'], format='audio/wav')
    col_dl,_=st.columns([1,3])
    with col_dl:
        st.download_button("⬇  DOWNLOAD WAV",data=res['wav_bytes'],
                           file_name="rf_smeter.wav",mime="audio/wav",key="dl_wav_top")
    st.markdown("---")

    # Tabs
    t1,t2,t3,t4,t5,t6=st.tabs(["OVERVIEW","SPECTRUM","WAVEFORMS","LEVEL INSPECTOR","DIGITAL MODES","DATA TABLE"])

    with t1:
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(chart_snr(res['table']),width='stretch',key='pc3')
        with c2: st.plotly_chart(chart_agc(res['table']),width='stretch',key='pc4')
        c3,c4=st.columns(2)
        with c3: st.plotly_chart(chart_flow(res['table']),width='stretch',key='pc5')
        with c4: st.plotly_chart(chart_rms_timeline(res['rf_segs'],levels,res['sr']),width='stretch',key='pc6')

        # Quick-listen: per-level audio players in a compact grid
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
        letter-spacing:0.12em;text-transform:uppercase;color:#4fc3f7;
        border-top:1px solid #1e2d3d;padding-top:0.8rem;margin-top:0.4rem;">
        🔊 QUICK-LISTEN — PER-LEVEL SEGMENTS</div>""", unsafe_allow_html=True)
        cols_per_row = 5
        level_rows = [levels[i:i+cols_per_row] for i in range(0,len(levels),cols_per_row)]
        for row in level_rows:
            pcols = st.columns(cols_per_row)
            for ci, n in enumerate(row):
                with pcols[ci]:
                    col_hex = SC.get(n,'#546e7a')
                    st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;'
                                f'font-size:0.7rem;color:{col_hex};text-align:center;'
                                f'margin-bottom:0.2rem;">S{n}'
                                f'{"  ·  "+res["table"][levels.index(n)]["SNR (dB)"]+" dB" if n>0 else "  ·  noise"}'
                                f'</div>', unsafe_allow_html=True)
                    st.audio(res['seg_wavs'][n], format='audio/wav')

    with t2:
        st.plotly_chart(chart_spectrum_all(res['spectra'],levels,band_low,band_high),width='stretch',key='pc7')
        st.markdown("### Per-Level Spectrum + Audio")
        cpr=2
        for i in range(0,len(levels),cpr):
            row=levels[i:i+cpr]; cols=st.columns(cpr)
            for ci,n in enumerate(row):
                with cols[ci]:
                    st.plotly_chart(chart_spectrum_single(res['spectra'],n,band_low,band_high),width='stretch',key=f'spec_single_{n}')
                    col_hex=SC.get(n,'#546e7a')
                    st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
                                f'color:{col_hex};margin-bottom:0.15rem;">▶ S{n} PLAYBACK</div>',
                                unsafe_allow_html=True)
                    st.audio(res['seg_wavs'][n], format='audio/wav')

    with t3:
        st.markdown("### Waveforms + Audio")
        cpr=2
        for i in range(0,len(levels),cpr):
            row=levels[i:i+cpr]; cols=st.columns(cpr)
            for ci,n in enumerate(row):
                with cols[ci]:
                    st.plotly_chart(chart_waveform(res['rf_segs'][n],res['sr'],n),width='stretch',key=f'waveform_{n}')
                    col_hex=SC.get(n,'#546e7a')
                    st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
                                f'color:{col_hex};margin-bottom:0.15rem;">▶ S{n} PLAYBACK</div>',
                                unsafe_allow_html=True)
                    st.audio(res['seg_wavs'][n], format='audio/wav')

    with t4:
        import streamlit.components.v1 as components
        import base64, json as _json

        # Build per-level data payload for the JS meter
        _level_data = []
        for _n in res['levels']:
            _row = next(r for r in res['table'] if r['Level']==f'S{_n}')
            _wav_b64 = base64.b64encode(res['seg_wavs'][_n]).decode()
            _level_data.append({
                'level': _n,
                'snr':   _row['SNR (dB)'],
                'sig':   _row['Pre-AGC Sig'],
                'agc':   _row['AGC Scale'],
                'noise': _row['Out Noise'],
                'outsig':_row['Out Signal'],
                'wav':   _wav_b64,
                'color': SC.get(_n, '#546e7a'),
            })

        _ld_json = _json.dumps(_level_data)

        _smeter_html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0a0e14;color:#c8d6e5;font-family:'Share Tech Mono',monospace;padding:10px;}
.container{display:grid;grid-template-columns:370px 1fr;gap:14px;}
.left-panel,.right-panel{display:flex;flex-direction:column;gap:10px;}
.card{background:#0d1117;border:1px solid #1e2d3d;border-radius:4px;padding:10px;}
.card-title{font-size:0.6rem;letter-spacing:0.14em;text-transform:uppercase;color:#4fc3f7;margin-bottom:8px;}
#smeter-canvas{width:100%;height:170px;display:block;border-radius:3px;}
.digital-display{display:grid;grid-template-columns:1fr 1fr;gap:5px;}
.dig-cell{background:#060a0f;border:1px solid #1e2d3d;border-radius:3px;padding:5px 8px;}
.dig-label{font-size:0.52rem;letter-spacing:0.1em;color:#4fc3f7;text-transform:uppercase;margin-bottom:1px;}
.dig-value{font-size:0.95rem;color:#e0f7fa;}
.level-grid{display:grid;grid-template-columns:repeat(10,1fr);gap:3px;}
.lvl-btn{background:#0d1117;border:1px solid #1e2d3d;border-radius:3px;color:#546e7a;
  font-family:'Share Tech Mono',monospace;font-size:0.68rem;padding:4px 0;text-align:center;cursor:pointer;transition:all .15s;}
.lvl-btn:hover{border-color:#4fc3f7;color:#4fc3f7;}
.lvl-btn.active{color:#0a0e14;font-weight:bold;}
.controls{display:flex;gap:7px;align-items:center;flex-wrap:wrap;}
.ctrl-btn{background:linear-gradient(135deg,#0d47a1,#01579b);border:1px solid #1565c0;border-radius:3px;
  color:#e3f2fd;font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:.08em;
  text-transform:uppercase;padding:4px 12px;cursor:pointer;transition:all .15s;}
.ctrl-btn:hover{border-color:#4fc3f7;}
.ctrl-btn.active{background:linear-gradient(135deg,#004d40,#00695c);border-color:#26c6da;}
.ctrl-label{font-size:0.6rem;color:#546e7a;}
.ctrl-range{width:90px;accent-color:#4fc3f7;}
#osc-canvas{width:100%;height:200px;display:block;border-radius:3px;}
#bar-canvas{width:100%;height:80px;display:block;border-radius:3px;}
audio{width:100%;height:30px;filter:invert(0.85) hue-rotate(185deg);}
.status-strip{display:flex;gap:10px;align-items:center;font-size:0.6rem;color:#546e7a;margin-top:7px;flex-wrap:wrap;}
.status-dot{width:6px;height:6px;border-radius:50%;background:#546e7a;display:inline-block;margin-right:3px;}
.status-dot.live{background:#26c6da;box-shadow:0 0 5px #26c6da;animation:pulse 1s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.peak-hold{font-size:0.56rem;color:#ff7043;}
.power-dial-wrap{display:flex;flex-direction:row;align-items:center;gap:10px;}
#dial-canvas{cursor:grab;display:block;flex-shrink:0;}
#dial-canvas:active{cursor:grabbing;}
.dial-readout{font-size:0.68rem;color:#e0f7fa;letter-spacing:0.08em;}
.dial-sub{font-size:0.52rem;color:#546e7a;letter-spacing:0.06em;}
.dial-row{display:flex;align-items:center;gap:10px;flex-wrap:nowrap;}
.dial-info{display:flex;flex-direction:column;gap:2px;}
.dial-labels{display:grid;grid-template-columns:repeat(2,1fr);gap:1px 8px;font-size:0.50rem;color:#546e7a;letter-spacing:0.05em;}
</style></head><body>
<div class="container">
  <div class="left-panel">
    <div class="card">
      <div class="card-title">📻 ANALOG S-METER</div>
      <canvas id="smeter-canvas" width="720" height="340"></canvas>
    </div>
    <div class="card">
      <div class="card-title">DIGITAL READOUT</div>
      <div class="digital-display">
        <div class="dig-cell"><div class="dig-label">S-Level</div><div class="dig-value" id="d-level">—</div></div>
        <div class="dig-cell"><div class="dig-label">SNR</div><div class="dig-value" id="d-snr">—</div></div>
        <div class="dig-cell"><div class="dig-label">Signal</div><div class="dig-value" id="d-sig">—</div></div>
        <div class="dig-cell"><div class="dig-label">AGC Scale</div><div class="dig-value" id="d-agc">—</div></div>
        <div class="dig-cell"><div class="dig-label">Out Noise</div><div class="dig-value" id="d-noise">—</div></div>
        <div class="dig-cell"><div class="dig-label">Out Signal</div><div class="dig-value" id="d-outsig">—</div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">SELECT LEVEL</div>
      <div class="level-grid" id="level-grid"></div>
    </div>
    <div class="card">
      <div class="card-title">📡 TRANSMISSION POWER</div>
      <div class="power-dial-wrap">
        <canvas id="dial-canvas" width="90" height="90"></canvas>
        <div class="dial-info">
          <div class="dial-readout" id="dial-readout">100%</div>
          <div class="dial-sub" id="dial-sub-lbl">MAX S9 · +54 dBm</div>
          <div class="dial-labels" style="margin-top:4px;">
            <span style="color:#ff7043">● S9·100%</span>
            <span style="color:#ef5350">● S8· 88%</span>
            <span style="color:#ec407a">● S7· 77%</span>
            <span style="color:#ab47bc">● S6· 66%</span>
            <span style="color:#7e57c2">● S5· 55%</span>
            <span style="color:#5c6bc0">● S4· 44%</span>
            <span style="color:#42a5f5">● S3· 33%</span>
            <span style="color:#29b6f6">● S2· 22%</span>
            <span style="color:#26c6da">● S1· 11%</span>
            <span style="color:#546e7a">● S0·  0%</span>
          </div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">CONTROLS</div>
      <div class="controls">
        <button class="ctrl-btn active" id="btn-sweep" onclick="toggleSweep()">⟳ AUTO SWEEP</button>
        <button class="ctrl-btn" id="btn-noise" onclick="toggleNoise()">⊼ NOISE SIM</button>
        <span class="ctrl-label">Speed</span>
        <input type="range" class="ctrl-range" id="sweep-speed" min="300" max="3000" value="900" step="100">
      </div>
      <div class="status-strip">
        <span><span class="status-dot live" id="status-dot"></span>LIVE</span>
        <span id="status-txt">Sweeping S0→S9</span>
        <span class="peak-hold" id="peak-hold-txt"></span>
      </div>
    </div>
  </div>
  <div class="right-panel">
    <div class="card">
      <div class="card-title">OSCILLOSCOPE — SCROLLING WAVEFORM</div>
      <canvas id="osc-canvas" width="1100" height="200"></canvas>
    </div>
    <div class="card">
      <div class="card-title">RMS POWER BARGRAPH</div>
      <canvas id="bar-canvas" width="1100" height="160"></canvas>
    </div>
    <div class="card">
      <div class="card-title" id="audio-title">🔊 SEGMENT AUDIO</div>
      <audio id="audio-player" controls></audio>
    </div>
  </div>
</div>
<script>
const LEVELS="""  + _ld_json + """;
let currentLevel=LEVELS[LEVELS.length-1].level;
let displayValue=currentLevel,targetValue=currentLevel;
let peakValue=0,peakDecay=0,sweepActive=true,noiseActive=false;
let sweepDir=1,sweepIdx=LEVELS.length-1,sweepTimer=null;
let noisePhase=0,rmsIdx=0,oscBuffer=[],barHistory=[];
const BAR_N=50;
for(let i=0;i<BAR_N;i++)barHistory.push(0);

// ── Power dial state ──
let powerPct=100;          // 0-100
let dialDragging=false;
let dialLastY=0;
const SC_COLORS={0:'#546e7a',1:'#26c6da',2:'#29b6f6',3:'#42a5f5',4:'#5c6bc0',
                 5:'#7e57c2',6:'#ab47bc',7:'#ec407a',8:'#ef5350',9:'#ff7043'};

function powerToMaxLevel(pct){
  // 0% = S0, 100% = S9, linear mapping
  return Math.round((pct/100)*9);
}

function clampLevelToPower(n){
  return Math.min(n, powerToMaxLevel(powerPct));
}

// ── Dial drawing ──
const dc=document.getElementById('dial-canvas');
const dx=dc.getContext('2d');
const DW=dc.width,DH=dc.height;
const DCX=DW/2,DCY=DH/2,DR=DW*.41;
// Arc: 210° to 330° (7π/6 to 11π/6)
const D_START=Math.PI*(7/6), D_END=Math.PI*(11/6), D_RANGE=D_END-D_START;

function pctToDialAngle(pct){return D_START+(pct/100)*D_RANGE;}

function getDialColor(pct){
  const lvl=powerToMaxLevel(pct);
  return SC_COLORS[lvl]||'#546e7a';
}

function drawDial(pct){
  dx.clearRect(0,0,DW,DH);
  const col=getDialColor(pct);
  const ang=pctToDialAngle(pct);

  // Outer bezel
  dx.beginPath(); dx.arc(DCX,DCY,DR*1.12,0,Math.PI*2);
  const bez=dx.createRadialGradient(DCX,DCY-DR*.2,DR*.1,DCX,DCY,DR*1.2);
  bez.addColorStop(0,'#2a3a4a'); bez.addColorStop(1,'#0d1117');
  dx.fillStyle=bez; dx.fill();
  dx.strokeStyle='#1e2d3d'; dx.lineWidth=1.5; dx.stroke();

  // Track (background arc)
  dx.beginPath(); dx.arc(DCX,DCY,DR,D_START,D_END);
  dx.strokeStyle='#1e2d3d'; dx.lineWidth=DR*.22; dx.lineCap='round'; dx.stroke();

  // Filled arc (power level)
  if(pct>0){
    const fg=dx.createLinearGradient(DCX-DR,DCY,DCX+DR,DCY);
    fg.addColorStop(0,'#26c6da80'); fg.addColorStop(0.5,'#7e57c270'); fg.addColorStop(1,col+'cc');
    dx.beginPath(); dx.arc(DCX,DCY,DR,D_START,ang);
    dx.strokeStyle=fg; dx.lineWidth=DR*.22; dx.lineCap='round'; dx.stroke();
    // Glow on tip
    dx.beginPath(); dx.arc(DCX,DCY,DR,Math.max(D_START,ang-0.08),ang);
    dx.strokeStyle=col; dx.lineWidth=DR*.08; dx.shadowColor=col; dx.shadowBlur=12;
    dx.stroke(); dx.shadowBlur=0;
  }

  // Scale ticks
  for(let s=0;s<=9;s++){
    const ta=D_START+(s/9)*D_RANGE;
    const tc=SC_COLORS[s];
    const r1=DR*1.05, r2=DR*.82;
    dx.beginPath();
    dx.moveTo(DCX+r1*Math.cos(ta),DCY+r1*Math.sin(ta));
    dx.lineTo(DCX+r2*Math.cos(ta),DCY+r2*Math.sin(ta));
    dx.strokeStyle=tc; dx.lineWidth=s===0||s===9?2:1.2;
    dx.lineCap='square'; dx.shadowBlur=0; dx.stroke();
  }

  // Knob body
  const kg=dx.createRadialGradient(DCX-DR*.15,DCY-DR*.15,DR*.05,DCX,DCY,DR*.58);
  kg.addColorStop(0,'#3a4a5a'); kg.addColorStop(1,'#0d1117');
  dx.beginPath(); dx.arc(DCX,DCY,DR*.56,0,Math.PI*2);
  dx.fillStyle=kg; dx.fill();
  dx.strokeStyle='#2a3a4a'; dx.lineWidth=1.5; dx.stroke();

  // Pointer line on knob
  dx.beginPath();
  dx.moveTo(DCX+DR*.18*Math.cos(ang),DCY+DR*.18*Math.sin(ang));
  dx.lineTo(DCX+DR*.50*Math.cos(ang),DCY+DR*.50*Math.sin(ang));
  dx.strokeStyle=col; dx.lineWidth=3; dx.lineCap='round';
  dx.shadowColor=col; dx.shadowBlur=8; dx.stroke(); dx.shadowBlur=0;

  // Pointer dot
  dx.beginPath(); dx.arc(DCX+DR*.50*Math.cos(ang),DCY+DR*.50*Math.sin(ang),3,0,Math.PI*2);
  dx.fillStyle=col; dx.shadowColor=col; dx.shadowBlur=6; dx.fill(); dx.shadowBlur=0;

  // Max-S badge in centre
  const maxL=powerToMaxLevel(pct);
  dx.font='bold '+Math.round(DR*.42)+'px Share Tech Mono,monospace';
  dx.fillStyle=col; dx.textAlign='center'; dx.textBaseline='middle';
  dx.fillText('S'+maxL,DCX,DCY-DR*.04);
  dx.font=Math.round(DR*.2)+'px Share Tech Mono,monospace';
  dx.fillStyle='#546e7a';
  dx.fillText('MAX',DCX,DCY+DR*.3);

  // Update text readout
  document.getElementById('dial-readout').textContent=pct+'%';
  document.getElementById('dial-readout').style.color=col;
  const dbLabels=['-∞','+6','+12','+18','+24','+30','+36','+42','+48','+54'];
  document.getElementById('dial-sub-lbl').textContent=
    'MAX S'+maxL+' · '+(maxL===0?'S0 NOISE':dbLabels[maxL]+' dBm');
}

// ── Dial interaction (drag up/down) ──
function dialPctFromDrag(dy){
  return Math.max(0,Math.min(100,powerPct-dy*.4));
}

dc.addEventListener('mousedown',e=>{dialDragging=true;dialLastY=e.clientY;e.preventDefault();});
dc.addEventListener('touchstart',e=>{dialDragging=true;dialLastY=e.touches[0].clientY;e.preventDefault();});
window.addEventListener('mousemove',e=>{
  if(!dialDragging)return;
  const dy=e.clientY-dialLastY; dialLastY=e.clientY;
  powerPct=Math.max(0,Math.min(100,powerPct-dy*.6));
  powerPct=Math.round(powerPct);
  onPowerChange();
});
window.addEventListener('touchmove',e=>{
  if(!dialDragging)return;
  const dy=e.touches[0].clientY-dialLastY; dialLastY=e.touches[0].clientY;
  powerPct=Math.max(0,Math.min(100,powerPct-dy*.6));
  powerPct=Math.round(powerPct);
  onPowerChange();
},{passive:false});
window.addEventListener('mouseup',()=>{dialDragging=false;});
window.addEventListener('touchend',()=>{dialDragging=false;});
dc.addEventListener('wheel',e=>{
  e.preventDefault();
  powerPct=Math.max(0,Math.min(100,powerPct-(e.deltaY>0?-1:1)*5));
  powerPct=Math.round(powerPct);
  onPowerChange();
},{passive:false});

function onPowerChange(){
  drawDial(powerPct);
  const maxL=powerToMaxLevel(powerPct);
  // If current level exceeds new power cap, drop to max allowed
  if(currentLevel>maxL) selectLevel(maxL);
  // Cap sweep range
  sweepIdx=Math.min(sweepIdx,LEVELS.filter(d=>d.level<=maxL).length-1);
}

function buildGrid(){
  const g=document.getElementById('level-grid');
  LEVELS.forEach(d=>{
    const b=document.createElement('button');
    b.className='lvl-btn'+(d.level===currentLevel?' active':'');
    b.id='lvl-'+d.level; b.textContent='S'+d.level;
    b.style.borderColor=d.color+'60';
    b.onclick=()=>selectLevel(d.level,true);
    g.appendChild(b);
  });
}

function selectLevel(n,manual=false){
  if(manual){sweepActive=false;document.getElementById('btn-sweep').classList.remove('active');}
  n=clampLevelToPower(n);
  currentLevel=n; rmsIdx=0; oscPlayhead=0;
  const d=LEVELS.find(x=>x.level===n);
  targetValue=n+(noiseActive?(Math.random()-.5)*.5:0);
  LEVELS.forEach(x=>{
    const b=document.getElementById('lvl-'+x.level);
    if(!b)return;
    b.classList.toggle('active',x.level===n);
    b.style.background=x.level===n?d.color:'#0d1117';
    b.style.color=x.level===n?'#0a0e14':x.color;
    b.style.borderColor=x.level===n?d.color:x.color+'60';
  });
  document.getElementById('d-level').textContent='S'+n;
  document.getElementById('d-level').style.color=d.color;
  document.getElementById('d-snr').textContent=d.snr+' dB';
  document.getElementById('d-sig').textContent=d.sig;
  document.getElementById('d-agc').textContent=d.agc;
  document.getElementById('d-noise').textContent=d.noise;
  document.getElementById('d-outsig').textContent=d.outsig;
  const p=document.getElementById('audio-player');
  p.src='data:audio/wav;base64,'+d.wav;
  document.getElementById('audio-title').textContent='🔊 S'+n+' SEGMENT AUDIO';
  document.getElementById('status-txt').textContent='Level S'+n+' — SNR '+d.snr+' dB';
}

function toggleSweep(){
  sweepActive=!sweepActive;
  document.getElementById('btn-sweep').classList.toggle('active',sweepActive);
  document.getElementById('status-txt').textContent=sweepActive?'Sweeping S0→S9':'Manual mode';
  if(sweepActive)rescheduleSweep();
}
function toggleNoise(){
  noiseActive=!noiseActive;
  document.getElementById('btn-noise').classList.toggle('active',noiseActive);
}
function doSweepStep(){
  if(!sweepActive)return;
  const maxL=powerToMaxLevel(powerPct);
  const lvls=LEVELS.filter(d=>d.level<=maxL).map(d=>d.level);
  if(lvls.length===0)return;
  sweepIdx=(sweepIdx+sweepDir+lvls.length)%lvls.length;
  if(sweepIdx>=lvls.length-1){sweepIdx=lvls.length-1;sweepDir=-1;}
  if(sweepIdx<=0){sweepIdx=0;sweepDir=1;}
  selectLevel(lvls[sweepIdx]);
}
function rescheduleSweep(){
  clearTimeout(sweepTimer);
  if(!sweepActive)return;
  const sp=parseInt(document.getElementById('sweep-speed').value);
  sweepTimer=setTimeout(()=>{doSweepStep();rescheduleSweep();},sp);
}
document.getElementById('sweep-speed').addEventListener('input',rescheduleSweep);
rescheduleSweep();

// ── Analog meter ──
const mc=document.getElementById('smeter-canvas');
const mx=mc.getContext('2d');
const W=mc.width,H=mc.height;
const CX=W/2,CY=H*.87,R=H*.80;
const A_S=Math.PI+Math.PI*.2,A_E=Math.PI*2-Math.PI*.2;
const A_R=A_E-A_S;
function l2a(v){return A_S+(Math.max(0,Math.min(v,9))/9)*A_R;}

function drawMeter(val,peak,col){
  mx.clearRect(0,0,W,H);
  // Background glow
  const bg=mx.createRadialGradient(CX,CY,R*.2,CX,CY,R*.95);
  bg.addColorStop(0,'#0d1117'); bg.addColorStop(1,'#060a0f');
  mx.fillStyle=bg; mx.fillRect(0,0,W,H);
  // Arc track
  mx.beginPath(); mx.arc(CX,CY,R*.9,A_S,A_E);
  mx.strokeStyle='#1e2d3d'; mx.lineWidth=R*.12; mx.stroke();
  // Coloured zone segments
  const zones=[[0,3,'#26c6da'],[3,6,'#7e57c2'],[6,9,'#ef5350']];
  zones.forEach(([a,b,zc])=>{
    mx.beginPath(); mx.arc(CX,CY,R*.9,l2a(a),l2a(b));
    mx.strokeStyle=zc+'22'; mx.lineWidth=R*.12; mx.stroke();
  });
  // Filled arc
  if(val>0){
    const ag=mx.createLinearGradient(CX-R,CY,CX+R,CY);
    ag.addColorStop(0,'#26c6da90');ag.addColorStop(.5,'#7e57c270');ag.addColorStop(1,'#ef535090');
    mx.beginPath(); mx.arc(CX,CY,R*.9,A_S,l2a(val));
    mx.strokeStyle=ag; mx.lineWidth=R*.06; mx.stroke();
    // Glow
    mx.shadowColor=col; mx.shadowBlur=18;
    mx.beginPath(); mx.arc(CX,CY,R*.9,A_S,l2a(val));
    mx.strokeStyle=col+'80'; mx.lineWidth=R*.025; mx.stroke();
    mx.shadowBlur=0;
  }
  // Peak tick
  if(peak>0.05){
    const pa=l2a(Math.min(peak,9));
    mx.beginPath();
    mx.moveTo(CX+R*.80*Math.cos(pa),CY+R*.80*Math.sin(pa));
    mx.lineTo(CX+R*.96*Math.cos(pa),CY+R*.96*Math.sin(pa));
    mx.strokeStyle='#ff7043'; mx.lineWidth=3; mx.stroke();
    document.getElementById('peak-hold-txt').textContent='PEAK: S'+peak.toFixed(1);
  }
  // Scale ticks
  for(let s=0;s<=9;s++){
    const a=l2a(s),cs=Math.cos(a),sn=Math.sin(a);
    const tc=s<3?'#26c6da':s<6?'#7e57c2':'#ef5350';
    mx.beginPath();
    mx.moveTo(CX+R*.82*cs,CY+R*.82*sn);
    mx.lineTo(CX+R*.65*cs,CY+R*.65*sn);
    mx.strokeStyle=tc; mx.lineWidth=2.5; mx.stroke();
    if(s<9){
      const a2=l2a(s+.5);
      mx.beginPath();
      mx.moveTo(CX+R*.82*Math.cos(a2),CY+R*.82*Math.sin(a2));
      mx.lineTo(CX+R*.75*Math.cos(a2),CY+R*.75*Math.sin(a2));
      mx.strokeStyle=tc+'80'; mx.lineWidth=1; mx.stroke();
    }
    // Label
    mx.font='bold '+Math.round(R*.078)+'px Share Tech Mono,monospace';
    mx.fillStyle=tc; mx.textAlign='center'; mx.textBaseline='middle';
    mx.fillText('S'+s,CX+R*.55*cs,CY+R*.55*sn);
    // dB sub-label
    const dbs=['-∞','+6','+12','+18','+24','+30','+36','+42','+48','+54'];
    mx.font=Math.round(R*.052)+'px Share Tech Mono,monospace';
    mx.fillStyle='#546e7a';
    mx.fillText(dbs[s],CX+R*.43*cs,CY+R*.43*sn);
  }
  // Needle shadow
  const na=l2a(Math.max(0,Math.min(val,9)));
  mx.beginPath(); mx.moveTo(CX+3,CY+3);
  mx.lineTo(CX+(R*.82)*Math.cos(na)+3,CY+(R*.82)*Math.sin(na)+3);
  mx.strokeStyle='rgba(0,0,0,.5)'; mx.lineWidth=5; mx.lineCap='round'; mx.stroke();
  // Needle
  const ng=mx.createLinearGradient(CX,CY,CX+(R*.82)*Math.cos(na),CY+(R*.82)*Math.sin(na));
  ng.addColorStop(0,'#8bafc7'); ng.addColorStop(1,col);
  mx.beginPath(); mx.moveTo(CX,CY); mx.lineTo(CX+(R*.82)*Math.cos(na),CY+(R*.82)*Math.sin(na));
  mx.strokeStyle=ng; mx.lineWidth=3.5; mx.lineCap='round';
  mx.shadowColor=col; mx.shadowBlur=10; mx.stroke(); mx.shadowBlur=0;
  // Counter-weight nub
  const bk=l2a(Math.max(0,Math.min(val,9))+Math.PI);
  mx.beginPath(); mx.moveTo(CX,CY);
  mx.lineTo(CX+R*.06*Math.cos(na+Math.PI),CY+R*.06*Math.sin(na+Math.PI));
  mx.strokeStyle='#546e7a'; mx.lineWidth=4; mx.stroke();
  // Pivot
  mx.beginPath(); mx.arc(CX,CY,R*.045,0,Math.PI*2);
  mx.fillStyle='#c8d6e5'; mx.fill();
  mx.beginPath(); mx.arc(CX,CY,R*.02,0,Math.PI*2);
  mx.fillStyle='#060a0f'; mx.fill();
  // Centre level badge
  const lv=Math.round(Math.max(0,Math.min(val,9)));
  mx.font='bold '+Math.round(R*.17)+'px Share Tech Mono,monospace';
  mx.fillStyle=col+'dd'; mx.textAlign='center'; mx.textBaseline='middle';
  mx.fillText('S'+lv,CX,CY-R*.24);
  const d=LEVELS.find(x=>x.level===lv)||LEVELS[0];
  mx.font=Math.round(R*.07)+'px Share Tech Mono,monospace';
  mx.fillStyle='#546e7a';
  mx.fillText(d.snr+' dB',CX,CY-R*.10);
}

// ── Web Audio API — live analyser hooked to the audio element ──
const oc=document.getElementById('osc-canvas');
const ox=oc.getContext('2d');
const OW=oc.width,OH=oc.height;
const bc=document.getElementById('bar-canvas');
const bx=bc.getContext('2d');
const BW=bc.width,BH=bc.height;

let audioCtx=null, analyserOsc=null, analyserBar=null, oscBuf=null, barBuf=null;

// Track elements already connected — createMediaElementSource can only be called once per element
const _connectedEls=new WeakMap();
function initAudio(){
  const ap=document.getElementById('audio-player');
  if(!audioCtx) audioCtx=new (window.AudioContext||window.webkitAudioContext)();
  if(audioCtx.state==='suspended') audioCtx.resume();
  if(_connectedEls.has(ap)) return; // already wired up — analyser still live
  const src=audioCtx.createMediaElementSource(ap);
  analyserOsc=audioCtx.createAnalyser();
  analyserOsc.fftSize=2048;
  analyserOsc.smoothingTimeConstant=0.0;
  analyserBar=audioCtx.createAnalyser();
  analyserBar.fftSize=256;
  analyserBar.smoothingTimeConstant=0.8;
  src.connect(analyserOsc);
  analyserOsc.connect(analyserBar);
  analyserBar.connect(audioCtx.destination);
  oscBuf=new Float32Array(analyserOsc.fftSize);
  barBuf=new Uint8Array(analyserBar.frequencyBinCount);
  _connectedEls.set(ap, true);
}

document.getElementById('audio-player').addEventListener('play', ()=>{
  initAudio();
  if(audioCtx.state==='suspended') audioCtx.resume();
});

function drawOsc(col){
  ox.clearRect(0,0,OW,OH);
  // Grid
  ox.strokeStyle='#1e2d3d'; ox.lineWidth=1;
  for(let i=1;i<4;i++){const y=OH*i/4;ox.beginPath();ox.moveTo(0,y);ox.lineTo(OW,y);ox.stroke();}
  for(let i=1;i<8;i++){const x=OW*i/8;ox.beginPath();ox.moveTo(x,0);ox.lineTo(x,OH);ox.stroke();}
  ox.strokeStyle='#2a3a4a';
  ox.beginPath();ox.moveTo(0,OH/2);ox.lineTo(OW,OH/2);ox.stroke();
  const ap=document.getElementById('audio-player');
  if(!analyserOsc||ap.paused){
    // Flat line when not playing
    ox.strokeStyle='#1e2d3d'; ox.lineWidth=1.5;
    ox.beginPath();ox.moveTo(0,OH/2);ox.lineTo(OW,OH/2);ox.stroke();
    return;
  }
  analyserOsc.getFloatTimeDomainData(oscBuf);
  const step=OW/(oscBuf.length-1);
  ox.beginPath();
  for(let i=0;i<oscBuf.length;i++){
    const sv=Math.max(-1,Math.min(1,oscBuf[i]));
    const x=i*step;
    const y=OH/2 - sv*(OH*.46);
    if(i===0)ox.moveTo(x,y);else ox.lineTo(x,y);
  }
  const gr=ox.createLinearGradient(0,0,OW,0);
  gr.addColorStop(0,'#1e2d3d'); gr.addColorStop(0.3,col+'99'); gr.addColorStop(1,col);
  ox.strokeStyle=gr; ox.lineWidth=1.8;
  ox.shadowColor=col; ox.shadowBlur=7; ox.stroke(); ox.shadowBlur=0;
}

function drawBar(col){
  bx.clearRect(0,0,BW,BH);
  bx.strokeStyle='#1e2d3d'; bx.lineWidth=1;
  [.25,.5,.75].forEach(frac=>{
    bx.beginPath();bx.moveTo(0,BH*(1-frac));bx.lineTo(BW,BH*(1-frac));bx.stroke();
  });
  const ap=document.getElementById('audio-player');
  if(!analyserBar||ap.paused){
    bx.font='10px Share Tech Mono,monospace'; bx.fillStyle='#546e7a'; bx.textAlign='left';
    bx.fillText('100%',4,12); bx.fillText('50%',4,BH/2+4); bx.fillText('0%',4,BH-2);
    return;
  }
  analyserBar.getByteFrequencyData(barBuf);
  const N=barBuf.length;
  const bw=BW/N-1;
  let peak=0;
  for(let i=0;i<N;i++){
    const v=barBuf[i]/255;
    if(v>peak)peak=v;
    const bh=Math.max(2,v*(BH*.92));
    const x=i*(BW/N)+.5;
    const y=BH-bh;
    const hue=v<.33?'#26c6da':v<.66?'#7e57c2':'#ef5350';
    const bg=bx.createLinearGradient(x,y,x,BH);
    bg.addColorStop(0,hue+'ff'); bg.addColorStop(1,hue+'44');
    bx.fillStyle=bg;
    bx.fillRect(x,y,Math.max(bw,1),bh);
  }
  const pH=peak*(BH*.92);
  bx.strokeStyle=col; bx.lineWidth=2;
  bx.beginPath();bx.moveTo(0,BH-pH);bx.lineTo(BW,BH-pH);bx.stroke();
  bx.font='10px Share Tech Mono,monospace'; bx.fillStyle='#546e7a'; bx.textAlign='left';
  bx.fillText('100%',4,12); bx.fillText('50%',4,BH/2+4); bx.fillText('0%',4,BH-2);
}

// ── Animation loop ──
const ATTACK=.18, DECAY=.07;
function tick(){
  const noise=noiseActive?(Math.random()-.5)*.5:0;
  const target=targetValue+noise;
  if(displayValue<target) displayValue+=(target-displayValue)*ATTACK;
  else displayValue+=(target-displayValue)*DECAY;
  if(displayValue>peakValue){peakValue=displayValue;peakDecay=0;}
  else{peakDecay++;if(peakDecay>80)peakValue=Math.max(0,peakValue-.03);}
  const d=LEVELS.find(x=>x.level===currentLevel)||LEVELS[0];
  drawMeter(displayValue,peakValue,d.color);
  drawOsc(d.color);
  drawBar(d.color);
  requestAnimationFrame(tick);
}

buildGrid();
drawDial(powerPct);
selectLevel(currentLevel);
requestAnimationFrame(tick);
</script></body></html>"""

        components.html(_smeter_html, height=640, scrolling=False)

    with t5:
        import streamlit.components.v1 as _comp2
        import base64 as _b64, json as _json2
        _status_map={
            0:'NO LINK · SILENCE',
            1:'LINK FAILURE · GARBLED',
            2:'LINK FAILURE · GARBLED',
            3:'CLIFF EDGE · MARGINAL',
            4:'ABOVE CLIFF · STABLE',
            5:'DIGITAL WINS · CLEAN',
            6:'DIGITAL WINS · CLEAN',
            7:'DIGITAL WINS · CLEAN',
            8:'ANALOGUE WINS · CODEC COMPRESSED',
            9:'ANALOGUE WINS · CODEC COMPRESSED',
        }
        _winner_map={
            0:'—', 1:'—', 2:'—', 3:'—',
            4:'DIGITAL', 5:'DIGITAL', 6:'DIGITAL', 7:'DIGITAL',
            8:'ANALOGUE', 9:'ANALOGUE',
        }
        _dig_data=[]
        for _n in res['levels']:
            _row=next(r for r in res['table'] if r['Level']==f'S{_n}')
            _dig_data.append({
                'level':_n,
                'snr':_row['SNR (dB)'],
                'status':_status_map.get(_n,'CLEAN DECODE'),
                'winner':_winner_map.get(_n,'—'),
                'stable': _n >= 4,
                'wav':_b64.b64encode(res['dig_wavs'][_n]).decode(),
                'wav_ana':_b64.b64encode(res['seg_wavs'][_n]).decode(),
                'color':SC.get(_n,'#546e7a'),
            })
        import json as _j2
        _dig_json=_j2.dumps(_dig_data)

        _dig_css = '''
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0a0e14;color:#c8d6e5;font-family:'Share Tech Mono',monospace;padding:12px;}
.card{background:#0d1117;border:1px solid #1e2d3d;border-radius:4px;padding:10px;margin-bottom:12px;}
.card-title{font-size:0.58rem;letter-spacing:0.14em;text-transform:uppercase;color:#4fc3f7;margin-bottom:8px;}
.vis-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;}
canvas{display:block;width:100%;border-radius:3px;}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;}
.slot{background:#060a0f;border:1px solid #1e2d3d;border-radius:4px;padding:7px 6px;cursor:pointer;transition:all .15s;}
.slot.active{border-color:var(--col);box-shadow:0 0 8px var(--col)44;}
.slot-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;}
.slot-lvl{font-size:0.72rem;font-weight:bold;}
.slot-meta{font-size:0.50rem;color:#546e7a;line-height:1.6;}
.slot-badges{display:flex;gap:3px;margin-top:4px;flex-wrap:wrap;}
.badge{font-size:0.46rem;padding:1px 5px;border-radius:2px;letter-spacing:0.06em;text-transform:uppercase;}
.badge-dig{background:#0a1a2a;border:1px solid #1565c0;color:#4fc3f7;}
.badge-ana{background:#1a0a1a;border:1px solid #7b1fa2;color:#ce93d8;}
audio{width:100%;height:24px;margin-top:4px;filter:invert(0.85) hue-rotate(185deg);}
.mode-row{display:flex;gap:8px;margin-bottom:10px;align-items:center;}
.mode-btn{background:#0d1117;border:1px solid #1e2d3d;border-radius:3px;color:#546e7a;
  font-family:'Share Tech Mono',monospace;font-size:0.6rem;letter-spacing:.08em;
  text-transform:uppercase;padding:4px 10px;cursor:pointer;transition:all .15s;}
.mode-btn:hover{border-color:#4fc3f7;color:#4fc3f7;}
.mode-btn.active{background:linear-gradient(135deg,#0d47a1,#01579b);border-color:#4fc3f7;color:#e3f2fd;}
.info-strip{font-size:0.54rem;color:#546e7a;margin-top:5px;letter-spacing:0.05em;}
'''

        _dig_js = '''
const LEVELS=__DIG_JSON__;
let audioCtx=null,analyserOsc=null,analyserBar=null,oscBuf=null,barBuf=null,activeAudio=null,activeLevel=null;
let currentMode='dig';
const oc=document.getElementById('osc-c'),ox=oc.getContext('2d'),OW=oc.width,OH=oc.height;
const bc=document.getElementById('bar-c'),bx=bc.getContext('2d'),BW=bc.width,BH=bc.height;

// WeakMap tracks elements already passed to createMediaElementSource (one-time-only per element)
const _srcMap=new WeakMap();

function initAudio(el){
  if(!audioCtx) audioCtx=new (window.AudioContext||window.webkitAudioContext)();
  if(audioCtx.state==='suspended') audioCtx.resume();
  // Always create fresh analysers so switching between elements works correctly
  analyserOsc=audioCtx.createAnalyser(); analyserOsc.fftSize=2048; analyserOsc.smoothingTimeConstant=0.0;
  analyserBar=audioCtx.createAnalyser(); analyserBar.fftSize=512;  analyserBar.smoothingTimeConstant=0.75;
  oscBuf=new Float32Array(analyserOsc.fftSize);
  barBuf=new Uint8Array(analyserBar.frequencyBinCount);
  // Re-use existing source node if element was already connected, else create one
  let src=_srcMap.get(el);
  if(!src){
    src=audioCtx.createMediaElementSource(el);
    _srcMap.set(el,src);
  }
  src.connect(analyserOsc); analyserOsc.connect(analyserBar); analyserBar.connect(audioCtx.destination);
}

function setMode(m){
  currentMode=m;
  ['dig','ana','both'].forEach(x=>document.getElementById('btn-'+x).classList.toggle('active',x===m));
  buildGrid();
}

function attachAudio(el,d,t){
  el.addEventListener('play',()=>{
    if(activeAudio&&activeAudio!==el){activeAudio.pause();}
    activeAudio=el; activeLevel=d;
    try{initAudio(el);}catch(e){}
    document.querySelectorAll('.slot').forEach(x=>x.classList.remove('active'));
    document.getElementById('slot-'+d.level).classList.add('active');
    const lbl=t==='dig'?'DIGITAL':'ANALOGUE';
    const snrTxt=d.level===0?'NOISE FLOOR':'SNR '+d.snr+' dB';
    document.getElementById('osc-title').textContent='OSCILLOSCOPE — S'+d.level+' '+lbl+' · '+d.status;
    document.getElementById('bar-title').textContent='FREQUENCY POWER — S'+d.level+' '+lbl;
    document.getElementById('osc-info').textContent='S'+d.level+' | '+snrTxt+' | '+d.status+(t==='dig'?' | NO RF NOISE IN DIGITAL DECODE':'');
    document.getElementById('bar-info').textContent='Live FFT | 256 bins | S'+d.level+' '+lbl;
  });
  el.addEventListener('pause',()=>{activeAudio=null;});
  el.addEventListener('ended',()=>{activeAudio=null;activeLevel=null;document.getElementById('slot-'+d.level).classList.remove('active');});
}

function buildGrid(){
  const g=document.getElementById('slot-grid'); g.innerHTML='';
  LEVELS.forEach(d=>{
    const s=document.createElement('div');
    s.className='slot'; s.id='slot-'+d.level;
    s.style.setProperty('--col',d.color);
    const snrTxt=d.level===0?'NOISE FLOOR':'SNR '+d.snr+' dB';
    const showDig=currentMode!=='ana', showAna=currentMode!=='dig';
    s.innerHTML=
      '<div class="slot-hdr">'+
        '<span class="slot-lvl" style="color:'+d.color+'">S'+d.level+'</span>'+
        '<span style="font-size:0.46rem;color:'+(d.winner==='ANALOGUE'?'#ce93d8':d.winner==='DIGITAL'?'#26c6da':'#ef5350')+';">'
          +(d.winner!=='—'?'▶ '+d.winner:d.stable?'STABLE':'FAIL')+'</span>'+
      '</div>'+
      '<div class="slot-meta" style="font-size:0.48rem;">'+snrTxt+'<br><span style="color:#546e7a80;">'+d.status+'</span></div>'+
      '<div class="slot-badges">'+
        (showDig?'<span class="badge badge-dig">DIGITAL</span>':'')+
        (showAna?'<span class="badge badge-ana">ANALOGUE</span>':'')+
      '</div>';
    if(showDig){
      const a=document.createElement('audio');
      a.id='aud-dig-'+d.level; a.src='data:audio/wav;base64,'+d.wav;
      a.controls=true; a.preload='none'; a.style.cssText='width:100%;height:24px;margin-top:4px;filter:invert(0.85) hue-rotate(185deg);';
      attachAudio(a,d,'dig'); s.appendChild(a);
    }
    if(showAna){
      const a=document.createElement('audio');
      a.id='aud-ana-'+d.level; a.src='data:audio/wav;base64,'+d.wav_ana;
      a.controls=true; a.preload='none'; a.style.cssText='width:100%;height:24px;margin-top:4px;filter:invert(0.85) hue-rotate(185deg);';
      attachAudio(a,d,'ana'); s.appendChild(a);
    }
    g.appendChild(s);
  });
}

function drawGrid(){
  const drawG=(ctx,w,h)=>{
    ctx.strokeStyle='#1e2d3d'; ctx.lineWidth=1;
    for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(0,h*i/4);ctx.lineTo(w,h*i/4);ctx.stroke();}
    for(let i=1;i<8;i++){ctx.beginPath();ctx.moveTo(w*i/8,0);ctx.lineTo(w*i/8,h);ctx.stroke();}
    ctx.strokeStyle='#2a3a4a'; ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();
  };
  drawG(ox,OW,OH); drawG(bx,BW,BH);
}

function drawOsc(){
  ox.clearRect(0,0,OW,OH); drawGrid();
  if(!analyserOsc||!activeAudio||activeAudio.paused){
    ox.strokeStyle='#1e2d3d60'; ox.lineWidth=1;
    ox.beginPath();ox.moveTo(0,OH/2);ox.lineTo(OW,OH/2);ox.stroke(); return;
  }
  analyserOsc.getFloatTimeDomainData(oscBuf);
  let start=0;
  for(let i=1;i<oscBuf.length-1;i++){if(oscBuf[i-1]<0&&oscBuf[i]>=0){start=i;break;}}
  const visN=Math.min(2048,oscBuf.length-start);
  ox.beginPath();
  for(let i=0;i<visN;i++){
    const sv=Math.max(-1,Math.min(1,oscBuf[start+i]));
    const x=i*(OW/(visN-1)), y=OH/2-sv*(OH*.46);
    if(i===0)ox.moveTo(x,y);else ox.lineTo(x,y);
  }
  const col=activeLevel?activeLevel.color:'#4fc3f7';
  const gr=ox.createLinearGradient(0,0,OW,0);
  gr.addColorStop(0,'#1e2d3d'); gr.addColorStop(0.3,col+'99'); gr.addColorStop(1,col);
  ox.strokeStyle=gr; ox.lineWidth=1.8; ox.shadowColor=col; ox.shadowBlur=6; ox.stroke(); ox.shadowBlur=0;
}

function drawBar(){
  bx.clearRect(0,0,BW,BH);
  bx.strokeStyle='#1e2d3d'; bx.lineWidth=1;
  [.25,.5,.75].forEach(f=>{bx.beginPath();bx.moveTo(0,BH*(1-f));bx.lineTo(BW,BH*(1-f));bx.stroke();});
  if(!analyserBar||!activeAudio||activeAudio.paused){
    bx.font='9px Share Tech Mono,monospace'; bx.fillStyle='#546e7a'; bx.textAlign='left';
    bx.fillText('100%',3,11); bx.fillText('50%',3,BH/2+4); bx.fillText('0%',3,BH-3); return;
  }
  analyserBar.getByteFrequencyData(barBuf);
  const N=barBuf.length, bw=BW/N-0.5;
  const col=activeLevel?activeLevel.color:'#4fc3f7';
  let peak=0;
  for(let i=0;i<N;i++){
    const v=barBuf[i]/255; if(v>peak)peak=v;
    const bh=Math.max(2,v*(BH*.92)), x=i*(BW/N)+.5, y=BH-bh;
    const hue=v<.33?'#26c6da':v<.66?col:'#ef5350';
    const bg=bx.createLinearGradient(x,y,x,BH);
    bg.addColorStop(0,hue+'ff'); bg.addColorStop(1,hue+'33');
    bx.fillStyle=bg; bx.fillRect(x,y,Math.max(bw,1),bh);
  }
  const pH=peak*(BH*.92);
  bx.strokeStyle=col; bx.lineWidth=2;
  bx.beginPath();bx.moveTo(0,BH-pH);bx.lineTo(BW,BH-pH);bx.stroke();
  bx.font='9px Share Tech Mono,monospace'; bx.fillStyle='#546e7a'; bx.textAlign='left';
  bx.fillText('100%',3,11); bx.fillText('50%',3,BH/2+4); bx.fillText('0%',3,BH-3);
}

function tick(){drawOsc();drawBar();requestAnimationFrame(tick);}
buildGrid();
requestAnimationFrame(tick);
'''

        _dig_js_final = _dig_js.replace('__DIG_JSON__', _dig_json)

        _dig_html = (
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            '<style>' + _dig_css + '</style></head><body>'
            '<div class="vis-row">'
            '  <div class="card">'
            '    <div class="card-title" id="osc-title">OSCILLOSCOPE — PLAY A LEVEL TO ACTIVATE</div>'
            '    <canvas id="osc-c" width="800" height="160"></canvas>'
            '    <div class="info-strip" id="osc-info">—</div>'
            '  </div>'
            '  <div class="card">'
            '    <div class="card-title" id="bar-title">FREQUENCY POWER — PLAY A LEVEL TO ACTIVATE</div>'
            '    <canvas id="bar-c" width="800" height="160"></canvas>'
            '    <div class="info-strip" id="bar-info">—</div>'
            '  </div>'
            '</div>'
            '<div class="card">'
            '  <div class="card-title">S-LEVEL DIGITAL TRANSMISSIONS</div>'
            '  <div class="mode-row">'
            '    <span style="font-size:0.58rem;color:#546e7a;">DISPLAY MODE:</span>'
            '    <button class="mode-btn active" id="btn-dig" onclick="setMode(&apos;dig&apos;)">DIGITAL</button>'
            '    <button class="mode-btn" id="btn-ana" onclick="setMode(&apos;ana&apos;)">ANALOGUE</button>'
            '    <button class="mode-btn" id="btn-both" onclick="setMode(&apos;both&apos;)">BOTH</button>'
            '  </div>'
            '  <div class="grid" id="slot-grid"></div>'
            '</div>'
            '<script>' + _dig_js_final + '</script>'
            '</body></html>'
        )

        _comp2.html(_dig_html, height=780, scrolling=False)

    with t6:
        df=pd.DataFrame([{k:v for k,v in r.items() if not k.startswith('_')} for r in res['table']])
        st.dataframe(df,use_container_width=True,hide_index=True)  # noqa
        dl1,dl2,_=st.columns([1,1,2])
        with dl1:
            st.download_button("⬇  EXPORT CSV",df.to_csv(index=False),"rf_smeter_data.csv","text/csv",key="dl_csv")
        with dl2:
            st.download_button("⬇  DOWNLOAD WAV",data=res['wav_bytes'],
                               file_name="rf_smeter.wav",mime="audio/wav",key="dl_wav_table")
