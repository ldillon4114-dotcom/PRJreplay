import subprocess, sys, os, json
from faster_whisper import WhisperModel
model=WhisperModel("small.en",device="cpu",compute_type="int8",cpu_threads=4)
def extract(name,src,cs,dur):
    wav=f"t_{name}.wav"
    subprocess.run(['/workspace/ffmpeg','-hide_banner','-loglevel','error','-ss',str(cs),'-t',str(dur),'-i',src,'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le','-y',wav],check=True,capture_output=True)
    return wav
def lines_of(wav):
    segs,_=model.transcribe(wav,language='en',vad_filter=True,vad_parameters={'min_silence_duration_ms':350},word_timestamps=True)
    words=[]
    for s in segs:
        for w in s.words: words.append((w.start,w.end,w.word))
    lines=[];cur=None
    for a,b,t in words:
        gap=None if not cur else a-cur[1]
        if cur and (gap<0.45 and cur[1]-cur[0]<6.5 and not cur[2].rstrip().endswith(('?','!')) or (gap is not None and gap<0.25 and cur[1]-cur[0]<9)):
            cur[1]=b;cur[2]+=' '+t
        else:
            if cur:lines.append(cur)
            cur=[round(a,2),round(b,2),t]
    if cur:lines.append(cur)
    return [l for l in lines if l[2].strip()]
for spec in json.loads(sys.argv[1]):
    name,src,cs,dur=spec
    try:
        ls=lines_of(extract(name,src,cs,dur))
        print(f"===== {name} clip@{cs} len={dur} =====")
        for a,b,t in ls: print(f"{a:7.2f}-{b:7.2f} {t}")
    except Exception as e: print(name,'ERR',e)
