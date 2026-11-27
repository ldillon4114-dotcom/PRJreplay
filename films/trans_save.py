import subprocess, sys, os, json
from faster_whisper import WhisperModel
model=WhisperModel("small.en",device="cpu",compute_type="int8",cpu_threads=4)
def run(name,src,cs,dur):
    wav=f"f_{name}.wav"
    subprocess.run(['/workspace/ffmpeg','-hide_banner','-loglevel','error','-ss',str(cs),'-t',str(dur),'-i',src,'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le','-y',wav],check=True,capture_output=True)
    segs,_=model.transcribe(wav,language='en',vad_filter=True,vad_parameters={'min_silence_duration_ms':300},word_timestamps=True)
    words=[]
    for s in segs:
        for w in s.words: words.append((w.start,w.end,w.word))
    lines=[];cur=None
    for a,b,t in words:
        gap=None if not cur else a-cur[1]
        dur_l=None if not cur else cur[1]-cur[0]
        if cur and ((gap<0.30 and dur_l<4.5) or (gap<0.15 and dur_l<6.5)):
            cur[1]=b;cur[2]+=' '+t
        else:
            if cur:lines.append(cur)
            cur=[round(a,2),round(b,2),t.strip()]
    if cur:lines.append(cur)
    lines=[l for l in lines if l[2].strip()]
    json.dump(lines,open(f"f_{name}.json","w"))
    print(f"===== {name} clip@{cs} =====")
    for a,b,t in lines: print(f"{a:6.2f}-{b:6.2f} {t}")
for spec in json.loads(sys.argv[1]): run(*spec)
