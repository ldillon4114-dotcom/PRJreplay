import subprocess, sys, os, json
from faster_whisper import WhisperModel
model=WhisperModel("small.en",device="cpu",compute_type="int8",cpu_threads=4)
def probe(name,src,start,dur=300):
    wav=f"p_{name}.wav"
    if src.endswith('.mp4') and os.path.exists(src): pre=[]
    else: pre=['-ss',str(start),'-t',str(dur)]
    cmd=['/workspace/ffmpeg','-hide_banner','-loglevel','error']+pre+(['-i',src] if pre else ['-ss',str(start),'-t',str(dur),'-i',src])+['-vn','-ac','1','-ar','16000','-c:a','pcm_s16le','-y',wav]
    subprocess.run(cmd,check=True,capture_output=True)
    segs,_=model.transcribe(wav,language='en',vad_filter=True,vad_parameters={'min_silence_duration_ms':400})
    lines=[];cur=None
    for s in segs:
        if cur and s.start-cur[1]<0.9 and s.end-cur[1]<10: cur[1]=s.end;cur[2]+=' '+s.text.strip()
        else:
            if cur:lines.append(cur)
            cur=[round(s.start,2),round(s.end,2),s.text.strip()]
    if cur:lines.append(cur)
    print(f"===== {name} @ {start} =====")
    for a,b,t in lines: print(f"{a+start:7.2f}-{b+start:7.2f} {t}")
for spec in [
 ('charadeA','/workspace/prjreplay/films/charade.mp4',720),
 ('charadeB','/workspace/prjreplay/films/charade.mp4',1800),
 ('bucket','https://archive.org/download/a-bucket-of-blood-1959/A%20Bucket%20of%20Blood%20(1959).mp4',240),
 ('santa','https://archive.org/download/SantaClausConquersTheMartians1964_201310/Santa%20Claus%20Conquers%20the%20Martians%20(1964).mp4',900),
 ('carnival','https://archive.org/download/carnival-of-souls-1962_202108/Carnival%20of%20Souls%20%201962.mp4',1800),
 ('brain','https://archive.org/download/the_brain_that_wouldnt_die_201712/the_brain_that_wouldnt_die.mp4',3420),
 ('shop','https://archive.org/download/TheLittleShopOfHorrors1960_765/TheLittleShopOfHorrors1960.mp4',840),
 ('teens','https://archive.org/download/teenagers-from-outer-space/Teenagers%20from%20Outer%20Space.mp4',2880),
]:
    try: probe(spec[0],spec[1],spec[2])
    except Exception as e: print(spec[0],'ERR',e)
