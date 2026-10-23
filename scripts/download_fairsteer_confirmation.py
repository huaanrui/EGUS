"""Fetch three additional pinned BBQ files; validate JSON before writing."""
import concurrent.futures
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REV="bea11bd97d79217245b5871acd247b9d6eb24598"


def fetch(cat):
    path=ROOT/f"data/raw/bbq/{REV}/data/{cat}.jsonl"
    if path.exists():return (cat,"cached")
    urls=[f"https://raw.githubusercontent.com/nyu-mll/BBQ/{REV}/data/{cat}.jsonl",f"https://cdn.jsdelivr.net/gh/nyu-mll/BBQ@{REV}/data/{cat}.jsonl"]
    for url in urls:
        for direct in (True,False):
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
                with opener.open(url,timeout=18) as response:blob=response.read()
                n=0
                for line in blob.decode("utf-8").splitlines():json.loads(line);n+=1
                if n<128:raise ValueError("Unexpectedly small pinned dataset")
                path.write_bytes(blob)
                return (cat,n,len(blob),url)
            except Exception as error:
                print(cat,type(error).__name__,flush=True)
    raise RuntimeError("Unable to fetch pinned category "+cat)


if __name__=="__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch,["Sexual_orientation","Race_x_gender","Race_x_SES"]):print(result,flush=True)
