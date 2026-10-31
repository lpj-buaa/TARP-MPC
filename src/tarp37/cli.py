import argparse
import json
from .collection import run_collection,read
from .assets import PROJECT


def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['smoke','pilot','formal'],default='smoke')
    p.add_argument('--run-id',required=True); p.add_argument('--resume',action='store_true'); p.add_argument('--workers',type=int,default=1)
    p.add_argument('--until',choices=['audit','collect','analyze'],default='analyze'); p.add_argument('--config')
    a=p.parse_args(); config=read(a.config or PROJECT/f'configs/{a.mode}.json')
    print(json.dumps(run_collection(config,a.run_id,a.resume,a.workers,a.until),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
