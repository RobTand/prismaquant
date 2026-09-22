"""Print the prepared panel PWC's fields.

Runs only as a script; nothing executes at import.
"""
import pickle,json
from pathlib import Path


def main():
 p=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare/production.pkl');x=pickle.loads(p.read_bytes());print(type(x).__name__)
 for k,v in vars(x).items():
  print(k,type(v).__name__,len(v) if hasattr(v,'__len__') else None)
  if isinstance(v,dict) and v:
   key=next(iter(v)); value=v[key];print('sample',repr(key),repr(value)[:2500])


if __name__ == '__main__':
    main()
