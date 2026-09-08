"""Bounded standard-library Chrome event analysis, executed through PB."""
import collections
import hashlib
import json
import pathlib
import re
import sys

path=pathlib.Path(sys.argv[1]); expected=sys.argv[2]; output=pathlib.Path(sys.argv[3])
def digest():
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert digest()==expected
counts=collections.Counter(); duration=collections.Counter(); kernels=collections.Counter();cpu=collections.Counter();events=0
with path.open() as stream:
 buffer=''
 while True:
  chunk=stream.read(65536)
  if not chunk:raise RuntimeError('missing traceEvents array')
  buffer+=chunk
  found=re.search(r'"traceEvents"\s*:\s*\[',buffer)
  if found:
   buffer=buffer[found.end():];break
  if len(buffer)>2**20:raise RuntimeError('trace header exceeds cap')
 decoder=json.JSONDecoder()
 while True:
  buffer=buffer.lstrip(' \r\n\t,')
  if buffer.startswith(']'):break
  try:
   event,end=decoder.raw_decode(buffer)
  except json.JSONDecodeError:
   chunk=stream.read(65536)
   if not chunk:raise RuntimeError('truncated trace event')
   buffer+=chunk
   if len(buffer)>8*2**20:raise RuntimeError('single trace event exceeds cap')
   continue
  buffer=buffer[end:]
  if not isinstance(event,dict):raise RuntimeError('non-object trace event')
  category=event.get('cat','metadata');counts[category]+=1;events+=1
  if event.get('ph')=='X':
   value=float(event.get('dur',0));duration[category]+=value
   if category=='kernel':kernels[event['name']]+=value
   if category=='cpu_op':cpu[event['name']]+=value
assert counts['kernel']>0 and counts['cpu_op']>0
assert digest()==expected
record=dict(path=str(path),sha256=expected,bytes=path.stat().st_size,event_count=events,
 categories=dict(counts),category_duration_us=dict(duration),
 top_kernel_total_us=kernels.most_common(15),top_cpu_op_total_us=cpu.most_common(15),
 scope='Durations are summed trace events; nested CPU events overlap and are not self time. No GPU annotation is counted as a kernel.')
output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
