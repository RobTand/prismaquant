"""Run the no-payload cache proof in the exact source image, CPU-only."""
import json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.launch_glm_routing_prefix import launch_argv,IMAGE,CONTENT,image_content_sha256
from prismaquant.staged_lease import sdk_submodule
actual=image_content_sha256(json.loads(subprocess.check_output(['docker','image','inspect',IMAGE],text=True))[0]);assert actual==CONTENT
environ=dict(os.environ);environ.update(sdk_submodule('core')._reader_identity_environment(environ['PRISMABUILD_ACTION_KEY']))
argv=launch_argv(Path.cwd(),environ,os.sched_getaffinity(0),[])
i=argv.index('--gpus');del argv[i:i+2]
for option in ('--memory','--memory-swap'):argv[argv.index(option)+1]='4g'
command=(['python3','-m','pytest',*sys.argv[2:]] if sys.argv[1:2]==['--pytest'] else ['python3','tools/verify_glm_prefix_source_cache.py',*sys.argv[1:]])
i=argv.index(IMAGE);argv=argv[:i]+['--env','CUDA_VISIBLE_DEVICES=','--env','NVIDIA_VISIBLE_DEVICES=void',IMAGE]+command
raise SystemExit(subprocess.call(argv))
