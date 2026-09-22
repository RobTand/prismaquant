import json
from pathlib import Path
from tools import launch_glm_routing_prefix as launcher
from tools import tessera_campaign_container as existing

def test_strict_stage_root_enters_container_read_only(tmp_path,monkeypatch):
 mapping=tmp_path/'map.json';mapping.write_text(json.dumps({'stage_root':'/stage/prewarm'}))
 monkeypatch.setattr(existing,'_stage_root_mounted',lambda p:True)
 argv=launcher.launch_argv(tmp_path,{'PRISMABUILD_RESIDENCY_MAP':str(mapping)},[2,4],[])
 assert 'type=bind,src=/stage/prewarm,dst=/stage/prewarm,readonly' in argv
 assert argv[argv.index('--cpuset-cpus')+1]=='2,4'

def test_unmounted_stage_refuses_before_docker(tmp_path,monkeypatch):
 import pytest
 mapping=tmp_path/'map.json';mapping.write_text(json.dumps({'stage_root':'/stage/prewarm'}))
 monkeypatch.setattr(existing,'_stage_root_mounted',lambda p:False)
 with pytest.raises(RuntimeError,match='not a mounted directory'):
  launcher.launch_argv(tmp_path,{'PRISMABUILD_RESIDENCY_MAP':str(mapping)},[2],[])
