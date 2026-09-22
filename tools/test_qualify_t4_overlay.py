import importlib.util
from pathlib import Path
import pytest
import torch
spec=importlib.util.spec_from_file_location('qualifier',Path(__file__).with_name('qualify_t4_overlay.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@pytest.fixture
def cell(tmp_path,monkeypatch):
 tensor=torch.arange(8,dtype=torch.bfloat16).reshape(2,4);render=tmp_path/'render.pt';torch.save(tensor,render);wire=tmp_path/'unit.tessera';wire.write_bytes(b'wire')
 monkeypatch.setattr(m,'_read_verified_wire_blob',lambda c:(b'wire',m.digest(b'wire')));monkeypatch.setattr(m,'verify_cached_unit',lambda *a:None);monkeypatch.setattr(m,'_decode_wire',lambda *a,**k:tensor.clone())
 return dict(qname='unit',format='TESSERA_E2M1_K2_R896',render=str(render),render_stat=m.stamp(render),wire=str(wire),wire_stat=m.stamp(wire),record={'identity':{},'blob_sha256':m.digest(b'wire')},source_weight={'shape':[2,4]},activation={},encoding_identity_sha256='a'*64,render_origin='encoded',render_comparison='independent_render_vs_wire',catalog_source_adoption={},adopted_source_hessian={})
def test_current_decode_must_equal_retained_render(cell,monkeypatch):
 monkeypatch.setattr(m,'_decode_wire',lambda *a,**k:torch.zeros(2,4,dtype=torch.bfloat16))
 with pytest.raises(AssertionError):m.qualify(cell)
def test_changed_file_fence_refuses_before_decode(cell,monkeypatch):
 Path(cell['render']).write_bytes(b'changed')
 monkeypatch.setattr(m,'_decode_wire',lambda *a,**k:pytest.fail('decoder must not run'))
 with pytest.raises(AssertionError):m.qualify(cell)
def test_qualified_record_carries_actual_file_and_tensor_hash(cell):
 result=m.qualify(cell)
 assert result['render_file_sha256']==m.digest(Path(cell['render']).read_bytes())
 assert result['rendered_weight']['shape']==[2,4]
 assert result['render_comparison']=='independent_render_vs_wire'
