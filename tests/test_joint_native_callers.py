"""Native caller policy and durable attempts, independent of scientific fake evidence."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from tools import dispatch_stage_a_split as stage
from tools import dispatch_capture_chain as capture
from tools import dispatch_joint_quanta as joint
from prismaquant import flux_jobs as jobs, tessera_campaign_rows as rows


def row(**extra):
    return {'argv':['/bin/true'],'cwd':'/mnt/shared/source','demand':{'cpu':1,'mem_gb':1,'gpu':0},
            'timeout_s':60,'env':{},'tags':[],**extra}


def test_gateway_preflights_real_neutral_rows_without_shelling_pb():
    with patch.object(jobs,'validate_job',return_value={}), patch.object(jobs,'submit_job') as submit:
        for intent in (row(data_manifest='/strict'), row(produced_outputs={'template':'/out'}),
                       row(progress_phases=['phase=900']), row(priority=-10), row(pilot={}),
                       row(demand={'cpu':1,'mem_gb':1,'gpu':1})):
            with pytest.raises(ValueError):
                joint.Gateway().submit(intent,state_dir=Path('/mnt/shared/never-created'))
        submit.assert_not_called()


def test_old_pb_history_cannot_resume_as_native(tmp_path):
    path=tmp_path/'state.json'
    path.write_text('{"event":"stage-a-submitted","action_key":"old"}\n')
    with pytest.raises(joint.DispatchRefused,match='cannot resume as Flux'):
        joint._read_state(path)


def _saved_attempt(tmp_path):
    entry={'name':'prep','kind':'prep','row':row()}
    stage._write_json(tmp_path/'round.json',{'rows':[entry]})
    state=tmp_path/'native/prep/attempt-1'
    state.mkdir(parents=True)
    (state/'manifest.json').write_text(json.dumps({'rows':[entry['row']]}))
    saved={'schema':stage.NATIVE_SUBMISSION_SCHEMA,'name':'prep','attempt':'attempt-1','row':entry['row'],'state_dir':str(state)}
    stage._write_json(stage.submission_path(tmp_path,'prep'),saved)
    return entry,saved


@pytest.mark.parametrize('status',['running','unknown','succeeded'])
def test_retry_refuses_nonfailure_without_retiring_evidence(tmp_path,status):
    _,saved=_saved_attempt(tmp_path)
    before=stage.submission_path(tmp_path,'prep').read_bytes()
    with patch.object(rows,'collect_rows',return_value=[{'status':status,'payload_exit':0,'query_error':None}]):
        with pytest.raises(stage.SplitDispatchRefused,match='cannot retry'):
            stage.retire_attempt(tmp_path,'prep')
    assert stage.submission_path(tmp_path,'prep').read_bytes()==before
    assert not (tmp_path/'submissions/prep.attempt-1.json').exists()


def test_explicit_failed_attempt_preserves_old_evidence(tmp_path):
    _,saved=_saved_attempt(tmp_path)
    failure={'status':'failed','payload_exit':7,'query_error':None,'job_id':'failed-native-id'}
    with patch.object(rows,'collect_rows',return_value=[failure]):
        stage.retire_attempt(tmp_path,'prep')
    archive=json.loads((tmp_path/'submissions/prep.attempt-1.json').read_bytes())
    assert archive['terminal_result']==failure and archive['row']==saved['row']
    assert (Path(saved['state_dir'])/'manifest.json').is_file()
    assert not stage.submission_path(tmp_path,'prep').exists()


def test_old_pb_submission_record_does_not_pass_ending(tmp_path):
    path=stage.submission_path(tmp_path,'prep')
    stage._write_json(path,{'name':'prep','detach':{'action_key':'old'}})
    with pytest.raises(stage.SplitDispatchRefused,match='cannot resume as Flux'):
        stage.native_ending(tmp_path,'prep')


def test_changed_sealed_row_cannot_reuse_earlier_success(tmp_path):
    entry,_=_saved_attempt(tmp_path)
    entry['row']['argv']=['/bin/false']
    stage._write_json(tmp_path/'round.json',{'rows':[entry]})
    with patch.object(rows,'collect_rows') as collect:
        with pytest.raises(stage.SplitDispatchRefused,match='differs from sealed scientific row'):
            stage.native_ending(tmp_path,'prep')
        collect.assert_not_called()


def test_capture_refuses_unsupported_future_quantum_before_prep_launch(tmp_path):
    document={'rows':[{'name':'prep','row':row()},{'name':'quantum','row':row(data_manifest='/strict')}]}
    with patch.object(capture,'load_round',return_value=document), patch.object(jobs,'validate_job',return_value={}), patch.object(jobs,'submit_job') as submit:
        with pytest.raises(capture.ChainDispatchRefused,match='strict staging'):
            capture.submit(tmp_path)
        submit.assert_not_called()
        assert not stage.submission_path(tmp_path,'prep').exists()


def test_recovery_refuses_row_cwd_that_differs_from_saved_source(tmp_path):
    state = tmp_path / 'native'
    state.mkdir()
    application_row = row(cwd='/mnt/shared/foreign-source')
    (state/'manifest.json').write_text(json.dumps({'rows':[application_row],
        'sources':[{'checkout':'/mnt/shared/pinned-source'}]}))
    intent = {'event':'native-intent','state_dir':str(state),'row':application_row}
    submission = jobs.FluxSubmission('saved-id',state/'rows/00000','a'*64)
    with patch.object(jobs,'load_submission',return_value=submission), patch.object(jobs,'validate_launch') as validate:
        with pytest.raises(joint.DispatchRefused,match='source differs'):
            joint.recover_events([intent],tmp_path/'events.jsonl')
        validate.assert_not_called()
    assert not (tmp_path/'events.jsonl').exists()


def test_native_ending_refuses_pointer_descriptor_with_foreign_id(tmp_path):
    _, saved = _saved_attempt(tmp_path)
    record = Path(saved['state_dir'])/'rows/00000'
    saved['submissions'] = [{'job_id':'foreign-id','record_dir':str(record),'launch_sha256':'a'*64}]
    stage._write_json(stage.submission_path(tmp_path,'prep'),saved)
    with patch.object(jobs,'load_submission',return_value=jobs.FluxSubmission('saved-id',record,'a'*64)), patch.object(rows,'collect_rows') as collect:
        with pytest.raises(stage.SplitDispatchRefused,match='descriptor differs'):
            stage.native_ending(tmp_path,'prep')
        collect.assert_not_called()
        with patch.object(rows,'submit_rows') as submit:
            with pytest.raises(stage.SplitDispatchRefused,match='descriptor differs'):
                stage.submit_row(tmp_path,{'name':'prep','row':saved['row']})
            submit.assert_not_called()
