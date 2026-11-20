"""The row collector asks Netdata for a whole-second window (#3557)."""
import urllib.parse

from tools import collect_row_netdata


def test_collector_requests_whole_second_window(monkeypatch):
    queries=[]
    contexts=sorted(collect_row_netdata.REQUIRED_CONTEXTS)
    def fetch(host, endpoint):
        if endpoint=='charts':
            return {'charts':{'c%d'%i:{'context':c} for i,c in enumerate(contexts)}}
        query={k:v[0] for k,v in urllib.parse.parse_qs(endpoint.split('?',1)[1]).items()}
        queries.append(query)
        return {'labels':['time','v'], 'data':[[s,1.0] for s in range(100,112)]}
    monkeypatch.setattr(collect_row_netdata, 'fetch', fetch)
    try:
        result=collect_row_netdata.main(['--after','100.4','--before','110.6'])
    except RuntimeError as error:
        result=str(error)
    assert result==0
    assert queries
    assert {float(q['after']) for q in queries}=={100.0}
    assert {float(q['before']) for q in queries}=={111.0}


def test_collector_refuses_host_without_gpu_power_chart(monkeypatch, capsys):
    def fetch(host, endpoint):
        assert (host, endpoint) == ("sparklina", "charts")
        return {"charts": {
            "cpu": {"context": "system.cpu"},
            "ram": {"context": "system.ram"},
            "io": {"context": "system.io"},
        }}

    monkeypatch.setattr(collect_row_netdata, "fetch", fetch)
    try:
        collect_row_netdata.main(["--after", "100", "--before", "110"])
    except SystemExit as error:
        assert error.code == (
            "required netdata contexts missing on sparklina: "
            "['nvidia_smi.gpu_power_draw']"
        )
    else:
        raise AssertionError("The collector did not stop.")
    assert capsys.readouterr().out == ""
