"""Additive source framing, not a pin migration or serving qualification."""
from __future__ import annotations

import hashlib
from importlib import metadata
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from prismaquant import digests, runtime_provenance, tessera_reader
from test_runtime_provenance import relation_fixture, relation_load

V1 = 'prismaquant.source_tree.v1'
V2 = 'prismaquant.source_tree.v2'
CPP = (b'constexpr char embedded[] = R"BIN(before\0after)BIN";\n'
       b'static_assert(sizeof(embedded) == 13);\n'
       b'static_assert(embedded[6] == 0);\n'
       b'static_assert(embedded[7] == \'a\');\n'
       b'int main() { return embedded[12]; }\n')


def legacy(files):
    result = hashlib.sha256()
    for name in sorted(files, key=Path):
        if Path(name).suffix in tessera_reader.SOURCE_SUFFIXES:
            result.update(name.encode('utf-8') + b'\0' + files[name] + b'\0')
    return result.hexdigest()


def framed(files):
    result = hashlib.sha256(V2.encode('ascii') + b'\0')
    for name in sorted(files, key=lambda name: name.encode('utf-8')):
        if Path(name).suffix in tessera_reader.SOURCE_SUFFIXES:
            encoded, content = name.encode('utf-8'), files[name]
            result.update(len(encoded).to_bytes(8, 'big') + encoded)
            result.update(len(content).to_bytes(8, 'big') + content)
    return result.hexdigest()


def reseal():
    path = Path(__file__).parents[1] / 'tools' / 'reseal_campaign_identity.py'
    spec = importlib.util.spec_from_file_location('reseal_source_profiles_1762', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profiles(caller, files, root):
    if caller == 'producer':
        return runtime_provenance._source_profiles(files)
    root.mkdir()
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    if caller == 'reader':
        return tessera_reader._source_tree_profiles(root)[0]
    return reseal().encoder_tree_profiles(root)[0]


@pytest.mark.parametrize('caller', ['producer', 'reader', 'reseal'])
def test_counterexample_keeps_v1_but_separates_v2(tmp_path, caller):
    merged = {'__init__.py': b'', 'a.py': b'first\0b.py\0second'}
    split = {'__init__.py': b'', 'a.py': b'first', 'b.py': b'second'}
    left = profiles(caller, merged, tmp_path / 'merged')
    right = profiles(caller, split, tmp_path / 'split')
    assert left[V1] == right[V1] == legacy(split)
    assert left[V2] == framed(merged) != framed(split) == right[V2]


@pytest.mark.parametrize('caller', ['producer', 'reader', 'reseal'])
def test_legitimate_cpp_nul_is_accepted_and_stable(tmp_path, caller):
    files = {'__init__.py': b'', 'embedded.cpp': CPP}
    expected = {V1: '02f1f416bc3b045161dd9e984b7431c3219c71e0812a2280f4c855cdedc4c0d9',
                V2: framed(files)}
    assert len(CPP) == 196 and CPP.count(b'\0') == 1
    assert profiles(caller, files, tmp_path / 'first') == expected
    assert profiles(caller, files, tmp_path / 'second') == expected


def test_byte_order_u64_lengths_and_profile_tag():
    files = {'é.py': b'\0', 'a/b.py': b'child', 'a.py': b'parent', '__init__.py': b''}
    observed = runtime_provenance._source_profiles(files)
    assert observed == {V1: legacy(files), V2: framed(files)}
    assert runtime_provenance._source_profiles(dict(reversed(list(files.items())))) == observed
    assert runtime_provenance._source_profiles({})[V2] == hashlib.sha256(V2.encode() + b'\0').hexdigest()


@pytest.mark.parametrize('left_v2,right_v2', [(True, False), (False, True), (False, False)])
def test_gate_legacy_only_side_is_explicit(left_v2, right_v2):
    left = {V1: 'a' * 64, **({V2: 'b' * 64} if left_v2 else {})}
    right = {V1: 'a' * 64, **({V2: 'c' * 64} if right_v2 else {})}
    result = digests.compare_source_profiles(left, right)
    assert result == {'status': 'legacy_framing', 'profile': V1, 'sha256': 'a' * 64}


def test_gate_v2_mismatch_fails_even_when_v1_matches():
    with pytest.raises(ValueError, match='v2.*mismatch'):
        digests.compare_source_profiles({V1: 'a' * 64, V2: 'b' * 64},
                                        {V1: 'a' * 64, V2: 'c' * 64})
    result = digests.compare_source_profiles({V1: 'a' * 64, V2: 'b' * 64},
                                            {V1: 'c' * 64, V2: 'b' * 64})
    assert result == {'status': 'framed_v2', 'profile': V2, 'sha256': 'b' * 64}


@pytest.mark.parametrize('bad', [{V1: 'not-a-sha'}, {V1: 'a' * 64, V2: 'bad'},
                                  {V1: 'a' * 64, 'unknown.source.v2': 'b' * 64}])
def test_gate_refuses_malformed_or_unrecognized_profiles(bad):
    with pytest.raises(ValueError):
        digests.compare_source_profiles(bad, {V1: 'a' * 64})


def test_reader_reports_legacy_framing_and_dual_profiles(tmp_path):
    from test_tessera_reader_namespace import package
    declared = package(tmp_path)
    reader = tessera_reader.load_declared_reader(declared)
    assert reader.identity['source_framing']['status'] == 'legacy_framing'
    both = reader.identity['source_profiles']
    assert both[V1] == declared['source_sha256']
    dual = tessera_reader.load_declared_reader({**declared, 'source_profiles': both})
    assert dual.identity['source_framing']['status'] == 'framed_v2'
    with pytest.raises(ValueError, match='v2.*mismatch'):
        tessera_reader.load_declared_reader({**declared, 'source_profiles': {**both, V2: '0' * 64}})


def test_runtime_relation_preserves_record_and_reports_legacy(relation_fixture):
    _, record, _ = relation_fixture
    old = json.dumps(record, sort_keys=True)
    result = relation_load(relation_fixture)
    assert json.dumps(record, sort_keys=True) == old
    assert result['source_framing']['status'] == 'legacy_framing'
    assert all(run['source_framing']['status'] == 'legacy_framing' for run in result['runs'].values())


def test_runtime_relation_v2_mismatch_is_not_a_legacy_pass(relation_fixture):
    from prismaquant.measured_runtime_prices import RuntimePriceError
    evidence, record, _ = relation_fixture
    run = record['runs']['native']
    package = evidence.get(run['post_package'])
    package['source_profiles'] = {V1: package['encoder_source_sha256'], V2: '0' * 64}
    evidence.replace(run['post_package'], package)
    engine = record['runs']['engine']
    engine['post_package'] = dict(run['post_package'])
    engine_raw = evidence.get(engine['runtime'])
    engine_raw['loaded_package'] = package
    evidence.replace(engine['runtime'], engine_raw)
    with pytest.raises(RuntimePriceError, match='v2.*mismatch'):
        relation_load(relation_fixture)


@pytest.mark.parametrize('kind', ['malformed_v2', 'unknown_profile', 'incoherent_v1'])
def test_runtime_relation_refuses_bad_advertised_package_profiles(relation_fixture, kind):
    from prismaquant.measured_runtime_prices import RuntimePriceError
    evidence, record, _ = relation_fixture
    ref = record['runs']['native']['post_package']
    package = evidence.get(ref)
    profiles = {V1: package['encoder_source_sha256']}
    if kind == 'malformed_v2':
        profiles[V2] = 'not-a-sha'
    elif kind == 'unknown_profile':
        profiles['unknown.source.v2'] = '0' * 64
    else:
        profiles[V1] = '0' * 64
    package['source_profiles'] = profiles
    evidence.replace(ref, package)
    engine = record['runs']['engine']
    engine['post_package'] = dict(ref)
    raw = evidence.get(engine['runtime'])
    raw['loaded_package'] = package
    evidence.replace(engine['runtime'], raw)
    with pytest.raises(RuntimePriceError, match='source.*profile'):
        relation_load(relation_fixture)


def test_runtime_relation_v2_compares_strong_profiles_not_cross_run_v1(relation_fixture):
    evidence, record, _ = relation_fixture
    files = {name: name.encode() for name in
             ('__init__.py', 'cached_unit.py', 'serving/runtime_contract.json')}
    both = {V1: legacy(files), V2: framed(files)}
    package_ref = record['runs']['native']['post_package']
    package = evidence.get(package_ref)
    package['source_profiles'] = both
    evidence.replace(package_ref, package)
    for name, run in record['runs'].items():
        run['post_package'] = dict(package_ref)
        raw = evidence.get(run['runtime'])
        base = raw if name == 'native' else raw['base']
        base['source']['tessera_package_sha256'] = '9' * 64 if name == 'native' else both[V1]
        base['source']['tessera_package_source_profiles'] = {
            V1: base['source']['tessera_package_sha256'], V2: both[V2]}
        if name == 'engine':
            raw['loaded_package'] = package
        evidence.replace(run['runtime'], raw)
    # JSON canonicalization sorts run IDs. Observe the differing native v1
    # before the full-engine metadata, so the pre-fix RED names that gate.
    record['runs']['zengine'] = record['runs'].pop('engine')
    record['full_engine_run_id'] = 'zengine'
    result = relation_load(relation_fixture)
    assert result['source_framing']['status'] == 'framed_v2'
    assert all(run['source_framing']['status'] == 'framed_v2' for run in result['runs'].values())


def test_reseal_hash_output_emits_both_without_rewriting(tmp_path, capsys):
    from types import SimpleNamespace
    root = tmp_path / 'src' / 'tessera'
    root.mkdir(parents=True)
    (root / '__init__.py').write_bytes(b'')
    tool = reseal()
    assert tool.cmd_hash_tree(SimpleNamespace(prismaquant=None, producer=str(tmp_path))) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['encoder_source_sha256'] == output['encoder_source_profiles'][V1]
    assert output['encoder_source_profiles'][V2] == framed({'__init__.py': b''})
    assert list(root.iterdir()) == [root / '__init__.py']


def test_actual_installed_git_v1_identity_is_unchanged():
    try:
        distribution = metadata.distribution('tessera-quant')
    except metadata.PackageNotFoundError:
        pytest.skip('needs an installed Git-provenanced Tessera distribution')
    direct_url = distribution.read_text('direct_url.json')
    if direct_url is None:
        pytest.skip('needs Tessera Git direct_url.json provenance')
    url = json.loads(direct_url)
    if url.get('vcs_info', {}).get('vcs') != 'git':
        pytest.skip('needs a non-editable Git-provenanced Tessera install')
    commit = url['vcs_info']['commit_id']
    assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
    assert not url.get('dir_info', {}).get('editable', False)
    root = Path(distribution.locate_file('tessera')).resolve()
    assert root.is_relative_to(Path(sys.prefix).resolve())
    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in tessera_reader.SOURCE_SUFFIXES}
    expected = legacy(files)
    assert runtime_provenance._source_digest(files) == expected
    assert tessera_reader._source_tree(root)[0] == expected
    assert reseal().encoder_tree_sha256(root) == (expected, len(files))
    import tessera.cached_unit as installed_encoder
    assert Path(installed_encoder.__file__).resolve().parent == root
    installed_encoder.encoder_source_sha256.cache_clear()
    assert installed_encoder.encoder_source_sha256() == expected
    print(json.dumps({'actual_package': str(root), 'commit': commit,
                      'files': len(files), 'legacy_sha256': expected}, sort_keys=True))
