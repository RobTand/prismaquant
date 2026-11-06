import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

from tools import tessera_campaign_container as launcher


class StageBindTest(unittest.TestCase):
    def test_remapped_ancestor_cannot_replace_the_pb_stage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            stage = root / 'stage' / 'prewarm'
            stage.mkdir(parents=True)
            (stage / 'owned.bin').write_bytes(b'PB-owned bytes')
            unrelated = root / 'unrelated'
            (unrelated / 'prewarm').mkdir(parents=True)
            (unrelated / 'prewarm' / 'owned.bin').write_bytes(b'wrong mounted bytes')
            map_path = root / 'map.json'
            map_path.write_text(json.dumps({'stage_root': str(stage)}))
            declared = [
                {'source': str(map_path), 'target': str(map_path), 'readonly': True},
                {'source': str(unrelated), 'target': str(stage.parent)},
            ]
            spec = {'container': {'mounts': declared}, 'env': {}}
            with patch.object(launcher, '_stage_root_mounted', return_value=True):
                env, added = launcher.residency_environment(
                    spec, {launcher.RESIDENCY_MAP_ENV: str(map_path)})
            self.assertEqual(env[launcher.RESIDENCY_MAP_ENV], str(map_path))
            # Resolve the mount Docker would use, without Docker or a fleet.
            path = PurePosixPath(stage / 'owned.bin')
            covering = [m for m in declared + added
                        if PurePosixPath(m['target']) in path.parents]
            mount = max(covering, key=lambda m: len(PurePosixPath(m['target']).parts))
            actual = Path(mount['source']) / path.relative_to(mount['target'])
            self.assertEqual(actual.read_bytes(), b'PB-owned bytes')


if __name__ == '__main__':
    unittest.main()
