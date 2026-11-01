"""Recording/delegation plumbing; callers retain their update signatures."""
from __future__ import annotations

from prismaquant import digests


def record_source_frames(module, monkeypatch, update_adapter):
    batches = []

    class RecordingProfile(update_adapter):
        def __init__(self):
            self.owner = digests.LengthFramedSourceSha256()
            self.rows = []
            batches.append(self.rows)

        def _record(self, name, payload):
            self.rows.append((name, payload))
            self.owner.update(name, payload)

        def hexdigest(self):
            return self.owner.hexdigest()

    monkeypatch.setattr(module, "LengthFramedSourceSha256", RecordingProfile, raising=False)
    return batches
