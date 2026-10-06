"""The one real-shaped ``docker inspect`` dict the launcher contract tests share.

Deliberately stdlib-only: contract tests that must not load PrismaQuant or
Torch import this fixture, so a future tightening of the content-identity
owner's accepted inspection shape is a single edit (#2193 follow-up). The
``Id`` hex is not a content input -- ``image_content_sha256`` reads only
``Os``, ``Architecture``, ``RootFS`` and ``Config`` -- so every consumer can
carry the same payload.
"""


def inspection():
    """A fresh minimal inspection payload the real content-digest owner accepts."""
    return {"Id": "sha256:" + "a" * 64, "Os": "linux", "Architecture": "arm64",
            "RootFS": {"Type": "layers", "Layers": ["sha256:" + "2" * 64]},
            "Config": {"Env": ["PATH=/bin"], "Entrypoint": ["/entry"], "Cmd": []}}
