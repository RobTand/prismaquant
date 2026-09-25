#!/usr/bin/env python3
"""Give Glm5NextMTP the hf_to_vllm_mapper its checkpoint namespace needs.

A quantization config is written in the CHECKPOINT namespace
(``model.language_model.layers.45.mlp.experts``). vLLM translates it into the
namespace of the module tree it built with the model class's
``hf_to_vllm_mapper`` (``model_executor/model_loader/utils.py``,
``configure_quant_config``). ``Glm5NextMTP`` builds its head as a text-only
model (``model.layers.45.*``) and strips ``model.language_model.`` to
``model.`` by hand inside ``load_weights``, but declares no mapper, so a
quantization target on layer 45 never matches the drafter's modules.

One edit, asserted to land exactly once: a class attribute
``hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.":
"model."})``. ``load_weights`` is not touched; it keeps its own strip.
"""

import hashlib
from pathlib import Path

#: mtp.py as the base image ships it. The edit refuses any other bytes, so a
#: moved base fails the build instead of receiving an edit written for another file.
BASE_MTP_SHA256 = "715768cd2ee83c8cdaff4574762753538f40643df3dca064310199fd9bce63ff"

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
target = SITE / "models/glm5next/nvidia/mtp.py"
raw = target.read_bytes()
if hashlib.sha256(raw).hexdigest() != BASE_MTP_SHA256:
    raise SystemExit(f"mtp.py is not the base image's (sha256 {hashlib.sha256(raw).hexdigest()})")
text = raw.decode()

old_import = "from vllm.model_executor.models.utils import maybe_prefix\n"
if text.count(old_import) != 1:
    raise SystemExit("expected one maybe_prefix import in mtp.py")
text = text.replace(
    old_import,
    "from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix\n",
)

old_class = "class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):\n"
if text.count(old_class) != 1:
    raise SystemExit("expected one Glm5NextMTP class statement in mtp.py")
text = text.replace(
    old_class,
    old_class
    + "    # GLM53_MTP_MAPPER: translate checkpoint-namespace quantization targets\n"
    + "    # (model.language_model.*) into the head's own module tree (model.*).\n"
    + "    hf_to_vllm_mapper = WeightsMapper(\n"
    + '        orig_to_new_prefix={"model.language_model.": "model."}\n'
    + "    )\n\n",
)
target.write_text(text)
print("applied glm53_mtp_mapper")
