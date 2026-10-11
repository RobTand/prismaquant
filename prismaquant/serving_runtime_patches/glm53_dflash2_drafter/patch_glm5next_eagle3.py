#!/usr/bin/env python3
"""Teach the GLM5-next target the EAGLE3 aux-hidden interface DFlash2 uses.

The V2 runner enables auxiliary hidden states for method ``dflash``
(``model_runner.py``) and at load calls
``set_eagle3_aux_hidden_state_layers``, which raises
``Model does not support EAGLE3 interface`` unless the target implements
``SupportsEagle3``. Neither GLM5-next wrapper does, so a DFlash2 serve dies
before any CUDA graph is captured. The runner then asserts the model returns
a ``(hidden_states, aux_hidden_states)`` tuple, which the current
``Glm5NextModel.forward`` never does.

Seven edits on ``models/glm5next/nvidia/model.py``. Each edit carries an
already-applied marker that is absent in the base file, so a rerun
completes a partial run and a foreign file is refused instead of
double-applied. The file is refused unless it is byte-identical to the
base image's. The aux capture mirrors the served DeepSeek-V4 EAGLE3
pattern: capture after the layer whose 1-based index the drafter names
(the DFlash2 draft config names ``target_layer_ids``; ``eagle3_utils``
adds 1), and return the tuple only when at least one layer is named, so
a no-spec serve keeps its exact current return.
"""

import hashlib
from pathlib import Path

#: models/glm5next/nvidia/model.py as the base image ships it (upstream
#: vLLM fd4a15126, untouched by the mtp-mapper and kpool-tail sets). The
#: edit refuses any other bytes, so a moved base fails the build instead
#: of receiving an edit written for another file.
BASE_MODEL_SHA256 = (
    "4436911ea5b32e90dfef487509f5caabecce334bb87be3304372ca089047577d"
)

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = "models/glm5next/nvidia/model.py"

OLD_IMPORT = (
    "from vllm.model_executor.models.interfaces import (\n"
    "    HasInnerState,\n"
    "    IsHybrid,\n"
    "    MixtureOfExperts,\n"
    "    SupportsPP,\n"
    ")\n"
)
NEW_IMPORT = (
    "# GLM53_DFLASH2_EAGLE3 interfaces: the DFlash2 target serves aux states.\n"
    "from vllm.model_executor.models.interfaces import (\n"
    "    EagleModelMixin,\n"
    "    HasInnerState,\n"
    "    IsHybrid,\n"
    "    MixtureOfExperts,\n"
    "    SupportsEagle3,\n"
    "    SupportsPP,\n"
    ")\n"
)

OLD_MODEL_CLASS = "class Glm5NextModel(nn.Module):\n"
NEW_MODEL_CLASS = (
    "# GLM53_DFLASH2_EAGLE3 mixin: holds the aux layer ids the runner sets.\n"
    "class Glm5NextModel(nn.Module, EagleModelMixin):\n"
)

OLD_ACTIVE = (
    "        self._active_layers = self.layers[self.start_layer : self.end_layer]\n"
    "\n"
    "        if get_pp_group().is_last_rank:\n"
)
NEW_ACTIVE = (
    "        self._active_layers = self.layers[self.start_layer : self.end_layer]\n"
    "        # GLM53_DFLASH2_EAGLE3 store: named by set_aux_hidden_state_layers.\n"
    "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
    "\n"
    "        if get_pp_group().is_last_rank:\n"
)

OLD_LOOP = (
    "        full_num_tokens = positions.shape[0]\n"
    "        if self.is_sequence_parallel:\n"
    "            hidden_states = sp_shard(hidden_states)\n"
    "\n"
    "        for layer in self._active_layers:\n"
    "            hidden_states, residual, post, comb = layer(\n"
    "                positions, hidden_states, residual, post, comb\n"
    "            )\n"
)
NEW_LOOP = (
    "        full_num_tokens = positions.shape[0]\n"
    "        if self.is_sequence_parallel:\n"
    "            hidden_states = sp_shard(hidden_states)\n"
    "\n"
    "        # GLM53_DFLASH2_EAGLE3 capture: 1-based layer ids, deepseek pattern.\n"
    "        aux_hidden_states: list[torch.Tensor] = []\n"
    "        for idx, layer in enumerate(\n"
    "            self._active_layers, start=self.start_layer\n"
    "        ):\n"
    "            hidden_states, residual, post, comb = layer(\n"
    "                positions, hidden_states, residual, post, comb\n"
    "            )\n"
    "            if idx + 1 not in self.aux_hidden_state_layers:\n"
    "                continue\n"
    "            # Mid-stack mHC defers hc_post; materialize then contract\n"
    "            # 4 streams -> [tokens, hidden] (deepseek_v4 eagle3 pattern).\n"
    "            if post is not None and hasattr(layer, \"hc_post\"):\n"
    "                value = hc_contract(\n"
    "                    layer.hc_post(hidden_states, residual, post, comb),\n"
    "                    layer.n,\n"
    "                )\n"
    "            else:\n"
    "                value = hidden_states\n"
    "                if value.ndim == 3:\n"
    "                    value = value.mean(dim=1)\n"
    "            if self.is_sequence_parallel:\n"
    "                value = sp_all_gather(value)[:full_num_tokens]\n"
    "            aux_hidden_states.append(value)\n"
)

OLD_RETURN = (
    "        hidden_states = self.norm(hidden_states)\n"
    "        return hidden_states\n"
)
NEW_RETURN = (
    "        hidden_states = self.norm(hidden_states)\n"
    "        # GLM53_DFLASH2_EAGLE3 return: tuple only when named, else as before.\n"
    "        if aux_hidden_states:\n"
    "            return hidden_states, aux_hidden_states\n"
    "        return hidden_states\n"
)

OLD_CAUSAL = (
    "class Glm5NextForCausalLM(\n"
    "    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid\n"
    "):\n"
)
NEW_CAUSAL = (
    "# GLM53_DFLASH2_EAGLE3 target: serves aux states to the DFlash2 drafter.\n"
    "class Glm5NextForCausalLM(\n"
    "    nn.Module,\n"
    "    HasInnerState,\n"
    "    SupportsPP,\n"
    "    MixtureOfExperts,\n"
    "    IsHybrid,\n"
    "    SupportsEagle3,\n"
    "):\n"
)

OLD_MLLM = (
    "class Glm5NextForConditionalGeneration(\n"
    "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid\n"
    "):\n"
)
NEW_MLLM = (
    "# GLM53_DFLASH2_EAGLE3 target: serves aux states to the DFlash2 drafter.\n"
    "class Glm5NextForConditionalGeneration(\n"
    "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, SupportsEagle3\n"
    "):\n"
)

EDITS = (
    # (tag, old, new, already-applied marker absent in the base file).
    ("interface imports", OLD_IMPORT, NEW_IMPORT,
     "GLM53_DFLASH2_EAGLE3 interfaces"),
    ("Glm5NextModel mixin", OLD_MODEL_CLASS, NEW_MODEL_CLASS,
     "GLM53_DFLASH2_EAGLE3 mixin"),
    ("aux layer store", OLD_ACTIVE, NEW_ACTIVE,
     "GLM53_DFLASH2_EAGLE3 store"),
    ("aux capture loop", OLD_LOOP, NEW_LOOP,
     "GLM53_DFLASH2_EAGLE3 capture"),
    ("tuple return", OLD_RETURN, NEW_RETURN,
     "GLM53_DFLASH2_EAGLE3 return"),
    ("CausalLM interface", OLD_CAUSAL, NEW_CAUSAL,
     "IsHybrid,\n    SupportsEagle3,\n"),
    ("ConditionalGeneration interface", OLD_MLLM, NEW_MLLM,
     "HasInnerState, IsHybrid, SupportsEagle3\n"),
)


def patched_source(data: bytes) -> bytes:
    """Apply the seven edits; refuse foreign bytes, complete partial runs."""
    text = data.decode()
    for tag, old, new, marker in EDITS:
        if marker in text:
            continue
        if text.count(old) != 1:
            raise ValueError(
                f"expected one {tag} target, found {text.count(old)}"
            )
        text = text.replace(old, new)
    return text.encode()


def main() -> None:
    target = SITE / TARGET
    raw = target.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BASE_MODEL_SHA256:
        raise SystemExit(f"{TARGET} is not the base image's (sha256 {digest})")
    target.write_text(patched_source(raw).decode())
    compile(target.read_text(), str(target), "exec")
    print("applied glm53_dflash2_drafter eagle3")


if __name__ == "__main__":
    main()
