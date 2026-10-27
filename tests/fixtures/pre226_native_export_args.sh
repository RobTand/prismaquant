: "${EXPORT_GPTQ:=auto}"
: "${EXPORT_SCALE_SWEEP:=auto}"
echo "[pipeline] [4/4] exporting to compressed-tensors ..."
EXPORT_ARGS=(
  python3 -m prismaquant.export_native_compressed
  --model "$MODEL_PATH"
  --layer-config "${WORK_DIR}/artifacts/layer_config.json"
  --output "${WORK_DIR}/exported"
  --device "$EXPORT_DEVICE"
  --shard-bytes "$EXPORT_SHARD_BYTES"
  --activation-cache-dir "${WORK_DIR}/act"
  "${EXPORT_PIN_ARGS[@]}"
)
case "$EXPORT_GPTQ" in
  0|false|False|FALSE|no|No|NO) EXPORT_ARGS+=(--no-gptq) ;;
  1|true|True|TRUE|yes|Yes|YES) EXPORT_ARGS+=(--gptq) ;;
  auto|"") ;;
  *)
    echo "[pipeline] ERROR: EXPORT_GPTQ must be auto, 0, or 1" >&2
    exit 2
    ;;
esac
case "$EXPORT_SCALE_SWEEP" in
  0|false|False|FALSE|no|No|NO) EXPORT_ARGS+=(--no-scale-sweep) ;;
  1|true|True|TRUE|yes|Yes|YES) EXPORT_ARGS+=(--scale-sweep) ;;
  auto|"") ;;
  *)
    echo "[pipeline] ERROR: EXPORT_SCALE_SWEEP must be auto, 0, or 1" >&2
    exit 2
    ;;
esac
