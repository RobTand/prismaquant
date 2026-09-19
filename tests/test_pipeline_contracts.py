from __future__ import annotations

from prismaquant.pipeline import (
    ArtifactSpec,
    MetricGateSpec,
    PipelineSpec,
    PipelineStageSpec,
    ResourceContract,
    default_production_pipeline_spec,
    load_pipeline_spec,
    main,
    parse_render_mechanisms,
    production_pipeline_spec_from_config,
    render_mechanism_stage_specs,
    write_pipeline_spec,
)


def test_metric_gate_selects_only_improved_linears():
    gate = MetricGateSpec(
        name="mse_improves",
        metric="output_mse",
        mode="per_item",
        direction="lower_is_better",
    )

    result = gate.evaluate(
        baseline={
            "layer.a": {"output_mse": 1.0},
            "layer.b": {"output_mse": 1.0},
            "layer.c": {"output_mse": 1.0},
        },
        candidate={
            "layer.a": {"output_mse": 0.9},
            "layer.b": {"output_mse": 1.0},
            "layer.c": {"output_mse": 1.1},
        },
    )

    assert result.passed is True
    assert result.accepted_keys() == ("layer.a",)
    assert result.rejected_keys() == ("layer.b", "layer.c")


def test_metric_gate_can_enforce_global_validation_improvement():
    gate = MetricGateSpec(
        name="kl_improves",
        metric="end_kl",
        mode="all",
        direction="lower_is_better",
    )

    passed = gate.evaluate(
        baseline={"end_kl": 0.12},
        candidate={"end_kl": 0.10},
    )
    failed = gate.evaluate(
        baseline={"end_kl": 0.12},
        candidate={"end_kl": 0.12},
    )

    assert passed.passed is True
    assert passed.accepted_keys() == ("__global__",)
    assert failed.passed is False
    assert failed.decisions[0].reason == "regressed_or_tied"


def test_metric_gate_can_allow_bounded_metric_regression():
    gate = MetricGateSpec(
        name="ppl_preserved",
        metric="ppl",
        mode="all",
        direction="lower_is_better",
        require_improvement=False,
        max_relative_regression=0.005,
    )

    tolerated = gate.evaluate(
        baseline={"ppl": 10.0},
        candidate={"ppl": 10.04},
    )
    rejected = gate.evaluate(
        baseline={"ppl": 10.0},
        candidate={"ppl": 10.10},
    )

    assert tolerated.passed is True
    assert tolerated.decisions[0].reason == "within_regression_budget"
    assert rejected.passed is False


def test_render_mechanisms_are_exposed_as_ordered_pipeline_stages():
    stages = render_mechanism_stage_specs((
        "gptq",
        "static_act_order",
        "joint_scale_opt",
        "four_over_six",
    ))

    assert tuple(stage.name for stage in stages) == (
        "render.four_over_six",
        "render.joint_scale_opt",
        "render.static_act_order",
        "render.gptq",
    )
    assert all(
        stage.resources[0].owner == "ProductionWeightCache"
        for stage in stages
    )
    assert stages[0].gates == ("gate.render.output_mse",)


def test_default_production_pipeline_contract_validates():
    spec = default_production_pipeline_spec()
    result = spec.validate()

    assert result.ok is True
    assert result.errors == ()
    stages = {stage.name: stage for stage in spec.stages}
    assert "render.static_act_order" in stages
    assert stages["cache.prefetch_assignment"].resources[0] == ResourceContract(
        resource="rendered_weights",
        owner="ProductionWeightCache",
        residency="required",
    )
    assert any(
        resource.owner == "PerturbedActivationCache"
        for resource in stages["validate.kl"].resources
    )


def test_pipeline_spec_round_trips_through_json(tmp_path):
    path = tmp_path / "pipeline.json"
    spec = default_production_pipeline_spec(render_mechanisms=("gptq",))

    write_pipeline_spec(spec, path)
    loaded = load_pipeline_spec(path)

    assert loaded.to_dict() == spec.to_dict()
    assert loaded.validate().ok is True


def test_render_mechanisms_parse_env_style_config():
    mechanisms = parse_render_mechanisms(
        "gptq,joint_scale_opt, gptq",
        disabled="joint_scale_opt",
    )

    assert mechanisms == ("gptq",)


def test_production_pipeline_spec_records_run_config():
    spec = production_pipeline_spec_from_config(
        render_mechanisms="gptq,joint_scale_opt",
        model_path="/models/qwen",
        work_dir="/runs/qwen",
        formats="NVFP4,BF16",
        target_bits=4.75,
        target_profile="vllm_packed_moe",
        calibration_modality="text-only",
        selection_mode="surrogate",
        production_cache="1",
        production_recache="1",
    )

    assert spec.validate().ok is True
    assert spec.metadata["render_mechanisms"] == ["joint_scale_opt", "gptq"]
    assert spec.metadata["target_profile"] == "vllm_packed_moe"
    assert spec.metadata["formats"] == "NVFP4,BF16"


def test_surrogate_run_spec_omits_unexecuted_validation_stages():
    spec = production_pipeline_spec_from_config(
        render_mechanisms="gptq",
        selection_mode="surrogate",
    )
    stage_names = [stage.name for stage in spec.stages]

    assert "validate.kl" not in stage_names
    assert "validate.vllm_smoke" not in stage_names
    assert spec.metadata["omitted_unexecuted_stages"] == [
        "validate.kl",
        "validate.vllm_smoke",
    ]
    assert spec.validate().ok is True


def test_validated_surrogate_spec_keeps_kl_validation_stage():
    spec = production_pipeline_spec_from_config(
        render_mechanisms="gptq",
        selection_mode="validated-surrogate",
    )
    stage_names = [stage.name for stage in spec.stages]

    assert "validate.kl" in stage_names
    assert "validate.vllm_smoke" not in stage_names
    assert spec.metadata["omitted_unexecuted_stages"] == [
        "validate.vllm_smoke",
    ]


def test_pipeline_cli_writes_validated_default_spec(tmp_path):
    path = tmp_path / "pipeline_spec.json"

    rc = main([
        "--write-default-production",
        str(path),
        "--validate",
        "--render-mechanisms",
        "gptq",
        "--target-profile",
        "research",
    ])
    loaded = load_pipeline_spec(path)

    assert rc == 0
    assert path.exists()
    assert loaded.validate().ok is True
    assert loaded.metadata["render_mechanisms"] == ["gptq"]


def test_pipeline_validation_rejects_parallel_rendered_weight_cache():
    spec = PipelineSpec(
        id="bad",
        artifacts=(ArtifactSpec("source", "model", provided=True),),
        stages=(PipelineStageSpec(
            name="bad.cache",
            component="bad",
            inputs=("source",),
            outputs=("cache",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="AdHocCache",
                residency="required",
            ),),
        ),),
    )

    result = spec.validate()
    assert result.ok is False
    assert any("rendered_weights must use" in error for error in result.errors)


def test_pipeline_validation_rejects_parallel_activation_cache():
    spec = PipelineSpec(
        id="bad-activation-cache",
        artifacts=(ArtifactSpec("source", "model", provided=True),),
        stages=(PipelineStageSpec(
            name="bad.activation",
            component="bad",
            inputs=("source",),
            outputs=("acts",),
            resources=(ResourceContract(
                resource="perturbed_activations",
                owner="AdHocActivationCache",
                residency="required",
            ),),
        ),),
    )

    result = spec.validate()
    assert result.ok is False
    assert any("perturbed_activations must use" in error for error in result.errors)


def test_pipeline_validation_requires_inputs_to_be_available_in_order():
    spec = PipelineSpec(
        id="bad-order",
        artifacts=(
            ArtifactSpec("source", "model", provided=True),
            ArtifactSpec("late", "payload"),
        ),
        stages=(
            PipelineStageSpec(
                name="uses.late",
                component="consumer",
                inputs=("source", "late"),
                outputs=("out",),
            ),
            PipelineStageSpec(
                name="produces.late",
                component="producer",
                inputs=("source",),
                outputs=("late",),
            ),
        ),
    )

    result = spec.validate()
    assert result.ok is False
    assert "uses.late: input 'late' is not available" in result.errors
