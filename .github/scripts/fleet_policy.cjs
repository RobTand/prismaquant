"use strict";

const { FLEET_CHECK, readBase, verifyCandidate, workflowTrigger } = require("./queue_policy.cjs");

async function admit({ github, context, core }) {
  const base = await readBase(github, context);
  const { data: run } = await github.rest.actions.getWorkflowRun({
    ...context.repo, run_id: context.payload.workflow_run.id,
  });
  const trigger = workflowTrigger(run, context, base);
  const binding = await verifyCandidate({ github, context, ...trigger });
  const checkIds = [];
  for (const headSha of new Set([binding.pull.head, binding.integrated])) {
    const { data: check } = await github.rest.checks.create({
      ...context.repo, name: FLEET_CHECK, head_sha: headSha, status: "in_progress",
      details_url: `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`,
      output: { title: "Fleet candidate admitted", summary: `Testing integrated ${binding.integrated} against main ${binding.base}` },
    });
    checkIds.push(check.id);
  }
  core.setOutput("binding", JSON.stringify(binding));
  core.setOutput("checks", JSON.stringify(checkIds));
  core.setOutput("integrated", binding.integrated);
}

async function finish({ github, context, binding, checkIds, result, verify = verifyCandidate }) {
  let conclusion = "failure";
  let summary = `Fleet job result: ${result}; candidate ${binding.integrated}, main ${binding.base}`;
  try {
    const fresh = await verify({ github, context, number: binding.pull.number, expectedHead: binding.pull.head });
    if (JSON.stringify(fresh) !== JSON.stringify(binding)) throw new Error("candidate binding changed during fleet run");
    if (result === "success") conclusion = "success";
  } catch (error) {
    summary += `; fresh admission failed: ${error.message}`;
  }
  for (const checkId of checkIds) {
    await github.rest.checks.update({ ...context.repo, check_run_id: checkId, status: "completed", conclusion,
      output: { title: `Fleet integrated CI: ${conclusion}`, summary } });
  }
  return conclusion;
}

module.exports = { admit, finish };
