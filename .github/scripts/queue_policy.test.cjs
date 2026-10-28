"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { batchBinding, readQueueStatus, verifyQueue, verifyCandidate, workflowTrigger } = require("./queue_policy.cjs");
const { admit, finish } = require("./fleet_policy.cjs");
const { run: linkedIssue } = require("./linked_issue.cjs");
const { readFileSync } = require("node:fs");

const REPOSITORY = "RobTand/prismaquant";
const BASE = "a".repeat(40);
const HEAD = "b".repeat(40);
const INTEGRATED = "c".repeat(40);
const context = { repo: { owner: "RobTand", repo: "prismaquant" }, serverUrl: "https://github.com", runId: 55 };

function fixture() {
  const member = {
    number: 3143, state: "open", merged: false, draft: false,
    user: { login: "contributor", id: 123, type: "User" }, labels: [],
    created_at: "2026-10-28T00:00:00Z", updated_at: "2026-10-28T00:00:00Z",
    head: { ref: "prismaquant-3140-provenance", sha: HEAD, repo: { full_name: REPOSITORY } },
    base: { ref: "main", sha: BASE, repo: { full_name: REPOSITORY } }, merge_commit_sha: INTEGRATED,
  };
  const pull = { ...structuredClone(member), number: 3144, draft: true,
    user: { login: "mergify[bot]", type: "Bot", id: 37929162 },
    head: { ref: "mergify/merge-queue/main/batch-1", sha: INTEGRATED, repo: { full_name: REPOSITORY } } };
  const status = JSON.parse(readFileSync(`${__dirname}/fixtures/mergify-status.json`, "utf8"));
  const state = { member, pull, status, base: BASE, parents: [BASE, HEAD], linked: true,
    pullReads: 0, queueReads: 0, outputs: {}, checks: [], updates: [] };
  state.run = { repository: { full_name: REPOSITORY }, head_repository: { full_name: REPOSITORY },
    path: ".github/workflows/linked-issue.yml", event: "pull_request_target", status: "completed",
    conclusion: "success", head_sha: HEAD,
    pull_requests: [{ number: member.number, head: { sha: HEAD }, base: { sha: BASE } }] };
  state.github = {
    graphql: async (_query, input) => ({ repository: { pullRequest: {
      number: input.number, body: "", closingIssuesReferences: { nodes: state.linked ? [{ __typename: "Issue",
        number: 3140, state: "OPEN", url: "https://github.com/RobTand/prismaquant/issues/3140",
        repository: { nameWithOwner: REPOSITORY } }] : [] },
    } } }),
    rest: {
      pulls: { get: async ({ pull_number: number }) => {
        state.pullReads += 1;
        const value = number === state.pull.number ? state.pull : state.member;
        if (state.onRead) state.onRead(state, number);
        return { data: structuredClone(value) };
      } },
      users: { getByUsername: async () => ({ data: { login: "mergify[bot]", type: "Bot", id: 37929162 } }) },
      repos: { getBranch: async () => ({ data: { commit: { sha: state.base } } }),
        getCommit: async () => ({ data: { sha: state.mergeSha ?? INTEGRATED, parents: state.parents.map((parent) => ({ sha: parent })) } }) },
      actions: { getWorkflowRun: async () => ({ data: state.run }) },
      checks: { create: async (value) => { state.checks.push(value); return { data: { id: state.checks.length } }; },
        update: async (value) => state.updates.push(value) },
    },
  };
  state.queueStatus = async () => {
    state.queueReads += 1;
    if (state.onQueue) state.onQueue(state);
    return structuredClone(state.status);
  };
  state.verify = () => verifyQueue({ github: state.github, context, pull: structuredClone(state.pull), queueStatus: state.queueStatus });
  return state;
}

test("authenticates a complete batch and binds original issue policy and merge parents", async () => {
  const state = fixture();
  const binding = await state.verify();
  assert.equal(binding.integrated, INTEGRATED);
  assert.equal(binding.member.head, HEAD);
  assert.equal(state.queueReads, 2);
  assert.equal(state.pullReads, 4);
});

const badCases = {
  "spoofed bot id": (state) => { state.pull.user.id = 123; },
  "spoofed bot type": (state) => { state.pull.user.type = "User"; },
  "spoofed author login": (state) => { state.pull.user.login = "contributor"; },
  "spoofed branch": (state) => { state.pull.head.ref = "prismaquant-3140"; },
  "fork queue": (state) => { state.pull.head.repo.full_name = "other/prismaquant"; },
  "missing membership": (state) => { delete state.status.batches[0].pull_requests; },
  "empty membership": (state) => { state.status.batches[0].pull_requests = []; },
  "multi-member batch": (state) => { state.status.batches[0].pull_requests = [3143, 3145]; },
  "invalid member": (state) => { state.status.batches[0].pull_requests = [null]; },
  "self member": (state) => { state.status.batches[0].pull_requests = [{ number: 3144 }]; },
  "no batch": (state) => { state.status.batches = []; },
  "multiple batches": (state) => { state.status.batches.push(structuredClone(state.status.batches[0])); },
  "split batch": (state) => { state.status.batches[0].sub_batches = [2]; },
  "dependent batch": (state) => { state.status.batches[0].parent_ids = ["550e8400-e29b-41d4-a716-446655440002"]; },
  "missing completeness envelope": (state) => { delete state.status.waiting_pull_requests; },
  "pagination": (state) => { state.status.pageInfo = { hasNextPage: true }; },
  "next cursor": (state) => { state.status.next_cursor = "next"; },
  "wrong queue PR": (state) => { state.status.batches[0].queue_pull_request_number = 1; },
  "wrong queue": (state) => { state.status.batches[0].queue_rule_name = "other"; },
  "wrong mode": (state) => { state.status.mode = "parallel"; },
  "stale batch": (state) => { state.status.batches[0].status.code = "merged"; },
  "label mismatch": (state) => { state.status.batches[0].queue_rule_name = "release"; },
  "closed member": (state) => { state.member.state = "closed"; },
  "draft member": (state) => { state.member.draft = true; },
  "fork member": (state) => { state.member.head.repo.full_name = "other/prismaquant"; },
  "unlinked member": (state) => { state.linked = false; },
  "wrong member branch": (state) => { state.member.head.ref = "unrelated"; },
  "stale member base": (state) => { state.member.base.sha = HEAD; },
  "stale queue base": (state) => { state.pull.base.sha = HEAD; },
  "wrong base parent": (state) => { state.parents[0] = HEAD; },
  "stale member head": (state) => { state.member.head.sha = BASE; },
  "missing merge parents": (state) => { state.parents = []; },
  "truncated membership with extra integrated parent": (state) => { state.parents.push("d".repeat(40)); },
  "integrated SHA mismatch": (state) => { state.mergeSha = HEAD; },
};

for (const [name, corrupt] of Object.entries(badCases)) {
  test(`fails closed: ${name}`, async () => {
    const state = fixture();
    corrupt(state);
    await assert.rejects(state.verify());
  });
}

for (const target of ["member head", "queue head", "member label", "member draft", "main", "batch identity"]) {
  test(`rejects freshness race: ${target}`, async () => {
    const state = fixture();
    state.onRead = (current) => {
      if (current.pullReads !== 3) return;
      if (target === "member head") current.member.head.sha = BASE;
      if (target === "queue head") current.pull.head.sha = HEAD;
      if (target === "member label") current.member.labels = [{ name: "release" }];
      if (target === "member draft") current.member.draft = true;
      if (target === "main") current.base = HEAD;
    };
    state.onQueue = (current) => {
      if (current.queueReads === 2 && target === "batch identity") current.status.batches[0].id = "550e8400-e29b-41d4-a716-446655440002";
    };
    await assert.rejects(state.verify());
  });
}

test("a release member must bind the release queue", async () => {
  const state = fixture();
  state.status.batches[0].queue_rule_name = "release";
  state.member.labels = [{ name: "release" }];
  assert.equal((await state.verify()).batch.queue, "release");
});

test("queue and GitHub API errors fail closed", async () => {
  const state = fixture();
  state.queueStatus = async () => { throw new Error("API unavailable"); };
  await assert.rejects(state.verify(), /API unavailable/);
  state.github.rest.pulls.get = async () => { throw new Error("GitHub unavailable"); };
  await assert.rejects(state.verify(), /GitHub unavailable/);
});

test("Mergify transport refuses missing credentials, errors, links and redirects", async () => {
  await assert.rejects(readQueueStatus(REPOSITORY, ""), /credential/);
  for (const [ok, link] of [[false, null], [true, '<next>; rel="next"']]) {
    await assert.rejects(readQueueStatus(REPOSITORY, "test-token", async (_url, options) => {
      assert.equal(options.redirect, "error");
      assert.equal(options.headers.Authorization, "Bearer test-token");
      return { ok, headers: { get: () => link } };
    }), /unavailable or paginated/);
  }
});

test("ordinary PR binds the actual GitHub test merge SHA", async () => {
  const state = fixture();
  const binding = await verifyCandidate({ github: state.github, context, number: state.member.number, expectedHead: HEAD });
  assert.equal(binding.integrated, INTEGRATED);
  assert.equal(state.queueReads, 0);
  await assert.rejects(verifyCandidate({ github: state.github, context, number: state.member.number, expectedHead: BASE }), /trigger head is stale/);
});

test("ordinary fork/draft and merge-ref mismatch never enter fleet", async () => {
  for (const corrupt of [
    (state) => { state.member.head.repo.full_name = "fork/prismaquant"; },
    (state) => { state.member.draft = true; },
    (state) => { state.mergeSha = HEAD; },
  ]) {
    const state = fixture();
    corrupt(state);
    await assert.rejects(verifyCandidate({ github: state.github, context, number: state.member.number, expectedHead: HEAD }));
  }
});

test("workflow admission checks server-loaded path/event/repo and PR/head/base association", () => {
  const state = fixture();
  assert.deepEqual(workflowTrigger(state.run, context, BASE), { number: 3143, expectedHead: HEAD });
  for (const [key, value] of [["path", ".github/workflows/attacker.yml"], ["event", "pull_request"],
    ["head_sha", BASE], ["conclusion", "failure"], ["pull_requests", []]]) {
    assert.throws(() => workflowTrigger({ ...state.run, [key]: value }, context, BASE));
  }
});

test("admission creates separately named checks on the head AND integrated SHA", async () => {
  const state = fixture();
  await admit({ github: state.github, context: { ...context, payload: { workflow_run: { id: 9 } } },
    core: { setOutput: (name, value) => { state.outputs[name] = value; } } });
  assert.deepEqual(state.checks.map((check) => check.head_sha), [HEAD, INTEGRATED]);
  assert.ok(state.checks.every((check) => check.name === "fleet integrated CI"));
  assert.equal(JSON.parse(state.outputs.binding).integrated, INTEGRATED);
});

for (const result of ["success", "failure", "cancelled", "skipped", "timed_out"]) {
  test(`publication requires fresh admission AND successful fleet result: ${result}`, async () => {
    const state = fixture();
    const binding = await state.verify();
    const conclusion = await finish({ github: state.github, context, binding, checkIds: [1], result,
      verify: async () => structuredClone(binding) });
    assert.equal(conclusion, result === "success" ? "success" : "failure");
    assert.equal(state.updates[0].conclusion, conclusion);
  });
}

test("a green fleet job cannot publish success for a stale or unverifiable candidate", async () => {
  for (const verify of [async () => ({}), async () => { throw new Error("API unavailable"); }]) {
    const state = fixture();
    const binding = await state.verify();
    assert.equal(await finish({ github: state.github, context, binding, checkIds: [1], result: "success", verify }), "failure");
  }
});

test("workflow keeps contributor execution out of the trusted metadata-only workflow", () => {
  const linked = readFileSync(`${__dirname}/../workflows/linked-issue.yml`, "utf8");
  const fleet = readFileSync(`${__dirname}/../workflows/fleet-ci.yml`, "utf8");
  assert.match(linked, /ref: \$\{\{ github.event.pull_request.base.sha \}\}/);
  assert.doesNotMatch(linked, /runs-on: \[self-hosted|npm install|pip install/);
  assert.match(fleet, /workflow_run:/);
  const execution = fleet.split("  fleet:\n")[1].split("  publish:\n")[0];
  assert.match(execution, /runs-on: \[self-hosted, linux, x64, fleet\]/);
  assert.doesNotMatch(execution, /secrets\.|write|pull_request_target|upload-artifact/);
  assert.match(execution, /persist-credentials: false/);
  assert.doesNotMatch(fleet, /pb-tests/);
});

test("the real linked-issue entry point only exempts authenticated batch members", async () => {
  for (const spoofed of [false, true]) {
    const state = fixture();
    if (spoofed) state.pull.user.id = 1;
    const statuses = [];
    const failures = [];
    state.github.rest.repos.createCommitStatus = async (status) => statuses.push(status);
    await linkedIssue({ github: state.github, context: { ...context, payload: { pull_request: state.pull } },
      queueStatus: state.queueStatus,
      core: { info: () => {}, error: () => {}, setFailed: (message) => failures.push(message) } });
    assert.equal(statuses.at(-1).sha, INTEGRATED);
    assert.equal(statuses.at(-1).context, "linked issue");
    assert.equal(statuses.at(-1).state, spoofed ? "failure" : "success");
    assert.equal(failures.length, spoofed ? 1 : 0);
  }
});
