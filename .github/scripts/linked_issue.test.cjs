"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const {
  CHECK_NAME,
  evaluateClosingIssues,
  pullRequestIdentity,
  run,
} = require("./linked_issue.cjs");

const REPOSITORY = "RobTand/prismaquant";
const HEAD_SHA = "a".repeat(40);

function workflowEvent(eventName, change = () => {}) {
  const root = path.resolve(__dirname, "../..");
  const event = JSON.parse(fs.readFileSync(path.join(__dirname,
    "fixtures/linked-issue-mergify-batch.json"), "utf8"));
  change(event.pull_request, event);
  const workflow = fs.readFileSync(path.join(root, ".github/workflows/linked-issue.yml"), "utf8");
  const match = workflow.match(/          script: \|\n((?:            .*\n?)*)/);
  assert.ok(match, "the production policy script must exist in the real workflow");
  const script = match[1].replace(/^ {12}/gm, "");
  assert.match(script, /await policy\.run\(\{ github, context, core \}\)/);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "pq-linked-event-"));
  const eventPath = path.join(directory, "event.json");
  fs.writeFileSync(eventPath, JSON.stringify(event));
  const driver = `
    const fs = require("node:fs");
    const payload = JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH, "utf8"));
    const context = { eventName: process.env.GITHUB_EVENT_NAME, payload,
      repo: { owner: "RobTand", repo: "prismaquant" },
      serverUrl: "https://github.com", runId: 1234 };
    const statuses = [], failures = [], lookups = [];
    const github = {
      graphql: async (_query, variables) => {
        lookups.push(variables);
        if (variables.issueNumber !== undefined) return { repository: { issue:
          variables.issueNumber === 3216 ? { __typename: "Issue", number: 3216,
            state: "OPEN", url: "https://github.com/RobTand/prismaquant/issues/3216",
            repository: { nameWithOwner: "RobTand/prismaquant" } } : null } };
        return { repository: { pullRequest: { number: payload.pull_request.number,
          body: payload.pull_request.body, closingIssuesReferences: { nodes: [] } } } };
      },
      rest: { repos: { createCommitStatus: async status => statuses.push(status) } }
    };
    const core = { info() {}, error() {}, setFailed(message) { failures.push(message); } };
    (async () => {
      ${script}
      process.stdout.write(JSON.stringify({ statuses, failures, lookupCount: lookups.length }));
      process.exitCode = failures.length ? 1 : 0;
    })().catch(() => { process.exitCode = 2; });
  `;
  try {
    const child = spawnSync(process.execPath, ["--v8-pool-size=1", "-e", driver], {
      cwd: root, encoding: "utf8", timeout: 10000,
      env: { PATH: process.env.PATH, TMPDIR: os.tmpdir(), UV_THREADPOOL_SIZE: "1",
        GITHUB_EVENT_PATH: eventPath, GITHUB_EVENT_NAME: eventName },
    });
    assert.ifError(child.error);
    assert.notEqual(child.status, 2, "the workflow entry point must complete");
    return { ...JSON.parse(child.stdout), returncode: child.status };
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

for (const eventName of ["pull_request", "pull_request_target"]) {
  test(`real workflow accepts the reference-free authenticated batch event: ${eventName}`, () => {
    const result = workflowEvent(eventName);
    assert.equal(result.returncode, 0);
    assert.deepEqual(result.failures, []);
    assert.equal(result.statuses.at(-1).state, "success");
    assert.equal(result.lookupCount, 0);
  });
}

for (const eventName of ["pull_request", "pull_request_target"]) {
  for (const [name, change] of [
    ["ordinary PR", pr => { pr.head.ref = "prismaquant-3216-topic"; }],
    ["empty batch suffix", pr => { pr.head.ref = "mergify/merge-queue/"; }],
    ["spoofed batch prefix", pr => { pr.user = { login: "contributor", id: 123, type: "User" }; }],
    ["wrong bot id", pr => { pr.user.id = 123; }],
    ["wrong bot login", pr => { pr.user.login = "different[bot]"; }],
    ["non-bot author", pr => { pr.user.type = "User"; }],
    ["foreign head repository", pr => { pr.head.repo.full_name = "someone/prismaquant"; }],
    ["missing head repository", pr => { delete pr.head.repo; }],
    ["non-draft branch", pr => { pr.draft = false; }],
    ["ordinary sender", (_pr, event) => { event.sender = { login: "contributor", id: 123, type: "User" }; }],
    ["wrong sender id", (_pr, event) => { event.sender.id = 123; }],
    ["wrong sender login", (_pr, event) => { event.sender.login = "different[bot]"; }],
    ["non-bot sender", (_pr, event) => { event.sender.type = "User"; }],
    ["missing sender", (_pr, event) => { delete event.sender; }],
  ]) {
    test(`real workflow still rejects a reference-free ${name}: ${eventName}`, () => {
      const result = workflowEvent(eventName, change);
      assert.equal(result.returncode, 1);
      assert.equal(result.failures.length, 1);
      assert.equal(result.statuses.at(-1).state, "failure");
      assert.ok(result.lookupCount > 0);
    });
  }
}

test("real workflow still validates an ordinary PR against its open parent", () => {
  const result = workflowEvent("pull_request", pr => {
    pr.head.ref = "prismaquant-3216-topic";
    pr.user = { login: "contributor", id: 123, type: "User" };
    pr.draft = false;
    pr.body = "Refs #3216";
  });
  assert.equal(result.returncode, 0);
  assert.equal(result.statuses.at(-1).state, "success");
  assert.equal(result.lookupCount, 2);
});

function issue(number, repository = REPOSITORY) {
  return {
    __typename: "Issue",
    number,
    state: "OPEN",
    url: `https://github.com/${repository}/issues/${number}`,
    repository: { nameWithOwner: repository },
  };
}

function response(nodes, number = 253, body = "") {
  return {
    repository: {
      pullRequest: {
        number,
        body,
        closingIssuesReferences: { nodes },
      },
    },
  };
}

function context() {
  return {
    repo: { owner: "RobTand", repo: "prismaquant" },
    payload: {
      pull_request: {
        number: 253,
        base: { ref: "main", repo: { full_name: REPOSITORY } },
        head: { sha: HEAD_SHA, ref: "prismaquant-252" },
        created_at: "2026-10-09T17:00:00Z",
      },
    },
    runId: 1234,
    serverUrl: "https://github.com",
  };
}

test("accepts a real same-repository issue", () => {
  assert.deepEqual(evaluateClosingIssues(response([issue(252)]), REPOSITORY, 253), {
    ok: true,
    issues: [
      {
        number: 252,
        state: "OPEN",
        url: "https://github.com/RobTand/prismaquant/issues/252",
      },
    ],
    totalClosingReferences: 1,
  });
});

test("rejects missing, cross-repository, and pull-request references", () => {
  const pullRequest = {
    ...issue(251),
    __typename: "PullRequest",
    url: "https://github.com/RobTand/prismaquant/pull/251",
  };
  const result = evaluateClosingIssues(
    response([issue(9, "RobTand/tessera"), pullRequest]),
    REPOSITORY,
    253,
  );
  assert.equal(result.ok, false);
  assert.deepEqual(result.issues, []);
  assert.equal(result.totalClosingReferences, 2);
  assert.equal(evaluateClosingIssues(response([]), REPOSITORY, 253).ok, false);
});

test("fails closed on malformed or mismatched API responses", () => {
  assert.throws(
    () => evaluateClosingIssues({}, REPOSITORY, 253),
    /GraphQL repository is missing or malformed/,
  );
  assert.throws(
    () => evaluateClosingIssues(response([issue(252)], 999), REPOSITORY, 253),
    /instead of #253/,
  );
  assert.throws(
    () =>
      evaluateClosingIssues(
        { repository: { pullRequest: { number: 253, closingIssuesReferences: {} } } },
        REPOSITORY,
        253,
      ),
    /nodes is missing or malformed/,
  );
});

test("takes the check SHA from the pull request head", () => {
  assert.deepEqual(pullRequestIdentity(context()), {
    owner: "RobTand",
    repo: "prismaquant",
    repository: REPOSITORY,
    pullNumber: 253,
    headSha: HEAD_SHA,
  });
});

test("an API error completes the actual-head check as failure", async () => {
  const statuses = [];
  const github = {
    graphql: async () => {
      throw new Error("GraphQL unavailable");
    },
    rest: {
      repos: {
        createCommitStatus: async (input) => statuses.push(input),
      },
    },
  };
  const failures = [];
  await run({
    github,
    context: context(),
    core: {
      error: () => {},
      info: () => {},
      setFailed: (message) => failures.push(message),
    },
  });

  assert.equal(statuses[0].sha, HEAD_SHA);
  assert.equal(statuses[0].context, CHECK_NAME);
  assert.equal(statuses[0].state, "pending");
  assert.equal(statuses.at(-1).sha, HEAD_SHA);
  assert.equal(statuses.at(-1).state, "failure");
  assert.match(failures.at(-1), /failed closed/);
});

test("a valid issue completes the actual-head check as success", async () => {
  const statuses = [];
  const github = {
    graphql: async () => response([issue(252)]),
    rest: {
      repos: {
        createCommitStatus: async (input) => statuses.push(input),
      },
    },
  };
  const failures = [];
  await run({
    github,
    context: context(),
    core: {
      error: () => {},
      info: () => {},
      setFailed: (message) => failures.push(message),
    },
  });

  assert.deepEqual(failures, []);
  assert.equal(statuses.at(-1).sha, HEAD_SHA);
  assert.equal(statuses.at(-1).context, CHECK_NAME);
  assert.equal(statuses.at(-1).state, "success");
  assert.match(statuses.at(-1).description, /#252/);
});

async function parentCheck(body, parent, options = {}) {
  const statuses = [];
  const failures = [];
  const lookups = [];
  const github = {
    graphql: async (_query, variables) => {
      if (variables.issueNumber === undefined) {
        return response(options.closing || [], 253, body);
      }
      lookups.push(variables);
      if (options.lookupError) throw new Error("parent lookup unavailable");
      return { repository: { issue: options.parents ? options.parents[variables.issueNumber] ?? null : parent } };
    },
    rest: { repos: { createCommitStatus: async (input) => statuses.push(input) } },
  };
  const event = context();
  // The API body, not a stale event snapshot, is the authority.
  event.payload.pull_request.body = options.eventBody || "Refs #999";
  if (Object.hasOwn(options, "branch")) event.payload.pull_request.head.ref = options.branch;
  if (Object.hasOwn(options, "createdAt")) event.payload.pull_request.created_at = options.createdAt;
  await run({ github, context: event,
    core: { info: () => {}, error: () => {}, setFailed: (message) => failures.push(message) } });
  return { statuses, failures, lookups };
}

for (const body of ["Refs #252", "Part of #252", "Refs RobTand/prismaquant#252", "Refs https://github.com/RobTand/prismaquant/issues/252"]) {
  test(`accepts open same-repository parent: ${body}`, async () => {
    const result = await parentCheck(body, issue(252));
    assert.equal(result.statuses.at(-1).state, "success");
    assert.equal(result.statuses.at(-1).sha, HEAD_SHA);
    assert.match(result.statuses.at(-1).description, /parent.*#252/i);
    assert.deepEqual(result.failures, []);
    assert.equal(result.lookups.length, 1);
    assert.equal(result.lookups[0].issueNumber, 252);
  });
}

test("rejects a closed referenced parent", async () => {
  const result = await parentCheck("Refs #252", { ...issue(252), state: "CLOSED" });
  assert.equal(result.statuses.at(-1).state, "failure");
  assert.equal(result.failures.length, 1);
});

test("rejects cross-repository Refs without looking in this repository", async () => {
  const result = await parentCheck("Refs RobTand/tessera#252", issue(252));
  assert.equal(result.statuses.at(-1).state, "failure");
  assert.deepEqual(result.lookups, []);
});

test("rejects no reference even if the stale event body names a parent", async () => {
  const result = await parentCheck("No linked work item", issue(252));
  assert.equal(result.statuses.at(-1).state, "failure");
  assert.deepEqual(result.lookups, []);
});

test("fails closed when parent lookup is unavailable", async () => {
  const result = await parentCheck("Refs #252", issue(252), { lookupError: true });
  assert.equal(result.statuses.at(-1).state, "failure");
  assert.match(result.failures.at(-1), /failed closed.*parent lookup unavailable/);
});

test("fails closed on a mismatched parent API response", async () => {
  const result = await parentCheck("Refs #252", issue(999));
  assert.equal(result.statuses.at(-1).state, "failure");
  assert.match(result.failures.at(-1), /failed closed/);
});

test("rejects a parent API node that is a pull request", async () => {
  const result = await parentCheck("Refs #252", { ...issue(252), __typename: "PullRequest" });
  assert.equal(result.statuses.at(-1).state, "failure");
});

test("rejects missing or foreign parent API nodes", async () => {
  for (const parent of [null, issue(252, "RobTand/tessera")]) {
    const result = await parentCheck("Refs #252", parent);
    assert.equal(result.statuses.at(-1).state, "failure");
  }
});

test("deduplicates explicit open-parent directives", async () => {
  const result = await parentCheck("Refs #252, #252\nPart of #252", issue(252));
  assert.equal(result.statuses.at(-1).state, "success");
  assert.equal(result.lookups.length, 1);
});

test("native closing references retain closed-issue behavior", async () => {
  const result = await parentCheck("", null, { closing: [{ ...issue(252), state: "CLOSED" }] });
  assert.equal(result.statuses.at(-1).state, "success");
  assert.deepEqual(result.lookups, []);
});

test("preserves native closing references without consulting parent refs", async () => {
  const result = await parentCheck("Refs RobTand/tessera#999", null, { closing: [issue(252)] });
  assert.equal(result.statuses.at(-1).state, "success");
  assert.deepEqual(result.lookups, []);
});

for (const body of ["Example #252", "> Refs #252", "```text\nRefs #252\n```", "Refs #0", "Refs #252garbage", "<!--\nRefs #252\n-->", "    Refs #252", "Refs #2147483648", "Refs https://example.com/RobTand/prismaquant/issues/252", "Refs https://github.com/RobTand/tessera/issues/252"]) {
  test(`does not accept a plain or quoted/malformed reference: ${JSON.stringify(body)}`, async () => {
    const result = await parentCheck(body, issue(252));
    assert.equal(result.statuses.at(-1).state, "failure");
    assert.deepEqual(result.lookups, []);
  });
}

for (const branch of ["prismaquant-123", "prismaquant-123-gpu", "prismaquant-123-gpu-2"]) {
  test(`accepts branch ${branch} for a verified closing issue`, async () => {
    const result = await parentCheck("", null, { branch, closing: [issue(123)] });
    assert.equal(result.statuses.at(-1).state, "success");
    assert.deepEqual(result.failures, []);
  });
  test(`accepts branch ${branch} for a verified open parent`, async () => {
    const result = await parentCheck("Refs #123", issue(123), { branch });
    assert.equal(result.statuses.at(-1).state, "success");
    assert.deepEqual(result.failures, []);
  });
}

for (const branch of ["fix/something", "prismaquant-124", "prismaquant-123-", "prismaquant-123-GPU", "prismaquant-123-/gpu"]) {
  test(`rejects branch ${branch} when only issue #123 is linked`, async () => {
    const result = await parentCheck("", null, { branch, closing: [issue(123)] });
    assert.equal(result.statuses.at(-1).state, "failure");
    assert.equal(result.statuses.at(-1).sha, HEAD_SHA);
    assert.match(result.statuses.at(-1).description, /branch rule.*prismaquant-<issue>/i);
    assert.match(result.failures.at(-1), /branch rule.*prismaquant-<issue>/i);
    assert.match(result.failures.at(-1), /prismaquant-123/);
  });
}

for (const options of [
  { branch: "ig/train/x-9" },
  { branch: "ig/2518-old" },
  { branch: "pg/PRISMAQUANT-2353/PRISMAQUANT-2353.a1" },
  { branch: "pg-train/3" },
  { branch: "release" },
  { branch: "release-v1" },
  { branch: "fix/something", createdAt: "2026-10-09T16:59:59Z" },
]) {
  test(`exempts only branch validation: ${JSON.stringify(options)}`, async () => {
    const accepted = await parentCheck("", null, { ...options, closing: [issue(123)] });
    assert.equal(accepted.statuses.at(-1).state, "success");
    const rejected = await parentCheck("", null, options);
    assert.equal(rejected.statuses.at(-1).state, "failure");
    assert.match(rejected.failures.at(-1), /no same-repository closing issue or open parent/);
  });
}

for (const createdAt of ["2026-10-09T17:00:00Z", "2026-10-09T17:00:01Z"]) {
  test(`enforces the branch rule at ${createdAt}`, async () => {
    const result = await parentCheck("Refs #123", issue(123), { branch: "fix/something", createdAt });
    assert.equal(result.statuses.at(-1).state, "failure");
    assert.match(result.failures.at(-1), /branch rule/i);
  });
}

test("matches any verified closing reference, not only the first", async () => {
  const result = await parentCheck("", null, { branch: "prismaquant-123", closing: [issue(124), issue(123)] });
  assert.equal(result.statuses.at(-1).state, "success");
  assert.deepEqual(result.lookups, []);
});

test("matches a later verified open parent", async () => {
  const result = await parentCheck("Refs #124, #123", null, {
    branch: "prismaquant-123", parents: { 124: issue(124), 123: issue(123) },
  });
  assert.equal(result.statuses.at(-1).state, "success");
  assert.deepEqual(result.failures, []);
  assert.match(result.statuses.at(-1).description, /parent.*#123/i);
});

test("matches an open parent when closing references name another issue", async () => {
  const result = await parentCheck("Part of #123", issue(123), {
    branch: "prismaquant-123", closing: [issue(124)],
  });
  assert.equal(result.statuses.at(-1).state, "success");
  assert.deepEqual(result.failures, []);
});

for (const parent of [null, { ...issue(123), state: "CLOSED" }, issue(123, "RobTand/tessera"), { ...issue(123), __typename: "PullRequest" }]) {
  test(`does not use an unverified branch parent: ${JSON.stringify(parent)}`, async () => {
    const result = await parentCheck("Refs #123", parent, { branch: "prismaquant-123", closing: [issue(124)] });
    assert.equal(result.statuses.at(-1).state, "failure");
    assert.match(result.failures.at(-1), /branch rule/i);
  });
}

test("does not use a foreign closing reference to match the branch", async () => {
  const result = await parentCheck("", null, {
    branch: "prismaquant-123", closing: [issue(124), issue(123, "RobTand/tessera")],
  });
  assert.equal(result.statuses.at(-1).state, "failure");
  assert.match(result.failures.at(-1), /branch rule/i);
});

test("fails closed if the branch parent lookup fails despite another closing issue", async () => {
  const result = await parentCheck("Refs #123", null, {
    branch: "prismaquant-123", closing: [issue(124)], lookupError: true,
  });
  assert.equal(result.statuses.at(-1).state, "failure");
  assert.match(result.failures.at(-1), /failed closed.*parent lookup unavailable/);
});

for (const options of [{ branch: undefined }, { createdAt: undefined }, { createdAt: "not-a-date" }]) {
  test(`fails closed on missing or malformed branch metadata: ${JSON.stringify(options)}`, async () => {
    const result = await parentCheck("", null, { ...options, closing: [issue(252)] });
    assert.equal(result.statuses.at(-1).state, "failure");
    assert.equal(result.statuses.at(-1).sha, HEAD_SHA);
    assert.match(result.failures.at(-1), /failed closed/);
  });
}

test("rejects a branch parent that is closed after another open parent", async () => {
  const result = await parentCheck("Refs #124, #123", null, {
    branch: "prismaquant-123", parents: { 124: issue(124), 123: { ...issue(123), state: "CLOSED" } },
  });
  assert.equal(result.statuses.at(-1).state, "failure");
  assert.match(result.failures.at(-1), /branch rule.*prismaquant-<issue>/i);
  assert.match(result.failures.at(-1), /prismaquant-124/);
});

test("an exempt branch still fails closed on a parent API error", async () => {
  const result = await parentCheck("Refs #123", null, { branch: "ig/train/x-9", lookupError: true });
  assert.equal(result.statuses.at(-1).state, "failure");
  assert.match(result.failures.at(-1), /failed closed.*parent lookup unavailable/);
});
