"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  CHECK_NAME,
  evaluateClosingIssues,
  pullRequestIdentity,
  run,
} = require("./linked_issue.cjs");

const REPOSITORY = "RobTand/prismaquant";
const HEAD_SHA = "a".repeat(40);

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
        head: { sha: HEAD_SHA },
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
      return { repository: { issue: parent } };
    },
    rest: { repos: { createCommitStatus: async (input) => statuses.push(input) } },
  };
  const event = context();
  // The API body, not a stale event snapshot, is the authority.
  event.payload.pull_request.body = options.eventBody || "Refs #999";
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
