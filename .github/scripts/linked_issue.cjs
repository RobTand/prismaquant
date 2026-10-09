"use strict";

const CHECK_NAME = "linked issue";
const TARGET_BRANCH = "main";
const BRANCH_RULE_CUTOFF = Date.parse("2026-10-09T17:00:00Z");

const CLOSING_ISSUES_QUERY = `
  query ClosingIssues($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        number
        body
        closingIssuesReferences(first: 100) {
          nodes {
            __typename
            number
            state
            url
            repository {
              nameWithOwner
            }
          }
        }
      }
    }
  }
`;

const PARENT_ISSUE_QUERY = `
  query ParentIssue($owner: String!, $repo: String!, $issueNumber: Int!) {
    repository(owner: $owner, name: $repo) {
      issue(number: $issueNumber) {
        __typename
        number
        state
        url
        repository { nameWithOwner }
      }
    }
  }
`;

function requireObject(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} is missing or malformed`);
  }
  return value;
}

function evaluateClosingIssues(response, expectedRepository, expectedPullNumber) {
  const root = requireObject(response, "GraphQL response");
  const repository = requireObject(root.repository, "GraphQL repository");
  const pullRequest = requireObject(
    repository.pullRequest,
    `pull request #${expectedPullNumber}`,
  );

  if (pullRequest.number !== expectedPullNumber) {
    throw new Error(
      `GraphQL returned pull request #${String(pullRequest.number)} instead of #${expectedPullNumber}`,
    );
  }

  const connection = requireObject(
    pullRequest.closingIssuesReferences,
    "closingIssuesReferences",
  );
  if (!Array.isArray(connection.nodes)) {
    throw new Error("closingIssuesReferences.nodes is missing or malformed");
  }

  const issues = [];
  for (const rawNode of connection.nodes) {
    const node = requireObject(rawNode, "closing issue node");
    const nodeRepository = requireObject(
      node.repository,
      "closing issue repository",
    );
    if (
      node.__typename === "Issue" &&
      nodeRepository.nameWithOwner === expectedRepository &&
      Number.isInteger(node.number) &&
      node.number > 0 &&
      typeof node.url === "string" &&
      node.url.length > 0
    ) {
      issues.push({
        number: node.number,
        state: node.state,
        url: node.url,
      });
    }
  }

  return {
    ok: issues.length > 0,
    issues,
    totalClosingReferences: connection.nodes.length,
  };
}

function parentIssueNumbers(body, expectedRepository, serverUrl) {
  if (typeof body !== "string") {
    throw new Error("pull request body is missing or malformed");
  }
  const numbers = new Set();
  let origin;
  try {
    origin = new URL(serverUrl).origin;
  } catch {
    throw new Error("GitHub server URL is missing or malformed");
  }
  let fence;
  const text = body.replace(/<!--[\s\S]*?(?:-->|$)/g, "");
  for (const line of text.split(/\r?\n/)) {
    const marker = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    if (marker) {
      if (!fence) fence = marker[1];
      else if (marker[1][0] === fence[0] && marker[1].length >= fence.length && !marker[2].trim()) fence = undefined;
      continue;
    }
    if (fence) continue;
    const directive = line.match(/^ {0,3}(?:[-*][ \t]+)?(?:Refs|Part[ \t]+of)[ \t]+(.+)$/i);
    if (!directive) continue;
    for (const token of directive[1].split(/[ \t,]+/)) {
      let repository = expectedRepository;
      let match = token.match(/^#([1-9][0-9]*)[.;:]?$/);
      if (!match) {
        const qualified = token.match(/^([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)#([1-9][0-9]*)[.;:]?$/);
        if (qualified) {
          repository = qualified[1];
          match = [qualified[0], qualified[2]];
        } else if (token.startsWith("https://")) {
          let url;
          try {
            url = new URL(token);
          } catch {
            continue;
          }
          const path = url.pathname.match(/^\/([^/]+\/[^/]+)\/issues\/([1-9][0-9]*)$/);
          if (url.origin === origin && !url.search && !url.hash && !url.username && !url.password && path) {
            repository = path[1];
            match = [token, path[2]];
          }
        }
      }
      if (!match || repository.toLowerCase() !== expectedRepository.toLowerCase()) continue;
      const number = Number(match[1]);
      if (Number.isSafeInteger(number) && number <= 2147483647) numbers.add(number);
      if (numbers.size > 100) throw new Error("too many parent issue references");
    }
  }
  return [...numbers];
}

function evaluateOpenParent(response, identity, expectedNumber) {
  const repository = requireObject(requireObject(response, "parent GraphQL response").repository, "parent GraphQL repository");
  if (repository.issue === null) return null;
  const node = requireObject(repository.issue, "parent issue node");
  if (node.number !== expectedNumber) {
    throw new Error(`GraphQL returned parent #${String(node.number)} instead of #${expectedNumber}`);
  }
  if (node.__typename !== "Issue") return null;
  const owner = requireObject(node.repository, "parent issue repository");
  if (owner.nameWithOwner !== identity.repository || node.state !== "OPEN" || typeof node.url !== "string" || !node.url) return null;
  return { number: node.number, state: node.state, url: node.url };
}

async function readOpenParent(github, response, identity, serverUrl, branchIssueNumber) {
  let first;
  for (const issueNumber of parentIssueNumbers(response.repository.pullRequest.body, identity.repository, serverUrl)) {
    const parent = await github.graphql(PARENT_ISSUE_QUERY, {
      owner: identity.owner, repo: identity.repo, issueNumber,
    });
    const issue = evaluateOpenParent(parent, identity, issueNumber);
    if (!issue) continue;
    first ??= issue;
    if (!branchIssueNumber || issue.number === branchIssueNumber) return issue;
  }
  return first ?? null;
}

function branchPolicy(pullRequest) {
  const branch = pullRequest.head.ref;
  const createdAt = pullRequest.created_at;
  if (typeof branch !== "string" || !branch) {
    throw new Error("pull request head branch is missing or malformed");
  }
  if (typeof createdAt !== "string" || !Number.isFinite(Date.parse(createdAt))) {
    throw new Error("pull request creation time is missing or malformed");
  }
  const exempt = branch.startsWith("ig/") || branch.startsWith("release") ||
    Date.parse(createdAt) < BRANCH_RULE_CUTOFF;
  const match = branch.match(/^prismaquant-([0-9]+)(-[a-z0-9][a-z0-9-]*)?$/);
  return { branch, exempt, issueNumber: match ? Number(match[1]) : null };
}

function pullRequestIdentity(context) {
  const pullRequest = requireObject(
    context && context.payload && context.payload.pull_request,
    "pull_request event payload",
  );
  const base = requireObject(pullRequest.base, "pull request base");
  const baseRepository = requireObject(base.repo, "pull request base repository");
  const head = requireObject(pullRequest.head, "pull request head");
  const headSha = head.sha;
  const expectedRepository = `${context.repo.owner}/${context.repo.repo}`;

  if (base.ref !== TARGET_BRANCH) {
    throw new Error(
      `refusing to evaluate base branch ${String(base.ref)}; expected ${TARGET_BRANCH}`,
    );
  }
  if (baseRepository.full_name !== expectedRepository) {
    throw new Error(
      `base repository ${String(baseRepository.full_name)} does not match ${expectedRepository}`,
    );
  }
  if (!Number.isInteger(pullRequest.number) || pullRequest.number <= 0) {
    throw new Error("pull request number is missing or malformed");
  }
  if (typeof headSha !== "string" || !/^[0-9a-f]{40}$/.test(headSha)) {
    throw new Error("pull request head SHA is missing or malformed");
  }

  return {
    owner: context.repo.owner,
    repo: context.repo.repo,
    repository: expectedRepository,
    pullNumber: pullRequest.number,
    headSha,
  };
}

async function readPullRequest(github, identity) {
  return github.graphql(CLOSING_ISSUES_QUERY, {
    owner: identity.owner,
    repo: identity.repo,
    number: identity.pullNumber,
  });
}

async function publishStatus(github, identity, targetUrl, state, description) {
  await github.rest.repos.createCommitStatus({
    owner: identity.owner,
    repo: identity.repo,
    sha: identity.headSha,
    state,
    context: CHECK_NAME,
    description,
    target_url: targetUrl,
  });
}

async function run({ github, context, core }) {
  let identity;
  let targetUrl;
  try {
    identity = pullRequestIdentity(context);
    targetUrl = `${context.serverUrl}/${identity.repository}/actions/runs/${context.runId}`;
    await publishStatus(
      github,
      identity,
      targetUrl,
      "pending",
      "Resolving closing issues or an open parent reference",
    );

    const response = await readPullRequest(github, identity);
    const result = evaluateClosingIssues(
      response,
      identity.repository,
      identity.pullNumber,
    );

    const policy = branchPolicy(context.payload.pull_request);
    let parent;
    const matchesBranch = () => result.issues.some((issue) => issue.number === policy.issueNumber);
    if (!result.ok || (!policy.exempt && policy.issueNumber && !matchesBranch())) {
      parent = await readOpenParent(github, response, identity, context.serverUrl,
        policy.exempt ? null : policy.issueNumber);
      if (parent) result.issues.push(parent);
    }

    if (!result.issues.length) {
      await publishStatus(github, identity, targetUrl, "failure", "No closing issue or open same-repository parent is linked");
      core.setFailed(`pull request #${identity.pullNumber} has no same-repository closing issue or open parent reference`);
      return;
    }

    if (!policy.exempt && !matchesBranch()) {
      const expected = result.issues.map((issue) => `prismaquant-${issue.number}`).join(", ");
      await publishStatus(github, identity, targetUrl, "failure",
        "Branch rule: expected prismaquant-<issue> for a verified linked issue");
      core.setFailed(`Branch rule: expected prismaquant-<issue> or prismaquant-<issue>-<word>. ` +
        `Use ${expected}, with an optional lowercase suffix. Head branch: ${policy.branch}.`);
      return;
    }

    const linked = result.issues
      .map((issue) => `#${issue.number} (${String(issue.state).toLowerCase()})`)
      .join(", ");
    await publishStatus(
      github,
      identity,
      targetUrl,
      "success",
      result.ok ? `Closes same-repository issue #${result.issues[0].number}` :
        `Refs open same-repository parent #${parent.number}`,
    );
    core.info(`accepted issue reference(s): ${linked}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (identity && targetUrl) {
      try {
        await publishStatus(
          github,
          identity,
          targetUrl,
          "failure",
          "Could not verify a linked issue; gate failed closed",
        );
      } catch (publishError) {
        core.error(
          `could not publish the failure to pull request head ${identity.headSha}: ${String(publishError)}`,
        );
      }
    }
    core.setFailed(`linked-issue verification failed closed: ${message}`);
  }
}

module.exports = {
  CHECK_NAME,
  CLOSING_ISSUES_QUERY,
  PARENT_ISSUE_QUERY,
  TARGET_BRANCH,
  parentIssueNumbers,
  evaluateOpenParent,
  evaluateClosingIssues,
  publishStatus,
  pullRequestIdentity,
  run,
};
