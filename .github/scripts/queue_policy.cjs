"use strict";

const { requireIssuePolicy } = require("./linked_issue.cjs");

const BOT_LOGIN = "mergify[bot]";
const QUEUE_PREFIX = "mergify/merge-queue/main/";
const FLEET_CHECK = "fleet integrated CI";

function insist(condition, message) {
  if (!condition) throw new Error(message);
}

function sha(value) {
  insist(typeof value === "string" && /^[0-9a-f]{40}$/.test(value), "invalid SHA");
  return value;
}

function isQueueCandidate(pull) {
  return pull.head?.ref?.startsWith("mergify/") || pull.user?.login === BOT_LOGIN;
}

function pullBinding(pull, repository) {
  insist(pull.state === "open" && !pull.merged && pull.base?.ref === "main", "pull request is not open against main");
  insist(pull.head?.repo?.full_name === repository && pull.base?.repo?.full_name === repository,
    "fork or mismatched repository");
  insist(Number.isSafeInteger(pull.number) && pull.number > 0, "invalid pull request number");
  insist(typeof pull.draft === "boolean" && Number.isSafeInteger(pull.user?.id) && pull.user.id > 0 &&
    typeof pull.updated_at === "string" && Number.isFinite(Date.parse(pull.updated_at)), "missing fresh pull metadata");
  insist(typeof pull.head.ref === "string" && pull.head.ref.length > 0 && Array.isArray(pull.labels), "missing ref/labels");
  insist(pull.labels.every((label) => typeof label?.name === "string"), "malformed labels");
  return { number: pull.number, head: sha(pull.head.sha), base: sha(pull.base.sha), ref: pull.head.ref,
    draft: pull.draft, author: pull.user?.id, updated: pull.updated_at,
    labels: pull.labels.map((label) => label.name).sort() };
}

function sameBinding(before, after) {
  insist(JSON.stringify(before) === JSON.stringify(after), "pull request binding changed");
}

function batchBinding(response, pull) {
  insist(response && Array.isArray(response.batches) && Array.isArray(response.waiting_pull_requests),
    "missing complete Mergify status");
  insist(!response.next && !response.next_cursor && !response.has_more && !response.pageInfo?.hasNextPage,
    "paginated or truncated queue status");
  insist(response.mode === "serial" && response.batches.length === 1, "expected one serial batch");
  const batch = response.batches[0];
  insist(batch && typeof batch.id === "string" && /^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i.test(batch.id), "invalid batch identity");
  insist(batch.queue_pull_request_number === pull.number, "batch does not bind this queue pull request");
  insist(["default", "release"].includes(batch.queue_rule_name) && batch.status?.code === "running",
    "batch is not checking in an admitted queue");
  insist(Array.isArray(batch.pull_requests) && batch.pull_requests.length === 1,
    "missing, empty, truncated or unsupported multi-member batch");
  insist(Array.isArray(batch.parent_ids) && batch.parent_ids.length === 0 &&
    (batch.sub_batches == null || (Array.isArray(batch.sub_batches) && batch.sub_batches.length === 0)),
  "dependent or split batch is unsupported");
  const member = batch.pull_requests[0]?.number;
  insist(Number.isSafeInteger(member) && member > 0 && member !== pull.number, "invalid batch member");
  return { id: batch.id, queue: batch.queue_rule_name, ref: pull.head.ref, number: pull.number, member };
}

async function readQueueStatus(repository, token = process.env.MERGIFY_API_TOKEN, fetcher = fetch) {
  insist(typeof token === "string" && token.length > 0, "Mergify read credential unavailable");
  const response = await fetcher(`https://api.mergify.com/v1/repos/${repository}/merge-queue/status?branch=main`, {
    headers: { Authorization: `Bearer ${token}`, Accept: "application/json", "Cache-Control": "no-cache" },
    redirect: "error",
    signal: AbortSignal.timeout(15000),
  });
  insist(response.ok && !response.headers.get("link"), "Mergify status unavailable or paginated");
  return response.json();
}

async function readPull(github, context, number) {
  const { data } = await github.rest.pulls.get({ ...context.repo, pull_number: number });
  insist(data.number === number, "GitHub returned another pull request");
  return data;
}

async function readBase(github, context) {
  const { data } = await github.rest.repos.getBranch({ ...context.repo, branch: "main" });
  return sha(data.commit.sha);
}

async function requireMerge(github, context, integrated, base, member) {
  const { data } = await github.rest.repos.getCommit({ ...context.repo, ref: integrated });
  insist(data.sha === integrated && Array.isArray(data.parents) && data.parents.length === 2 &&
    data.parents[0].sha === base && data.parents[1].sha === member, "integrated SHA does not bind exact base/member parents");
}

async function verifyQueue({ github, context, pull, queueStatus = readQueueStatus }) {
  const repository = `${context.repo.owner}/${context.repo.repo}`;
  const eventBinding = pullBinding(pull, repository);
  const current = await readPull(github, context, pull.number);
  sameBinding(eventBinding, pullBinding(current, repository));
  const { data: bot } = await github.rest.users.getByUsername({ username: BOT_LOGIN });
  insist(bot.login === BOT_LOGIN && bot.type === "Bot" && Number.isSafeInteger(bot.id) && bot.id > 0 &&
    current.user?.login === BOT_LOGIN && current.user?.type === "Bot" && current.user?.id === bot.id &&
    current.head.ref.startsWith(QUEUE_PREFIX), "queue author/ref is not authenticated Mergify");
  const before = batchBinding(await queueStatus(repository), current);
  const base = await readBase(github, context);
  insist(eventBinding.base === base, "queue base is stale");
  const member = await readPull(github, context, before.member);
  const memberBinding = pullBinding(member, repository);
  insist(!isQueueCandidate(member) && memberBinding.base === base && !member.draft, "invalid or stale original member");
  insist((before.queue === "release") === member.labels?.some((label) => label.name === "release"), "member queue label mismatch");
  await requireIssuePolicy(github, { ...context, payload: { pull_request: member } });
  await requireMerge(github, context, eventBinding.head, base, memberBinding.head);
  sameBinding(memberBinding, pullBinding(await readPull(github, context, member.number), repository));
  sameBinding(eventBinding, pullBinding(await readPull(github, context, pull.number), repository));
  insist(await readBase(github, context) === base, "main advanced during verification");
  sameBinding(before, batchBinding(await queueStatus(repository), current));
  return { pull: eventBinding, base, integrated: eventBinding.head, batch: before, member: memberBinding };
}

async function verifyCandidate({ github, context, number, expectedHead, queueStatus }) {
  const repository = `${context.repo.owner}/${context.repo.repo}`;
  const pull = await readPull(github, context, number);
  const binding = pullBinding(pull, repository);
  insist(binding.head === sha(expectedHead), "trigger head is stale");
  if (isQueueCandidate(pull)) return verifyQueue({ github, context, pull, queueStatus });
  insist(!pull.draft, "draft pull requests are not admitted to fleet");
  await requireIssuePolicy(github, { ...context, payload: { pull_request: pull } });
  const base = await readBase(github, context);
  insist(binding.base === base, "pull request base is stale");
  const integrated = sha(pull.merge_commit_sha);
  const { data: mergeRef } = await github.rest.repos.getCommit({ ...context.repo, ref: `refs/pull/${number}/merge` });
  insist(mergeRef.sha === integrated, "GitHub merge ref SHA mismatch");
  await requireMerge(github, context, integrated, base, binding.head);
  sameBinding(binding, pullBinding(await readPull(github, context, number), repository));
  insist(await readBase(github, context) === base, "main advanced during verification");
  return { pull: binding, base, integrated };
}

function workflowTrigger(run, context, base) {
  const repository = `${context.repo.owner}/${context.repo.repo}`;
  insist(run.repository?.full_name === repository && run.head_repository?.full_name === repository &&
    run.path === ".github/workflows/linked-issue.yml" && run.event === "pull_request_target" &&
    run.status === "completed" && run.conclusion === "success",
  "untrusted or stale policy workflow run");
  insist(Array.isArray(run.pull_requests) && run.pull_requests.length === 1, "missing complete workflow PR association");
  const pull = run.pull_requests[0];
  insist(Number.isSafeInteger(pull.number) && pull.number > 0 && pull.head?.sha === sha(run.head_sha) &&
    pull.base?.sha === base, "workflow PR/head/base association mismatch");
  return { number: pull.number, expectedHead: run.head_sha };
}

module.exports = { FLEET_CHECK, isQueueCandidate, pullBinding, batchBinding, readQueueStatus,
  verifyQueue, verifyCandidate, workflowTrigger, readBase };
