import { spawn, spawnSync } from "node:child_process";
import { existsSync, promises as fs } from "node:fs";
import { createServer } from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test, expect } from "@playwright/test";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..", "..");
const frontendRoot = path.resolve(repoRoot, "frontend");

const realE2eEnabled = process.env.BATCHALIGN_REAL_SERVER_E2E === "1";
const dashboardDir =
  process.env.BATCHALIGN_DASHBOARD_DIR ?? path.join(frontendRoot, "dist");

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function resolveBinaryPath() {
  const candidates = process.env.BATCHALIGN_BIN
    ? [process.env.BATCHALIGN_BIN]
    : [
        path.join(repoRoot, "target", "debug", "batchalign3"),
        path.join(repoRoot, "target", "release", "batchalign3"),
      ];

  const found = candidates.find((candidate) => existsSync(candidate));
  if (!found) {
    throw new Error(`could not find batchalign3 binary; checked ${candidates.join(", ")}`);
  }
  return found;
}

function pythonImportsWorker(pythonPath) {
  const probe = spawnSync(pythonPath, ["-c", "import batchalign.worker"], {
    stdio: "ignore",
  });
  return probe.status === 0;
}

function resolvePythonPath() {
  if (process.env.BATCHALIGN_PYTHON) {
    return process.env.BATCHALIGN_PYTHON;
  }

  let cursor = repoRoot;
  for (;;) {
    const candidate = path.join(cursor, ".venv", "bin", "python3");
    if (existsSync(candidate) && pythonImportsWorker(candidate)) {
      return candidate;
    }
    const parent = path.dirname(cursor);
    if (parent === cursor) {
      break;
    }
    cursor = parent;
  }

  if (pythonImportsWorker("python3")) {
    return "python3";
  }

  throw new Error("could not resolve a Python runtime with batchalign.worker installed");
}

async function pickFreePort() {
  return await new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("failed to allocate localhost port"));
        return;
      }
      const { port } = address;
      server.close((err) => {
        if (err) {
          reject(err);
        } else {
          resolve(port);
        }
      });
    });
  });
}

async function waitForServerHealth(baseUrl, child, logs, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(
        [
          `server exited before healthy (exit ${child.exitCode})`,
          `stdout:\n${logs.stdout || "<empty>"}`,
          `stderr:\n${logs.stderr || "<empty>"}`,
        ].join("\n\n")
      );
    }

    try {
      const response = await fetch(`${baseUrl}/health`);
      if (response.ok) {
        return await response.json();
      }
    } catch {
      // Retry until timeout.
    }

    await sleep(250);
  }

  throw new Error(
    [
      `timed out waiting for ${baseUrl}/health`,
      `stdout:\n${logs.stdout || "<empty>"}`,
      `stderr:\n${logs.stderr || "<empty>"}`,
    ].join("\n\n")
  );
}

async function stopServer(child) {
  if (child.exitCode !== null || child.killed) {
    return;
  }

  child.kill("SIGTERM");
  const done = await Promise.race([
    new Promise((resolve) => child.once("exit", resolve)).then(() => true),
    sleep(5_000).then(() => false),
  ]);
  if (done) {
    return;
  }

  child.kill("SIGKILL");
  await new Promise((resolve) => child.once("exit", resolve));
}

async function startRealServer() {
  await fs.access(path.join(dashboardDir, "index.html"));

  const port = await pickFreePort();
  const homeDir = await fs.mkdtemp(path.join(os.tmpdir(), "batchalign-react-e2e-"));
  const configPath = path.join(homeDir, "server.yaml");
  const binaryPath = resolveBinaryPath();
  const workerPython = resolvePythonPath();

  await fs.writeFile(
    configPath,
    [
      "host: 127.0.0.1",
      `port: ${port}`,
      "warmup_commands: []",
      "max_concurrent_jobs: 2",
    ].join("\n"),
    "utf8"
  );

  const args = [
    "serve",
    "start",
    "--foreground",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
    "--config",
    configPath,
    "--python",
    workerPython,
  ];

  const logs = { stdout: "", stderr: "" };
  const child = spawn(binaryPath, args, {
    cwd: repoRoot,
    env: {
      ...process.env,
      HOME: homeDir,
      BATCHALIGN_DASHBOARD_DIR: dashboardDir,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  child.stdout.on("data", (chunk) => {
    logs.stdout += chunk.toString();
  });
  child.stderr.on("data", (chunk) => {
    logs.stderr += chunk.toString();
  });

  const baseUrl = `http://127.0.0.1:${port}`;
  const health = await waitForServerHealth(baseUrl, child, logs);

  return {
    baseUrl,
    child,
    logs,
    health,
    close: async () => {
      await stopServer(child);
      await fs.rm(homeDir, { recursive: true, force: true });
    },
  };
}

async function fetchJobInfo(request, baseUrl, jobId) {
  const response = await request.get(`${baseUrl}/jobs/${jobId}`);
  expect(response.ok()).toBeTruthy();
  return await response.json();
}

async function waitForJobStatus(request, baseUrl, jobId, predicate, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs;
  let lastStatus = "unknown";

  while (Date.now() < deadline) {
    const response = await request.get(`${baseUrl}/jobs/${jobId}`);
    if (response.ok()) {
      const info = await response.json();
      lastStatus = info.status;
      if (predicate(info.status, info)) {
        return info;
      }
    }
    await sleep(300);
  }

  throw new Error(`timed out waiting for job ${jobId}; last status=${lastStatus}`);
}

function makeChatFile(filename, utterance) {
  return {
    filename,
    content: [
      "@UTF8",
      "@Begin",
      "@Languages:\teng",
      "@Participants:\tPAR Participant",
      "@ID:\teng|test|PAR|||||Participant|||",
      `*PAR:\t${utterance} .`,
      "@End",
      "",
    ].join("\n"),
  };
}

function buildChatFiles(prefix, count) {
  return Array.from({ length: count }, (_, index) =>
    makeChatFile(
      `${prefix}-${String(index).padStart(2, "0")}.cha`,
      `hello from react dashboard sample ${index}`
    )
  );
}

function uniqueSourceDir(label) {
  return `/tmp/react-e2e-${label}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function submitMorphotagJob(request, baseUrl, { files, lang = "eng", sourceDir }) {
  const payload = {
    command: "morphotag",
    lang,
    num_speakers: 1,
    files,
    media_files: [],
    media_mapping: "",
    media_subdir: "",
    source_dir: sourceDir,
    options: {
      command: "morphotag",
      override_media_cache: true,
      retokenize: false,
      skipmultilang: false,
      merge_abbrev: false,
    },
    paths_mode: false,
    source_paths: [],
    output_paths: [],
    display_names: [],
    debug_traces: false,
    before_paths: [],
  };

  const response = await request.post(`${baseUrl}/jobs`, {
    data: payload,
    headers: { "content-type": "application/json" },
  });

  expect(response.ok()).toBeTruthy();
  return await response.json();
}

async function fetchJobResults(request, baseUrl, jobId) {
  const response = await request.get(`${baseUrl}/jobs/${jobId}/results`);
  expect(response.ok()).toBeTruthy();
  return await response.json();
}

async function cleanupJob(request, baseUrl, jobId) {
  const response = await request.get(`${baseUrl}/jobs/${jobId}`);
  if (response.status() === 404) {
    return;
  }

  expect(response.ok()).toBeTruthy();
  const info = await response.json();
  if (info.status === "queued" || info.status === "running") {
    const cancelResponse = await request.post(`${baseUrl}/jobs/${jobId}/cancel`);
    expect(cancelResponse.ok()).toBeTruthy();
    await waitForJobStatus(
      request,
      baseUrl,
      jobId,
      (status) => status !== "queued" && status !== "running",
      180_000
    );
  }

  const deleteResponse = await request.delete(`${baseUrl}/jobs/${jobId}`);
  expect([200, 404].includes(deleteResponse.status())).toBeTruthy();
}

test.describe("real Rust server e2e (React dashboard)", () => {
  test.describe.configure({ mode: "serial" });
  test.skip(
    !realE2eEnabled,
    "set BATCHALIGN_REAL_SERVER_E2E=1 to enable real-server dashboard tests"
  );

  let harness;
  const createdJobs = [];

  test.beforeAll(async () => {
    harness = await startRealServer();
    expect(harness.health.capabilities).toContain("morphotag");
  });

  test.afterEach(async ({ request }) => {
    while (createdJobs.length > 0) {
      const jobId = createdJobs.pop();
      await cleanupJob(request, harness.baseUrl, jobId);
    }
  });

  test.afterAll(async () => {
    await harness?.close();
  });

  test("real-model jobs submit, show progress, surface results, and delete cleanly", async ({
    page,
    request,
  }) => {
    test.setTimeout(720_000);
    const rootResponse = await request.get(`${harness.baseUrl}/`, { maxRedirects: 0 });
    expect(rootResponse.status()).toBe(307);

    const files = buildChatFiles("real-success", 6);
    const submitted = await submitMorphotagJob(request, harness.baseUrl, {
      files,
      sourceDir: uniqueSourceDir("success"),
    });
    const jobId = submitted.job_id;
    createdJobs.push(jobId);

    await waitForJobStatus(request, harness.baseUrl, jobId, (status) => status === "running");

    await page.goto(`${harness.baseUrl}/dashboard`, { waitUntil: "domcontentloaded" });
    const jobLink = page.locator(`a[href="/dashboard/jobs/${jobId}"]`);
    await expect(jobLink).toBeVisible({ timeout: 30_000 });
    await jobLink.click();
    await expect(page).toHaveURL(new RegExp(`${harness.baseUrl}/dashboard/jobs/${jobId}$`));

    await expect(page.getByText("Options")).toBeVisible();
    await page.getByRole("button", { name: "raw JSON" }).click();
    await expect(page.getByText('"override_media_cache": true')).toBeVisible();
    await expect
      .poll(async () => await page.locator(".progress-striped, .progress-indeterminate").count())
      .toBeGreaterThan(0);

    await waitForJobStatus(
      request,
      harness.baseUrl,
      jobId,
      (status) => status === "completed",
      600_000
    );
    await page.reload({ waitUntil: "domcontentloaded" });

    await expect(page.getByRole("button", { name: "Delete" })).toBeVisible();
    await expect(page.getByText("real-success-00.cha")).toBeVisible();

    const finalInfo = await fetchJobInfo(request, harness.baseUrl, jobId);
    expect(finalInfo.status).toBe("completed");
    expect(finalInfo.completed_files).toBe(files.length);

    const results = await fetchJobResults(request, harness.baseUrl, jobId);
    expect(results.status).toBe("completed");
    expect(results.files).toHaveLength(files.length);
    for (const result of results.files) {
      expect(result.error ?? null).toBeNull();
      expect(result.content).toContain("%mor:");
      expect(result.content).toContain("%gra:");
    }

    await page.getByRole("button", { name: "Delete" }).click();
    await expect(page).toHaveURL(new RegExp(`${harness.baseUrl}/dashboard/?$`));
    await expect(page.locator(`a[href="/dashboard/jobs/${jobId}"]`)).toHaveCount(0);

    const afterDelete = await request.get(`${harness.baseUrl}/jobs/${jobId}`);
    expect(afterDelete.status()).toBe(404);
  });

  test("failed real-model jobs surface errors and can be restarted from the dashboard", async ({
    page,
    request,
  }) => {
    test.setTimeout(360_000);
    const files = [makeChatFile("real-failure.cha", "this job should fail cleanly")];
    const submitted = await submitMorphotagJob(request, harness.baseUrl, {
      files,
      lang: "zzz",
      sourceDir: uniqueSourceDir("failure"),
    });
    const jobId = submitted.job_id;
    createdJobs.push(jobId);

    await page.goto(`${harness.baseUrl}/dashboard`, { waitUntil: "domcontentloaded" });
    const jobLink = page.locator(`a[href="/dashboard/jobs/${jobId}"]`);
    await expect(jobLink).toBeVisible({ timeout: 30_000 });
    await jobLink.click();
    await expect(page).toHaveURL(new RegExp(`${harness.baseUrl}/dashboard/jobs/${jobId}$`));

    await waitForJobStatus(
      request,
      harness.baseUrl,
      jobId,
      (status) => status === "failed",
      120_000
    );
    await page.reload({ waitUntil: "domcontentloaded" });

    await expect(page.getByRole("button", { name: "Restart" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Delete" })).toBeVisible();
    await expect(page.getByRole("cell", { name: "real-failure.cha", exact: true })).toBeVisible();
    await expect(page.locator("table").getByText("failed to parse ready signal")).toBeVisible();

    const failedResults = await fetchJobResults(request, harness.baseUrl, jobId);
    expect(failedResults.status).toBe("failed");
    expect(failedResults.files).toHaveLength(1);
    expect(failedResults.files[0].error).toContain("failed to parse ready signal");

    const restartResponse = page.waitForResponse(
      (response) =>
        response.url() === `${harness.baseUrl}/jobs/${jobId}/restart` &&
        response.request().method() === "POST" &&
        response.status() === 200
    );
    await page.getByRole("button", { name: "Restart" }).click();
    await restartResponse;

    await waitForJobStatus(
      request,
      harness.baseUrl,
      jobId,
      (status) => status === "queued" || status === "running",
      30_000
    );
    await waitForJobStatus(
      request,
      harness.baseUrl,
      jobId,
      (status) => status === "failed",
      120_000
    );
    await page.reload({ waitUntil: "domcontentloaded" });

    await expect(page.getByRole("button", { name: "Restart" })).toBeVisible();
    await expect(page.locator("table").getByText("failed to parse ready signal")).toBeVisible();

    await page.getByRole("button", { name: "Delete" }).click();
    await expect(page).toHaveURL(new RegExp(`${harness.baseUrl}/dashboard/?$`));
    await expect(page.locator(`a[href="/dashboard/jobs/${jobId}"]`)).toHaveCount(0);
  });
});
