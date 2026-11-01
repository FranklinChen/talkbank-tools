// affects: frontend/src/**
// affects: frontend/dist/**
import { createServer } from "node:http";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test, expect } from "@playwright/test";
import { WebSocketServer } from "ws";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendRoot = path.resolve(__dirname, "..", "..");
const buildDir = path.join(frontendRoot, "dist");

const contentTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".ico", "image/x-icon"],
  [".png", "image/png"],
]);

function withCors(headers = {}) {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,DELETE,OPTIONS",
    "access-control-allow-headers": "content-type",
    ...headers,
  };
}

function makeFileStatus(filename, status, overrides = {}) {
  return {
    filename,
    status,
    error: null,
    error_category: null,
    started_at: status === "done" ? "2026-02-25T00:00:01.000Z" : null,
    finished_at: status === "done" ? "2026-02-25T00:00:02.000Z" : null,
    progress_current: status === "processing" ? 1 : null,
    progress_total: status === "processing" ? 4 : null,
    progress_stage: status === "processing" ? "analyzing_morphosyntax" : null,
    progress_label: status === "processing" ? "Analyzing morphosyntax" : null,
    ...overrides,
  };
}

function buildFailedJobFileStatuses() {
  const files = [];

  for (let index = 0; index < 42; index += 1) {
    const dir = index < 24 ? "alpha" : "beta";
    files.push(makeFileStatus(`${dir}/sample-${String(index).padStart(2, "0")}.cha`, "done"));
  }

  for (let index = 42; index < 54; index += 1) {
    const sample = String(index).padStart(2, "0");
    files.push(
      makeFileStatus(`beta/sample-${sample}.cha`, "error", {
        error: [
          `Validation mismatch for sample-${sample}.cha: parser lost speaker alignment while rebuilding the CHAT AST after retokenization.`,
          `Second line detail for sample-${sample}.cha explains the exact mismatch and preserves the full diagnostic payload.`,
        ].join("\n"),
        error_category: "validation",
      })
    );
  }

  files.push(
    makeFileStatus("gamma/sample-54.cha", "error", {
      error: "CHAT declaration is missing an @ID tier for speaker PAR.",
      error_category: "parse_error",
    }),
    makeFileStatus("gamma/sample-55.cha", "error", {
      error: "CHAT declaration is missing an @ID tier for speaker PAR.",
      error_category: "parse_error",
    }),
    makeFileStatus("gamma/sample-56.cha", "error", {
      error: "Referenced media file missing from media root.",
      error_category: "input_missing",
    }),
    makeFileStatus("gamma/sample-57.cha", "error", {
      error: "Referenced media file missing from media root.",
      error_category: "input_missing",
    }),
    makeFileStatus("gamma/sample-58.cha", "error", {
      error: "SQLite store returned a synthetic write failure while flushing the result bundle.",
      error_category: "system",
    }),
    makeFileStatus("gamma/sample-59.cha", "error", {
      error: "Morphosyntax worker exited unexpectedly after returning a malformed payload.",
      error_category: "worker_crash",
    })
  );

  return files;
}

function makeJob(jobId, status, { completed, total, submittedAt, fileStatuses, error, sourceDir }) {
  const resolvedFileStatuses =
    fileStatuses ??
    [
      makeFileStatus(
        `${jobId}.cha`,
        status === "completed"
          ? "done"
          : status === "failed"
            ? "error"
            : status === "running"
              ? "processing"
              : "queued",
        {
          error:
            status === "failed"
              ? "synthetic failure while analyzing this file: tokenization and morphosyntax pipeline both reported inconsistent state for dashboard testing."
              : null,
          error_category: status === "failed" ? "worker_crash" : null,
          progress_current: completed,
          progress_total: total,
        }
      ),
    ];
  const completedFiles =
    completed ?? resolvedFileStatuses.filter((file) => file.status === "done").length;
  const totalFiles = total ?? resolvedFileStatuses.length;
  const errorFiles = resolvedFileStatuses.filter((file) => file.status === "error").length;

  return {
    job_id: jobId,
    status,
    command: "morphotag",
    options: {
      command: "morphotag",
      retokenize: false,
      skipmultilang: false,
      merge_abbrev: false,
      override_cache: false,
    },
    lang: "eng",
    source_dir: sourceDir ?? "/tmp/source",
    total_files: totalFiles,
    completed_files: completedFiles,
    error_files: errorFiles,
    submitted_at: submittedAt,
    completed_at:
      status === "completed" || status === "failed" ? "2026-02-25T00:10:00Z" : null,
    duration_s: status === "completed" || status === "failed" ? 2.4 : null,
    num_workers: 2,
    error: error ?? (status === "failed" ? "synthetic failure" : null),
    file_statuses: resolvedFileStatuses,
  };
}

function listItem(job) {
  return {
    job_id: job.job_id,
    status: job.status,
    command: job.command,
    lang: job.lang,
    source_dir: job.source_dir,
    total_files: job.total_files,
    completed_files: job.completed_files,
    error_files: job.error_files,
    submitted_at: job.submitted_at,
    completed_at: job.completed_at,
    duration_s: job.duration_s,
    num_workers: job.num_workers,
    submitted_by: "tester",
    submitted_by_name: "Test User",
  };
}

function json(res, status, payload) {
  const body = Buffer.from(JSON.stringify(payload), "utf-8");
  res.writeHead(status, withCors({ "content-type": "application/json; charset=utf-8" }));
  res.end(body);
}

async function staticFileResponse(pathname) {
  const cleaned = pathname === "/" ? "/index.html" : pathname;
  const relative = decodeURIComponent(cleaned).replace(/^\/+/, "");
  const fullPath = path.resolve(buildDir, relative);
  if (!fullPath.startsWith(buildDir)) {
    return { status: 403, body: Buffer.from("forbidden"), type: "text/plain; charset=utf-8" };
  }

  try {
    const body = await fs.readFile(fullPath);
    const type = contentTypes.get(path.extname(fullPath)) ?? "application/octet-stream";
    return { status: 200, body, type };
  } catch {
    const indexPath = path.join(buildDir, "index.html");
    const body = await fs.readFile(indexPath);
    return { status: 200, body, type: "text/html; charset=utf-8" };
  }
}

async function startHarness() {
  await fs.access(path.join(buildDir, "index.html"));

  const submissions = [];
  const jobs = new Map([
    [
      "runjob",
      makeJob("runjob", "running", {
        completed: 1,
        total: 4,
        submittedAt: "2026-02-25T00:00:00Z",
      }),
    ],
    [
      "donejob",
      makeJob("donejob", "completed", {
        completed: 2,
        total: 2,
        submittedAt: "2026-02-25T00:05:00Z",
      }),
    ],
    [
      "diagjob",
      // A transcription whose second file was written with admission
      // diagnostics: the job completed, the file is neither done nor failed.
      makeJob("diagjob", "completed", {
        submittedAt: "2026-02-25T00:07:00Z",
        fileStatuses: [
          makeFileStatus("clean.cha", "done"),
          // Clean, with a note align left out on purpose: the wire shape of
          // `OutputExclusionRecord`, information beside a done file.
          makeFileStatus("noted.cha", "done", {
            exclusions: [
              {
                kind: "not_in_recording",
                excluded_utterances: 2,
                excluded_words: 9,
                first_excluded: [
                  {
                    utterance: 3,
                    main_tier: "*MOT:\twent to the park . [+ diary]",
                    words: 5,
                    postcode: "diary",
                  },
                ],
              },
            ],
            // The server's review (`FileReview`), rendered in Rust and
            // pinned there; opaque text here, because the dashboard only
            // shows what it is sent.
            review: {
              left_out: { headline: "LEFT-OUT-HEADLINE", lines: ["LEFT-OUT-LINE-1", "  LEFT-OUT-LINE-2"] },
            },
          }),
          makeFileStatus("noisy.cha", "diagnosed", {
            started_at: "2026-02-25T00:00:01.000Z",
            finished_at: "2026-02-25T00:00:02.000Z",
            // The wire shape of `FileOutputDiagnostics`: the judged
            // findings (bar, count, first findings, where the full list
            // is) and the shortfalls.
            diagnostics: {
              findings: {
                bar: "construction",
                finding_count: 2,
                findings_by_code: [
                  { code: "E220", count: 1 },
                  { code: "E241", count: 1 },
                ],
                first_findings: [
                  { code: "E220", level: "main_tier_valid", message: "a digit inside a word" },
                  {
                    code: "E241",
                    level: "main_tier_valid",
                    message: "a reserved marker in the wrong case",
                  },
                ],
                full_findings: { kind: "inline" },
              },
              shortfalls: [{ kind: "stage_skipped", stage: "morphosyntax" }],
            },
            // The server's review, opaque here (see the note above).
            review: {
              needs_review: { headline: "NEEDS-REVIEW-HEADLINE", lines: ["REVIEW-LINE-1", "  REVIEW-LINE-2"] },
            },
          }),
        ],
      }),
    ],
    [
      "failjob",
      makeJob("failjob", "failed", {
        submittedAt: "2026-02-25T00:08:00Z",
        fileStatuses: buildFailedJobFileStatuses(),
        error: "18 files failed validation and processing checks.",
        sourceDir: "/tmp/failure-case",
      }),
    ],
  ]);

  let baseUrl = "";
  const sockets = new Set();
  const serverSockets = new Set();

  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://localhost");

    if (req.method === "OPTIONS") {
      res.writeHead(204, withCors());
      res.end();
      return;
    }

    if (url.pathname === "/health" && req.method === "GET") {
      json(res, 200, {
        status: "ok",
        version: "test",
        capabilities: ["morphotag"],
        workers_available: 2,
      });
      return;
    }

    if (url.pathname === "/fleet" && req.method === "GET") {
      json(res, 200, [{ name: "local", url: baseUrl }]);
      return;
    }

    if (url.pathname === "/jobs" && req.method === "POST") {
      let raw = "";
      for await (const chunk of req) raw += chunk;
      const submission = JSON.parse(raw);
      submissions.push(submission);
      // Exercise the real command/language admission contract rather than
      // fabricating a successful job for a malformed native-client request.
      const perFile = ["morphotag", "translate"].includes(submission.command);
      if (perFile !== (submission.lang === "per-file")) {
        json(res, 400, { detail: "This command must use the matching job/per-file language policy." });
        return;
      }
      const files = submission.source_paths.map((source) => makeFileStatus(source, "done"));
      const job = makeJob("desktopjob", "completed", {
        completed: files.length, total: files.length, fileStatuses: files,
        submittedAt: "2026-02-25T00:00:00Z", sourceDir: submission.source_dir,
      });
      jobs.set(job.job_id, job);
      json(res, 200, job);
      return;
    }
    if (url.pathname === "/jobs" && req.method === "GET") {
      json(res, 200, Array.from(jobs.values()).map(listItem));
      return;
    }

    const cancelMatch = url.pathname.match(/^\/jobs\/([^/]+)\/cancel$/);
    if (cancelMatch && req.method === "POST") {
      const job = jobs.get(cancelMatch[1]);
      if (!job) {
        json(res, 404, { status: "not_found", message: "job not found" });
        return;
      }
      job.status = "cancelled";
      job.error = "cancelled by user";
      broadcast({ type: "job_update", job: listItem(job) });
      json(res, 200, { status: "ok", message: "cancelled" });
      return;
    }

    const restartMatch = url.pathname.match(/^\/jobs\/([^/]+)\/restart$/);
    if (restartMatch && req.method === "POST") {
      const job = jobs.get(restartMatch[1]);
      if (!job) {
        json(res, 404, { status: "not_found", message: "job not found" });
        return;
      }
      job.status = "queued";
      job.error = null;
      job.completed_files = 0;
      job.file_statuses = [
        {
          ...job.file_statuses[0],
          status: "queued",
          error: null,
          error_category: null,
          progress_current: 0,
          progress_stage: null,
          progress_label: null,
        },
      ];
      broadcast({ type: "job_update", job: listItem(job) });
      json(res, 200, job);
      return;
    }

    const outputFoldersMatch = url.pathname.match(/^\/jobs\/([^/]+)\/output-folders$/);
    if (outputFoldersMatch && req.method === "GET") {
      json(res, jobs.has(outputFoldersMatch[1]) ? 200 : 404, {
        job_id: outputFoldersMatch[1], access: { state: "not_recorded" },
      });
      return;
    }

    const jobMatch = url.pathname.match(/^\/jobs\/([^/]+)$/);
    if (jobMatch && req.method === "GET") {
      const job = jobs.get(jobMatch[1]);
      if (!job) {
        json(res, 404, { status: "not_found", message: "job not found" });
        return;
      }
      json(res, 200, job);
      return;
    }

    if (jobMatch && req.method === "DELETE") {
      const existed = jobs.delete(jobMatch[1]);
      if (!existed) {
        json(res, 404, { status: "not_found", message: "job not found" });
        return;
      }
      broadcast({ type: "job_deleted", job_id: jobMatch[1] });
      json(res, 200, { status: "ok", message: "deleted" });
      return;
    }

    const file = await staticFileResponse(url.pathname);
    res.writeHead(file.status, withCors({ "content-type": file.type }));
    res.end(file.body);
  });

  server.on("connection", (socket) => {
    serverSockets.add(socket);
    socket.on("close", () => serverSockets.delete(socket));
  });

  const wsServer = new WebSocketServer({ noServer: true });

  function broadcast(event) {
    const payload = JSON.stringify(event);
    for (const ws of sockets) {
      if (ws.readyState === 1) {
        ws.send(payload);
      }
    }
  }

  server.on("upgrade", (req, socket, head) => {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    if (url.pathname !== "/ws") {
      socket.destroy();
      return;
    }
    wsServer.handleUpgrade(req, socket, head, (ws) => {
      wsServer.emit("connection", ws);
    });
  });

  wsServer.on("connection", (ws) => {
    sockets.add(ws);
    ws.on("close", () => sockets.delete(ws));
    ws.on("message", (raw) => {
      if (raw.toString() === "ping") {
        try {
          ws.send("pong");
        } catch {
          // Ignore races with socket close.
        }
      }
    });

    ws.send(
      JSON.stringify({
        type: "snapshot",
        jobs: Array.from(jobs.values()).map(listItem),
        health: { status: "ok" },
      })
    );
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });

  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("failed to resolve server address");
  }
  baseUrl = `http://127.0.0.1:${address.port}`;

  return {
    baseUrl,
    submissions,
    close: async () => {
      for (const ws of sockets) {
        try {
          ws.terminate();
        } catch {
          // Ignore races with already-closed sockets.
        }
      }
      sockets.clear();
      await new Promise((resolve) => wsServer.close(resolve));
      await new Promise((resolve, reject) => {
        server.close((err) => {
          if (err) {
            reject(err);
          } else {
            resolve();
          }
        });
        for (const socket of serverSockets) {
          socket.destroy();
        }
      });
    },
  };
}

async function installDesktopTauriStub(page, commandResponses) {
  await page.addInitScript((responses) => {
    const queuedResponses = new Map(
      Object.entries(responses).map(([command, value]) => [
        command,
        Array.isArray(value) ? [...value] : [value],
      ]),
    );
    const invokeCalls = [];
    const callbacks = new Map();
    const eventListeners = new Map();
    const heldReplies = new Map();
    let nextCallbackId = 1;
    let nextEventId = 1;

    function cloneResponse(value) {
      if (value === null || value === undefined) {
        return value;
      }
      if (typeof value === "object") {
        return JSON.parse(JSON.stringify(value));
      }
      return value;
    }

    function consumeResponse(command) {
      const queue = queuedResponses.get(command);
      if (!queue || queue.length === 0) {
        throw new Error(`Unexpected Tauri invoke: ${command}`);
      }
      const value = queue.length > 1 ? queue.shift() : queue[0];
      if (value?.nativeError) throw new Error(value.nativeError);
      if (value?.nativeGate) return new Promise(resolve => {
        heldReplies.set(value.nativeGate, () => resolve(cloneResponse(value.response)));
      });
      if (value?.nativeDelayMs) return new Promise(resolve => setTimeout(() => resolve(cloneResponse(value.response)), value.nativeDelayMs));
      return cloneResponse(value);
    }

    function removeEventListener(event, eventId) {
      const listeners = eventListeners.get(event) ?? [];
      eventListeners.set(
        event,
        listeners.filter((listener) => listener.eventId !== eventId),
      );
    }

    window.__desktopInvokeCalls = invokeCalls;
    window.__hasHeldDesktopReply = key => heldReplies.has(key);
    // Deterministic transport scheduling: tests release the exact reply whose
    // ownership is under review, without relying on wall-clock delay windows.
    window.__releaseDesktopReply = key => {
      const release = heldReplies.get(key);
      if (!release) throw new Error(`No held desktop reply: ${key}`);
      heldReplies.delete(key);
      release();
    };
    window.__emitDesktopEvent = (event, payload) => {
      for (const listener of eventListeners.get(event) ?? []) {
        const callback = callbacks.get(listener.callbackId);
        callback?.({ event, id: listener.eventId, payload });
      }
    };
    window.__TAURI_INTERNALS__ = {
      invoke(command, args) {
        invokeCalls.push({ command, args: args ?? null });
        if (command === "admit_job_paths" && !queuedResponses.has(command)) {
          // Positive wire fixture only. Real filesystem policy is exercised by
          // native tests; never duplicate that policy in this browser double.
          const proposal = args.proposal;
          const normalize = value => value.replace(/\\/g, "/").replace(/\/+$/, "");
          const root = normalize(proposal.source_root);
          const output = proposal.destination.mode === "in_place" ? root : normalize(proposal.destination.folder);
          const audio = ["transcribe", "benchmark"].includes(proposal.command);
          return { source_root: proposal.source_root, inputs: proposal.inputs, output_root: output,
            outputs: proposal.inputs.map(source => {
              const relative = normalize(source).slice(root.length + 1);
              return output + "/" + (audio ? relative.replace(/\.[^/.]+$/, ".cha") : relative);
            }),
          };
        }
        if (command === "plugin:event|listen") {
          const eventId = nextEventId++;
          const listeners = eventListeners.get(args.event) ?? [];
          listeners.push({ eventId, callbackId: args.handler });
          eventListeners.set(args.event, listeners);
          return eventId;
        }
        if (command === "plugin:event|unlisten") {
          removeEventListener(args.event, args.eventId);
          return null;
        }
        return consumeResponse(command);
      },
      transformCallback(callback) {
        const id = nextCallbackId++;
        callbacks.set(id, callback);
        return id;
      },
      unregisterCallback(id) {
        callbacks.delete(id);
      },
    };
    window.__TAURI_EVENT_PLUGIN_INTERNALS__ = {
      unregisterListener(event, eventId) {
        removeEventListener(event, eventId);
      },
    };
  }, commandResponses);
}

async function readDesktopInvokeCalls(page) {
  return page.evaluate(() => window.__desktopInvokeCalls);
}

async function installOutputFolderDesktop(page, harness, responses = {}) {
  await installDesktopTauriStub(page, {
    server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
    is_first_launch: false,
    open_output_directory: null,
    ...responses,
  });
}

test("desktop reopened job opens persisted outputs across reload and page-origin differences", async ({ page }, testInfo) => {
  const api = await startHarness();
  const frontend = await startHarness();
  const reads = [];
  try {
    await installOutputFolderDesktop(page, api);
    await page.route("**/jobs/donejob/output-folders", async route => {
      reads.push(route.request().url());
      await route.fulfill({ headers: withCors(), json: { job_id: "donejob", access: { state: "recorded", directories: ["/recorded/results"] } } });
    });
    await page.goto(frontend.baseUrl + "/dashboard/jobs/donejob");
    await page.getByRole("button", { name: "Open Output Folder", exact: true }).click();
    await expect.poll(() => readDesktopInvokeCalls(page)).toEqual(expect.arrayContaining([
      { command: "open_output_directory", args: { path: "/recorded/results" } },
    ]));
    await page.screenshot({ path: testInfo.outputPath("reopened-output-folders.png") });
    await page.reload();
    await page.getByRole("button", { name: "Open Output Folder", exact: true }).click();
    await expect.poll(() => readDesktopInvokeCalls(page)).toEqual(expect.arrayContaining([
      { command: "open_output_directory", args: { path: "/recorded/results" } },
    ]));
    expect(reads).toEqual([api.baseUrl + "/jobs/donejob/output-folders", api.baseUrl + "/jobs/donejob/output-folders"]);
    expect(api.submissions).toHaveLength(0);
  } finally { await Promise.all([api.close(), frontend.close()]); }
});

test("desktop output folders keep distinct recorded directories instead of guessing a common root", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installOutputFolderDesktop(page, harness);
    await page.route("**/jobs/donejob/output-folders", route => route.fulfill({ json: {
      job_id: "donejob", access: { state: "recorded", directories: ["/outputs/one", "/elsewhere/two", "/outputs/one"] },
    } }));
    await page.goto(harness.baseUrl + "/dashboard/jobs/donejob");
    await expect(page.getByRole("button", { name: /^Open Output Folder / })).toHaveCount(2);
    await page.getByRole("button", { name: "Open Output Folder /outputs/one", exact: true }).click();
    await page.getByRole("button", { name: "Open Output Folder /elsewhere/two", exact: true }).click();
    await expect.poll(async () => (await readDesktopInvokeCalls(page)).filter(call => call.command === "open_output_directory")).toEqual([
      { command: "open_output_directory", args: { path: "/outputs/one" } },
      { command: "open_output_directory", args: { path: "/elsewhere/two" } },
    ]);
  } finally { await harness.close(); }
});

for (const [state, message] of [
  ["content_mode", "This job used uploaded content; it has no recorded local output folder."],
  ["not_recorded", "No output folder was recorded for this job."],
]) {
  test(`desktop output folders explain ${state} without borrowing the input location`, async ({ page }) => {
    const harness = await startHarness();
    try {
      await installOutputFolderDesktop(page, harness);
      await page.route("**/jobs/donejob/output-folders", route => route.fulfill({ json: { job_id: "donejob", access: { state } } }));
      await page.goto(harness.baseUrl + "/dashboard/jobs/donejob");
      await expect(page.getByText(message, { exact: true })).toBeVisible();
      await expect(page.getByRole("button", { name: /Open Output Folder/ })).toHaveCount(0);
      expect((await readDesktopInvokeCalls(page)).filter(call => call.command === "open_output_directory")).toEqual([]);
    } finally { await harness.close(); }
  });
}

for (const [name, body] of [
  ["wrong job", { job_id: "other-job", access: { state: "recorded", directories: ["/wrong-job"] } }],
  ["empty population", { job_id: "donejob", access: { state: "recorded", directories: [] } }],
  ["empty path", { job_id: "donejob", access: { state: "recorded", directories: [""] } }],
  ["wrong path type", { job_id: "donejob", access: { state: "recorded", directories: [123] } }],
  ["unsupported state", { job_id: "donejob", access: { state: "future", directories: ["/guessed"] } }],
  ["stored refusal", { job_id: "donejob", access: { state: "refused", message: "The stored output mapping is invalid." } }],
]) {
  test(`desktop output lookup rejects ${name} before native opening`, async ({ page }) => {
    const harness = await startHarness();
    try {
      await installOutputFolderDesktop(page, harness);
      await page.route("**/jobs/donejob/output-folders", route => route.fulfill({ json: body }));
      await page.goto(harness.baseUrl + "/dashboard/jobs/donejob");
      await expect(page.getByRole("alert")).toBeVisible();
      await expect(page.getByRole("button", { name: /Open Output Folder/ })).toHaveCount(0);
      expect((await readDesktopInvokeCalls(page)).filter(call => call.command === "open_output_directory")).toEqual([]);
    } finally { await harness.close(); }
  });
}

test("desktop output lookup has visible read failure and explicit retry", async ({ page }) => {
  const harness = await startHarness();
  let reads = 0;
  try {
    await installOutputFolderDesktop(page, harness);
    await page.route("**/jobs/donejob/output-folders", route => {
      reads++;
      return reads === 1 ? route.fulfill({ status: 503, json: { detail: "Output records temporarily unavailable." } })
        : route.fulfill({ json: { job_id: "donejob", access: { state: "recorded", directories: ["/recorded/retry"] } } });
    });
    await page.goto(harness.baseUrl + "/dashboard/jobs/donejob");
    await expect(page.getByRole("alert")).toContainText("Output records temporarily unavailable.");
    expect(reads).toBe(1);
    await page.getByRole("button", { name: "Retry output lookup", exact: true }).click();
    await expect(page.getByRole("button", { name: "Open Output Folder", exact: true })).toBeVisible();
    expect(reads).toBe(2);
  } finally { await harness.close(); }
});

test("desktop output opener reports a missing folder and can explicitly retry", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installOutputFolderDesktop(page, harness, { open_output_directory: [{ nativeError: "The output folder no longer exists." }, null] });
    await page.route("**/jobs/donejob/output-folders", route => route.fulfill({ json: {
      job_id: "donejob", access: { state: "recorded", directories: ["/recorded/moved"] },
    } }));
    await page.goto(harness.baseUrl + "/dashboard/jobs/donejob");
    await page.getByRole("button", { name: "Open Output Folder", exact: true }).click();
    await expect(page.getByRole("alert")).toContainText("The output folder no longer exists.");
    await page.getByRole("button", { name: "Open Output Folder", exact: true }).click();
    await expect(page.getByRole("alert")).toHaveCount(0);
    await expect.poll(async () => (await readDesktopInvokeCalls(page)).filter(call => call.command === "open_output_directory").length).toBe(2);
  } finally { await harness.close(); }
});

test("desktop output opening is single-flight and its old completion cannot settle another job", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installOutputFolderDesktop(page, harness, { open_output_directory: [{ nativeGate: "old-output", response: null }, null] });
    await page.route("**/jobs/*/output-folders", route => {
      const id = new URL(route.request().url()).pathname.split("/")[2];
      return route.fulfill({ json: { job_id: id, access: { state: "recorded", directories: ["/recorded/" + id] } } });
    });
    await page.goto(harness.baseUrl + "/dashboard/jobs/donejob");
    await page.getByRole("button", { name: "Open Output Folder", exact: true }).dblclick();
    await expect(page.getByRole("button", { name: "Open Output Folder", exact: true })).toBeDisabled();
    await expect.poll(async () => (await readDesktopInvokeCalls(page)).filter(call => call.command === "open_output_directory").length).toBe(1);
    // Retain the document while changing the router input, so the old held
    // transport really can complete after its React owner has been retired.
    await page.evaluate(() => {
      window.history.pushState(null, "", "/dashboard/jobs/failjob");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    await expect(page.getByText("/recorded/failjob", { exact: true })).toBeVisible();
    await releaseDesktopReply(page, "old-output");
    await expect(page.getByRole("button", { name: "Open Output Folder", exact: true })).toBeEnabled();
    await page.getByRole("button", { name: "Open Output Folder", exact: true }).click();
    await expect.poll(async () => (await readDesktopInvokeCalls(page)).filter(call => call.command === "open_output_directory")).toEqual([
      { command: "open_output_directory", args: { path: "/recorded/donejob" } },
      { command: "open_output_directory", args: { path: "/recorded/failjob" } },
    ]);
  } finally { await harness.close(); }
});

test("desktop remote job cannot borrow the local output opener", async ({ page }) => {
  const local = await startHarness();
  const remote = await startHarness();
  let reads = 0;
  try {
    await installOutputFolderDesktop(page, local);
    await page.route("**/jobs/remotejob", route => route.request().url().startsWith(local.baseUrl)
      ? route.fulfill({ status: 404, json: { detail: "Job not found" } })
      : route.fulfill({ headers: withCors(), json: makeJob("remotejob", "completed", { completed: 1, total: 1, submittedAt: "2026-02-25T00:05:00Z" }) }));
    await page.route("**/jobs/remotejob/output-folders", route => { reads++; return route.abort(); });
    await page.goto(local.baseUrl + "/dashboard/jobs/remotejob?servers=" + new URL(remote.baseUrl).host);
    await expect(page.getByText("Output folders can be opened on the processing computer.", { exact: true })).toBeVisible();
    expect(reads).toBe(0);
    expect((await readDesktopInvokeCalls(page)).filter(call => call.command === "open_output_directory")).toEqual([]);
  } finally { await Promise.all([local.close(), remote.close()]); }
});

test("browser dashboard does not expose or query native output folders", async ({ page }) => {
  const harness = await startHarness();
  let reads = 0;
  try {
    await page.route("**/jobs/donejob/output-folders", route => { reads++; return route.abort(); });
    await page.goto(harness.baseUrl + "/dashboard/jobs/donejob");
    await expect(page.getByRole("button", { name: "Delete", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: /Open Output Folder/ })).toHaveCount(0);
    expect(reads).toBe(0);
  } finally { await harness.close(); }
});

async function releaseDesktopReply(page, key) {
  // A rendered "choosing" state can precede the adapter's dynamic import and
  // native invoke. Observe the actual held transport before releasing it.
  await expect.poll(() => page.evaluate(key => window.__hasHeldDesktopReply(key), key)).toBe(true);
  await page.evaluate(key => window.__releaseDesktopReply(key), key);
}

/** An independently controlled job record: action replies never change it. */
async function showRunningDesktopJob(page, harness, observe = () => "running") {
  await installDesktopTauriStub(page, {
    server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
    is_first_launch: false, "plugin:dialog|open": ["/input", "/output"], discover_files: [["/input/a.cha"]],
  });
  const job = status => makeJob("canceljob", status, {
    total: 1, completed: status === "completed" ? 1 : 0, sourceDir: "/input",
    fileStatuses: [makeFileStatus("/input/a.cha", status === "completed" ? "done" : "processing")],
  });
  let submitted = 0;
  await page.route("**/jobs", async route => {
    if (route.request().method() !== "POST") return route.continue();
    submitted++;
    await route.fulfill({ status: 200, json: job("running") });
  });
  await page.route("**/jobs/canceljob", async route => {
    const observation = observe();
    if (observation === "unavailable") await route.fulfill({ status: 503, json: { detail: "Progress temporarily unavailable" } });
    else await route.fulfill({ status: 200, json: job(observation) });
  });
  await page.goto(harness.baseUrl + "/process");
  await page.getByRole("button", { name: /Add Grammar/ }).click();
  await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
  await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
  await page.getByRole("button", { name: "Start Processing", exact: true }).click();
  await expect(page.getByRole("button", { name: "Cancel", exact: true })).toBeVisible();
  return { submitted: () => submitted };
}

async function emitDesktopEvent(page, event, payload) {
  await page.evaluate(
    ({ name, eventPayload }) => {
      window.__emitDesktopEvent(name, eventPayload);
    },
    { name: event, eventPayload: payload },
  );
}

test("dashboard list/detail/actions work against the mock harness", async ({ page, request }) => {
  const harness = await startHarness();
  try {
    await page.goto(`${harness.baseUrl}/dashboard`, { waitUntil: "domcontentloaded" });

    const runningCard = page.locator('a[href="/dashboard/jobs/runjob"]');
    const completedCard = page.locator('a[href="/dashboard/jobs/donejob"]');

    await expect(runningCard).toBeVisible();
    await expect(completedCard).toBeVisible();

    await runningCard.click();
    await expect(page).toHaveURL(new RegExp(`${harness.baseUrl}/dashboard/jobs/runjob$`));
    await expect(page.getByText("Options")).toBeVisible();
    await expect(page.getByText("morphotag")).toBeVisible();
    await expect(page.getByRole("button", { name: "Cancel" })).toBeVisible();
    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(page.getByRole("button", { name: "Restart" })).toBeVisible();

    const cancelled = await request.get(`${harness.baseUrl}/jobs/runjob`);
    expect(cancelled.ok()).toBeTruthy();
    expect((await cancelled.json()).status).toBe("cancelled");

    await page.getByRole("button", { name: "Restart" }).click();
    await expect(page.getByRole("button", { name: "Cancel" })).toBeVisible();

    const restarted = await request.get(`${harness.baseUrl}/jobs/runjob`);
    expect(restarted.ok()).toBeTruthy();
    expect((await restarted.json()).status).toBe("queued");

    await page.goto(`${harness.baseUrl}/dashboard/jobs/donejob`, {
      waitUntil: "domcontentloaded",
    });
    await expect(page.getByRole("button", { name: "Delete" })).toBeVisible();
    await page.getByRole("button", { name: "Delete" }).click();

    await expect(page).toHaveURL(new RegExp(`${harness.baseUrl}/dashboard/?$`));
    await expect(page.locator('a[href="/dashboard/jobs/donejob"]')).toHaveCount(0);

    const deleted = await request.get(`${harness.baseUrl}/jobs/donejob`);
    expect(deleted.status()).toBe(404);
  } finally {
    await harness.close();
  }
});

test("a file written with diagnostics renders as written output, not as an error", async ({ page }) => {
  const harness = await startHarness();
  try {
    await page.goto(`${harness.baseUrl}/dashboard/jobs/diagjob`, {
      waitUntil: "domcontentloaded",
    });
    await expect(page.getByText("noisy.cha")).toBeVisible();
    // The server's review is shown as sent: the dashboard keeps no copy of
    // the wording.
    await expect(page.getByText("NEEDS-REVIEW-HEADLINE").first()).toBeVisible();
    // Opening the file shows each detail line under the headline.
    await page.getByText("NEEDS-REVIEW-HEADLINE").first().click();
    await expect(page.getByText("REVIEW-LINE-2", { exact: false }).first()).toBeVisible();
    await expect(page.getByRole("tab", { name: /Diagnosed/ })).toBeVisible();
    await expect(page.getByRole("tab", { name: /Errors/ })).toBeDisabled();
    // A done file's exclusions are listed as information, not as a fault.
    await expect(
      page.getByText("LEFT-OUT-HEADLINE").first(),
    ).toBeVisible();
    // Opening it lists what was left out.
    await page.getByText("LEFT-OUT-HEADLINE").first().click();
    await expect(page.getByText("LEFT-OUT-LINE-2", { exact: false }).first()).toBeVisible();
  } finally {
    await harness.close();
  }
});

test("web runtime keeps root navigation on the browser dashboard surface", async ({ page }) => {
  const harness = await startHarness();
  try {
    await page.goto(harness.baseUrl, { waitUntil: "domcontentloaded" });

    await expect(page.locator('a[href="/dashboard/jobs/runjob"]')).toBeVisible();
    await expect(page.locator('a[href="/process"]')).toHaveCount(0);
    await expect(page.getByRole("link", { name: "Dashboard", exact: true })).toBeVisible();
  } finally {
    await harness.close();
  }
});

test("desktop runtime gates first launch through the config seam", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: true,
      server_status: {
        state: "ready", url: harness.baseUrl, build_hash: "test-build",
      },
      write_config: { message: "saved" },
    });

    await page.goto(`${harness.baseUrl}/?server=${encodeURIComponent(harness.baseUrl)}`, {
      waitUntil: "domcontentloaded",
    });

    await expect(page.getByRole("heading", { name: "Welcome to BA3 Runner" })).toBeVisible();
    await expect(page).toHaveTitle("BA3 Runner");
    await page.getByRole("button", { name: "Let's Get Started" }).click();
    await page.getByRole("button", { name: "Finish Setup" }).click();
    await page.getByRole("button", { name: "Get Started" }).click();

    await expect(page).toHaveURL(/\/process(?:\?|$)/);
    await expect(page.getByText("What would you like to do?")).toBeVisible();

    const calls = await readDesktopInvokeCalls(page);
    expect(calls).toEqual(
      expect.arrayContaining([
        { command: "is_first_launch", args: {} },
        {
          command: "write_config",
          args: {
            config: {
              engine: "whisper",
              rev_key: null,
            },
          },
        },
      ]),
    );
  } finally {
    await harness.close();
  }
});

test("desktop settings retire a pending read before reopening without editable defaults", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: false, server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      read_config: [{ nativeGate: "old-settings", response: { engine: "rev", rev_key: "old-value" } }, { engine: "whisper", rev_key: null }],
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    let dialog = page.getByRole("dialog", { name: "Settings" });
    await expect(dialog.getByText("Reading settings...", { exact: true })).toBeVisible();
    await expect(dialog.getByRole("button", { name: "Save", exact: true })).toHaveCount(0);
    await expect(dialog.locator("input")).toHaveCount(0);
    await expect.poll(() => page.evaluate(() => window.__hasHeldDesktopReply("old-settings"))).toBe(true);
    await dialog.getByRole("button", { name: "Close settings" }).click();
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    dialog = page.getByRole("dialog", { name: "Settings" });
    await expect(dialog.getByRole("button", { name: /^Whisper/ })).toHaveAttribute("aria-pressed", "true");
    await releaseDesktopReply(page, "old-settings");
    await expect(dialog.getByRole("button", { name: /^Whisper/ })).toHaveAttribute("aria-pressed", "true");
    await expect(dialog.getByRole("button", { name: /^Rev.AI/ })).toHaveAttribute("aria-pressed", "false");
    expect((await readDesktopInvokeCalls(page)).filter(call => call.command === "write_config")).toHaveLength(0);
  } finally { await harness.close(); }
});

test("desktop settings read failure permits retry without exposing a write or losing keyboard focus", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: false, server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      read_config: [{ nativeError: "Permission denied" }, { engine: "whisper", rev_key: null }],
    });
    await page.goto(harness.baseUrl + "/process");
    const opener = page.getByRole("button", { name: "Settings", exact: true });
    await opener.click();
    const dialog = page.getByRole("dialog", { name: "Settings" });
    await expect(dialog.getByRole("alert")).toContainText("Permission denied");
    await expect(dialog.getByRole("button", { name: "Save", exact: true })).toHaveCount(0);
    await dialog.getByRole("button", { name: "Retry reading settings" }).click();
    await expect(dialog.getByRole("button", { name: "Save", exact: true })).toBeVisible();
    await page.keyboard.press("Tab");
    await expect(dialog.getByRole("button", { name: "Close settings" })).toBeFocused();
    await page.keyboard.press("Shift+Tab");
    await expect(dialog.getByRole("button", { name: "Reload settings" })).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(opener).toBeFocused();
    expect((await readDesktopInvokeCalls(page)).some(call => call.command === "write_config")).toBe(false);
  } finally { await harness.close(); }
});

test("desktop settings reject incomplete read transport instead of inventing defaults", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: false, server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" }, read_config: {},
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Settings" });
    await expect(dialog.getByRole("alert")).toContainText("incomplete settings");
    await expect(dialog.getByRole("button", { name: "Save", exact: true })).toHaveCount(0);
  } finally { await harness.close(); }
});

test("desktop settings keep unsupported engines explicit until a supported choice", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: false, server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      read_config: { engine: "other-engine", rev_key: null }, write_config: { message: "saved" },
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Settings" });
    await expect(dialog.getByRole("status")).toContainText("Your configured engine is other-engine");
    await expect(dialog.getByRole("button", { name: "Save", exact: true })).toBeDisabled();
    await expect(dialog.getByRole("button", { name: /^Whisper/ })).toHaveAttribute("aria-pressed", "false");
    await dialog.getByRole("button", { name: /^Whisper/ }).click();
    await dialog.getByRole("button", { name: "Save", exact: true }).click();
    await expect(dialog.getByText("Settings saved.", { exact: true })).toBeVisible();
    expect((await readDesktopInvokeCalls(page)).filter(call => call.command === "write_config")[0].args.config).toEqual({ engine: "whisper", rev_key: null });
  } finally { await harness.close(); }
});

test("desktop settings save once, retain untouched credentials and retire successful replacement intent", async ({ page }, testInfo) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: false, server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      read_config: { engine: "rev", rev_key: "stored-before-load" },
      write_config: [{ nativeGate: "save-settings", response: { message: "saved" } }, { message: "saved" }],
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Settings" });
    const key = dialog.getByLabel("Rev.AI API key", { exact: true });
    await expect(key).toHaveValue("");
    await expect(dialog.getByText("Settings saved.", { exact: true })).toHaveCount(0);
    await page.screenshot({ path: testInfo.outputPath("settings-retain-key.png") });
    await dialog.getByRole("button", { name: "Save", exact: true }).dblclick();
    await expect(dialog.getByRole("button", { name: "Close settings" })).toBeDisabled();
    await expect(dialog.getByRole("button", { name: /^Whisper/ })).toBeDisabled();
    await page.keyboard.press("Escape");
    await expect(dialog).toBeVisible();
    await expect.poll(async () => (await readDesktopInvokeCalls(page)).filter(call => call.command === "write_config").length).toBe(1);
    await releaseDesktopReply(page, "save-settings");
    await expect(dialog.getByText("Settings saved.", { exact: true })).toBeVisible();
    await key.fill("replacement-value");
    await expect(dialog.getByText("Settings saved.", { exact: true })).toHaveCount(0);
    await dialog.getByRole("button", { name: "Save", exact: true }).click();
    await expect(dialog.getByText("Settings saved.", { exact: true })).toBeVisible();
    await expect(key).toHaveValue("");
    await dialog.getByRole("button", { name: "Save", exact: true }).click();
    await expect(dialog.getByText("Settings saved.", { exact: true })).toBeVisible();
    await dialog.getByRole("button", { name: /^Whisper/ }).click();
    await dialog.getByRole("button", { name: "Save", exact: true }).click();
    await expect(dialog.getByText("Settings saved.", { exact: true })).toBeVisible();
    const edits = (await readDesktopInvokeCalls(page)).filter(call => call.command === "write_config").map(call => call.args.config);
    expect(edits).toEqual([
      { engine: "rev", rev_key: null }, { engine: "rev", rev_key: "replacement-value" },
      { engine: "rev", rev_key: null }, { engine: "whisper", rev_key: null },
    ]);
  } finally { await harness.close(); }
});

test("desktop settings reject a control character in a replacement before native write", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: false, server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      read_config: { engine: "rev", rev_key: null },
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Settings" });
    await dialog.getByLabel("Rev.AI API key", { exact: true }).fill("bad\u0001value");
    await dialog.getByRole("button", { name: "Save", exact: true }).click();
    await expect(dialog.getByRole("alert")).toContainText("nonempty single line");
    expect((await readDesktopInvokeCalls(page)).some(call => call.command === "write_config")).toBe(false);
  } finally { await harness.close(); }
});

test("desktop settings require a save acknowledgement and can reload the actual configuration", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: false, server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      read_config: [{ engine: "rev", rev_key: "old-value" }, { engine: "whisper", rev_key: null }], write_config: {},
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Settings" });
    await dialog.getByRole("button", { name: "Save", exact: true }).click();
    await expect(dialog.getByRole("alert")).toContainText("did not confirm");
    await expect(dialog.getByText("Settings saved.", { exact: true })).toHaveCount(0);
    await dialog.getByRole("button", { name: "Reload settings" }).click();
    await expect(dialog.getByRole("button", { name: /^Whisper/ })).toHaveAttribute("aria-pressed", "true");
    await expect(dialog.getByText("Settings saved.", { exact: true })).toHaveCount(0);
  } finally { await harness.close(); }
});

for (const failure of [{ nativeError: "Configuration write denied" }, {}]) {
  test("desktop first-run save failure remains actionable without premature completion: " + JSON.stringify(failure), async ({ page }) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        is_first_launch: true, server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        write_config: [failure, { nativeGate: "retry-setup", response: { message: "saved" } }],
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: "Let's Get Started" }).click();
      await page.getByRole("button", { name: "Finish Setup", exact: true }).click();
      await expect(page.getByRole("alert")).toContainText("Could not confirm setup saved");
      await expect(page.getByRole("button", { name: "Get Started", exact: true })).toHaveCount(0);
      await page.getByRole("button", { name: "Finish Setup", exact: true }).dblclick();
      await expect(page.getByText("Saving setup...", { exact: true })).toBeVisible();
      await expect(page.getByRole("button", { name: "Back", exact: true })).toBeDisabled();
      await expect.poll(async () => (await readDesktopInvokeCalls(page)).filter(call => call.command === "write_config").length).toBe(2);
      await releaseDesktopReply(page, "retry-setup");
      await expect(page.getByRole("button", { name: "Get Started", exact: true })).toBeVisible();
    } finally { await harness.close(); }
  });
}

test("desktop first-run Rev.AI confirmation belongs to its credential step", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: true, server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" }, write_config: { message: "saved" },
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: "Let's Get Started" }).click();
    await page.getByRole("button", { name: /^Rev.AI/ }).click();
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await page.getByLabel("API key", { exact: true }).fill("replacement-value");
    await page.getByLabel("Confirm API key", { exact: true }).fill("different-value");
    await page.getByRole("button", { name: "Save & Finish", exact: true }).click();
    await expect(page.getByRole("alert")).toContainText("Keys do not match");
    expect((await readDesktopInvokeCalls(page)).some(call => call.command === "write_config")).toBe(false);
    await page.getByRole("button", { name: "Back", exact: true }).click();
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await expect(page.getByLabel("API key", { exact: true })).toHaveValue("replacement-value");
    await page.getByLabel("Confirm API key", { exact: true }).fill("replacement-value");
    await page.getByRole("button", { name: "Save & Finish", exact: true }).click();
    await expect(page.getByText("Default ASR engine: Rev.AI", { exact: true })).toBeVisible();
    expect((await readDesktopInvokeCalls(page)).filter(call => call.command === "write_config").map(call => call.args.config)).toEqual([{ engine: "rev", rev_key: "replacement-value" }]);
  } finally { await harness.close(); }
});

for (const failure of [{ nativeError: "Configuration permission denied" }, "invalid-check"]) {
  test("desktop first-launch check can be retried without inventing absence: " + JSON.stringify(failure), async ({ page }) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        is_first_launch: [failure, false], server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      });
      await page.goto(harness.baseUrl + "/process");
      await expect(page.getByRole("alert")).toContainText("Could not read configuration");
      await expect(page.getByRole("heading", { name: "Welcome to BA3 Runner" })).toHaveCount(0);
      await page.getByRole("button", { name: "Retry configuration check" }).click();
      await expect(page.getByRole("button", { name: /Add Grammar/ })).toBeVisible();
      expect((await readDesktopInvokeCalls(page)).some(call => call.command === "write_config")).toBe(false);
    } finally { await harness.close(); }
  });
}

test("desktop process flow routes folder discovery through the file seam", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: false,
      server_status: {
        state: "ready", url: harness.baseUrl, build_hash: "test-build",
      },
      "plugin:dialog|open": ["/Users/test/input", "/Users/test/output"],
      discover_files: [[
        "/Users/test/input/alpha.cha",
        "/Users/test/input/nested/beta.cha",
      ]],
    });

    await page.goto(`${harness.baseUrl}/process?server=${encodeURIComponent(harness.baseUrl)}`, {
      waitUntil: "domcontentloaded",
    });

    await expect(page.getByText("What would you like to do?")).toBeVisible();
    await page.getByRole("button", { name: /Add Grammar/i }).click();
    await expect(page.getByRole("button", { name: /Click to choose a folder/i })).toBeVisible();
    await page.getByRole("button", { name: /Click to choose a folder/i }).click();
    await expect(page.getByText("2 files found")).toBeVisible();
    await expect(page.getByText("/Users/test/input", { exact: true })).toBeVisible();

    await expect(page.getByRole("button", { name: /Click to choose output folder/i })).toBeVisible();
    await page.getByRole("button", { name: /Click to choose output folder/i }).click();
    await expect(page.getByText("/Users/test/output")).toBeVisible();

    const calls = await readDesktopInvokeCalls(page);
    expect(calls).toEqual(
      expect.arrayContaining([
        {
          command: "plugin:dialog|open",
          args: {
            options: {
              directory: true,
              multiple: false,
              title: "Select Add Grammar input folder",
            },
          },
        },
        {
          command: "discover_files",
          args: {
            dir: "/Users/test/input",
            extensions: ["cha"],
          },
        },
        {
          command: "plugin:dialog|open",
          args: {
            options: {
              directory: true,
              multiple: false,
              title: "Choose output folder",
            },
          },
        },
      ]),
    );
  } finally {
    await harness.close();
  }
});

test("desktop starts only a missing managed service and never offers to stop it", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      is_first_launch: false,
      server_status: { state: "stopped" },
      start_server: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
    });

    await page.goto(
      `${harness.baseUrl}/process?server=${encodeURIComponent("http://127.0.0.1:9")}`,
      { waitUntil: "domcontentloaded" },
    );

    await expect(page.getByText("What would you like to do?")).toBeVisible();

    const calls = await readDesktopInvokeCalls(page);
    expect(calls).toEqual(
      expect.arrayContaining([
        { command: "start_server", args: {} },
      ]),
    );

    await expect(page.getByText("Processing service ready", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Stop", exact: true })).toHaveCount(0);
  } finally {
    await harness.close();
  }
});

test("dashboard groups errors, paginates files, and filters failed job details", async ({
  page,
}) => {
  const harness = await startHarness();
  try {
    await page.goto(`${harness.baseUrl}/dashboard`, { waitUntil: "domcontentloaded" });

    const failedCard = page.locator('a[href="/dashboard/jobs/failjob"]');
    await expect(failedCard).toBeVisible();
    await failedCard.click();

    await expect(page).toHaveURL(new RegExp(`${harness.baseUrl}/dashboard/jobs/failjob$`));
    await expect(page.getByRole("button", { name: "Restart" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Delete" })).toBeVisible();
    await expect(page.getByText("18 files failed validation and processing checks.")).toBeVisible();
    await expect(page.getByText("18 errors")).toBeVisible();
    // Labelled by each error's own category: a refused input is the
    // input's to fix, never reported as a pipeline bug.
    await expect(
      page.getByText(
        "The input was refused before any output was written. Each file's error below names what to change in it."
      )
    ).toBeVisible();
    await expect(page.getByText(/Input Refused \(12\)/)).toBeVisible();
    await expect(page.getByText("This is a pipeline bug, not your input.")).toHaveCount(0);

    const validationGroup = page
      .locator("button")
      .filter({ hasText: "Validation mismatch for sample-42.cha" })
      .first();
    await validationGroup.click();
    await expect(page.getByRole("button", { name: "sample-42.cha", exact: true })).toBeVisible();

    await expect(page.getByText("Showing 1-50 of 60")).toBeVisible();
    await page.getByRole("button", { name: "Next" }).first().click();
    await expect(page.getByText("Showing 51-60 of 60")).toBeVisible();
    await expect(page.getByText("sample-58.cha")).toBeVisible();

    const search = page.getByPlaceholder("Search files...");
    await search.fill("SAMPLE-03");
    await expect(page.getByText("sample-03.cha")).toBeVisible();
    await expect(page.getByRole("cell", { name: "sample-58.cha", exact: true })).toHaveCount(0);

    await search.fill("");
    // Wait for the full file list to re-render after clearing the search
    // filter, so the Errors tab count is non-zero and the button is enabled.
    await expect(page.getByText("Showing 1-50 of 60")).toBeVisible();
    await page.getByRole("tab", { name: /Errors/ }).click();
    await expect(page.getByText("sample-03.cha")).toHaveCount(0);
    await expect(page.getByRole("cell", { name: "sample-52.cha", exact: true })).toBeVisible();

    await page.getByRole("button", { name: "sample-52.cha", exact: true }).click();
    await expect(
      page.locator("pre").filter({
        hasText:
          "Second line detail for sample-52.cha explains the exact mismatch and preserves the full diagnostic payload.",
      }),
    ).toBeVisible();
  } finally {
    await harness.close();
  }
});

test("desktop refuses incompatible CLI/service state before showing job controls", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, { server_status: { state: "refused", message: "Different builds; update the managed service." } });
    await page.goto(harness.baseUrl);
    await expect(page.getByRole("alert")).toContainText("Different builds");
    await expect(page.getByRole("button", { name: /Transcribe Audio/ })).toHaveCount(0);
    expect((await readDesktopInvokeCalls(page)).some(call => call.command === "start_server")).toBe(false);
  } finally { await harness.close(); }
});

test("desktop submits one paired plan, observes completion and opens its actual output", async ({ page }) => {
  const harness = await startHarness();
  const frontend = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false,
      "plugin:dialog|open": ["/input", "/output"],
      discover_files: [["/input/nested/alpha.cha"]],
      "open_output_directory": null,
    });
    // The API is deliberately NOT the page origin. Native root navigation
    // must retain the admitted endpoint after the router drops query params.
    await page.goto(frontend.baseUrl);
    await expect(page).toHaveURL(frontend.baseUrl + "/process");
    await page.getByRole("button", { name: /Add Grammar/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await expect(page.getByRole("button", { name: "Start Processing", exact: true })).toBeDisabled();
    await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
    await page.getByRole("button", { name: "Start Processing", exact: true }).dblclick();
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
    expect(harness.submissions).toHaveLength(1);
    expect(frontend.submissions).toHaveLength(0);
    expect(harness.submissions[0]).toMatchObject({
      command: "morphotag", lang: "per-file", source_dir: "/input",
      source_paths: ["/input/nested/alpha.cha"], output_paths: ["/output/nested/alpha.cha"],
      options: { command: "morphotag" },
    });
    await page.getByRole("button", { name: "Open Output Folder" }).click();
    await expect.poll(() => readDesktopInvokeCalls(page)).toEqual(expect.arrayContaining([
      { command: "open_output_directory", args: expect.objectContaining({ path: "/output" }) },
    ]));
  } finally { await Promise.all([harness.close(), frontend.close()]); }
});

for (const refusal of [
  { body: JSON.stringify({ detail: "The selected file is outside the admitted input folder." }), message: "The selected file is outside the admitted input folder." },
  { status: 409, body: JSON.stringify({ detail: { message: "An active job already owns this output." } }), message: "An active job already owns this output." },
  { status: 403, body: JSON.stringify({ detail: "Local filesystem access is not allowed." }), message: "Local filesystem access is not allowed." },
  { body: "Request refused by the local service.", message: "Request refused by the local service." },
  { body: "", message: "The server returned no readable explanation." },
]) {
  test("desktop preserves submission refusal: " + refusal.message, async ({ page }) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        is_first_launch: false,
        "plugin:dialog|open": ["/input", "/output"],
        discover_files: [["/input/alpha.cha"]],
      });
      await page.route("**/jobs", async route => {
        if (route.request().method() === "POST") await route.fulfill({ status: refusal.status ?? 400, body: refusal.body });
        else await route.continue();
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: /Add Grammar/ }).click();
      await expect(page.getByRole("textbox", { name: "Language", exact: true })).toHaveCount(0);
      await expect(page.getByText("Uses each transcript’s @Languages header.")).toBeVisible();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await page.getByRole("button", { name: "Start Processing", exact: true }).click();
      await expect(page.getByRole("alert")).toContainText(refusal.message);
      await expect(page.getByRole("button", { name: "Start Processing", exact: true })).toBeEnabled();
      await expect(page.getByRole("button", { name: "Open Output Folder" })).toHaveCount(0);
    } finally { await harness.close(); }
  });
}

for (const response of [
  { kind: "json", body: {} },
  { kind: "json", body: { job_id: 19 } },
  { kind: "json", body: { job_id: "../other" } },
  { kind: "unreadable" },
  { kind: "http", status: 503 },
  { kind: "http", status: 408 },
  { kind: "lost" },
]) {
  test("desktop keeps a possibly accepted submission unconfirmed: " + JSON.stringify(response), async ({ page }, testInfo) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        is_first_launch: false, "plugin:dialog|open": ["/input", "/output"], discover_files: [["/input/a.cha"]],
      });
      const progressReads = [];
      page.on("request", request => {
        if (/\/jobs\/[^/]+$/.test(new URL(request.url()).pathname)) progressReads.push(request.url());
      });
      await page.route("**/jobs", async route => {
        if (route.request().method() !== "POST") return route.continue();
        // The service really accepts the request before its response is lost
        // or damaged. A client must not turn that uncertainty into another POST.
        await route.fetch();
        switch (response.kind) {
          case "json": await route.fulfill({ status: 200, json: response.body }); break;
          case "unreadable": await route.fulfill({ status: 200, contentType: "application/json", body: "{broken" }); break;
          case "http": await route.fulfill({ status: response.status, json: { detail: "The response was interrupted after submission." } }); break;
          case "lost": await route.abort("failed"); break;
        }
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: /Add Grammar/ }).click();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await page.getByRole("button", { name: "Start Processing", exact: true }).dblclick();
      const recovery = page.getByRole("region", { name: "Unconfirmed submission" });
      await expect(recovery).toBeVisible();
      await expect(recovery).toContainText("The service may already have started this job");
      await expect(recovery).toContainText("Add Grammar — 1 input file");
      await expect(recovery.getByText("/input", { exact: true })).toBeVisible();
      await expect(recovery.getByText("/output", { exact: true })).toBeVisible();
      await expect(recovery.getByRole("link", { name: "Open task list" })).toHaveAttribute("href", "/dashboard");
      await expect(page.getByRole("button", { name: "Start Processing", exact: true })).toHaveCount(0);
      await expect(page.getByRole("button", { name: "Open Output Folder" })).toHaveCount(0);
      await expect(page.getByRole("button", { name: "Retry progress" })).toHaveCount(0);
      expect(harness.submissions).toHaveLength(1);
      expect(progressReads).toEqual([]);
      if (response.kind === "lost") await page.screenshot({ path: testInfo.outputPath("unconfirmed-submission.png"), fullPage: true });
      // Leaving recovery is a deliberate new configuration, not retrying the
      // captured plan. No POST occurs just because that action is chosen.
      await recovery.getByRole("button", { name: "I checked the tasks; configure another job" }).click();
      await expect(page.getByRole("button", { name: /Add Grammar/ })).toBeVisible();
      expect(harness.submissions).toHaveLength(1);
    } finally { await harness.close(); }
  });
}

test("desktop follows a confirmed job identity without inventing progress from its acknowledgement", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false, "plugin:dialog|open": ["/input", "/output"], discover_files: [["/input/a.cha"]],
    });
    await page.route("**/jobs", async route => {
      if (route.request().method() !== "POST") return route.continue();
      const accepted = await (await route.fetch()).json();
      await route.fulfill({ status: 200, json: { job_id: accepted.job_id } });
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Add Grammar/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
    await page.getByRole("button", { name: "Start Processing", exact: true }).click();
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
    expect(harness.submissions).toHaveLength(1);
    await expect(page.getByRole("region", { name: "Unconfirmed submission" })).toHaveCount(0);
  } finally { await harness.close(); }
});

for (const scenario of [
  { command: "align", label: /Align to Audio/, input: "/input/a.cha", output: "/output/a.cha", lang: "fra" },
  { command: "translate", label: /^Translate/, input: "/input/a.cha", output: "/output/a.cha", lang: "per-file" },
  { command: "utseg", label: /Segment Utterances/, input: "/input/a.cha", output: "/output/a.cha", lang: "spa" },
  { command: "benchmark", label: /Score Accuracy/, input: "/input/clip.wav", output: "/output/clip.cha", lang: "eng" },
]) {
  test("desktop basic command forwards its own inputs and parameters: " + scenario.command, async ({ page }) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        is_first_launch: false, "plugin:dialog|open": ["/input", "/output"], discover_files: [[scenario.input]],
        ...(scenario.command === "benchmark" ? { read_config: { engine: "whisper", rev_key: null } } : {}),
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: scenario.label }).click();
      const language = page.getByRole("textbox", { name: "Language", exact: true });
      if (scenario.lang === "per-file") await expect(language).toHaveCount(0);
      else await language.fill(scenario.lang);
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      if (scenario.command === "benchmark") {
        await expect(page.getByRole("button", { name: "Modify in place", exact: true })).toBeDisabled();
        await expect(page.getByText("Choose recordings with matching reference transcripts beside them", { exact: false })).toBeVisible();
        await page.getByRole("spinbutton", { name: "Number of speakers", exact: true }).fill("2");
      } else await expect(page.getByRole("spinbutton", { name: "Number of speakers", exact: true })).toHaveCount(0);
      await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await page.getByRole("button", { name: "Start Processing", exact: true }).click();
      await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
      expect(harness.submissions).toHaveLength(1);
      expect(harness.submissions[0]).toMatchObject({
        command: scenario.command, lang: scenario.lang, source_dir: "/input",
        source_paths: [scenario.input], output_paths: [scenario.output],
        options: { command: scenario.command },
      });
      const preferenceReads = (await readDesktopInvokeCalls(page)).filter(call => call.command === "read_config");
      if (scenario.command === "benchmark") {
        expect(harness.submissions[0]).toMatchObject({ num_speakers: 2, options: { asr_engine: "whisper" } });
        expect(preferenceReads).toHaveLength(1);
      } else {
        expect(harness.submissions[0].options).not.toHaveProperty("asr_engine");
        expect(preferenceReads).toHaveLength(0);
      }
    } finally { await harness.close(); }
  });
}

test("desktop preferences failure before submission permits correction without an uncertain POST", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false, read_config: { nativeError: "Could not read engine preferences" },
      "plugin:dialog|open": ["/input", "/output"], discover_files: [["/input/clip.wav"]],
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Transcribe Audio/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
    await page.getByRole("button", { name: "Start Processing", exact: true }).click();
    await expect(page.getByRole("alert")).toContainText("Could not read engine preferences");
    await expect(page.getByRole("button", { name: "Start Processing", exact: true })).toBeEnabled();
    await expect(page.getByRole("region", { name: "Unconfirmed submission" })).toHaveCount(0);
    expect(harness.submissions).toHaveLength(0);
  } finally { await harness.close(); }
});

for (const policy of [
  { command: /Add Grammar/, description: "CHAT transcripts", extensions: ["cha"] },
  { command: /Transcribe Audio/, description: "audio or video recordings", extensions: ["wav", "mp3", "mp4", "m4a", "flac", "ogg", "webm"] },
]) {
  test("desktop explains empty discovery for " + policy.description, async ({ page }) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        is_first_launch: false, "plugin:dialog|open": ["/empty", "/output"], discover_files: [[]],
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: policy.command }).click();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      await expect(page.getByRole("button", { name: /0 files found.*\/empty/ })).toBeVisible();
      await expect(page.getByRole("status").filter({ hasText: `No ${policy.description} found` })).toBeVisible();
      await expect(page.getByText("Looks for " + policy.description + ":", { exact: false })).toContainText(policy.extensions.map(ext => "." + ext).join(", "));
      await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await expect(page.getByRole("button", { name: "Start Processing", exact: true })).toBeDisabled();
      expect(harness.submissions).toHaveLength(0);
      expect(await readDesktopInvokeCalls(page)).toEqual(expect.arrayContaining([
        { command: "discover_files", args: { dir: "/empty", extensions: policy.extensions } },
      ]));
      expect((await readDesktopInvokeCalls(page)).some(call => call.command === "admit_job_paths")).toBe(false);
    } finally { await harness.close(); }
  });
}

for (const invalid of [null, [""]]) {
  test("desktop rejects incomplete discovery before it becomes a selected population: " + JSON.stringify(invalid), async ({ page }) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        is_first_launch: false, "plugin:dialog|open": "/input", discover_files: [invalid],
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: /Add Grammar/ }).click();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      await expect(page.getByRole("alert")).toContainText("incomplete file list");
      await expect(page.getByRole("button", { name: "Start Processing", exact: true })).toBeDisabled();
      await expect(page.getByText("Review 1 input file", { exact: true })).toHaveCount(0);
      expect(harness.submissions).toHaveLength(0);
    } finally { await harness.close(); }
  });
}

test("desktop reviews a bounded page while submitting the complete discovered population", async ({ page }, testInfo) => {
  const harness = await startHarness();
  const inputs = Array.from({ length: 51 }, (_, index) => `/input/nested/clip-${String(index + 1).padStart(2, "0")}.cha`);
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false, "plugin:dialog|open": ["/input", "/output"], discover_files: [inputs],
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Add Grammar/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await page.getByText("Review 51 input files", { exact: true }).focus();
    await page.keyboard.press("Enter");
    const review = page.getByRole("region", { name: "Input files to process" });
    await expect(review.getByText("Showing 1–25 of 51.", { exact: false })).toBeVisible();
    await expect(review.getByRole("listitem")).toHaveCount(25);
    await expect(review.getByRole("button", { name: "Previous files" })).toBeDisabled();
    await page.screenshot({ path: testInfo.outputPath("input-file-review.png"), fullPage: true });
    await review.getByRole("button", { name: "Next files" }).click();
    await expect(review.getByText(inputs[25], { exact: true })).toBeVisible();
    await expect(review.getByRole("listitem")).toHaveCount(25);
    await review.getByRole("button", { name: "Next files" }).click();
    await expect(review.getByText("Showing 51–51 of 51.", { exact: false })).toBeVisible();
    await expect(review.getByRole("listitem")).toHaveCount(1);
    await expect(review.getByText(inputs[50], { exact: true })).toBeVisible();
    await expect(review.getByRole("button", { name: "Next files" })).toBeDisabled();
    await review.getByRole("button", { name: "Previous files" }).click();
    await expect(review.getByText("Showing 26–50 of 51.", { exact: false })).toBeVisible();
    await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
    await page.getByRole("button", { name: "Start Processing", exact: true }).click();
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
    expect(harness.submissions).toHaveLength(1);
    expect(harness.submissions[0].source_paths).toEqual(inputs);
  } finally { await harness.close(); }
});

test("desktop retires a file review page when the same folder is discovered again", async ({ page }) => {
  const harness = await startHarness();
  const inputs = Array.from({ length: 26 }, (_, index) => `/input/old-${index}.cha`);
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false, "plugin:dialog|open": "/input", discover_files: [inputs, ["/input/new.cha"]],
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Add Grammar/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await page.getByText("Review 26 input files", { exact: true }).click();
    await page.getByRole("button", { name: "Next files", exact: true }).click();
    await expect(page.getByText(inputs[25], { exact: true })).toBeVisible();
    await page.getByRole("button", { name: /26 files found.*\/input/ }).click();
    await page.getByText("Review 1 input file", { exact: true }).click();
    const review = page.getByRole("region", { name: "Input files to process" });
    await expect(review.getByText("Showing 1–1 of 1.", { exact: false })).toBeVisible();
    await expect(review.getByText("/input/new.cha", { exact: true })).toBeVisible();
    await expect(review.getByRole("listitem")).toHaveCount(1);
    await expect(review.getByRole("button", { name: "Next files" })).toBeDisabled();
    await expect(review.getByRole("button", { name: "Previous files" })).toBeDisabled();
  } finally { await harness.close(); }
});

test("desktop suspends the previous input throughout a new dialog and scan", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false,
      "plugin:dialog|open": ["/old", "/output", { nativeGate: "dialog", response: "/new" }],
      discover_files: [["/old/a.cha"], { nativeGate: "scan", response: ["/new/b.cha"] }],
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Add Grammar/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
    const start = page.getByRole("button", { name: "Start Processing", exact: true });
    await expect(start).toBeEnabled();
    await page.getByRole("button", { name: /1 file found.*\/old/ }).dblclick();
    await expect(page.getByRole("status").filter({ hasText: "Choosing input folder" })).toBeVisible();
    await expect(start).toBeDisabled();
    await expect(page.getByRole("button", { name: "Modify in place", exact: true })).toBeDisabled();
    await releaseDesktopReply(page, "dialog");
    await expect(page.getByRole("status").filter({ hasText: "Scanning input folder: /new" })).toBeVisible();
    await expect(start).toBeDisabled();
    expect(harness.submissions).toHaveLength(0);
    expect((await readDesktopInvokeCalls(page)).filter(call => call.command === "plugin:dialog|open")).toHaveLength(3);
    await releaseDesktopReply(page, "scan");
    await expect(page.getByRole("button", { name: /1 file found.*\/new/ })).toBeVisible();
    await start.click();
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
    expect(harness.submissions).toHaveLength(1);
    expect(harness.submissions[0].source_paths).toEqual(["/new/b.cha"]);
  } finally { await harness.close(); }
});

test("desktop failed input rediscovery clears the old submit-capable selection", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false, "plugin:dialog|open": ["/old", "/output", "/unreadable", "/recovered"],
      discover_files: [["/old/a.cha"], { nativeError: "Cannot read /unreadable: permission denied" }, ["/recovered/c.cha"]],
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Add Grammar/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
    await page.getByRole("button", { name: /1 file found.*\/old/ }).click();
    await expect(page.getByRole("alert")).toContainText("Cannot read /unreadable: permission denied");
    const start = page.getByRole("button", { name: "Start Processing", exact: true });
    await expect(start).toBeDisabled();
    expect(harness.submissions).toHaveLength(0);
    expect((await readDesktopInvokeCalls(page)).some(call => call.command === "admit_job_paths")).toBe(false);
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await expect(start).toBeEnabled();
    await start.click();
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
    expect(harness.submissions[0].source_paths).toEqual(["/recovered/c.cha"]);
  } finally { await harness.close(); }
});

for (const target of ["input", "output"]) {
  test(`desktop cancelling the ${target} dialog retains the earlier explicit choices`, async ({ page }) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        is_first_launch: false, "plugin:dialog|open": ["/input", "/output", { nativeGate: "cancel", response: null }],
        discover_files: [["/input/a.cha"]],
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: /Add Grammar/ }).click();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await page.getByRole("button", { name: target === "input" ? /1 file found.*\/input/ : "/output", exact: target === "output" }).click();
      const start = page.getByRole("button", { name: "Start Processing", exact: true });
      await expect(start).toBeDisabled();
      await expect(page.getByRole("button", { name: "Modify in place", exact: true })).toBeDisabled();
      await releaseDesktopReply(page, "cancel");
      await expect(start).toBeEnabled();
      await start.click();
      await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
      expect(harness.submissions[0]).toMatchObject({ source_paths: ["/input/a.cha"], output_paths: ["/output/a.cha"] });
    } finally { await harness.close(); }
  });

  test(`desktop ignores the abandoned ${target} dialog before starting discovery or changing choices`, async ({ page }) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        is_first_launch: false,
        "plugin:dialog|open": target === "input"
          ? [{ nativeGate: "abandoned", response: "/stale" }, "/new", "/output"]
          : ["/old", { nativeGate: "abandoned", response: "/stale" }, "/new", "/output"],
        discover_files: target === "input" ? [["/new/b.cha"]] : [["/old/a.cha"], ["/new/b.cha"]],
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: /Add Grammar/ }).click();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      if (target === "output") await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await expect(page.getByRole("status").filter({ hasText: `Choosing ${target} folder` })).toBeVisible();
      await page.getByRole("button", { name: "Back", exact: true }).click();
      await page.getByRole("button", { name: /Add Grammar/ }).click();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await releaseDesktopReply(page, "abandoned");
      await page.getByRole("button", { name: "Start Processing", exact: true }).click();
      await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
      expect(harness.submissions[0]).toMatchObject({ source_paths: ["/new/b.cha"], output_paths: ["/output/b.cha"] });
      expect((await readDesktopInvokeCalls(page)).filter(call => call.command === "discover_files").some(call => call.args.dir === "/stale")).toBe(false);
    } finally { await harness.close(); }
  });
}

test("desktop abandoned scan cannot settle or release a newer folder interaction", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false, "plugin:dialog|open": ["/old", "/new", "/output"],
      discover_files: [{ nativeGate: "old-scan", response: ["/old/a.cha"] }, { nativeGate: "new-scan", response: ["/new/b.cha"] }],
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Add Grammar/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await expect(page.getByRole("status").filter({ hasText: "Scanning input folder: /old" })).toBeVisible();
    await page.getByRole("button", { name: "Back", exact: true }).click();
    await page.getByRole("button", { name: /Add Grammar/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await expect(page.getByRole("status").filter({ hasText: "Scanning input folder: /new" })).toBeVisible();
    await releaseDesktopReply(page, "old-scan");
    await expect(page.getByRole("status").filter({ hasText: "Scanning input folder: /new" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Start Processing", exact: true })).toBeDisabled();
    await releaseDesktopReply(page, "new-scan");
    await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
    await page.getByRole("button", { name: "Start Processing", exact: true }).click();
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
    expect(harness.submissions[0].source_paths).toEqual(["/new/b.cha"]);
  } finally { await harness.close(); }
});

test("desktop transcription in-place makes CHAT paths and forwards the selected engine", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false, read_config: { engine: "whisper", rev_key: null },
      "plugin:dialog|open": "C:\\input",
      discover_files: [["C:\\input\\nested\\clip.wav"]],
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Transcribe Audio/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await page.getByRole("button", { name: "Modify in place", exact: true }).click();
    await page.getByRole("button", { name: "Start Processing", exact: true }).click();
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
    expect(harness.submissions[0]).toMatchObject({
      source_paths: ["C:\\input\\nested\\clip.wav"],
      output_paths: ["C:/input/nested/clip.cha"], options: { command: "transcribe", asr_engine: "whisper" },
    });
  } finally { await harness.close(); }
});

test("desktop rejects output collisions before any job is submitted", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false,
      "plugin:dialog|open": ["/input", "/output"],
      discover_files: [["/input/clip.wav", "/input/clip.mp3"]],
      admit_job_paths: { nativeError: "Several inputs would write the same output name." },
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Transcribe Audio/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
    await page.getByRole("button", { name: "Start Processing", exact: true }).click();
    await expect(page.getByRole("alert")).toContainText("same output");
    expect(harness.submissions).toHaveLength(0);
  } finally { await harness.close(); }
});

// Leaving the launcher must retire unsent work, including when a later launcher
// already owns another configuration. Held native replies model both pre-POST
// boundaries; completing the replacement job is the synchronization barrier.
for (const boundary of ["paths", "preferences"]) {
  test(`desktop retires unsent work on navigation during ${boundary}`, async ({ page }) => {
    const harness = await startHarness();
    const paths = root => ({ source_root: root, inputs: [root + "/clip.wav"],
      output_root: "/output", outputs: ["/output/clip.cha"] });
    try {
      await installDesktopTauriStub(page, {
        server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        is_first_launch: false,
        "plugin:dialog|open": ["/old", "/output", "/new", "/output"],
        discover_files: [["/old/clip.wav"], ["/new/clip.wav"]],
        admit_job_paths: boundary === "paths"
          ? [{ nativeGate: "unsent", response: paths("/old") }, paths("/new")]
          : [paths("/old"), paths("/new")],
        read_config: boundary === "preferences"
          ? [{ nativeGate: "unsent", response: { engine: "whisper", rev_key: null } }, { engine: "whisper", rev_key: null }]
          : { engine: "whisper", rev_key: null },
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: /Transcribe Audio/ }).click();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await page.getByRole("button", { name: "Start Processing", exact: true }).click();
      await expect.poll(() => page.evaluate(() => window.__hasHeldDesktopReply("unsent"))).toBe(true);
      // Router navigation preserves the pending native Promise in this document.
      await page.evaluate(() => {
        window.history.pushState(null, "", "/dashboard");
        window.dispatchEvent(new PopStateEvent("popstate"));
      });
      await expect(page.getByRole("button", { name: /Transcribe Audio/ })).toHaveCount(0);
      await page.evaluate(() => {
        window.history.pushState(null, "", "/process");
        window.dispatchEvent(new PopStateEvent("popstate"));
      });
      await page.getByRole("button", { name: /Transcribe Audio/ }).click();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await releaseDesktopReply(page, "unsent");
      await page.getByRole("button", { name: "Start Processing", exact: true }).click();
      await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
      expect(harness.submissions.map(job => job.source_paths)).toEqual([["/new/clip.wav"]]);
      const reads = (await readDesktopInvokeCalls(page)).filter(call => call.command === "read_config");
      expect(reads).toHaveLength(boundary === "paths" ? 1 : 2);
    } finally { await harness.close(); }
  });
}

test("desktop leaving after POST preserves the job without cancelling or resubmitting", async ({ page }) => {
  const harness = await startHarness();
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const cancellations = [];
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false, "plugin:dialog|open": ["/input", "/output"], discover_files: [["/input/a.cha"]],
    });
    await page.route("**/jobs/*/cancel", route => { cancellations.push(route.request().url()); return route.continue(); });
    await page.route("**/jobs", async route => {
      if (route.request().method() !== "POST") return route.continue();
      const response = await route.fetch();
      await gate;
      await route.fulfill({ response });
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Add Grammar/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
    await page.getByRole("button", { name: "Start Processing", exact: true }).click();
    await expect.poll(() => harness.submissions.length).toBe(1);
    await page.evaluate(() => {
      window.history.pushState(null, "", "/dashboard");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    await expect(page.getByText("Submitting job...", { exact: true })).toHaveCount(0);
    await page.evaluate(() => {
      window.history.pushState(null, "", "/process");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    await page.getByRole("button", { name: /Translate/ }).click();
    const response = page.waitForResponse(response => response.request().method() === "POST" && new URL(response.url()).pathname === "/jobs");
    release();
    await response;
    await expect(page.getByRole("heading", { name: "Translate", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Start Processing", exact: true })).toBeDisabled();
    expect(harness.submissions).toHaveLength(1);
    expect(cancellations).toEqual([]);
    const record = await (await page.request.get(harness.baseUrl + "/jobs/desktopjob")).json();
    expect(record).toMatchObject({ job_id: "desktopjob", status: "completed" });
  } finally { release(); await harness.close(); }
});

test("desktop owns delayed filesystem admission and submits its canonical paths once", async ({ page }) => {
  const harness = await startHarness();
  try {
    await installDesktopTauriStub(page, {
      server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
      is_first_launch: false, "plugin:dialog|open": ["/input-alias", "/output-alias"],
      discover_files: [["/input-alias/a.cha"]],
      admit_job_paths: { nativeGate: "canonical-paths", response: {
        source_root: "/real/input", inputs: ["/real/input/a.cha"],
        output_root: "/real/output", outputs: ["/real/output/a.cha"],
      } },
    });
    await page.goto(harness.baseUrl + "/process");
    await page.getByRole("button", { name: /Add Grammar/ }).click();
    await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
    await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
    await page.getByRole("button", { name: "Start Processing", exact: true }).dblclick();
    await expect(page.getByRole("status").filter({ hasText: "Checking input and output files" })).toBeVisible();
    expect(harness.submissions).toHaveLength(0);
    await releaseDesktopReply(page, "canonical-paths");
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
    expect(harness.submissions).toHaveLength(1);
    expect(harness.submissions[0]).toMatchObject({ source_dir: "/real/input", source_paths: ["/real/input/a.cha"], output_paths: ["/real/output/a.cha"] });
    expect((await readDesktopInvokeCalls(page)).filter(call => call.command === "admit_job_paths")).toHaveLength(1);
  } finally { await harness.close(); }
});

for (const observation of [
  { nativeError: "Choose a separate output folder outside the input tree." },
  { nativeError: "An output would overwrite a selected input: /output/a.cha" },
  { source_root: "/input", inputs: ["/input/a.cha"], output_root: "/output", outputs: [] },
]) {
  test("desktop blocks native refusal or incomplete path admission: " + JSON.stringify(observation), async ({ page }) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        is_first_launch: false, "plugin:dialog|open": ["/input", "/output"],
        discover_files: [["/input/a.cha"]], admit_job_paths: observation,
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: /Add Grammar/ }).click();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await page.getByRole("button", { name: "Start Processing", exact: true }).click();
      await expect(page.getByRole("alert")).toContainText(observation.nativeError ?? "incomplete input/output admission");
      expect(harness.submissions).toHaveLength(0);
    } finally { await harness.close(); }
  });
}

test("desktop cancellation owns one request and survives a progress reconnect until the final job record", async ({ page }) => {
  const harness = await startHarness();
  let observation = "running";
  let requests = 0;
  const acknowledgement = Promise.withResolvers();
  try {
    const fixture = await showRunningDesktopJob(page, harness, () => observation);
    await page.route("**/jobs/canceljob/cancel", async route => {
      requests++;
      await acknowledgement.promise;
      await route.fulfill({ status: 200, json: { status: "cancelled", message: "Job canceljob cancelled." } });
    });
    await page.getByRole("button", { name: "Cancel", exact: true }).dblclick();
    await expect(page.getByText("Requesting cancellation...", { exact: true })).toBeVisible();
    await expect.poll(() => requests).toBe(1);
    acknowledgement.resolve();
    await expect(page.getByText("Cancellation acknowledged.", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toHaveCount(0);
    observation = "unavailable";
    await expect(page.getByRole("button", { name: "Retry progress" })).toBeVisible();
    await expect(page.getByText("Cancellation acknowledged.", { exact: true })).toBeVisible();
    observation = "running";
    await page.getByRole("button", { name: "Retry progress" }).click();
    await expect(page.getByRole("button", { name: "Retry progress" })).toHaveCount(0);
    await expect(page.getByText("Cancellation acknowledged.", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Cancel", exact: true })).toHaveCount(0);
    expect(requests).toBe(1);
    observation = "cancelled";
    await expect(page.getByText("Job cancelled. Review the file results below; some output may need attention.", { exact: true })).toBeVisible();
    expect(requests).toBe(1);
    expect(fixture.submitted()).toBe(1);
  } finally { acknowledgement.resolve(); await harness.close(); }
});

test("desktop cancellation failure permits an explicit retry and retires its old error", async ({ page }) => {
  const harness = await startHarness();
  let requests = 0;
  try {
    await showRunningDesktopJob(page, harness);
    await page.route("**/jobs/canceljob/cancel", async route => {
      requests++;
      if (requests === 1) await route.fulfill({ status: 503, json: { detail: "Cancellation service temporarily unavailable" } });
      else await route.fulfill({ status: 200, json: { status: "cancelled", message: "Job canceljob cancelled." } });
    });
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(page.getByRole("alert")).toContainText("Cancellation service temporarily unavailable");
    expect(requests).toBe(1);
    await page.getByRole("button", { name: "Retry cancellation", exact: true }).dblclick();
    await expect(page.getByText("Cancellation acknowledged.", { exact: true })).toBeVisible();
    await expect(page.getByRole("alert")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Retry cancellation" })).toHaveCount(0);
    expect(requests).toBe(2);
  } finally { await harness.close(); }
});

for (const response of [{ status: "ok", message: "unknown wire status" }, { status: "cancelled" }]) {
  test("desktop refuses an unadmitted cancellation acknowledgement: " + JSON.stringify(response), async ({ page }) => {
    const harness = await startHarness();
    try {
      await showRunningDesktopJob(page, harness);
      await page.route("**/jobs/canceljob/cancel", route => route.fulfill({ status: 200, json: response }));
      await page.getByRole("button", { name: "Cancel", exact: true }).click();
      await expect(page.getByRole("alert")).toContainText("Could not confirm cancellation");
      await expect(page.getByText("Cancellation acknowledged.", { exact: true })).toHaveCount(0);
      await expect(page.getByRole("button", { name: "Retry cancellation" })).toBeVisible();
      await expect(page.getByRole("button", { name: "Open Output Folder" })).toHaveCount(0);
    } finally { await harness.close(); }
  });
}

test("desktop cancellation racing completion preserves the service's actual outcome", async ({ page }) => {
  const harness = await startHarness();
  let observation = "running";
  try {
    await showRunningDesktopJob(page, harness, () => observation);
    await page.route("**/jobs/canceljob/cancel", route => route.fulfill({ status: 200, json: { status: "completed", message: "Job already finished." } }));
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(page.getByText("The service reports that the job already finished (completed).", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toHaveCount(0);
    observation = "completed";
    await expect(page.getByRole("button", { name: "Open Output Folder" })).toBeVisible();
    await expect(page.getByText("Cancellation acknowledged.", { exact: true })).toHaveCount(0);
    await expect(page.getByText("Job cancelled.", { exact: false })).toHaveCount(0);
  } finally { await harness.close(); }
});

for (const state of ["interrupted", "writeback_failed", "missing-status"]) {
  test("desktop keeps " + state + " distinct from successful completion", async ({ page }) => {
    const harness = await startHarness();
    try {
      await installDesktopTauriStub(page, {
        server_status: { state: "ready", url: harness.baseUrl, build_hash: "test-build" },
        is_first_launch: false,
        "plugin:dialog|open": ["/input", "/output"],
        discover_files: [["/input/alpha.cha"]],
      });
      await page.route("**/jobs/desktopjob", async route => {
        const job = makeJob("desktopjob", state === "missing-status" ? undefined : state, {
          total: 1, completed: 0, fileStatuses: [makeFileStatus("alpha.cha", "interrupted")],
          submittedAt: "2026-02-25T00:00:00Z",
        });
        await route.fulfill({ json: job });
      });
      await page.goto(harness.baseUrl + "/process");
      await page.getByRole("button", { name: /Add Grammar/ }).click();
      await page.getByRole("button", { name: "Click to choose a folder", exact: true }).click();
      await page.getByRole("button", { name: "Click to choose output folder", exact: true }).click();
      await page.getByRole("button", { name: "Start Processing", exact: true }).click();
      if (state === "missing-status") {
        await expect(page.getByRole("button", { name: "Retry progress" })).toBeVisible();
        await expect(page.getByText("The server returned incomplete job progress.")).toBeVisible();
      } else {
        await expect(page.getByText("Job " + state + ".", { exact: false })).toBeVisible();
        await expect(page.getByRole("button", { name: "Process More Files" })).toBeVisible();
      }
      expect(harness.submissions).toHaveLength(1);
    } finally { await harness.close(); }
  });
}
