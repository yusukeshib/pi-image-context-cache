import { afterEach, describe, expect, test } from "bun:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ImageContent } from "@earendil-works/pi-ai";
import { chmodSync, existsSync, lstatSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, symlinkSync, utimesSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import imageContextCacheExtension, {
  cacheImage,
  cacheStats,
  type ExtensionDependencies,
  hashImage,
  isCacheHitEntryData,
  pruneImageContext,
  readCachedPreview,
} from "./index.ts";

const tempDirs: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(path.join(os.tmpdir(), "pi-image-cache-test-"));
  tempDirs.push(dir);
  return dir;
}

function image(text = "pixel"): ImageContent {
  return {
    type: "image",
    data: Buffer.concat([Buffer.from("BM", "ascii"), Buffer.from(text)]).toString("base64"),
    mimeType: "image/bmp",
  };
}

function pngImage(): ImageContent {
  return {
    type: "image",
    data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
    mimeType: "image/png",
  };
}

function messages(...items: unknown[]): AgentMessage[] {
  return items as AgentMessage[];
}

function user(content: unknown[]) {
  return { role: "user", content, timestamp: 1 };
}

function assistant(content: unknown[] = [{ type: "text", text: "seen" }]) {
  return {
    role: "assistant",
    content,
    api: "openai-responses",
    provider: "openai",
    model: "test",
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "stop",
    timestamp: 2,
  };
}

function toolResult(toolCallId: string, content: unknown[]) {
  return {
    role: "toolResult",
    toolCallId,
    toolName: "read",
    content,
    isError: false,
    timestamp: 2,
  };
}

function extensionHarness(
  cacheDir: string,
  visionMode: "direct" | "isolated" = "direct",
  analyzeVision?: ExtensionDependencies["analyzeVision"],
) {
  process.env.PI_IMAGE_CONTEXT_CACHE_DIR = cacheDir;
  process.env.PI_IMAGE_CONTEXT_CACHE_VISION_MODE = visionMode;
  const handlers = new Map<string, (event: any, ctx?: any) => Promise<any>>();
  const commands = new Map<string, any>();
  const entries: Array<{ customType: string; data: any }> = [];
  imageContextCacheExtension({
    on(name: string, handler: (event: any, ctx?: any) => Promise<any>) {
      handlers.set(name, handler);
    },
    registerCommand(name: string, command: any) {
      commands.set(name, command);
    },
    registerEntryRenderer() {},
    appendEntry(customType: string, data: any) {
      entries.push({ customType, data });
    },
  } as any, { analyzeVision });
  const start = async (branch: any[] = [], persisted: any[] = []) => handlers.get("session_start")?.({}, {
    hasUI: true,
    sessionManager: { getEntries: () => persisted, getBranch: () => branch },
    ui: { notify() {} },
  });
  return { handlers, commands, entries, start };
}

afterEach(() => {
  delete process.env.PI_IMAGE_CONTEXT_CACHE_DIR;
  delete process.env.PI_IMAGE_CONTEXT_CACHE_TTL_TURNS;
  delete process.env.PI_IMAGE_CONTEXT_CACHE_MAX_AGE_DAYS;
  delete process.env.PI_IMAGE_CONTEXT_CACHE_MAX_IMAGE_BYTES;
  delete process.env.PI_IMAGE_CONTEXT_CACHE_MAX_BYTES;
  delete process.env.PI_IMAGE_CONTEXT_CACHE_VISION_MODE;
  delete process.env.PI_IMAGE_CONTEXT_CACHE_VISION_MODEL;
  delete process.env.PI_IMAGE_CONTEXT_CACHE_VISION_MAX_TOKENS;
  for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("image cache", () => {
  test("stores content-addressed images with private metadata", () => {
    const dir = tempDir();
    const value = image();
    const first = cacheImage(value, dir, "/tmp/source.png");
    const second = cacheImage(value, dir, "/tmp/source.png");

    expect(first.sha256).toBe(hashImage(value));
    expect(second.path).toBe(first.path);
    expect(existsSync(first.path)).toBe(true);
    expect(cacheStats(dir)).toEqual({
      images: 1,
      bytes: Buffer.from(value.data, "base64").length,
      metadataFiles: 1,
    });
  });

  test("caches but keeps a fresh image for the first model call", () => {
    const dir = tempDir();
    const value = image();
    const result = pruneImageContext(messages(user([{ type: "text", text: "look" }, value])), {
      cacheDir: dir,
      ttlTurns: 1,
    });

    expect((result[0] as any).content[1]).toEqual(value);
    expect(cacheStats(dir).images).toBe(1);
  });

  test("evicts an image after one assistant turn and leaves a cache reference", () => {
    const dir = tempDir();
    const value = image();
    const result = pruneImageContext(messages(user([value]), assistant()), {
      cacheDir: dir,
      ttlTurns: 1,
    });

    const replacement = (result[0] as any).content[0];
    expect(replacement.type).toBe("text");
    expect(replacement.text).toContain("Image payload evicted after 1 assistant turn");
    expect(replacement.text).toContain(hashImage(value));
    expect(cacheStats(dir).images).toBe(1);
  });

  test("preserves non-image blocks and the original read source", () => {
    const dir = tempDir();
    const value = image();
    const result = pruneImageContext(
      messages(
        assistant([
          {
            type: "toolCall",
            id: "call-1",
            name: "read",
            arguments: { path: "/tmp/render.png" },
          },
        ]),
        toolResult("call-1", [{ type: "text", text: "Read image" }, value]),
        assistant(),
      ),
      { cacheDir: dir, ttlTurns: 1 },
    );

    const content = (result[1] as any).content;
    expect(content[0]).toEqual({ type: "text", text: "Read image" });
    expect(content[1].text).toContain('Original source: "/tmp/render.png"');
  });

  test("deduplicates repeated images on disk", () => {
    const dir = tempDir();
    const value = image();
    const result = pruneImageContext(
      messages(user([value]), assistant(), user([value]), assistant()),
      { cacheDir: dir, ttlTurns: 1 },
    );

    expect((result[0] as any).content[0].type).toBe("text");
    expect((result[2] as any).content[0].type).toBe("text");
    expect(cacheStats(dir).images).toBe(1);
  });

  test("never evicts when the cache copy cannot be written", () => {
    const dir = tempDir();
    const invalidCachePath = path.join(dir, "not-a-directory");
    writeFileSync(invalidCachePath, "occupied");
    const value = image();
    const result = pruneImageContext(messages(user([value]), assistant()), {
      cacheDir: invalidCachePath,
      ttlTurns: 1,
    });

    expect((result[0] as any).content[0]).toEqual(value);
  });

  test("full-content hashes distinguish images with equal boundaries and lengths", () => {
    const dir = tempDir();
    const firstBytes = Buffer.alloc(512, 1);
    const secondBytes = Buffer.from(firstBytes);
    secondBytes[256] = 2;
    const first = image(firstBytes.toString("binary"));
    const second = image(secondBytes.toString("binary"));

    const firstRecord = cacheImage(first, dir);
    const secondRecord = cacheImage(second, dir);

    expect(firstRecord.sha256).not.toBe(secondRecord.sha256);
    expect(firstRecord.path).not.toBe(secondRecord.path);
    expect(cacheStats(dir).images).toBe(2);
  });

  test("clamps TTL zero so a fresh prompt image is still visible once", () => {
    const dir = tempDir();
    const value = image();
    const result = pruneImageContext(messages(user([value])), { cacheDir: dir, ttlTurns: 0 });

    expect((result[0] as any).content[0]).toEqual(value);
    expect(cacheStats(dir).images).toBe(1);
  });

  test("deduplicates a fresh prompt attachment after its SHA was seen", () => {
    const dir = tempDir();
    const value = image();
    const result = pruneImageContext(
      messages(user([value])),
      { cacheDir: dir, ttlTurns: 1 },
      undefined,
      { seenHashes: new Set([hashImage(value)]) },
    );

    expect((result[0] as any).content[0].type).toBe("text");
    expect((result[0] as any).content[0].text).toContain("Image cache hit");
  });

  test("keeps a fresh tool-result image until a successful assistant response", () => {
    const dir = tempDir();
    const value = image();
    const fresh = pruneImageContext(
      messages(
        assistant([{ type: "toolCall", id: "call-1", name: "read", arguments: { path: "/tmp/a.png" } }]),
        toolResult("call-1", [value]),
      ),
      { cacheDir: dir, ttlTurns: 1 },
    );
    const afterAbort = pruneImageContext(
      messages(
        user([value]),
        { ...assistant(), stopReason: "aborted" },
      ),
      { cacheDir: dir, ttlTurns: 1 },
    );

    expect((fresh[1] as any).content[0]).toEqual(value);
    expect((afterAbort[0] as any).content[0]).toEqual(value);
  });

  test("rejects image symlinks and does not evict without a verified copy", () => {
    const dir = tempDir();
    const value = image();
    const target = path.join(dir, "target");
    writeFileSync(target, "not the image");
    symlinkSync(target, path.join(realpathSync(dir), `${hashImage(value)}.bmp`));

    const result = pruneImageContext(messages(user([value]), assistant()), { cacheDir: dir, ttlTurns: 1 });

    expect((result[0] as any).content[0]).toEqual(value);
  });

  test("atomically replaces a metadata symlink without modifying its target", () => {
    const dir = tempDir();
    cacheImage(image("seed"), dir);
    const value = image();
    const target = path.join(dir, "metadata-target");
    writeFileSync(target, "do not overwrite");
    const metadataPath = path.join(dir, `${hashImage(value)}.json`);
    symlinkSync(target, metadataPath);

    cacheImage(value, dir);

    expect(readFileSync(target, "utf8")).toBe("do not overwrite");
    expect(lstatSync(metadataPath).isSymbolicLink()).toBe(false);
  });

  test("repairs cache directory permissions", () => {
    const dir = tempDir();
    chmodSync(dir, 0o755);

    cacheImage(image(), dir);

    expect(lstatSync(dir).mode & 0o077).toBe(0);
  });

  test("reads a verified cached preview through a no-follow descriptor", () => {
    const dir = tempDir();
    const value = pngImage();
    const record = cacheImage(value, dir);
    const data = { ...record, timestamp: Date.now() };

    expect(readCachedPreview(data, dir)).toBe(value.data);
  });

  test("rejects symlinked and hash-mismatched expanded previews", () => {
    const dir = tempDir();
    const outside = tempDir();
    const value = pngImage();
    const record = cacheImage(value, dir);
    const data = { ...record, timestamp: Date.now() };
    const outsideFile = path.join(outside, "outside.png");
    writeFileSync(outsideFile, Buffer.from(value.data, "base64"), { mode: 0o600 });
    rmSync(record.path, { force: true });
    symlinkSync(outsideFile, record.path);
    expect(readCachedPreview(data, dir)).toBeUndefined();

    rmSync(record.path, { force: true });
    writeFileSync(record.path, Buffer.alloc(record.bytes, 0x58), { mode: 0o600 });
    expect(readCachedPreview(data, dir)).toBeUndefined();
  });

  test("rejects malformed persisted cache-card data", () => {
    expect(isCacheHitEntryData({ sha256: 3, path: null })).toBe(false);
    expect(
      isCacheHitEntryData({
        sha256: "a".repeat(64),
        path: "/tmp/a.png",
        mimeType: "text/plain",
        bytes: -1,
        timestamp: Number.NaN,
      }),
    ).toBe(false);
  });

  test("registers Pi hooks, appends one TUI-only cache card, and deduplicates it", async () => {
    const dir = tempDir();
    process.env.PI_IMAGE_CONTEXT_CACHE_DIR = dir;
    process.env.PI_IMAGE_CONTEXT_CACHE_VISION_MODE = "direct";
    const handlers = new Map<string, (event: any, ctx?: any) => Promise<any>>();
    const commands = new Map<string, any>();
    const renderers = new Map<string, (entry: any, options: any, theme: any) => any>();
    const entries: Array<{ customType: string; data: any }> = [];
    imageContextCacheExtension({
      on(name: string, handler: (event: any, ctx?: any) => Promise<any>) {
        handlers.set(name, handler);
      },
      registerCommand(name: string, command: any) {
        commands.set(name, command);
      },
      registerEntryRenderer(name: string, renderer: (entry: any, options: any, theme: any) => any) {
        renderers.set(name, renderer);
      },
      appendEntry(customType: string, data: any) {
        entries.push({ customType, data });
      },
    } as any);

    await handlers.get("session_start")?.({}, {
      hasUI: true,
      sessionManager: { getEntries: () => [], getBranch: () => [] },
      ui: { notify() {} },
    });
    const value = pngImage();
    const firstResult = await handlers.get("tool_result")?.({
      toolName: "read",
      toolCallId: "call-1",
      input: { path: "/tmp/input.png" },
      content: [{ type: "text", text: "Read image" }, value],
    });
    expect(firstResult).toBeUndefined();

    const freshContext = {
      messages: messages(
        assistant([{ type: "toolCall", id: "call-1", name: "read", arguments: { path: "/tmp/input.png" } }]),
        toolResult("call-1", [value]),
      ),
    };
    const transformed = await handlers.get("context")?.(freshContext);
    expect((transformed.messages[1] as any).content[0]).toEqual(value);
    await handlers.get("turn_end")?.({ message: assistant() });

    const duplicateResult = await handlers.get("tool_result")?.({
      toolName: "read",
      toolCallId: "call-2",
      input: { path: "/tmp/input.png" },
      content: [{ type: "text", text: "Read image" }, value],
    });
    expect(duplicateResult.content[0]).toEqual({ type: "text", text: "Read image" });
    expect(duplicateResult.content[1].type).toBe("text");
    expect(duplicateResult.content[1].text).toContain("Image cache hit — payload not resent");
    expect(duplicateResult.content.some((block: any) => block.type === "image")).toBe(false);

    const changedImageResult = await handlers.get("tool_result")?.({
      toolName: "read",
      toolCallId: "call-3",
      input: { path: "/tmp/input.png" },
      content: [image("changed contents")],
    });
    expect(changedImageResult).toBeUndefined();

    const cachedPath = path.join(realpathSync(dir), `${hashImage(value)}.png`);
    const forcedResult = await handlers.get("tool_result")?.({
      toolName: "read",
      toolCallId: "call-4",
      input: { path: cachedPath },
      content: [value],
    });
    expect(forcedResult).toBeUndefined();
    expect(await handlers.get("tool_result")?.({
      toolName: "read",
      toolCallId: "call-5",
      input: { path: "/tmp/file.txt" },
      content: [{ type: "text", text: "plain text" }],
    })).toBeUndefined();

    expect(cacheStats(dir).images).toBe(2);
    expect(entries).toHaveLength(0);
    await handlers.get("turn_end")?.({ message: assistant() });
    expect(entries).toHaveLength(1);
    expect(entries[0]!.data.sha256).toBe(hashImage(value));
    expect(JSON.stringify(entries[0]!.data)).not.toContain(value.data);
    expect(commands.has("image-cache")).toBe(true);

    const renderer = renderers.get(entries[0]!.customType)!;
    const component = renderer(
      { data: entries[0]!.data },
      { expanded: false },
      { fg: (_name: string, text: string) => text, bg: (_name: string, text: string) => text },
    );
    expect(component.render(120).join("\n")).toContain("Image cache hit");

    const expandedPreview = renderer(
      { data: entries[0]!.data },
      { expanded: true },
      { fg: (_name: string, text: string) => text, bg: (_name: string, text: string) => text },
    );
    const expandedText = expandedPreview.render(120).join("\n");
    expect(expandedText).not.toContain("Cached preview is unavailable");
    expect(expandedText).not.toContain("Cached preview could not be rendered");

    const nextDuplicate = await handlers.get("tool_result")?.({
      toolName: "read",
      toolCallId: "call-6",
      input: { path: "/tmp/input.png" },
      content: [value],
    });
    expect(nextDuplicate.content[0].text).toContain("Image cache hit");
    await handlers.get("turn_end")?.({ message: assistant() });
    expect(entries).toHaveLength(2);
    expect(entries[0]!.data.hitId).not.toBe(entries[1]!.data.hitId);

    rmSync(entries[0]!.data.path, { force: true });
    const missingPreview = renderer(
      { data: entries[0]!.data },
      { expanded: true },
      { fg: (_name: string, text: string) => text, bg: (_name: string, text: string) => text },
    );
    expect(missingPreview.render(120).join("\n")).toContain("Cached preview is unavailable");
  });

  test("restores seen SHA state from successful session history after reload", async () => {
    const dir = tempDir();
    process.env.PI_IMAGE_CONTEXT_CACHE_DIR = dir;
    process.env.PI_IMAGE_CONTEXT_CACHE_VISION_MODE = "direct";
    const handlers = new Map<string, (event: any, ctx?: any) => Promise<any>>();
    const value = pngImage();
    imageContextCacheExtension({
      on(name: string, handler: (event: any, ctx?: any) => Promise<any>) {
        handlers.set(name, handler);
      },
      registerCommand() {},
      registerEntryRenderer() {},
      appendEntry() {},
    } as any);

    await handlers.get("session_start")?.({}, {
      hasUI: true,
      sessionManager: {
        getEntries: () => [],
        getBranch: () => [
          { type: "message", message: assistant([{ type: "toolCall", id: "old", name: "read", arguments: { path: "/tmp/a.png" } }]) },
          { type: "message", message: toolResult("old", [value]) },
          { type: "message", message: assistant() },
        ],
      },
      ui: { notify() {} },
    });
    const duplicate = await handlers.get("tool_result")?.({
      toolName: "read",
      toolCallId: "new",
      input: { path: "/tmp/a.png" },
      content: [value],
    });

    expect(duplicate.content[0].text).toContain("Image cache hit");
  });

  test("emits per-event cards while suppressing duplicate callbacks and retries", async () => {
    const dir = tempDir();
    process.env.PI_IMAGE_CONTEXT_CACHE_DIR = dir;
    process.env.PI_IMAGE_CONTEXT_CACHE_VISION_MODE = "direct";
    const handlers = new Map<string, (event: any, ctx?: any) => Promise<any>>();
    const appended: any[] = [];
    const value = image();
    const record = cacheImage(value, dir);
    imageContextCacheExtension({
      on(name: string, handler: (event: any, ctx?: any) => Promise<any>) {
        handlers.set(name, handler);
      },
      registerCommand() {},
      registerEntryRenderer() {},
      appendEntry(_type: string, data: any) {
        appended.push(data);
      },
    } as any);

    await handlers.get("session_start")?.({}, {
      hasUI: true,
      sessionManager: {
        getEntries: () => [{ type: "custom", customType: "pi-image-context-cache-hit", data: { ...record, timestamp: 1 } }],
        getBranch: () => [
          { type: "message", message: assistant([{ type: "toolCall", id: "seen", name: "read", arguments: { path: "/tmp/input.png" } }]) },
          { type: "message", message: toolResult("seen", [value]) },
          { type: "message", message: assistant() },
        ],
      },
      ui: { notify() {} },
    });
    const repeatedToolEvent = {
      toolName: "read",
      toolCallId: "call-restored",
      input: { path: "/tmp/input.png" },
      content: [value],
    };
    const duplicate = await handlers.get("tool_result")?.(repeatedToolEvent);
    await handlers.get("tool_result")?.(repeatedToolEvent);

    const repeatedContextEvent = { messages: messages({ ...user([value]), timestamp: 10 }) };
    await handlers.get("context")?.(repeatedContextEvent);
    await handlers.get("context")?.(repeatedContextEvent);
    await handlers.get("context")?.({ messages: messages({ ...user([value]), timestamp: 11 }) });
    await handlers.get("turn_end")?.({ message: assistant() });

    expect(duplicate.content[0].text).toContain("Image cache hit");
    expect(appended).toHaveLength(3);
    expect(new Set(appended.map((entry) => entry.hitId)).size).toBe(3);
    expect(appended.some((entry) => entry.hitId.includes("call-restored"))).toBe(true);
  });

  test("rejects malformed base64, unsupported MIME, signature mismatch, and oversized images", () => {
    const dir = tempDir();
    expect(() => cacheImage({ type: "image", data: "%%%", mimeType: "image/png" }, dir)).toThrow();
    expect(() => cacheImage({ ...pngImage(), data: `${pngImage().data}=` }, dir)).toThrow();
    expect(() => cacheImage({ ...pngImage(), mimeType: "image/svg+xml" }, dir)).toThrow();
    expect(() => cacheImage({ ...pngImage(), mimeType: "image/jpeg" }, dir)).toThrow();
    expect(() => cacheImage(pngImage(), dir, undefined, { maxImageBytes: 10 })).toThrow();
    expect(cacheStats(dir).images).toBe(0);
  });

  test("enforces total cache quota by evicting older images", () => {
    const dir = tempDir();
    const first = cacheImage(image("one"), dir, undefined, { maxCacheBytes: 12 });
    const second = cacheImage(image("second"), dir, undefined, { maxCacheBytes: 12 });

    expect(first.sha256).not.toBe(second.sha256);
    expect(existsSync(first.path)).toBe(false);
    expect(existsSync(second.path)).toBe(true);
    expect(cacheStats(dir).images).toBe(1);
  });

  test("safe clear preserves unrelated files and refuses dangerous directories", async () => {
    const dir = tempDir();
    const harness = extensionHarness(dir);
    await harness.start();
    cacheImage(image(), dir);
    const unrelated = path.join(realpathSync(dir), "keep-me.txt");
    writeFileSync(unrelated, "keep");
    const notices: string[] = [];

    await harness.commands.get("image-cache").handler("clear", {
      hasUI: false,
      ui: { notify(message: string) { notices.push(message); } },
    });

    expect(readFileSync(unrelated, "utf8")).toBe("keep");
    expect(cacheStats(dir).images).toBe(0);
    expect(notices[0]).toContain("cleared");
    expect(() => cacheStats(os.homedir())).toThrow();
  });

  test("refuses an existing non-cache directory without changing it", () => {
    const dir = tempDir();
    chmodSync(dir, 0o755);
    const unrelated = path.join(dir, "unrelated.txt");
    writeFileSync(unrelated, "keep");

    expect(() => cacheImage(image(), dir)).toThrow("non-empty cache directory");
    expect(readFileSync(unrelated, "utf8")).toBe("keep");
    expect(lstatSync(dir).mode & 0o077).not.toBe(0);
  });

  test("expired metadata is removed before branch state restoration can refresh it", async () => {
    const dir = tempDir();
    const value = pngImage();
    const record = cacheImage(value, dir);
    const metadataPath = path.join(realpathSync(dir), `${record.sha256}.json`);
    const metadata = JSON.parse(readFileSync(metadataPath, "utf8"));
    metadata.lastAccessedAt = "2000-01-01T00:00:00.000Z";
    writeFileSync(metadataPath, `${JSON.stringify(metadata)}\n`, { mode: 0o600 });
    const old = new Date("2000-01-01T00:00:00.000Z");
    utimesSync(metadataPath, old, old);
    utimesSync(record.path, old, old);
    process.env.PI_IMAGE_CONTEXT_CACHE_MAX_AGE_DAYS = "1";
    const harness = extensionHarness(dir);

    await harness.start([
      { type: "message", message: user([value]) },
      { type: "message", message: assistant() },
    ]);

    expect(existsSync(record.path)).toBe(false);
    expect(existsSync(metadataPath)).toBe(false);
  });

  test("branch navigation rebuilds seen-image state from only the active branch", async () => {
    const dir = tempDir();
    const value = pngImage();
    const harness = extensionHarness(dir);
    const branchA = [
      { type: "message", message: user([value]) },
      { type: "message", message: assistant() },
    ];
    const context = (branch: any[]) => ({
      sessionManager: { getBranch: () => branch },
      hasUI: true,
      ui: { notify() {} },
    });
    await harness.start(branchA);

    const hitOnA = await harness.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "a1", input: { path: "/tmp/a.png" }, content: [value],
    });
    expect(hitOnA.content[0].text).toContain("Image cache hit");

    await harness.handlers.get("session_tree")?.({}, context([]));
    const freshOnB = await harness.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "b1", input: { path: "/tmp/a.png" }, content: [value],
    });
    expect(freshOnB).toBeUndefined();

    await harness.handlers.get("session_tree")?.({}, context(branchA));
    const hitAgainOnA = await harness.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "a2", input: { path: "/tmp/a.png" }, content: [value],
    });
    expect(hitAgainOnA.content[0].text).toContain("Image cache hit");

    await harness.handlers.get("session_compact")?.({}, {
      sessionManager: { getBranch: () => branchA, buildContextEntries: () => [] },
      hasUI: true,
      ui: { notify() {} },
    });
    const freshAfterCompaction = await harness.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "compact", input: { path: "/tmp/a.png" }, content: [value],
    });
    expect(freshAfterCompaction).toBeUndefined();
  });

  test("failed turns neither mark images seen nor persist pending cards", async () => {
    const dir = tempDir();
    const value = pngImage();
    const harness = extensionHarness(dir);
    await harness.start();
    const duplicatePrompt = { ...user([value, value]), timestamp: 50 };
    const transformed = await harness.handlers.get("context")?.({ messages: messages(duplicatePrompt) });
    expect(transformed.messages[0].content.filter((block: any) => block.type === "image")).toHaveLength(1);
    await harness.handlers.get("turn_end")?.({ message: { ...assistant(), stopReason: "error" } });
    expect(harness.entries).toHaveLength(0);

    const fresh = await harness.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "after-error", input: { path: "/tmp/a.png" }, content: [value],
    });
    expect(fresh).toBeUndefined();

    await harness.start([], [{
      type: "custom",
      customType: "pi-image-context-cache-hit",
      data: { ...cacheImage(value, dir), timestamp: 1 },
    }]);
    const stillFresh = await harness.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "after-reload", input: { path: "/tmp/a.png" }, content: [value],
    });
    expect(stillFresh).toBeUndefined();
  });

  test("parallel fresh tool results persist only one image payload", async () => {
    const dir = tempDir();
    const value = pngImage();
    const harness = extensionHarness(dir);
    await harness.start();
    const first = await harness.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "parallel-1", input: { path: "/tmp/a.png" }, content: [value],
    });
    const second = await harness.handlers.get("tool_result")?.({
      toolName: "read",
      toolCallId: "parallel-2",
      input: { path: "/tmp/b.png" },
      content: [{ type: "text", text: "before" }, value, { type: "text", text: "after" }],
    });

    expect(first).toBeUndefined();
    expect(second.content[0].text).toBe("before");
    expect(second.content[1].type).toBe("text");
    expect(second.content[2].text).toBe("after");
    expect(second.content.some((block: any) => block.type === "image")).toBe(false);
  });

  test("restores seen state from image-bearing custom messages", async () => {
    const dir = tempDir();
    const value = pngImage();
    const harness = extensionHarness(dir);
    await harness.start([
      { type: "custom_message", content: [value] },
      { type: "message", message: assistant() },
    ]);

    const duplicate = await harness.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "custom-message", input: { path: "/tmp/a.png" }, content: [value],
    });
    expect(duplicate.content[0].text).toContain("Image cache hit");
  });

  test("honors TTL values greater than one", () => {
    const dir = tempDir();
    const value = pngImage();
    const oneTurn = pruneImageContext(messages(user([value]), assistant()), { cacheDir: dir, ttlTurns: 2 });
    const twoTurns = pruneImageContext(messages(user([value]), assistant(), assistant()), { cacheDir: dir, ttlTurns: 2 });

    expect((oneTurn[0] as any).content[0].type).toBe("image");
    expect((twoTurns[0] as any).content[0].type).toBe("text");
  });

  test("isolates prompt attachments and caches Vision results by image and question", async () => {
    const dir = tempDir();
    let calls = 0;
    const harness = extensionHarness(dir, "isolated", async (_image, question) => {
      calls += 1;
      return { summary: `analysis for: ${question}`, model: "test/vision", cost: 0.01 };
    });
    await harness.start();
    const ctx = { signal: undefined, sessionManager: { getBranch: () => [] } };
    const event = { text: "What is shown?", images: [pngImage()], source: "interactive" };

    const first = await harness.handlers.get("input")?.(event, ctx);
    expect(first.action).toBe("transform");
    expect(first.images).toHaveLength(0);
    expect(first.text).toContain("Image analyzed in isolated Vision context");
    expect(first.text).toContain("analysis for: What is shown?");
    expect(first.text).not.toContain(pngImage().data);
    expect(calls).toBe(1);
    await harness.handlers.get("turn_end")?.({ message: assistant() });
    expect(harness.entries).toHaveLength(1);
    expect(harness.entries[0]!.data.kind).toBe("vision");

    const second = await harness.handlers.get("input")?.(event, ctx);
    expect(second.text).toContain("cached: true");
    expect(calls).toBe(1);
    await harness.handlers.get("input")?.({ ...event, text: "Read all visible text" }, ctx);
    expect(calls).toBe(2);
  });

  test("isolates read-tool images, fails closed, and allows explicit main-model bypass", async () => {
    const dir = tempDir();
    const value = pngImage();
    const harness = extensionHarness(dir, "isolated", async () => ({
      summary: "A precise isolated description",
      model: "test/vision",
    }));
    const branch = [{ type: "message", message: { ...user([{ type: "text", text: "Describe it" }]), content: [{ type: "text", text: "Describe it" }] } }];
    await harness.start(branch);
    const ctx = { signal: undefined, sessionManager: { getBranch: () => branch } };
    const isolated = await harness.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "isolated", input: { path: "/tmp/a.png" }, content: [value],
    }, ctx);
    expect(isolated.content.some((block: any) => block.type === "image")).toBe(false);
    expect(isolated.content[0].text).toContain("A precise isolated description");
    expect(JSON.stringify(isolated.content)).not.toContain(value.data);

    const record = cacheImage(value, dir);
    const bypass = await harness.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "bypass", input: { path: record.path }, content: [value],
    }, ctx);
    expect(bypass).toBeUndefined();

    const failing = extensionHarness(tempDir(), "isolated", async () => { throw new Error("vision unavailable"); });
    await failing.start(branch);
    const failure = await failing.handlers.get("tool_result")?.({
      toolName: "read", toolCallId: "failure", input: { path: "/tmp/a.png" }, content: [value],
    }, ctx);
    expect(failure.content.some((block: any) => block.type === "image")).toBe(false);
    expect(failure.content[0].text).toContain("payload was NOT sent");
    expect(failure.content[0].text).toContain("vision unavailable");
  });

  test("clear invalidates Vision memory and abandoned input cards", async () => {
    const dir = tempDir();
    let calls = 0;
    const harness = extensionHarness(dir, "isolated", async () => {
      calls += 1;
      return { summary: "result", model: "test/vision" };
    });
    await harness.start();
    const ctx = { signal: undefined, sessionManager: { getBranch: () => [] } };
    const event = { text: "question", images: [pngImage()], source: "interactive" };
    await harness.handlers.get("input")?.(event, ctx);
    await harness.handlers.get("input")?.({ text: "replacement", source: "interactive" }, ctx);
    await harness.handlers.get("turn_end")?.({ message: assistant() });
    expect(harness.entries).toHaveLength(0);

    await harness.handlers.get("input")?.(event, ctx);
    expect(calls).toBe(1);
    await harness.commands.get("image-cache").handler("clear", {
      hasUI: false,
      ui: { notify() {} },
    });
    await harness.handlers.get("input")?.(event, ctx);
    expect(calls).toBe(2);
  });

  test("bounds query-specific Vision artifacts with the total cache quota", async () => {
    const dir = tempDir();
    process.env.PI_IMAGE_CONTEXT_CACHE_MAX_BYTES = "700";
    const harness = extensionHarness(dir, "isolated", async (_image, question) => ({
      summary: `description ${question}`.repeat(10),
      model: "test/vision",
    }));
    await harness.start();
    const ctx = { signal: undefined, sessionManager: { getBranch: () => [] } };
    for (let index = 0; index < 6; index += 1) {
      await harness.handlers.get("input")?.({
        text: `question-${index}`,
        images: [image()],
        source: "interactive",
      }, ctx);
    }
    const total = readdirSync(realpathSync(dir))
      .filter((name) => name !== ".pi-image-context-cache")
      .reduce((sum, name) => sum + lstatSync(path.join(realpathSync(dir), name)).size, 0);
    expect(total).toBeLessThanOrEqual(700);
  });

  test("keeps operational state isolated between extension instances", async () => {
    const value = pngImage();
    const first = extensionHarness(tempDir());
    const second = extensionHarness(tempDir());
    await first.start();
    await second.start();
    const event = { toolName: "read", toolCallId: "one", input: { path: "/tmp/a.png" }, content: [value] };
    expect(await first.handlers.get("tool_result")?.(event)).toBeUndefined();
    expect(await first.handlers.get("tool_result")?.({ ...event, toolCallId: "two" })).toBeDefined();
    expect(await second.handlers.get("tool_result")?.({ ...event, toolCallId: "three" })).toBeUndefined();
  });

  test("deduplicates concurrent isolated Vision analysis", async () => {
    const dir = tempDir();
    let calls = 0;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const harness = extensionHarness(dir, "isolated", async () => {
      calls += 1;
      await gate;
      return { summary: "shared result", model: "test/vision" };
    });
    await harness.start();
    const ctx = { signal: undefined, sessionManager: { getBranch: () => [] } };
    const event = { text: "Same question", images: [pngImage()], source: "interactive" };
    const first = harness.handlers.get("input")?.(event, ctx);
    const second = harness.handlers.get("input")?.(event, ctx);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(calls).toBe(1);
    release();
    const results = await Promise.all([first, second]);
    expect(results[0].text).toContain("shared result");
    expect(results[1].text).toContain("shared result");
  });
});
