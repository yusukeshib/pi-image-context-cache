import { afterEach, describe, expect, test } from "bun:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ImageContent } from "@earendil-works/pi-ai";
import { chmodSync, existsSync, lstatSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import imageContextCacheExtension, {
  cacheImage,
  cacheStats,
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
    data: Buffer.from(text).toString("base64"),
    mimeType: "image/png",
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

afterEach(() => {
  delete process.env.PI_IMAGE_CONTEXT_CACHE_DIR;
  delete process.env.PI_IMAGE_CONTEXT_CACHE_TTL_TURNS;
  delete process.env.PI_IMAGE_CONTEXT_CACHE_MAX_AGE_DAYS;
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
    expect(cacheStats(dir)).toEqual({ images: 1, bytes: 5, metadataFiles: 1 });
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
    expect(content[1].text).toContain("Original source: /tmp/render.png");
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
    symlinkSync(target, path.join(dir, `${hashImage(value)}.png`));

    const result = pruneImageContext(messages(user([value]), assistant()), { cacheDir: dir, ttlTurns: 1 });

    expect((result[0] as any).content[0]).toEqual(value);
  });

  test("atomically replaces a metadata symlink without modifying its target", () => {
    const dir = tempDir();
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
      sessionManager: { getEntries: () => [] },
      ui: { notify() {} },
    });
    const value = pngImage();
    await handlers.get("tool_result")?.({
      toolName: "read",
      toolCallId: "call-1",
      input: { path: "/tmp/input.png" },
      content: [value],
    });
    const contextEvent = {
      messages: messages(
        assistant([{ type: "toolCall", id: "call-1", name: "read", arguments: { path: "/tmp/input.png" } }]),
        toolResult("call-1", [value]),
        assistant(),
      ),
    };
    const transformed = await handlers.get("context")?.(contextEvent);

    expect(cacheStats(dir).images).toBe(1);
    expect((transformed.messages[1] as any).content[0].text).toContain("Original source: /tmp/input.png");
    expect(entries).toHaveLength(0);
    await handlers.get("turn_end")?.({});
    expect(entries).toHaveLength(1);
    expect(entries[0]!.data.sha256).toBe(hashImage(value));
    expect(JSON.stringify(entries[0]!.data)).not.toContain(value.data);

    await handlers.get("context")?.(contextEvent);
    await handlers.get("turn_end")?.({});
    expect(entries).toHaveLength(1);
    expect(commands.has("image-cache")).toBe(true);

    const renderer = renderers.get(entries[0]!.customType)!;
    const component = renderer(
      { data: entries[0]!.data },
      { expanded: false },
      { fg: (_name: string, text: string) => text, bg: (_name: string, text: string) => text },
    );
    expect(component.render(120).join("\n")).toContain("Image context cached");

    const expandedPreview = renderer(
      { data: entries[0]!.data },
      { expanded: true },
      { fg: (_name: string, text: string) => text, bg: (_name: string, text: string) => text },
    );
    const expandedText = expandedPreview.render(120).join("\n");
    expect(expandedText).not.toContain("Cached preview is unavailable");
    expect(expandedText).not.toContain("Cached preview could not be rendered");

    rmSync(entries[0]!.data.path, { force: true });
    const missingPreview = renderer(
      { data: entries[0]!.data },
      { expanded: true },
      { fg: (_name: string, text: string) => text, bg: (_name: string, text: string) => text },
    );
    expect(missingPreview.render(120).join("\n")).toContain("Cached preview is unavailable");
  });

  test("restores rendered SHA deduplication from persisted custom entries", async () => {
    const dir = tempDir();
    process.env.PI_IMAGE_CONTEXT_CACHE_DIR = dir;
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
      },
      ui: { notify() {} },
    });
    await handlers.get("context")?.({ messages: messages(user([value]), assistant()) });
    await handlers.get("turn_end")?.({});

    expect(appended).toHaveLength(0);
  });
});
