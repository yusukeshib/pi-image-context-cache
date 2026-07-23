import { afterEach, describe, expect, test } from "bun:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ImageContent } from "@earendil-works/pi-ai";
import { chmodSync, existsSync, lstatSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import imageContextCacheExtension, { cacheImage, cacheStats, hashImage, pruneImageContext } from "./index.ts";

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

  test("registers Pi hooks that cache tool images and prune later context", async () => {
    const dir = tempDir();
    process.env.PI_IMAGE_CONTEXT_CACHE_DIR = dir;
    const handlers = new Map<string, (event: any, ctx?: any) => Promise<any>>();
    const commands = new Map<string, any>();
    imageContextCacheExtension({
      on(name: string, handler: (event: any, ctx?: any) => Promise<any>) {
        handlers.set(name, handler);
      },
      registerCommand(name: string, command: any) {
        commands.set(name, command);
      },
    } as any);

    const value = image();
    await handlers.get("tool_result")?.({
      toolName: "read",
      toolCallId: "call-1",
      input: { path: "/tmp/input.png" },
      content: [value],
    });
    const transformed = await handlers.get("context")?.({
      messages: messages(
        assistant([{ type: "toolCall", id: "call-1", name: "read", arguments: { path: "/tmp/input.png" } }]),
        toolResult("call-1", [value]),
        assistant(),
      ),
    });

    expect(cacheStats(dir).images).toBe(1);
    expect((transformed.messages[1] as any).content[0].text).toContain("Original source: /tmp/input.png");
    expect(commands.has("image-cache")).toBe(true);
  });
});
