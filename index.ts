import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createHash, randomBytes } from "node:crypto";
import type { Stats } from "node:fs";
import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";

const DEFAULT_CACHE_DIR = path.join(os.homedir(), ".cache", "pi-image-context");
const DEFAULT_TTL_TURNS = 1;
const DEFAULT_MAX_AGE_DAYS = 30;

interface CacheRecord {
  sha256: string;
  path: string;
  mimeType: string;
  bytes: number;
  source?: string;
}

interface CacheMetadata extends CacheRecord {
  createdAt: string;
  lastAccessedAt: string;
}

interface MemoizedCacheRecord {
  metadata: CacheMetadata;
  identity: string;
}

interface ImageCacheOptions {
  cacheDir: string;
  ttlTurns: number;
  maxAgeDays: number;
}

interface CacheStats {
  images: number;
  bytes: number;
  metadataFiles: number;
}

type MessageWithContent = {
  content?: string | unknown[];
  toolCallId?: string;
};

const recordByHash = new Map<string, MemoizedCacheRecord>();
const sourceByToolCall = new Map<string, string>();

function nonNegativeInteger(value: string | undefined, fallback: number): number {
  if (value === undefined || value.trim() === "") return fallback;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}

function expandHome(value: string): string {
  if (value === "~") return os.homedir();
  if (value.startsWith(`~${path.sep}`)) return path.join(os.homedir(), value.slice(2));
  return value;
}

function loadOptions(): ImageCacheOptions {
  return {
    cacheDir: path.resolve(expandHome(process.env.PI_IMAGE_CONTEXT_CACHE_DIR || DEFAULT_CACHE_DIR)),
    ttlTurns: Math.max(1, nonNegativeInteger(process.env.PI_IMAGE_CONTEXT_CACHE_TTL_TURNS, DEFAULT_TTL_TURNS)),
    maxAgeDays: nonNegativeInteger(process.env.PI_IMAGE_CONTEXT_CACHE_MAX_AGE_DAYS, DEFAULT_MAX_AGE_DAYS),
  };
}

function isImageContent(value: unknown): value is ImageContent {
  if (!value || typeof value !== "object") return false;
  const block = value as Partial<ImageContent>;
  return block.type === "image" && typeof block.data === "string" && typeof block.mimeType === "string";
}

function imageExtension(mimeType: string): string {
  switch (mimeType.toLowerCase()) {
    case "image/jpeg":
      return "jpg";
    case "image/png":
      return "png";
    case "image/gif":
      return "gif";
    case "image/webp":
      return "webp";
    case "image/bmp":
      return "bmp";
    default:
      return "img";
  }
}

export function hashImage(image: ImageContent): string {
  return createHash("sha256").update(image.data, "base64").digest("hex");
}

function assertOwnedPath(stat: Stats, target: string): void {
  const getuid = process.getuid;
  if (typeof getuid === "function" && stat.uid !== getuid()) {
    throw new Error(`Image cache path is not owned by the current user: ${target}`);
  }
}

function ensureCacheDir(cacheDir: string): void {
  mkdirSync(cacheDir, { recursive: true, mode: 0o700 });
  const stat = lstatSync(cacheDir);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`Image cache path must be a real directory: ${cacheDir}`);
  }
  assertOwnedPath(stat, cacheDir);
  if ((stat.mode & 0o077) !== 0) chmodSync(cacheDir, 0o700);
}

function privateFileIdentity(file: string): string {
  const stat = lstatSync(file);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error(`Image cache entry must be a regular file: ${file}`);
  }
  assertOwnedPath(stat, file);
  if ((stat.mode & 0o077) !== 0) chmodSync(file, 0o600);
  return `${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeMs}`;
}

function ensureImageFile(file: string, data: Buffer, sha256: string): string {
  let created = false;
  try {
    writeFileSync(file, data, { flag: "wx", mode: 0o600 });
    created = true;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== "EEXIST") throw error;
  }
  const identity = privateFileIdentity(file);
  if (!created) {
    const existingHash = createHash("sha256").update(readFileSync(file)).digest("hex");
    if (existingHash !== sha256) {
      throw new Error(`Existing image cache entry failed hash verification: ${file}`);
    }
  }
  return identity;
}

function readMetadata(file: string): Partial<CacheMetadata> | undefined {
  if (!existsSync(file)) return undefined;
  try {
    privateFileIdentity(file);
    return JSON.parse(readFileSync(file, "utf8")) as Partial<CacheMetadata>;
  } catch {
    return undefined;
  }
}

function writeMetadataAtomic(file: string, metadata: CacheMetadata): void {
  const temporary = path.join(
    path.dirname(file),
    `.${path.basename(file)}.${process.pid}.${randomBytes(6).toString("hex")}.tmp`,
  );
  try {
    writeFileSync(temporary, `${JSON.stringify(metadata, null, 2)}\n`, { flag: "wx", mode: 0o600 });
    renameSync(temporary, file);
    chmodSync(file, 0o600);
  } finally {
    rmSync(temporary, { force: true });
  }
}

export function cacheImage(image: ImageContent, cacheDir: string, source?: string): CacheRecord {
  const resolvedCacheDir = path.resolve(cacheDir);
  ensureCacheDir(resolvedCacheDir);
  const sha256 = hashImage(image);
  const memoKey = `${resolvedCacheDir}\0${sha256}`;
  const memoized = recordByHash.get(memoKey);
  if (memoized && existsSync(memoized.metadata.path)) {
    const identity = privateFileIdentity(memoized.metadata.path);
    if (identity === memoized.identity) {
      const now = Date.now();
      const lastAccess = Date.parse(memoized.metadata.lastAccessedAt);
      const needsSource = Boolean(source && !memoized.metadata.source);
      const needsTouch = !Number.isFinite(lastAccess) || now - lastAccess >= 60 * 60 * 1000;
      if (!needsSource && !needsTouch) return memoized.metadata;
      const updated = {
        ...memoized.metadata,
        source: source || memoized.metadata.source,
        lastAccessedAt: new Date(now).toISOString(),
      };
      writeMetadataAtomic(path.join(resolvedCacheDir, `${sha256}.json`), updated);
      recordByHash.set(memoKey, { metadata: updated, identity });
      return updated;
    }
  }

  const decoded = Buffer.from(image.data, "base64");
  const imagePath = path.join(resolvedCacheDir, `${sha256}.${imageExtension(image.mimeType)}`);
  const metadataPath = path.join(resolvedCacheDir, `${sha256}.json`);
  const identity = ensureImageFile(imagePath, decoded, sha256);
  const previous = readMetadata(metadataPath);
  const now = new Date().toISOString();
  const metadata: CacheMetadata = {
    sha256,
    path: imagePath,
    mimeType: image.mimeType,
    bytes: decoded.byteLength,
    source: source || previous?.source,
    createdAt: previous?.createdAt || now,
    lastAccessedAt: now,
  };
  writeMetadataAtomic(metadataPath, metadata);
  recordByHash.set(memoKey, { metadata, identity });
  return metadata;
}

function sourceFromToolInput(toolName: string, input: Record<string, unknown>): string | undefined {
  if (toolName !== "read") return undefined;
  return typeof input.path === "string" ? input.path : undefined;
}

function toolCallSources(messages: AgentMessage[]): Map<string, string> {
  const result = new Map(sourceByToolCall);
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    for (const block of message.content) {
      if (block.type !== "toolCall" || block.name !== "read") continue;
      const source = typeof block.arguments.path === "string" ? block.arguments.path : undefined;
      if (source) result.set(block.id, source);
    }
  }
  return result;
}

function placeholder(record: CacheRecord, ttlTurns: number): TextContent {
  const source = record.source ? `\nOriginal source: ${record.source}` : "";
  return {
    type: "text",
    text:
      `[Image payload evicted after ${ttlTurns} assistant turn${ttlTurns === 1 ? "" : "s"}.]` +
      `\nCached image: ${record.path}${source}` +
      `\nSHA-256: ${record.sha256}` +
      `\nMIME: ${record.mimeType}; bytes: ${record.bytes}` +
      `\nUse the read tool on the cached image when visual inspection is needed again.`,
  };
}

export function pruneImageContext(
  messages: AgentMessage[],
  options: Pick<ImageCacheOptions, "cacheDir" | "ttlTurns">,
): AgentMessage[] {
  const sources = toolCallSources(messages);
  const ttlTurns = Math.max(1, options.ttlTurns);
  const assistantTurnsAfter = new Array<number>(messages.length).fill(0);
  let turns = 0;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    assistantTurnsAfter[index] = turns;
    const message = messages[index];
    if (message?.role === "assistant" && message.stopReason !== "error" && message.stopReason !== "aborted") {
      turns += 1;
    }
  }

  return messages.map((message, index) => {
    const candidate = message as unknown as MessageWithContent;
    if (!Array.isArray(candidate.content) || !candidate.content.some(isImageContent)) return message;

    const source = candidate.toolCallId ? sources.get(candidate.toolCallId) : undefined;
    const shouldEvict = assistantTurnsAfter[index]! >= ttlTurns;
    let changed = false;
    const content = candidate.content.map((block) => {
      if (!isImageContent(block)) return block;
      try {
        const record = cacheImage(block, options.cacheDir, source);
        if (!shouldEvict) return block;
        changed = true;
        return placeholder(record, ttlTurns);
      } catch {
        // Never remove an image unless a recoverable cache copy exists.
        return block;
      }
    });
    return changed ? ({ ...message, content } as AgentMessage) : message;
  });
}

export function cacheStats(cacheDir: string): CacheStats {
  if (!existsSync(cacheDir)) return { images: 0, bytes: 0, metadataFiles: 0 };
  let images = 0;
  let bytes = 0;
  let metadataFiles = 0;
  for (const name of readdirSync(cacheDir)) {
    const file = path.join(cacheDir, name);
    let stat;
    try {
      stat = statSync(file);
    } catch {
      continue;
    }
    if (!stat.isFile()) continue;
    if (name.endsWith(".json")) metadataFiles += 1;
    else {
      images += 1;
      bytes += stat.size;
    }
  }
  return { images, bytes, metadataFiles };
}

function removeExpired(cacheDir: string, maxAgeDays: number): number {
  if (!existsSync(cacheDir) || maxAgeDays === 0) return 0;
  ensureCacheDir(cacheDir);
  const cutoff = Date.now() - maxAgeDays * 24 * 60 * 60 * 1000;
  let removed = 0;
  for (const name of readdirSync(cacheDir)) {
    const match = /^([a-f0-9]{64})\.json$/.exec(name);
    if (!match) continue;
    const metadataPath = path.join(cacheDir, name);
    try {
      const metadata = readMetadata(metadataPath);
      if (!metadata || metadata.sha256 !== match[1] || typeof metadata.mimeType !== "string") continue;
      const lastAccess = Date.parse(metadata.lastAccessedAt || metadata.createdAt || "");
      if (!Number.isFinite(lastAccess) || lastAccess >= cutoff) continue;
      const imagePath = path.join(cacheDir, `${match[1]}.${imageExtension(metadata.mimeType)}`);
      rmSync(imagePath, { force: true });
      rmSync(metadataPath, { force: true });
      removed += 1;
    } catch {
      // Ignore malformed metadata instead of deleting an unknown image.
    }
  }
  return removed;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

export default function imageContextCacheExtension(pi: ExtensionAPI) {
  const options = loadOptions();

  pi.on("session_start", async (_event, ctx) => {
    sourceByToolCall.clear();
    recordByHash.clear();
    const removed = removeExpired(options.cacheDir, options.maxAgeDays);
    if (removed > 0 && ctx.hasUI) {
      ctx.ui.notify(`Image context cache: removed ${removed} expired image(s).`, "info");
    }
  });

  pi.on("tool_result", async (event) => {
    const images = event.content.filter(isImageContent);
    if (images.length === 0) return;
    const source = sourceFromToolInput(event.toolName, event.input);
    if (source) sourceByToolCall.set(event.toolCallId, source);
    for (const image of images) {
      try {
        cacheImage(image, options.cacheDir, source);
      } catch {
        // Caching is best-effort. Never break the originating tool result.
      }
    }
  });

  pi.on("context", async (event) => ({
    messages: pruneImageContext(event.messages, options),
  }));

  pi.registerCommand("image-cache", {
    description: "Show or clear the pi image context cache: /image-cache [stats|clear]",
    handler: async (args, ctx) => {
      const action = args.trim() || "stats";
      if (action === "stats") {
        const stats = cacheStats(options.cacheDir);
        ctx.ui.notify(
          `Image context cache: ${stats.images} image(s), ${formatBytes(stats.bytes)}\n${options.cacheDir}\nTTL: ${options.ttlTurns} assistant turn(s)`,
          "info",
        );
        return;
      }
      if (action === "clear") {
        const confirmed = !ctx.hasUI || (await ctx.ui.confirm("Clear image context cache?", options.cacheDir));
        if (!confirmed) return;
        rmSync(options.cacheDir, { recursive: true, force: true });
        recordByHash.clear();
        sourceByToolCall.clear();
        ctx.ui.notify("Image context cache cleared.", "info");
        return;
      }
      ctx.ui.notify("Usage: /image-cache [stats|clear]", "warning");
    },
  });
}
