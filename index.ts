import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Box, Image as TuiImage, Text } from "@earendil-works/pi-tui";
import { createHash, randomBytes } from "node:crypto";
import type { Stats } from "node:fs";
import {
  chmodSync,
  closeSync,
  constants as fsConstants,
  existsSync,
  fstatSync,
  lstatSync,
  mkdirSync,
  openSync,
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
const CACHE_HIT_ENTRY_TYPE = "pi-image-context-cache-hit";

export interface CacheRecord {
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

export interface CacheHitEntryData extends CacheRecord {
  timestamp: number;
  hitId?: string;
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

function cacheHitPlaceholder(record: CacheRecord): TextContent {
  const source = record.source ? `\nOriginal source: ${record.source}` : "";
  return {
    type: "text",
    text:
      `[Image cache hit — payload not resent.]` +
      `\nCached image: ${record.path}${source}` +
      `\nSHA-256: ${record.sha256}` +
      `\nMIME: ${record.mimeType}; bytes: ${record.bytes}` +
      `\nThe model has already inspected this image in this session. To force visual re-inspection, use the read tool on the cached image path above.`,
  };
}

interface ImageContextDeduplication {
  seenHashes: ReadonlySet<string>;
  onFreshImage?: (record: CacheRecord) => void;
  onCacheHit?: (record: CacheRecord, eventId: string) => void;
}

function isExplicitCacheRead(source: string | undefined, record: CacheRecord): boolean {
  return typeof source === "string" && path.resolve(source) === record.path;
}

export function pruneImageContext(
  messages: AgentMessage[],
  options: Pick<ImageCacheOptions, "cacheDir" | "ttlTurns">,
  onEvict?: (record: CacheRecord) => void,
  deduplication?: ImageContextDeduplication,
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

  const offeredThisRequest = new Set<string>();

  return messages.map((message, index) => {
    const candidate = message as unknown as MessageWithContent;
    if (!Array.isArray(candidate.content) || !candidate.content.some(isImageContent)) return message;

    const source = candidate.toolCallId ? sources.get(candidate.toolCallId) : undefined;
    const messageTimestamp = typeof (message as { timestamp?: unknown }).timestamp === "number"
      ? String((message as { timestamp: number }).timestamp)
      : "untimed";
    const eventId = candidate.toolCallId || `${message.role}:${messageTimestamp}:${index}`;
    const shouldEvict = assistantTurnsAfter[index]! >= ttlTurns;
    let changed = false;
    const content = candidate.content.map((block) => {
      if (!isImageContent(block)) return block;
      try {
        const record = cacheImage(block, options.cacheDir, source);
        if (!shouldEvict) {
          const alreadyOffered = offeredThisRequest.has(record.sha256);
          const cacheHit = Boolean(deduplication?.seenHashes.has(record.sha256) || alreadyOffered);
          if (cacheHit && !isExplicitCacheRead(source, record)) {
            changed = true;
            deduplication?.onCacheHit?.(record, eventId);
            return cacheHitPlaceholder(record);
          }
          offeredThisRequest.add(record.sha256);
          deduplication?.onFreshImage?.(record);
          return block;
        }
        changed = true;
        onEvict?.(record);
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

export function isCacheHitEntryData(value: unknown): value is CacheHitEntryData {
  if (!value || typeof value !== "object") return false;
  const data = value as Partial<CacheHitEntryData>;
  return (
    typeof data.sha256 === "string" &&
    /^[a-f0-9]{64}$/.test(data.sha256) &&
    typeof data.path === "string" &&
    path.isAbsolute(data.path) &&
    typeof data.mimeType === "string" &&
    /^image\/[a-z0-9.+-]+$/i.test(data.mimeType) &&
    typeof data.bytes === "number" &&
    Number.isSafeInteger(data.bytes) &&
    data.bytes >= 0 &&
    typeof data.timestamp === "number" &&
    Number.isFinite(data.timestamp) &&
    (data.source === undefined || typeof data.source === "string") &&
    (data.hitId === undefined || typeof data.hitId === "string")
  );
}

export function readCachedPreview(data: CacheHitEntryData, cacheDir: string): string | undefined {
  const expectedPath = path.join(cacheDir, `${data.sha256}.${imageExtension(data.mimeType)}`);
  if (path.resolve(data.path) !== expectedPath || typeof fsConstants.O_NOFOLLOW !== "number") return undefined;

  let descriptor: number | undefined;
  try {
    descriptor = openSync(expectedPath, fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW);
    const stat = fstatSync(descriptor);
    if (!stat.isFile()) return undefined;
    const getuid = process.getuid;
    if (typeof getuid === "function" && stat.uid !== getuid()) return undefined;
    if ((stat.mode & 0o077) !== 0 || stat.size !== data.bytes) return undefined;
    const bytes = readFileSync(descriptor);
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    return sha256 === data.sha256 ? bytes.toString("base64") : undefined;
  } catch {
    return undefined;
  } finally {
    if (descriptor !== undefined) closeSync(descriptor);
  }
}

export default function imageContextCacheExtension(pi: ExtensionAPI) {
  const options = loadOptions();
  const seenHashes = new Set<string>();
  const pendingSeenHashes = new Set<string>();
  const pendingEntries = new Map<string, CacheHitEntryData>();
  const queueCacheCard = (record: CacheRecord, hitId: string) => {
    if (pendingEntries.has(hitId)) return;
    pendingEntries.set(hitId, { ...record, timestamp: Date.now(), hitId });
  };

  pi.registerEntryRenderer<CacheHitEntryData>(CACHE_HIT_ENTRY_TYPE, (entry, { expanded }, theme) => {
    const data = entry.data;
    const box = new Box(1, 0, (text) => theme.bg("customMessageBg", text));
    if (!isCacheHitEntryData(data)) {
      box.addChild(new Text(theme.fg("warning", "♻ Image context cache entry has invalid data"), 0, 0));
      return box;
    }

    box.addChild(
      new Text(
        `${theme.fg("success", "♻")} ${theme.fg("accent", "Image cache hit")} ${theme.fg("muted", `· ${formatBytes(data.bytes)} not resent`)}`,
        0,
        0,
      ),
    );
    if (!expanded) return box;

    box.addChild(new Text(theme.fg("dim", `${data.sha256.slice(0, 12)}… · ${data.path}`), 0, 0));
    if (data.source) box.addChild(new Text(theme.fg("dim", `source: ${data.source}`), 0, 0));
    const imageData = readCachedPreview(data, options.cacheDir);
    if (!imageData) {
      box.addChild(new Text(theme.fg("warning", "Cached preview is unavailable."), 0, 0));
      return box;
    }
    try {
      box.addChild(
        new TuiImage(
          imageData,
          data.mimeType,
          { fallbackColor: (text) => theme.fg("muted", text) },
          { maxWidthCells: 72, maxHeightCells: 20, filename: path.basename(data.path) },
        ),
      );
    } catch {
      box.addChild(new Text(theme.fg("warning", "Cached preview could not be rendered."), 0, 0));
    }
    return box;
  });

  pi.on("session_start", async (_event, ctx) => {
    sourceByToolCall.clear();
    recordByHash.clear();
    seenHashes.clear();
    pendingSeenHashes.clear();
    pendingEntries.clear();
    for (const entry of ctx.sessionManager.getEntries()) {
      if (entry.type !== "custom" || entry.customType !== CACHE_HIT_ENTRY_TYPE) continue;
      if (!isCacheHitEntryData(entry.data)) continue;
      seenHashes.add(entry.data.sha256);
    }

    let successfulAssistantAfter = false;
    const branch = ctx.sessionManager.getBranch();
    for (let index = branch.length - 1; index >= 0; index -= 1) {
      const entry = branch[index];
      if (entry?.type !== "message") continue;
      const message = entry.message;
      if (message.role === "assistant") {
        if (message.stopReason !== "error" && message.stopReason !== "aborted") successfulAssistantAfter = true;
        continue;
      }
      const candidate = message as unknown as MessageWithContent;
      if (!successfulAssistantAfter || !Array.isArray(candidate.content)) continue;
      for (const block of candidate.content) {
        if (!isImageContent(block)) continue;
        try {
          seenHashes.add(cacheImage(block, options.cacheDir).sha256);
        } catch {
          // A failed cache copy must never enable deduplication.
        }
      }
    }
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
    const offeredInResult = new Set<string>();
    let changed = false;
    const content = event.content.map((block) => {
      if (!isImageContent(block)) return block;
      try {
        const record = cacheImage(block, options.cacheDir, source);
        const cacheHit = seenHashes.has(record.sha256) || offeredInResult.has(record.sha256);
        if (cacheHit && !isExplicitCacheRead(source, record)) {
          changed = true;
          queueCacheCard(record, `tool:${event.toolCallId}:${record.sha256}`);
          return cacheHitPlaceholder(record);
        }
        offeredInResult.add(record.sha256);
        return block;
      } catch {
        // Caching is best-effort. Never remove an image without a verified cache copy.
        return block;
      }
    });
    return changed ? { content } : undefined;
  });

  pi.on("context", async (event) => ({
    messages: pruneImageContext(event.messages, options, undefined, {
      seenHashes,
      onFreshImage: (record) => pendingSeenHashes.add(record.sha256),
      onCacheHit: (record, eventId) => queueCacheCard(record, `context:${eventId}:${record.sha256}`),
    }),
  }));

  pi.on("turn_end", async (event) => {
    if (
      event.message.role === "assistant" &&
      event.message.stopReason !== "error" &&
      event.message.stopReason !== "aborted"
    ) {
      for (const sha256 of pendingSeenHashes) seenHashes.add(sha256);
    }
    pendingSeenHashes.clear();
    for (const data of pendingEntries.values()) {
      pi.appendEntry<CacheHitEntryData>(CACHE_HIT_ENTRY_TYPE, data);
    }
    pendingEntries.clear();
  });

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
        seenHashes.clear();
        pendingSeenHashes.clear();
        pendingEntries.clear();
        ctx.ui.notify("Image context cache cleared.", "info");
        return;
      }
      ctx.ui.notify("Usage: /image-cache [stats|clear]", "warning");
    },
  });
}
