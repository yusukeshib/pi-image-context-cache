import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { complete, type ImageContent, type TextContent, type UserMessage } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Box, Image as TuiImage, Text } from "@earendil-works/pi-tui";
import { createHash, randomBytes } from "node:crypto";
import type { Stats } from "node:fs";
import {
  chmodSync,
  closeSync,
  constants as fsConstants,
  existsSync,
  fchmodSync,
  fstatSync,
  fsyncSync,
  linkSync,
  lstatSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";

const DEFAULT_CACHE_DIR = path.join(os.homedir(), ".cache", "pi-image-context");
const DEFAULT_TTL_TURNS = 1;
const DEFAULT_MAX_AGE_DAYS = 30;
const DEFAULT_MAX_IMAGE_BYTES = 50 * 1024 * 1024;
const DEFAULT_MAX_CACHE_BYTES = 1024 * 1024 * 1024;
const CACHE_SENTINEL = ".pi-image-context-cache-v1";
const CACHE_HIT_ENTRY_TYPE = "pi-image-context-cache-hit";
const MAX_SOURCE_BYTES = 4096;
const MAX_VISION_QUESTION_CHARS = 8192;
const MAX_VISION_SUMMARY_BYTES = 64 * 1024;
const MAX_METADATA_BYTES = 256 * 1024;
const VISION_PROMPT_VERSION = "v1";

const IMAGE_EXTENSIONS = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/gif": "gif",
  "image/webp": "webp",
  "image/bmp": "bmp",
} as const;

type SupportedImageMime = keyof typeof IMAGE_EXTENSIONS;

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
  maxImageBytes: number;
  maxCacheBytes: number;
  visionMode: "isolated" | "direct";
  visionModel?: string;
  visionMaxTokens: number;
}

interface VisionAnalysis {
  summary: string;
  model: string;
  cost?: number;
  cached: boolean;
}

interface VisionCacheData extends VisionAnalysis {
  version: 1;
  sha256: string;
  queryHash: string;
  question: string;
  keyModel: string;
  createdAt: string;
}

export interface ExtensionDependencies {
  analyzeVision?: (
    image: ImageContent,
    question: string,
    ctx: ExtensionContext,
    signal?: AbortSignal,
  ) => Promise<Omit<VisionAnalysis, "cached">>;
}

interface CachePolicy {
  maxImageBytes: number;
  maxCacheBytes: number;
}

interface CacheStats {
  images: number;
  bytes: number;
  metadataFiles: number;
}

export interface CacheHitEntryData extends CacheRecord {
  timestamp: number;
  hitId?: string;
  kind?: "cache-hit" | "vision";
  model?: string;
  cached?: boolean;
}

type MessageWithContent = {
  content?: string | unknown[];
  toolCallId?: string;
};

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

function positiveInteger(value: string | undefined, fallback: number): number {
  const parsed = nonNegativeInteger(value, fallback);
  return parsed > 0 ? parsed : fallback;
}

function loadOptions(): ImageCacheOptions {
  return {
    cacheDir: path.resolve(expandHome(process.env.PI_IMAGE_CONTEXT_CACHE_DIR || DEFAULT_CACHE_DIR)),
    ttlTurns: Math.max(1, nonNegativeInteger(process.env.PI_IMAGE_CONTEXT_CACHE_TTL_TURNS, DEFAULT_TTL_TURNS)),
    maxAgeDays: nonNegativeInteger(process.env.PI_IMAGE_CONTEXT_CACHE_MAX_AGE_DAYS, DEFAULT_MAX_AGE_DAYS),
    maxImageBytes: positiveInteger(process.env.PI_IMAGE_CONTEXT_CACHE_MAX_IMAGE_BYTES, DEFAULT_MAX_IMAGE_BYTES),
    maxCacheBytes: positiveInteger(process.env.PI_IMAGE_CONTEXT_CACHE_MAX_BYTES, DEFAULT_MAX_CACHE_BYTES),
    visionMode: process.env.PI_IMAGE_CONTEXT_CACHE_VISION_MODE === "direct" ? "direct" : "isolated",
    visionModel: process.env.PI_IMAGE_CONTEXT_CACHE_VISION_MODEL || undefined,
    visionMaxTokens: positiveInteger(process.env.PI_IMAGE_CONTEXT_CACHE_VISION_MAX_TOKENS, 2048),
  };
}

function isImageContent(value: unknown): value is ImageContent {
  if (!value || typeof value !== "object") return false;
  const block = value as Partial<ImageContent>;
  return block.type === "image" && typeof block.data === "string" && typeof block.mimeType === "string";
}

function supportedImageMime(mimeType: string): SupportedImageMime | undefined {
  const normalized = mimeType.toLowerCase();
  return normalized in IMAGE_EXTENSIONS ? (normalized as SupportedImageMime) : undefined;
}

function imageExtension(mimeType: string): string {
  const supported = supportedImageMime(mimeType);
  if (!supported) throw new Error(`Unsupported image MIME type: ${mimeType}`);
  return IMAGE_EXTENSIONS[supported];
}

function hasMatchingImageSignature(bytes: Buffer, mimeType: SupportedImageMime): boolean {
  switch (mimeType) {
    case "image/png":
      return bytes.length >= 8 && bytes.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
    case "image/jpeg":
      return bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
    case "image/gif":
      return bytes.length >= 6 && ["GIF87a", "GIF89a"].includes(bytes.subarray(0, 6).toString("ascii"));
    case "image/webp":
      return bytes.length >= 12 && bytes.subarray(0, 4).toString("ascii") === "RIFF" && bytes.subarray(8, 12).toString("ascii") === "WEBP";
    case "image/bmp":
      return bytes.length >= 2 && bytes[0] === 0x42 && bytes[1] === 0x4d;
  }
}

function decodeImage(image: ImageContent, maxBytes: number): { bytes: Buffer; mimeType: SupportedImageMime; sha256: string } {
  const mimeType = supportedImageMime(image.mimeType);
  if (!mimeType) throw new Error(`Unsupported image MIME type: ${image.mimeType}`);
  const encoded = image.data;
  if (encoded.length === 0 || encoded.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(encoded)) {
    throw new Error("Image payload is not canonical base64.");
  }
  const estimatedBytes = Math.ceil(encoded.length * 3 / 4);
  if (estimatedBytes > maxBytes + 2) throw new Error(`Image exceeds ${maxBytes} byte limit.`);
  const bytes = Buffer.from(encoded, "base64");
  if (bytes.length === 0 || bytes.length > maxBytes) throw new Error(`Image exceeds ${maxBytes} byte limit.`);
  if (bytes.toString("base64") !== encoded) {
    throw new Error("Image payload is not canonical base64.");
  }
  if (!hasMatchingImageSignature(bytes, mimeType)) {
    throw new Error(`Image bytes do not match declared MIME type ${mimeType}.`);
  }
  return { bytes, mimeType, sha256: createHash("sha256").update(bytes).digest("hex") };
}

export function hashImage(image: ImageContent): string {
  return decodeImage(image, DEFAULT_MAX_IMAGE_BYTES).sha256;
}

function normalizedSource(source: string | undefined): string | undefined {
  if (!source) return undefined;
  const clean = source.replace(/[\p{Cc}\p{Cf}]/gu, "�");
  const home = os.homedir();
  const redacted = clean === home ? "~" : clean.startsWith(`${home}${path.sep}`) ? `~${clean.slice(home.length)}` : clean;
  return truncateUtf8(redacted, MAX_SOURCE_BYTES);
}

function quoted(value: string): string {
  return JSON.stringify(value);
}

function displaySafe(value: string): string {
  return value.replace(/[\u0000-\u001f\u007f-\u009f]/g, "�");
}

function assertOwnedPath(stat: Stats, target: string): void {
  const getuid = process.getuid;
  if (typeof getuid === "function" && stat.uid !== getuid()) {
    throw new Error(`Image cache path is not owned by the current user: ${target}`);
  }
}

function assertSafeCacheLocation(cacheDir: string): void {
  const resolved = path.resolve(cacheDir);
  const root = path.parse(resolved).root;
  if (resolved === root || resolved === path.resolve(os.homedir())) {
    throw new Error(`Refusing dangerous image cache directory: ${resolved}`);
  }
}

function assertTrustedAncestry(cacheDir: string): void {
  if (process.platform === "win32") return;
  const uid = process.getuid?.();
  let current = path.dirname(cacheDir);
  const root = path.parse(current).root;
  while (true) {
    const stat = lstatSync(current);
    const sticky = (stat.mode & 0o1000) !== 0;
    if ((uid !== undefined && stat.uid !== uid && stat.uid !== 0) || ((stat.mode & 0o022) !== 0 && !sticky)) {
      throw new Error(`Image cache has untrusted writable ancestry: ${current}`);
    }
    if (current === root) break;
    current = path.dirname(current);
  }
}

function openPrivateFile(file: string): { descriptor: number; stat: Stats } {
  if (typeof fsConstants.O_NOFOLLOW !== "number") {
    throw new Error("This platform cannot safely open private cache files.");
  }
  const descriptor = openSync(file, fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW);
  try {
    const stat = fstatSync(descriptor);
    if (!stat.isFile()) throw new Error(`Image cache entry must be a regular file: ${file}`);
    assertOwnedPath(stat, file);
    if ((stat.mode & 0o077) !== 0) fchmodSync(descriptor, 0o600);
    return { descriptor, stat };
  } catch (error) {
    closeSync(descriptor);
    throw error;
  }
}

function readPrivateFile(file: string, maxBytes = DEFAULT_MAX_IMAGE_BYTES): { bytes: Buffer; identity: string } {
  const { descriptor, stat } = openPrivateFile(file);
  try {
    if (stat.size > maxBytes) throw new Error(`Private cache file exceeds ${maxBytes} byte limit: ${file}`);
    const bytes = readFileSync(descriptor);
    const after = fstatSync(descriptor);
    if (after.dev !== stat.dev || after.ino !== stat.ino || after.size !== bytes.length || after.mtimeMs !== stat.mtimeMs) {
      throw new Error(`Image cache entry changed while reading: ${file}`);
    }
    return { bytes, identity: `${after.dev}:${after.ino}:${after.size}:${after.mtimeMs}` };
  } finally {
    closeSync(descriptor);
  }
}

function privateFileIdentity(file: string): string {
  const { descriptor, stat } = openPrivateFile(file);
  try {
    return `${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeMs}`;
  } finally {
    closeSync(descriptor);
  }
}

function writePrivateTemp(file: string, data: string | Buffer): void {
  if (typeof fsConstants.O_NOFOLLOW !== "number") throw new Error("O_NOFOLLOW is required for image caching.");
  const descriptor = openSync(
    file,
    fsConstants.O_WRONLY | fsConstants.O_CREAT | fsConstants.O_EXCL | fsConstants.O_NOFOLLOW,
    0o600,
  );
  try {
    writeFileSync(descriptor, data);
    fchmodSync(descriptor, 0o600);
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}

function ensureCacheDir(cacheDir: string): string {
  assertSafeCacheLocation(cacheDir);
  mkdirSync(cacheDir, { recursive: true, mode: 0o700 });
  const canonical = realpathSync(cacheDir);
  assertSafeCacheLocation(canonical);
  const stat = lstatSync(canonical);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`Image cache path must be a real directory: ${canonical}`);
  }
  assertOwnedPath(stat, canonical);
  assertTrustedAncestry(canonical);
  const sentinel = path.join(canonical, CACHE_SENTINEL);
  if (!existsSync(sentinel)) {
    const unexpected = readdirSync(canonical).filter(
      (name) => !METADATA_NAME.test(name) && !IMAGE_NAME.test(name) && !TEMP_NAME.test(name) && !VISION_NAME.test(name),
    );
    if (unexpected.length > 0) {
      throw new Error(`Refusing to initialize a non-empty cache directory without a sentinel: ${canonical}`);
    }
    if ((stat.mode & 0o077) !== 0) chmodSync(canonical, 0o700);
    writePrivateTemp(sentinel, "pi-image-context-cache:v1\n");
  } else if ((stat.mode & 0o077) !== 0) {
    chmodSync(canonical, 0o700);
  }
  const sentinelData = readPrivateFile(sentinel, 64).bytes.toString("utf8");
  if (sentinelData !== "pi-image-context-cache:v1\n") throw new Error(`Invalid image cache sentinel: ${sentinel}`);
  return canonical;
}

function ensureImageFile(file: string, data: Buffer, sha256: string): string {
  if (existsSync(file)) {
    const existing = readPrivateFile(file, data.length);
    const existingHash = createHash("sha256").update(existing.bytes).digest("hex");
    if (existingHash !== sha256) throw new Error(`Existing image cache entry failed hash verification: ${file}`);
    return existing.identity;
  }

  const temporary = path.join(
    path.dirname(file),
    `.${path.basename(file)}.${process.pid}.${randomBytes(6).toString("hex")}.tmp`,
  );
  try {
    writePrivateTemp(temporary, data);
    try {
      linkSync(temporary, file);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    }
  } finally {
    rmSync(temporary, { force: true });
  }
  const existing = readPrivateFile(file, data.length);
  const existingHash = createHash("sha256").update(existing.bytes).digest("hex");
  if (existingHash !== sha256) throw new Error(`Image cache entry failed hash verification: ${file}`);
  return existing.identity;
}

function isCacheMetadata(value: unknown, metadataPath: string): value is CacheMetadata {
  if (!value || typeof value !== "object") return false;
  const metadata = value as Partial<CacheMetadata>;
  const match = /^([a-f0-9]{64})\.json$/.exec(path.basename(metadataPath));
  const mimeType = typeof metadata.mimeType === "string" ? supportedImageMime(metadata.mimeType) : undefined;
  if (!match || !mimeType) return false;
  const expectedImagePath = path.join(path.dirname(metadataPath), `${match[1]}.${imageExtension(mimeType)}`);
  const createdAt = typeof metadata.createdAt === "string" ? Date.parse(metadata.createdAt) : Number.NaN;
  const lastAccessedAt = typeof metadata.lastAccessedAt === "string" ? Date.parse(metadata.lastAccessedAt) : Number.NaN;
  return (
    metadata.sha256 === match[1] &&
    metadata.path === expectedImagePath &&
    metadata.mimeType === mimeType &&
    typeof metadata.bytes === "number" &&
    Number.isSafeInteger(metadata.bytes) &&
    metadata.bytes > 0 &&
    Number.isFinite(createdAt) &&
    Number.isFinite(lastAccessedAt) &&
    createdAt <= Date.now() + 24 * 60 * 60 * 1000 &&
    lastAccessedAt <= Date.now() + 24 * 60 * 60 * 1000 &&
    (metadata.source === undefined || typeof metadata.source === "string")
  );
}

function readMetadata(file: string): CacheMetadata | undefined {
  try {
    const parsed: unknown = JSON.parse(readPrivateFile(file, MAX_METADATA_BYTES).bytes.toString("utf8"));
    return isCacheMetadata(parsed, file) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function writeMetadataAtomic(file: string, metadata: unknown): void {
  const temporary = path.join(
    path.dirname(file),
    `.${path.basename(file)}.${process.pid}.${randomBytes(6).toString("hex")}.tmp`,
  );
  try {
    writePrivateTemp(temporary, `${JSON.stringify(metadata, null, 2)}\n`);
    renameSync(temporary, file);
    if (typeof fsConstants.O_DIRECTORY === "number") {
      const directory = openSync(path.dirname(file), fsConstants.O_RDONLY | fsConstants.O_DIRECTORY);
      try {
        fsyncSync(directory);
      } finally {
        closeSync(directory);
      }
    }
  } finally {
    rmSync(temporary, { force: true });
  }
}

export function cacheImage(
  image: ImageContent,
  cacheDir: string,
  source?: string,
  policy: Partial<CachePolicy> = {},
  memo?: Map<string, MemoizedCacheRecord>,
): CacheRecord {
  const activeMemo = memo ?? new Map<string, MemoizedCacheRecord>();
  const limits: CachePolicy = {
    maxImageBytes: policy.maxImageBytes ?? DEFAULT_MAX_IMAGE_BYTES,
    maxCacheBytes: policy.maxCacheBytes ?? DEFAULT_MAX_CACHE_BYTES,
  };
  const decoded = decodeImage(image, Math.min(limits.maxImageBytes, limits.maxCacheBytes));
  const resolvedCacheDir = ensureCacheDir(path.resolve(cacheDir));
  const memoKey = `${resolvedCacheDir}\0${decoded.sha256}`;
  const eventSource = normalizedSource(source);
  const memoized = activeMemo.get(memoKey);
  if (memoized && existsSync(memoized.metadata.path)) {
    const identity = privateFileIdentity(memoized.metadata.path);
    if (identity === memoized.identity) {
      const now = Date.now();
      const lastAccess = Date.parse(memoized.metadata.lastAccessedAt);
      const needsTouch = !Number.isFinite(lastAccess) || now - lastAccess >= 60 * 60 * 1000;
      if (!needsTouch) return eventSource ? { ...memoized.metadata, source: eventSource } : memoized.metadata;
      const updated = { ...memoized.metadata, lastAccessedAt: new Date(now).toISOString() };
      writeMetadataAtomic(path.join(resolvedCacheDir, `${decoded.sha256}.json`), updated);
      activeMemo.set(memoKey, { metadata: updated, identity });
      return eventSource ? { ...updated, source: eventSource } : updated;
    }
  }

  const imagePath = path.join(resolvedCacheDir, `${decoded.sha256}.${imageExtension(decoded.mimeType)}`);
  const metadataPath = path.join(resolvedCacheDir, `${decoded.sha256}.json`);
  const identity = ensureImageFile(imagePath, decoded.bytes, decoded.sha256);
  const previous = readMetadata(metadataPath);
  const now = new Date().toISOString();
  const metadata: CacheMetadata = {
    sha256: decoded.sha256,
    path: imagePath,
    mimeType: decoded.mimeType,
    bytes: decoded.bytes.byteLength,
    source: previous?.source || eventSource,
    createdAt: typeof previous?.createdAt === "string" ? previous.createdAt : now,
    lastAccessedAt: now,
  };
  writeMetadataAtomic(metadataPath, metadata);
  activeMemo.set(memoKey, { metadata, identity });
  maintainCache(resolvedCacheDir, 0, limits.maxCacheBytes, decoded.sha256);
  return eventSource ? { ...metadata, source: eventSource } : metadata;
}

function visionCachePath(cacheDir: string, sha256: string, queryHash: string): string {
  return path.join(cacheDir, `${sha256}.vision.${queryHash}.json`);
}

function visionQueryHash(question: string, keyModel: string): string {
  return createHash("sha256").update(`${VISION_PROMPT_VERSION}\0${keyModel}\0${question}`).digest("hex");
}

function isVisionCacheData(
  value: unknown,
  sha256: string,
  queryHash: string,
  expectedQuestion?: string,
  expectedKeyModel?: string,
): value is VisionCacheData {
  if (!value || typeof value !== "object") return false;
  const data = value as Partial<VisionCacheData>;
  return (
    data.version === 1 &&
    data.sha256 === sha256 &&
    data.queryHash === queryHash &&
    typeof data.question === "string" &&
    data.question.length <= MAX_VISION_QUESTION_CHARS &&
    typeof data.keyModel === "string" &&
    data.keyModel.length <= 512 &&
    (expectedQuestion === undefined || data.question === expectedQuestion) &&
    (expectedKeyModel === undefined || data.keyModel === expectedKeyModel) &&
    visionQueryHash(data.question, data.keyModel) === queryHash &&
    typeof data.summary === "string" &&
    Buffer.byteLength(data.summary, "utf8") <= MAX_VISION_SUMMARY_BYTES &&
    typeof data.model === "string" &&
    data.model.length <= 512 &&
    (data.cost === undefined || (typeof data.cost === "number" && Number.isFinite(data.cost) && data.cost >= 0)) &&
    typeof data.createdAt === "string" &&
    Number.isFinite(Date.parse(data.createdAt))
  );
}

function readVisionCache(
  cacheDir: string,
  sha256: string,
  queryHash: string,
  question: string,
  keyModel: string,
  maxAgeDays: number,
): VisionCacheData | undefined {
  try {
    const file = visionCachePath(cacheDir, sha256, queryHash);
    const parsed: unknown = JSON.parse(readPrivateFile(file, MAX_METADATA_BYTES).bytes.toString("utf8"));
    if (!isVisionCacheData(parsed, sha256, queryHash, question, keyModel)) return undefined;
    const createdAt = Date.parse(parsed.createdAt);
    const now = Date.now();
    if (createdAt > now + 24 * 60 * 60 * 1000) return undefined;
    if (maxAgeDays !== 0 && createdAt < now - maxAgeDays * 24 * 60 * 60 * 1000) return undefined;
    return parsed;
  } catch {
    return undefined;
  }
}

function truncateUtf8(value: string, maxBytes: number): string {
  const bytes = Buffer.from(value, "utf8");
  if (bytes.length <= maxBytes) return value;
  return `${bytes.subarray(0, Math.max(0, maxBytes - 3)).toString("utf8").replace(/�+$/, "")}…`;
}

function validatedVisionAnalysis(value: Omit<VisionAnalysis, "cached">): Omit<VisionAnalysis, "cached"> {
  if (typeof value.summary !== "string" || !value.summary.trim()) throw new Error("Vision analysis returned no text.");
  if (typeof value.model !== "string" || !value.model.trim() || value.model.length > 512) {
    throw new Error("Vision analysis returned an invalid model identifier.");
  }
  if (value.cost !== undefined && (typeof value.cost !== "number" || !Number.isFinite(value.cost) || value.cost < 0)) {
    throw new Error("Vision analysis returned an invalid cost.");
  }
  return { summary: truncateUtf8(value.summary.trim(), MAX_VISION_SUMMARY_BYTES), model: value.model, cost: value.cost };
}

function latestUserQuestion(ctx: ExtensionContext): string {
  const branch = ctx.sessionManager.getBranch();
  for (let index = branch.length - 1; index >= 0; index -= 1) {
    const entry = branch[index];
    if (entry?.type !== "message" || entry.message.role !== "user") continue;
    const content = entry.message.content;
    const text = typeof content === "string"
      ? content
      : content.filter((block): block is TextContent => block.type === "text").map((block) => block.text).join("\n");
    if (text.trim()) return truncateUtf8(text.trim(), MAX_VISION_QUESTION_CHARS);
  }
  return "Describe this image thoroughly, including visible text, layout, objects, colors, and details relevant to follow-up questions.";
}

async function defaultVisionAnalyzer(
  image: ImageContent,
  question: string,
  ctx: ExtensionContext,
  options: ImageCacheOptions,
  signal?: AbortSignal,
): Promise<Omit<VisionAnalysis, "cached">> {
  let model = ctx.model;
  if (options.visionModel) {
    const separator = options.visionModel.indexOf("/");
    if (separator <= 0) throw new Error("PI_IMAGE_CONTEXT_CACHE_VISION_MODEL must be provider/model.");
    model = ctx.modelRegistry.find(options.visionModel.slice(0, separator), options.visionModel.slice(separator + 1));
  }
  if (!model) throw new Error("No Vision model is configured.");
  if (!model.input?.includes("image")) throw new Error(`Model ${model.provider}/${model.id} does not accept images.`);
  const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
  if (!auth.ok) throw new Error(auth.error);
  if (!auth.apiKey) throw new Error(`No API key for ${model.provider}/${model.id}.`);

  const userMessage: UserMessage = {
    role: "user",
    content: [
      { type: "text", text: `User request:\n${question}\n\nAnalyze the attached image. Return a precise, self-contained answer for another model that will not receive the image. Include OCR text verbatim when present, spatial relationships, uncertainty, and details useful for follow-up questions.` },
      image,
    ],
    timestamp: Date.now(),
  };
  const response = await complete(
    model,
    {
      systemPrompt: "You are an isolated Vision analyst. Analyze only the supplied image and request. Do not assume the downstream model can see the image. Be precise and concise, but preserve important visual and textual details.",
      messages: [userMessage],
    },
    {
      apiKey: auth.apiKey,
      headers: auth.headers,
      env: auth.env,
      maxTokens: options.visionMaxTokens,
      signal,
    },
  );
  if (response.stopReason === "aborted") throw new Error("Vision analysis was aborted.");
  if (response.stopReason === "error") throw new Error(response.errorMessage || "Vision analysis failed.");
  const summary = response.content
    .filter((block): block is TextContent => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();
  if (!summary) throw new Error("Vision analysis returned no text.");
  return {
    summary: truncateUtf8(summary, MAX_VISION_SUMMARY_BYTES),
    model: `${model.provider}/${model.id}`,
    cost: response.usage?.cost.total,
  };
}

function isolatedVisionPlaceholder(record: CacheRecord, analysis: VisionAnalysis): TextContent {
  const source = record.source ? `\nOriginal source: ${quoted(record.source)}` : "";
  const cost = analysis.cost === undefined ? "" : `; isolated cost: $${analysis.cost.toFixed(4)}`;
  return {
    type: "text",
    text:
      `[Image analyzed in isolated Vision context — payload not sent to main model.]` +
      `\nCached image: ${quoted(record.path)}${source}` +
      `\nSHA-256: ${record.sha256}` +
      `\nVision model: ${analysis.model}; cached: ${analysis.cached}${cost}` +
      `\n\nVision analysis:\n${analysis.summary}` +
      `\n\nTo bypass isolation and let the main model inspect pixels directly, use the read tool on the cached image path above.`,
  };
}

function isolationFailurePlaceholder(record: CacheRecord, error: unknown): TextContent {
  const message = error instanceof Error ? error.message : String(error);
  return {
    type: "text",
    text:
      `[Isolated Vision analysis failed — image payload was NOT sent to the main model.]` +
      `\nCached image: ${quoted(record.path)}` +
      `\nSHA-256: ${record.sha256}` +
      `\nError: ${truncateUtf8(message, 2000)}` +
      `\nTo explicitly inspect the image in the main context, use the read tool on the cached image path above.`,
  };
}

function sourceFromToolInput(toolName: string, input: Record<string, unknown>): string | undefined {
  if (toolName !== "read") return undefined;
  return typeof input.path === "string" ? input.path : undefined;
}

function toolCallSources(messages: AgentMessage[]): Map<string, string> {
  const result = new Map<string, string>();
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
  const source = record.source ? `\nOriginal source: ${quoted(record.source)}` : "";
  return {
    type: "text",
    text:
      `[Image payload evicted after ${ttlTurns} assistant turn${ttlTurns === 1 ? "" : "s"}.]` +
      `\nCached image: ${quoted(record.path)}${source}` +
      `\nSHA-256: ${record.sha256}` +
      `\nMIME: ${record.mimeType}; bytes: ${record.bytes}` +
      `\nUse the read tool on the cached image when visual inspection is needed again.`,
  };
}

function cacheHitPlaceholder(record: CacheRecord): TextContent {
  const source = record.source ? `\nOriginal source: ${quoted(record.source)}` : "";
  return {
    type: "text",
    text:
      `[Image cache hit — payload not resent.]` +
      `\nCached image: ${quoted(record.path)}${source}` +
      `\nSHA-256: ${record.sha256}` +
      `\nMIME: ${record.mimeType}; bytes: ${record.bytes}` +
      `\nThe model has already inspected this image in this session. To force visual re-inspection, use the read tool on the cached image path above.`,
  };
}

interface ContextRecordMemo {
  data: string;
  mimeType: string;
  record: CacheRecord;
}

interface ImageContextDeduplication {
  seenHashes: ReadonlySet<string>;
  onFreshImage?: (record: CacheRecord) => void;
  onCacheHit?: (record: CacheRecord, eventId: string) => void;
  contextRecords?: Map<string, ContextRecordMemo>;
  cacheMemo?: Map<string, MemoizedCacheRecord>;
  policy?: Partial<CachePolicy>;
}

function recordIsRecoverable(record: CacheRecord, cacheDir: string): boolean {
  try {
    const expected = path.join(realpathSync(cacheDir), `${record.sha256}.${imageExtension(record.mimeType)}`);
    if (record.path !== expected) return false;
    const identity = privateFileIdentity(record.path);
    const size = Number(identity.split(":")[2]);
    return size === record.bytes;
  } catch {
    return false;
  }
}

function isExplicitCacheRead(source: string | undefined, record: CacheRecord): boolean {
  return typeof source === "string" && path.resolve(source) === record.path;
}

export function pruneImageContext(
  messages: AgentMessage[],
  options: Pick<ImageCacheOptions, "cacheDir" | "ttlTurns"> & Partial<Pick<ImageCacheOptions, "maxImageBytes" | "maxCacheBytes">>,
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
    const content = candidate.content.map((block, blockIndex) => {
      if (!isImageContent(block)) return block;
      try {
        const recordKey = `${eventId}:${blockIndex}`;
        const memoized = deduplication?.contextRecords?.get(recordKey);
        const record = memoized?.data === block.data && memoized.mimeType === block.mimeType && recordIsRecoverable(memoized.record, options.cacheDir)
          ? { ...memoized.record, source: normalizedSource(source) || memoized.record.source }
          : cacheImage(
              block,
              options.cacheDir,
              source,
              {
                maxImageBytes: options.maxImageBytes,
                maxCacheBytes: options.maxCacheBytes,
                ...deduplication?.policy,
              },
              deduplication?.cacheMemo,
            );
        deduplication?.contextRecords?.set(recordKey, { data: block.data, mimeType: block.mimeType, record });
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

const METADATA_NAME = /^([a-f0-9]{64})\.json$/;
const IMAGE_NAME = /^([a-f0-9]{64})\.(jpg|png|gif|webp|bmp)$/;
const TEMP_NAME = /^\.[a-f0-9]{64}\..+\.\d+\.[a-f0-9]+\.tmp$/;
const VISION_NAME = /^([a-f0-9]{64})\.vision\.([a-f0-9]{64})\.json$/;

function removeCacheHash(cacheDir: string, sha256: string): number {
  let removed = 0;
  for (const extension of [...new Set(Object.values(IMAGE_EXTENSIONS)), "json"]) {
    const file = path.join(cacheDir, `${sha256}.${extension}`);
    if (!existsSync(file)) continue;
    rmSync(file, { force: true });
    removed += 1;
  }
  return removed;
}

function clearRecognizedCache(cacheDir: string): number {
  if (!existsSync(cacheDir)) return 0;
  cacheDir = ensureCacheDir(cacheDir);
  let removed = 0;
  for (const name of readdirSync(cacheDir)) {
    if (!METADATA_NAME.test(name) && !IMAGE_NAME.test(name) && !TEMP_NAME.test(name) && !VISION_NAME.test(name)) continue;
    rmSync(path.join(cacheDir, name), { force: true });
    removed += 1;
  }
  return removed;
}

export function cacheStats(cacheDir: string): CacheStats {
  if (!existsSync(cacheDir)) return { images: 0, bytes: 0, metadataFiles: 0 };
  cacheDir = ensureCacheDir(cacheDir);
  let images = 0;
  let bytes = 0;
  let metadataFiles = 0;
  for (const name of readdirSync(cacheDir)) {
    const match = METADATA_NAME.exec(name);
    if (!match) continue;
    const metadataPath = path.join(cacheDir, name);
    const metadata = readMetadata(metadataPath);
    if (!metadata) continue;
    try {
      const stat = lstatSync(metadata.path);
      if (stat.isSymbolicLink() || !stat.isFile() || stat.size !== metadata.bytes) continue;
      assertOwnedPath(stat, metadata.path);
      images += 1;
      bytes += stat.size;
      metadataFiles += 1;
    } catch {
      continue;
    }
  }
  return { images, bytes, metadataFiles };
}

function maintainCache(cacheDir: string, maxAgeDays: number, maxCacheBytes: number, keepHash?: string): number {
  if (!existsSync(cacheDir)) return 0;
  cacheDir = ensureCacheDir(cacheDir);
  const now = Date.now();
  const cutoff = maxAgeDays === 0 ? Number.NEGATIVE_INFINITY : now - maxAgeDays * 24 * 60 * 60 * 1000;
  const validHashes = new Set<string>();
  const entries: Array<{ metadata: CacheMetadata; lastAccess: number; bytes: number }> = [];
  let removed = 0;
  let totalBytes = 0;

  for (const name of readdirSync(cacheDir)) {
    const match = METADATA_NAME.exec(name);
    if (!match) continue;
    const metadataPath = path.join(cacheDir, name);
    const metadata = readMetadata(metadataPath);
    if (!metadata) {
      removed += removeCacheHash(cacheDir, match[1]!);
      continue;
    }
    let stat: Stats;
    try {
      stat = lstatSync(metadata.path);
      if (stat.isSymbolicLink() || !stat.isFile() || stat.size !== metadata.bytes) throw new Error("invalid image");
      assertOwnedPath(stat, metadata.path);
    } catch {
      removed += removeCacheHash(cacheDir, metadata.sha256);
      continue;
    }
    const lastAccess = Date.parse(metadata.lastAccessedAt);
    if (maxAgeDays !== 0 && lastAccess < cutoff && metadata.sha256 !== keepHash) {
      removed += removeCacheHash(cacheDir, metadata.sha256);
      continue;
    }
    const metadataBytes = lstatSync(metadataPath).size;
    const entryBytes = stat.size + metadataBytes;
    validHashes.add(metadata.sha256);
    totalBytes += entryBytes;
    entries.push({ metadata, lastAccess, bytes: entryBytes });
  }

  entries.sort((a, b) => a.lastAccess - b.lastAccess);
  for (const entry of entries) {
    if (totalBytes <= maxCacheBytes) break;
    if (entry.metadata.sha256 === keepHash) continue;
    totalBytes -= entry.bytes;
    validHashes.delete(entry.metadata.sha256);
    removed += removeCacheHash(cacheDir, entry.metadata.sha256);
  }

  const orphanCutoff = now - 60 * 60 * 1000;
  const visionEntries: Array<{ file: string; bytes: number; createdAt: number }> = [];
  for (const name of readdirSync(cacheDir)) {
    const file = path.join(cacheDir, name);
    try {
      const stat = lstatSync(file);
      if (stat.isSymbolicLink()) {
        rmSync(file, { force: true });
        removed += 1;
        continue;
      }
      const visionMatch = VISION_NAME.exec(name);
      if (visionMatch) {
        const cached = (() => {
          try {
            const parsed: unknown = JSON.parse(readPrivateFile(file, MAX_METADATA_BYTES).bytes.toString("utf8"));
            return isVisionCacheData(parsed, visionMatch[1]!, visionMatch[2]!) ? parsed : undefined;
          } catch {
            return undefined;
          }
        })();
        const createdAt = cached ? Date.parse(cached.createdAt) : Number.NaN;
        if (!cached || (maxAgeDays !== 0 && createdAt < cutoff)) {
          rmSync(file, { force: true });
          removed += 1;
        } else {
          totalBytes += stat.size;
          visionEntries.push({ file, bytes: stat.size, createdAt });
        }
      } else {
        const imageMatch = IMAGE_NAME.exec(name);
        if (imageMatch && !validHashes.has(imageMatch[1]!) && stat.mtimeMs < orphanCutoff) {
          rmSync(file, { force: true });
          removed += 1;
        } else if (TEMP_NAME.test(name) && stat.mtimeMs < orphanCutoff) {
          rmSync(file, { force: true });
          removed += 1;
        }
      }
    } catch {
      continue;
    }
  }
  visionEntries.sort((a, b) => a.createdAt - b.createdAt);
  for (const entry of visionEntries) {
    if (totalBytes <= maxCacheBytes) break;
    rmSync(entry.file, { force: true });
    totalBytes -= entry.bytes;
    removed += 1;
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
    supportedImageMime(data.mimeType) !== undefined &&
    typeof data.bytes === "number" &&
    Number.isSafeInteger(data.bytes) &&
    data.bytes > 0 &&
    data.bytes <= DEFAULT_MAX_IMAGE_BYTES &&
    typeof data.timestamp === "number" &&
    Number.isFinite(data.timestamp) &&
    (data.source === undefined || (typeof data.source === "string" && data.source.length <= MAX_SOURCE_BYTES + 1)) &&
    (data.hitId === undefined || (typeof data.hitId === "string" && data.hitId.length <= 1024)) &&
    (data.kind === undefined || data.kind === "cache-hit" || data.kind === "vision") &&
    (data.model === undefined || (typeof data.model === "string" && data.model.length <= 512)) &&
    (data.cached === undefined || typeof data.cached === "boolean")
  );
}

export function readCachedPreview(data: CacheHitEntryData, cacheDir: string): string | undefined {
  try {
    if (!isCacheHitEntryData(data)) return undefined;
    const expectedPath = path.join(realpathSync(cacheDir), `${data.sha256}.${imageExtension(data.mimeType)}`);
    if (path.resolve(data.path) !== expectedPath) return undefined;
    const cached = readPrivateFile(expectedPath);
    if (cached.bytes.length !== data.bytes || cached.bytes.length > DEFAULT_MAX_IMAGE_BYTES) return undefined;
    const sha256 = createHash("sha256").update(cached.bytes).digest("hex");
    const mimeType = supportedImageMime(data.mimeType);
    if (sha256 !== data.sha256 || !mimeType || !hasMatchingImageSignature(cached.bytes, mimeType)) return undefined;
    return cached.bytes.toString("base64");
  } catch {
    return undefined;
  }
}

export default function imageContextCacheExtension(pi: ExtensionAPI, dependencies: ExtensionDependencies = {}) {
  const options = loadOptions();
  const seenHashes = new Set<string>();
  const pendingSeenHashes = new Set<string>();
  const pendingEntries = new Map<string, CacheHitEntryData>();
  const offeredToolHashes = new Set<string>();
  const cacheMemo = new Map<string, MemoizedCacheRecord>();
  const contextRecords = new Map<string, ContextRecordMemo>();
  const visionMemo = new Map<string, VisionAnalysis>();
  const visionInFlight = new Map<string, Promise<VisionAnalysis>>();
  let visionGeneration = 0;
  let inputSequence = 0;
  const cachePolicy: CachePolicy = {
    maxImageBytes: options.maxImageBytes,
    maxCacheBytes: options.maxCacheBytes,
  };
  const queueCacheCard = (
    record: CacheRecord,
    hitId: string,
    details: Pick<CacheHitEntryData, "kind" | "model" | "cached"> = {},
  ) => {
    if (pendingEntries.has(hitId)) return;
    pendingEntries.set(hitId, { ...record, ...details, timestamp: Date.now(), hitId });
  };

  const analyzeIsolated = async (
    image: ImageContent,
    record: CacheRecord,
    question: string,
    ctx: ExtensionContext,
  ): Promise<VisionAnalysis> => {
    const normalizedQuestion = truncateUtf8(question.trim() || "Describe this image thoroughly.", MAX_VISION_QUESTION_CHARS);
    const modelKey = options.visionModel || (ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "unconfigured");
    const queryHash = visionQueryHash(normalizedQuestion, modelKey);
    const key = `${record.sha256}:${queryHash}`;
    const memoized = visionMemo.get(key);
    if (memoized) return { ...memoized, cached: true };
    const cacheDir = ensureCacheDir(options.cacheDir);
    const disk = readVisionCache(
      cacheDir,
      record.sha256,
      queryHash,
      normalizedQuestion,
      modelKey,
      options.maxAgeDays,
    );
    if (disk) {
      const cached: VisionAnalysis = { summary: disk.summary, model: disk.model, cost: disk.cost, cached: true };
      visionMemo.set(key, cached);
      return cached;
    }
    const running = visionInFlight.get(key);
    if (running) return running;
    const generation = visionGeneration;
    const promise = (async () => {
      const analyzed = dependencies.analyzeVision
        ? await dependencies.analyzeVision(image, normalizedQuestion, ctx, ctx.signal)
        : await defaultVisionAnalyzer(image, normalizedQuestion, ctx, options, ctx.signal);
      if (generation !== visionGeneration) throw new Error("Vision cache was cleared while analysis was running.");
      const valid = validatedVisionAnalysis(analyzed);
      const result: VisionAnalysis = { ...valid, cached: false };
      const data: VisionCacheData = {
        ...result,
        version: 1,
        sha256: record.sha256,
        queryHash,
        question: normalizedQuestion,
        keyModel: modelKey,
        createdAt: new Date().toISOString(),
      };
      writeMetadataAtomic(visionCachePath(cacheDir, record.sha256, queryHash), data);
      maintainCache(cacheDir, options.maxAgeDays, options.maxCacheBytes, record.sha256);
      visionMemo.set(key, result);
      return result;
    })();
    visionInFlight.set(key, promise);
    try {
      return await promise;
    } finally {
      visionInFlight.delete(key);
    }
  };

  pi.registerEntryRenderer<CacheHitEntryData>(CACHE_HIT_ENTRY_TYPE, (entry, { expanded }, theme) => {
    const data = entry.data;
    const box = new Box(1, 0, (text) => theme.bg("customMessageBg", text));
    if (!isCacheHitEntryData(data)) {
      box.addChild(new Text(theme.fg("warning", "♻ Image context cache entry has invalid data"), 0, 0));
      return box;
    }

    const title = data.kind === "vision" ? "Isolated Vision" : "Image cache hit";
    const suffix = data.kind === "vision"
      ? `· ${data.model || "vision model"}${data.cached ? " · cached" : ""}`
      : `· ${formatBytes(data.bytes)} not resent`;
    box.addChild(
      new Text(
        `${theme.fg("success", "♻")} ${theme.fg("accent", title)} ${theme.fg("muted", suffix)}`,
        0,
        0,
      ),
    );
    if (!expanded) return box;

    box.addChild(new Text(theme.fg("dim", `${data.sha256.slice(0, 12)}… · ${displaySafe(data.path)}`), 0, 0));
    if (data.source) box.addChild(new Text(theme.fg("dim", `source: ${displaySafe(data.source)}`), 0, 0));
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

  const restoreBranchState = (ctx: ExtensionContext) => {
    seenHashes.clear();
    pendingSeenHashes.clear();
    pendingEntries.clear();
    offeredToolHashes.clear();
    contextRecords.clear();
    let successfulAssistantAfter = false;
    const manager = ctx.sessionManager as typeof ctx.sessionManager & { buildContextEntries?: () => ReturnType<typeof ctx.sessionManager.getBranch> };
    const branch = manager.buildContextEntries?.() ?? ctx.sessionManager.getBranch();
    for (let index = branch.length - 1; index >= 0; index -= 1) {
      const entry = branch[index];
      if (!entry) continue;
      if (entry.type === "message" && entry.message.role === "assistant") {
        if (entry.message.stopReason !== "error" && entry.message.stopReason !== "aborted") successfulAssistantAfter = true;
        continue;
      }
      const candidate = entry.type === "message"
        ? (entry.message as unknown as MessageWithContent)
        : entry.type === "custom_message"
          ? ({ content: entry.content } as MessageWithContent)
          : undefined;
      if (!candidate || !successfulAssistantAfter || !Array.isArray(candidate.content)) continue;
      for (const block of candidate.content) {
        if (!isImageContent(block)) continue;
        try {
          seenHashes.add(decodeImage(block, options.maxImageBytes).sha256);
        } catch {
          // Invalid or over-limit historical images must not affect semantic state.
        }
      }
    }
  };

  pi.on("session_start", async (_event, ctx) => {
    const removed = maintainCache(options.cacheDir, options.maxAgeDays, options.maxCacheBytes);
    restoreBranchState(ctx);
    if (removed > 0 && ctx.hasUI) {
      ctx.ui.notify(`Image context cache: removed ${removed} stale artifact(s).`, "info");
    }
  });

  pi.on("session_tree", async (_event, ctx) => {
    restoreBranchState(ctx);
  });

  pi.on("session_compact", async (_event, ctx) => {
    restoreBranchState(ctx);
  });

  pi.on("input", async (event, ctx) => {
    for (const id of pendingEntries.keys()) {
      if (id.startsWith("vision-input:")) pendingEntries.delete(id);
    }
    if (options.visionMode !== "isolated" || !event.images?.length) return { action: "continue" as const };
    const sequence = ++inputSequence;
    const results: string[] = [];
    for (let index = 0; index < event.images.length; index += 1) {
      const image = event.images[index]!;
      try {
        const record = cacheImage(image, options.cacheDir, undefined, cachePolicy, cacheMemo);
        try {
          const analysis = await analyzeIsolated(image, record, event.text, ctx);
          results.push(isolatedVisionPlaceholder(record, analysis).text);
          queueCacheCard(record, `vision-input:${sequence}:${index}:${record.sha256}`, {
            kind: "vision",
            model: analysis.model,
            cached: analysis.cached,
          });
        } catch (error) {
          results.push(isolationFailurePlaceholder(record, error).text);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        results.push(`[Image isolation failed before analysis — payload removed.]\nError: ${truncateUtf8(message, 2000)}`);
      }
    }
    return {
      action: "transform" as const,
      text: [event.text, ...results].filter(Boolean).join("\n\n"),
      images: [],
    };
  });

  pi.on("tool_result", async (event, ctx) => {
    if (!event.content.some(isImageContent)) return;
    const source = sourceFromToolInput(event.toolName, event.input);
    const offeredInResult = new Set<string>();
    const question = options.visionMode === "isolated" ? latestUserQuestion(ctx) : "";
    let changed = false;
    const content: Array<TextContent | ImageContent> = [];
    for (const block of event.content) {
      if (!isImageContent(block)) {
        content.push(block);
        continue;
      }
      let record: CacheRecord;
      try {
        record = cacheImage(block, options.cacheDir, source, cachePolicy, cacheMemo);
      } catch (error) {
        if (options.visionMode === "isolated") {
          changed = true;
          const message = error instanceof Error ? error.message : String(error);
          content.push({
            type: "text",
            text: `[Image isolation failed before analysis — payload was NOT sent to the main model.]\nError: ${truncateUtf8(message, 2000)}`,
          });
        } else {
          content.push(block);
        }
        continue;
      }

      if (isExplicitCacheRead(source, record)) {
        offeredInResult.add(record.sha256);
        offeredToolHashes.add(record.sha256);
        content.push(block);
        continue;
      }

      if (options.visionMode === "isolated") {
        changed = true;
        try {
          const analysis = await analyzeIsolated(block, record, question, ctx);
          content.push(isolatedVisionPlaceholder(record, analysis));
          queueCacheCard(record, `vision-tool:${event.toolCallId}:${record.sha256}`, {
            kind: "vision",
            model: analysis.model,
            cached: analysis.cached,
          });
          seenHashes.add(record.sha256);
        } catch (error) {
          content.push(isolationFailurePlaceholder(record, error));
        }
        continue;
      }

      const cacheHit =
        seenHashes.has(record.sha256) || offeredToolHashes.has(record.sha256) || offeredInResult.has(record.sha256);
      if (cacheHit) {
        changed = true;
        queueCacheCard(record, `tool:${event.toolCallId}:${record.sha256}`, { kind: "cache-hit" });
        content.push(cacheHitPlaceholder(record));
      } else {
        offeredInResult.add(record.sha256);
        offeredToolHashes.add(record.sha256);
        content.push(block);
      }
    }
    return changed ? { content } : undefined;
  });

  pi.on("context", async (event) => ({
    messages: pruneImageContext(event.messages, options, undefined, {
      seenHashes,
      onFreshImage: (record) => pendingSeenHashes.add(record.sha256),
      onCacheHit: (record, eventId) => queueCacheCard(record, `context:${eventId}:${record.sha256}`),
      contextRecords,
      cacheMemo,
      policy: cachePolicy,
    }),
  }));

  pi.on("agent_end", async () => {
    for (const id of pendingEntries.keys()) {
      if (id.startsWith("vision-input:")) pendingEntries.delete(id);
    }
  });

  pi.on("turn_end", async (event) => {
    const successful =
      event.message.role === "assistant" &&
      event.message.stopReason !== "error" &&
      event.message.stopReason !== "aborted";
    if (successful) {
      for (const sha256 of pendingSeenHashes) seenHashes.add(sha256);
      for (const data of pendingEntries.values()) {
        pi.appendEntry<CacheHitEntryData>(CACHE_HIT_ENTRY_TYPE, data);
      }
    }
    pendingSeenHashes.clear();
    pendingEntries.clear();
    offeredToolHashes.clear();
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
        const removed = clearRecognizedCache(options.cacheDir);
        cacheMemo.clear();
        contextRecords.clear();
        visionGeneration += 1;
        visionMemo.clear();
        visionInFlight.clear();
        seenHashes.clear();
        pendingSeenHashes.clear();
        pendingEntries.clear();
        offeredToolHashes.clear();
        ctx.ui.notify(`Image context cache cleared (${removed} artifact(s)).`, "info");
        return;
      }
      ctx.ui.notify("Usage: /image-cache [stats|clear]", "warning");
    },
  });
}
