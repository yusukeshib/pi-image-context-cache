# pi-image-context-cache

A [pi](https://github.com/earendil-works/pi) extension that caches image results locally and removes stale base64 image payloads from repeated LLM requests.

Images remain visible for their first model turn. On later turns, the extension replaces each image in the outgoing context with a compact cache reference. The persisted pi session is not rewritten.

## Why

Image tool results can be several megabytes. Without eviction, the same image is sent again on every later model request, increasing cache-read cost and context pressure. This extension keeps the first visual inspection intact while making later requests text-only.

## Install

```sh
pi install npm:@yusukeshib/pi-image-context-cache
```

Or install directly from GitHub while developing:

```sh
pi install git:github.com/yusukeshib/pi-image-context-cache
```

## Behavior

1. Image blocks from prompt attachments, custom messages, and tool results are SHA-256 hashed and cached under `~/.cache/pi-image-context/`.
2. A new image remains in context until the model has had one assistant turn to inspect it.
3. Subsequent provider requests receive a text placeholder containing the cache path, original read path when known, MIME type, size, and hash.
4. Re-reading the cached file makes the image fresh for another turn.
5. Images are only evicted after a cache copy succeeds.

The cache uses directory mode `0700` and file mode `0600`. Images older than 30 days are removed at session start.

## Commands

```text
/image-cache          # show cache statistics
/image-cache stats    # same
/image-cache clear    # clear after confirmation
```

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `PI_IMAGE_CONTEXT_CACHE_DIR` | `~/.cache/pi-image-context` | Cache directory |
| `PI_IMAGE_CONTEXT_CACHE_TTL_TURNS` | `1` | Number of successful later assistant messages before eviction; values below `1` are clamped to `1` |
| `PI_IMAGE_CONTEXT_CACHE_MAX_AGE_DAYS` | `30` | Remove older cache entries at startup; `0` disables expiry |

## Scope and limitations

- The extension changes only the context sent to the model; it does not rewrite existing session JSONL files.
- “Seen” is inferred from a later non-error/non-aborted assistant message. This is branch-safe and retry-safe for normal Pi flows, but another extension loaded later could still remove an image before the provider request.
- The first version preserves the original image bytes. It does not resize or recompress images.
- Source paths are recorded only for the built-in `read` tool. Other image-producing tools still receive a cache path.
- Cache files may contain sensitive screenshots. They remain local and private, but should still be handled as sensitive data.

## Development

```sh
bun test
```
