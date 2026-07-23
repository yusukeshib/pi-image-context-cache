# pi-image-context-cache

A [pi](https://github.com/earendil-works/pi) extension that keeps image payloads out of the main agent context by default, caches image bytes locally, and delegates visual analysis to an isolated Vision request.

## Why

Image tool results can be several megabytes. Sending the same image through the main conversation increases provider cost, session size, and context pressure. This extension uses a hybrid model:

- Normal image attachments and reads are analyzed in an isolated Vision request.
- The main model receives only a detailed text result.
- Vision results are cached by image SHA-256, question, prompt version, and model.
- Reading the generated cache path explicitly bypasses isolation when the main model genuinely needs direct pixel access.

## Install

```sh
pi install npm:@yusukeshib/pi-image-context-cache
```

Or install directly from GitHub:

```sh
pi install git:github.com/yusukeshib/pi-image-context-cache
```

## Behavior

1. Supported image blocks are strictly validated, SHA-256 hashed, and stored under `~/.cache/pi-image-context/`.
2. Prompt attachments are removed before the main agent starts and replaced with isolated Vision analysis.
3. Images returned by the built-in `read` tool are replaced before the tool result is persisted or sent to the main model.
4. Vision analysis is cached by `image SHA + user question + model + prompt version`; identical concurrent requests share one provider call.
5. A different question about the same image triggers a new isolated analysis.
6. Vision failure is fail-closed: the image is not silently sent to the main model. The result explains how to opt into direct inspection.
7. Reading the absolute generated cache path is an explicit bypass and sends that image to the main model for one turn.
8. In `direct` mode, the first image reaches the main model and later duplicate payloads are replaced with compact cache references.
9. Every successful cache-hit/Vision event can add a TUI-only transcript card. Cards never enter LLM context and never store base64.
10. Branch navigation and compaction rebuild seen-image state from the effective active branch.

## Commands

```text
/image-cache          # show cache statistics
/image-cache stats    # same
/image-cache clear    # safely remove only recognized cache artifacts
```

`clear` never recursively removes the configured directory and preserves unrelated files.

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `PI_IMAGE_CONTEXT_CACHE_DIR` | `~/.cache/pi-image-context` | Private cache directory |
| `PI_IMAGE_CONTEXT_CACHE_VISION_MODE` | `isolated` | `isolated` or `direct` |
| `PI_IMAGE_CONTEXT_CACHE_VISION_MODEL` | current model | Optional `provider/model` used for isolated analysis |
| `PI_IMAGE_CONTEXT_CACHE_VISION_MAX_TOKENS` | `2048` | Maximum isolated Vision output tokens |
| `PI_IMAGE_CONTEXT_CACHE_TTL_TURNS` | `1` | Successful assistant turns before old direct-mode payloads are evicted; minimum `1` |
| `PI_IMAGE_CONTEXT_CACHE_MAX_IMAGE_BYTES` | `52428800` | Hard per-image decoded-byte limit |
| `PI_IMAGE_CONTEXT_CACHE_MAX_BYTES` | `1073741824` | Total cached-image quota with oldest-access eviction |
| `PI_IMAGE_CONTEXT_CACHE_MAX_AGE_DAYS` | `30` | Cache/analysis expiry; `0` disables age expiry |

## Security and privacy

- Cache directories use mode `0700`; files use `0600` on supported Unix platforms.
- Cache files are opened with `O_NOFOLLOW`, validated by descriptor, and verified by size and SHA before use.
- MIME types are allowlisted and checked against image signatures.
- Writes use private temporary files and atomic publication.
- A sentinel and dangerous-path checks protect cache maintenance and clear operations.
- Source paths are control-character sanitized, length-limited, and the home prefix is redacted in metadata.
- Isolated Vision still sends the image to the selected Vision provider. It isolates the image from the **main conversation**, not from the provider itself.
- Isolated Vision calls have their own provider cost. The extension includes that cost in the generated text when the provider reports it, but it is not part of the main assistant turn’s usage accounting.
- Platforms without `O_NOFOLLOW` fail closed for caching/preview rather than claiming equivalent filesystem guarantees.

## Scope and limitations

- Existing historical session JSONL files are not rewritten. New isolated `read` results do not persist image base64.
- Prompt attachments are transformed before agent processing, so their image payloads are not added to the main user message in isolated mode.
- The cache preserves original compressed bytes; it does not resize or recompress images.
- Signature checks reject obvious MIME mismatches but are not a full hardened image decoder.
- The explicit cache-path bypass trades cost for maximum task-specific visual accuracy.

## Development

```sh
npm install
npm run check
npm pack --dry-run
```
