# OpenRouter production model settings

## Maximum-reasoning Pass@1 campaign (manifest schema v1, queue schema v2)

The current campaign roster is `production/models.full-cap.json`. It contains
nine models at their highest supported reasoning setting and expands to nine
configurations, 36 framework shards, and 2,520 fresh authoritative requests.

One endpoint is selected per model using the deterministic quality-first
ranking. The complete evidence and route policy live in the v2 registry and
generated preflight snapshot. Candidate imports from an earlier prompt version
are forbidden.

## Model rationale

The roster was refreshed against OpenRouter's `GET /api/v1/models` response on
2026-08-14 and checked against the model pages and unified reasoning
documentation. Re-query the endpoint catalog before a production run because
model routing and supported settings can change.

| Requested model | Pinned OpenRouter ID | Production setting | Catalog evidence |
|---|---|---|---|
| ChatGPT 5.6 Sol | `openai/gpt-5.6-sol` | `max` | Optional reasoning; supported efforts are `max`, `xhigh`, `high`, `medium`, `low`, `none`; default is `medium`. |
| Grok 4.6 | `x-ai/grok-4.6` | `xhigh` | Reasoning is mandatory; `xhigh` is the highest supported effort. |
| Opus 5 | `anthropic/claude-opus-5` | `max` | Supported efforts are `max`, `xhigh`, `high`, `medium`, and `low` (default `high`); `max` is the highest. |
| Fable 5 | `anthropic/claude-fable-5` | `max` | Reasoning is mandatory; supported efforts are `max`, `xhigh`, `high`, `medium`, and `low` (default `high`); `max` is the highest. |
| Gemini 3.1 Pro | `google/gemini-3.1-pro-preview` | `high` | The non-preview slug is unavailable. Reasoning is mandatory and `high` is the highest supported effort. |
| Kimi K3 | `moonshotai/kimi-k3` | `max` | Reasoning is enabled by default and the catalog exposes `max`, `high`, and `low`; the highest named level is pinned. |
| GLM 5.2 | `z-ai/glm-5.2` | `max` | The author API documents `max` as deep reasoning and the default/highest effective effort. |
| Gemma 31B | `google/gemma-4-31b-it` | `enabled` | This is the current 31B Gemma model. Reasoning defaults off and the catalog exposes enablement, not named effort levels. |
| Nemotron 3 Ultra | `nvidia/nemotron-3-ultra-550b-a55b` | `high` | Supported named efforts are `high` and `medium`; `high` is the highest. The model also accepts a direct reasoning-token budget. |

Every production row requests the highest reasoning control exposed by the
catalog. Models with named levels use their highest level; models with
reasoning but no named levels use explicit `enabled`. The production runner
does not accept a provider-default sentinel.

Endpoint qualification uses an author-documented output ceiling when one is
available. Grok 4.6, Kimi K3, Gemma 4 31B, and Nemotron 3 Ultra do not publish
a separate native maximum output allowance, so the benchmark assigns them a
128,000-token minimum endpoint floor. Missing or nonnumeric endpoint completion
limits remain disqualifying; context length is not treated as an output limit.
Grok 4.6 is the sole frozen exception because its first-party `xai` route does
not disclose the field. That route remains pinned with fallbacks disabled and
any `finish_reason=length` below 128,000 reported completion tokens is an
infrastructure failure, not a model result.

GLM 5.2 is pinned explicitly to `max_tokens=131072`. Z.AI's model overview
uses the shorthand “128K,” but the precise Core Parameters table defines the
default as 65,536 and the maximum as 131,072. Omitting `max_tokens` would
therefore request only the documented default, and encoding the shorthand as
decimal 128,000 would leave 3,072 native output tokens unused.

Sources:

- [OpenRouter reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [GPT-5.6 Sol](https://openrouter.ai/openai/gpt-5.6-sol)
- [Grok 4.6](https://openrouter.ai/x-ai/grok-4.6)
- [Claude Opus 5](https://openrouter.ai/anthropic/claude-opus-5)
- [Claude Fable 5](https://openrouter.ai/anthropic/claude-fable-5)
- [Gemini 3.1 Pro Preview](https://openrouter.ai/google/gemini-3.1-pro-preview)
- [Kimi K3](https://openrouter.ai/moonshotai/kimi-k3)
- [GLM 5.2](https://openrouter.ai/z-ai/glm-5.2)
- [Z.AI GLM 5.2 precise `max_tokens` table](https://docs.z.ai/guides/overview/concept-param)
- [Gemma 4 31B](https://openrouter.ai/google/gemma-4-31b-it)
- [Nemotron 3 Ultra](https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b)
