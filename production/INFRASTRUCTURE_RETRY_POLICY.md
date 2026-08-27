# Pass@1 infrastructure retry and deferred-recovery policy

This policy is operational. It does not change a benchmark prompt, model sample,
endpoint route, output ceiling, reasoning setting, temperature behavior, or accepted
model-quality outcome.

## Frozen request-attempt policy

Each Pass@1 logical request may make at most six physical infrastructure attempts:
the initial request plus five retries. The retries use exponential backoff with up to
25% positive jitter. The default uncapped sequence before jitter is 1, 2, 4, 8, and
16 seconds. `Retry-After` is authoritative when present, including when it exceeds
the normal 60-second exponential-backoff cap.

Only these failures are retryable:

- HTTP 408, 429, 500, 502, 503, and 504;
- connection resets and other explicitly transient connection failures;
- transport timeouts and temporary DNS failures;
- truncated or invalid transport response bodies;
- structured provider errors that use one of the retryable HTTP codes; and
- code-less structured errors that explicitly report temporary provider
  unavailability, overload, or upstream rate limiting.

HTTP 4xx responses outside the list, Cloudflare-specific 520-527 responses,
certificate failures, permanent URL/configuration errors, and arbitrary harness
exceptions are not retried. A route-verification failure, missing provider-reported
cost, or below-ceiling output-capacity failure is also not part of the transient
retry taxonomy. Those failures remain non-authoritative infrastructure/provenance
records and pause only the affected lane for diagnosis.

A valid response is never retried. In particular, `finish_reason=length` with
provider-reported completion tokens equal to the configured full output ceiling is
a legitimate model outcome. A length termination below that ceiling is an endpoint
capacity-contract failure, not a model outcome.

All six attempts retain the exact frozen endpoint tag, route revision, output-token
parameter and ceiling, reasoning setting, and temperature behavior. Provider
fallbacks remain disabled and exact-route metadata remains mandatory.

## Durable deferral and circuit breaker

When all six transient attempts fail, the logical request is written as
`deferred_infrastructure`. It is not accepted, scored, converted to a candidate-less
outcome, or counted in the benchmark denominator. The mutable recovery snapshot is
`deferred-infrastructure-state.json`; its append-only transition history is
`deferred-infrastructure-ledger.jsonl`.

The deferred entry records the configuration-scoped logical key, job, endpoint tag, route
revision, all six attempt errors, attempt count, first and last deferral timestamps,
next eligible retry time (never less than 30 minutes after exhaustion), defer count,
and source segment. Controller restarts load
this state and reconcile it against route-verified accepted JSONL records before
scheduling anything.

One exhausted request is removed from the normal ready queue while unrelated work on
the same healthy endpoint continues. When the lane's normal ready queue is empty, the
controller performs one bounded deferred sweep. It schedules only keys that still
lack an accepted result.

Two distinct logical requests that consecutively exhaust all six attempts open a
route-scoped circuit. The controller stops assigning new work to that endpoint,
preserves successful in-flight results, durably defers exhausted in-flight results,
and records a cooldown of at least 1,800 seconds. Unrelated model lanes continue.
After the cooldown, one deferred request is used as the half-open probe. A verified
success closes the circuit; another exhausted transient request reopens it for a new
30-minute cooldown. Persistent failures remain deferred until the controller's
single absolute 24-hour recovery deadline. A model still unavailable at that
deadline blocks the model and withholds the complete five-model comparison.

## Attempt ledger and deduplication

`provider-attempt-ledger.jsonl` is append-only and contains one event for each
physical provider attempt. Events include the configuration ID and logical key, endpoint,
route, output ceiling, reasoning and temperature settings, timestamps, status,
error, provider generation ID, usage, and cost whenever the provider reported them.
Deterministic event IDs prevent a controller restart from appending the same attempt
event twice.

Before every normal segment and deferred sweep, the controller re-reads accepted
records with strict route and cost provenance. Accepted logical keys are excluded
from scheduling. A segment that nevertheless contains a second accepted outcome for
an existing logical key is rejected as a controller error.
