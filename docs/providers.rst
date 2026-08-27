Providers
=========

A provider is the component that generates code for a given quantum circuit
task. It receives a structured request containing a prompt, a framework name,
and an expected entry point, and returns generated code or an error.

QCircuitEval ships three built-in providers: ``smoke`` for testing,
``openrouter`` for model evaluation through OpenRouter, and ``coda`` for
evaluation through the Coda Build agent. You can also write your own provider by
implementing the :class:`~qceval.providers.base.Provider` protocol.

This page explains the provider protocol, documents the built-in providers in
detail, and shows how to write and register a custom provider.


The Provider Protocol
---------------------

Providers are defined by a structural typing protocol, not a base class. Any
object with a ``name`` attribute and a ``generate`` method that accepts a
:class:`~qceval.models.ProviderRequest` and returns a
:class:`~qceval.models.ProviderResponse` satisfies the protocol:

.. code-block:: python

    class Provider(Protocol):
        name: str

        def generate(self, request: ProviderRequest) -> ProviderResponse: ...

The use of a protocol rather than an abstract base class is deliberate: it means
you do not need to import or inherit from anything in ``qceval`` to write a
provider. Any class with the right shape will work.

Providers may also implement the optional batch protocol:

.. code-block:: python

    class BatchProvider(Provider, Protocol):
        def generate_many(self, requests: Sequence[ProviderRequest]) -> list[ProviderResponse]: ...

The runner uses ``generate_many`` when available and otherwise falls back to
calling ``generate`` concurrently with a thread pool. Responses must be returned
in the same order as the requests.


ProviderRequest
^^^^^^^^^^^^^^^

The request dataclass contains everything a provider needs to generate code:

.. code-block:: python

    @dataclass(frozen=True)
    class ProviderRequest:
        task_id: str
        framework: Framework      # "qiskit", "cirq", "pennylane", or "cudaq"
        prompt: str
        entry_point: str
        model: str | None = None
        metadata: Mapping[str, Any] = field(default_factory=dict)
        sample_index: int = 0
        attempt_index: int = 0
        messages: tuple[ProviderMessage, ...] = ()

``task_id``
    A suite-local task identifier. Core tasks use zero-padded IDs such as
    ``"01"``; QEC tasks use IDs such as ``"qec01"``.

``framework``
    The quantum framework the generated code should target. The prompt already
    specifies this, but the field is available for providers that want to adjust
    their generation strategy per framework.

``prompt``
    The full prompt text, including the function signature, docstring, and any
    grading notes from the bundled task.

``entry_point``
    The name of the function the generated code must define. The evaluator will
    look for ``def <entry_point>`` in the generated code.

``model``
    The model identifier from the CLI ``--model`` flag. May be ``None`` for
    providers that do not use external models.

``metadata``
    Additional task metadata, including the suite, task category, canonical
    class field, and (for the smoke provider) the canonical solution. The
    ``canonical_class`` value is retained for executor and smoke-provider
    compatibility; it is not score authority. Providers may use or ignore
    metadata, but they do not select verifier routes or grading targets.

``sample_index``
    Zero-based repeated-sample identity for Pass@K runs.

``attempt_index``
    Zero-based feedback-repair attempt identity.

``messages``
    Ordered chat history for providers that support multi-turn repair. When
    non-empty, it includes the initial user prompt and alternating assistant
    code / user feedback messages. One-shot providers may ignore it and read
    ``prompt``.


ProviderResponse
^^^^^^^^^^^^^^^^

The response dataclass carries the generated code and optional diagnostics:

.. code-block:: python

    @dataclass(frozen=True)
    class ProviderResponse:
        code: str | None
        model: str | None = None
        metadata: Mapping[str, Any] = field(default_factory=dict)
        usage: TokenUsage | None = None
        raw_response: Any | None = None
        error: str | None = None

``code``
    The generated Python code. Set this to ``None`` when generation fails.

``model``
    The model that actually produced the response. This may differ from the
    requested model if the provider routes to a fallback.

``metadata``
    Provider-specific metadata (e.g., ``{"provider": "openrouter"}``). Recorded
    in the output for auditing.

``usage``
    Token counts, if available. The :class:`~qceval.models.TokenUsage` dataclass
    has fields for ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
    ``reasoning_tokens``, ``cached_tokens``, and ``cost_usd``. ``cost_usd`` is
    the provider-reported request charge in US dollars. All fields are optional.

``raw_response``
    The raw API response payload for debugging and auditing. Set to ``None``
    for providers that do not have a raw response.

``error``
    A human-readable error message. When this is set, the response is considered
    failed and the runner records ``provider_failed`` for the task.

The ``ok`` property returns ``True`` when ``error`` is ``None`` and ``code`` is
a nonempty string after stripping whitespace. Empty and whitespace-only
responses are provider failures (``provider_failed``) and are not evaluated.


Built-in Providers
------------------


The Smoke Provider
^^^^^^^^^^^^^^^^^^

The smoke provider (``qceval.providers.smoke.SmokeProvider``) is a deterministic,
offline provider intended for testing the evaluation pipeline. It never calls
an external API. It has three modes, selected by the ``--smoke-mode`` CLI flag or
the ``mode`` constructor argument:

**canonical** (default)
    Returns the bundled canonical solution when the task provides one. For
    tasks without canonical source, it can generate deterministic compatibility
    responses from task metadata.
    This exercises generation and executor integration. It does not bypass
    behavior contracts, framework lowering, verifier support, or resource
    limits, so a canonical response is not guaranteed to pass. QEC canonical
    responses still execute every declared case and metadata check.

**empty**
    Returns an empty string as the generated code for every task. Empty code
    is not ``ok``, so the runner records ``provider_failed`` for every task
    and skips evaluation. This exercises the empty-response provider-failure
    path.

**error**
    Returns a provider error for every task without generating any code. This
    exercises the provider-failure path: the runner records ``provider_failed``
    for every task and skips evaluation entirely.

The smoke provider also computes synthetic token usage by splitting the prompt
and generated code on whitespace and counting words. This is not accurate but
ensures that the ``usage`` field is always populated, which is useful for testing
output consumers. Matrix runs also copy ``reasoning_effort``,
``reasoning_enabled``, and ``configuration_id`` into smoke response metadata,
so a credential-free smoke matrix exercises the same configuration lineage as
a provider-backed matrix.


The OpenRouter Provider
^^^^^^^^^^^^^^^^^^^^^^^

The OpenRouter provider (``qceval.providers.openrouter.OpenRouterProvider``)
sends the task prompt to an `OpenRouter <https://openrouter.ai/>`_
chat-completions endpoint and extracts code from the response.

Unpinned matrix runs keep ``configuration_id`` in the QCircuitEval result and
sweep manifest but do not pass it to OpenRouter: OpenRouter route provenance
requires a complete endpoint pin. Mixed-model registry sweeps normally omit
pins because endpoint tags differ by model.

**Configuration:**

Provider settings are passed through CLI flags or the provider config
dictionary. When a CLI credential flag is omitted, the CLI checks the matching
process environment variable and then ``.env`` in the current working
directory. Provider classes used directly still receive credentials through
their explicit config.

- ``api_key``: The OpenRouter API key. Required. Passed via
  ``--openrouter-api-key``, read from a protected file with
  ``--openrouter-api-key-file``, or resolved from ``OPENROUTER_API_KEY`` /
  ``.env`` by the CLI.
- ``base_url``: The chat-completions endpoint. Defaults to
  ``https://openrouter.ai/api/v1/chat/completions``. Override with
  ``--openrouter-base-url`` for proxies or local servers.
- ``timeout``: HTTP request timeout in seconds. Defaults to 120.
- ``temperature``: Sampling temperature. Defaults to 0.2.
- ``reasoning_effort``: Optional OpenRouter reasoning effort.
- ``reasoning_enabled``: Optional enable switch for models without named
  reasoning effort levels.

Official production configurations always set one of these controls explicitly
to the model's pinned highest available setting. Omitting both remains available
to ad hoc API and CLI users, but delegates the choice to the provider and is not
a reproducible production protocol.

**Request format:**

The provider sends a single-message chat completion request for one-shot runs:

.. code-block:: json

    {
        "model": "<model>",
        "messages": [{"role": "user", "content": "<prompt>"}],
        "temperature": 0.2,
        "stream": false,
        "reasoning": {"exclude": true}
    }

The ``reasoning.exclude`` field tells models that support extended thinking to
omit reasoning text from the response, keeping the output focused on code.
When configured, the same object also carries either ``effort`` or ``enabled``.
Reasoning token counts remain available in usage telemetry.

For feedback repair, the provider serializes ``ProviderRequest.messages`` as
the OpenRouter ``messages`` array in the original order. Providers that only
support one-shot generation may ignore ``messages``; OpenRouter uses it when
present and falls back to the single user prompt when it is empty.

**Code extraction:**

The provider extracts code from the response text using the following priority:

1. A fenced ``python`` code block containing ``def <entry_point>``.
2. A generic fenced code block containing ``def <entry_point>``.
3. The first fenced ``python`` code block.
4. The first generic fenced code block.
5. The raw response text, stripped of leading and trailing whitespace.

This heuristic handles the common case where models wrap code in markdown fences
and may include explanatory text before or after the code.

**Error handling:**

- If the API key is not configured, the provider returns an error without making
  a network request.
- If the model is not specified in the request, the provider returns an error.
- HTTP errors are caught and reported with the status code and the first 500
  bytes of the response body.
- All other exceptions are caught and reported with the exception type and
  message.

In all error cases, the provider returns a ``ProviderResponse`` with
``code=None`` and a descriptive ``error`` string. It never raises an exception.

**Retry behavior:**

The OpenRouter provider retries transient HTTP errors with bounded exponential
backoff and jitter.  Retryable status codes are 408, 429, 500, 502, 503, and 504.
Non-retryable codes (400, 401, 403, 404, 422) fail immediately.

Connection errors, DNS failures, and socket timeouts are also retried.

The retry formula is:

.. code-block:: text

    delay = min(retry_base_delay * 2^attempt, retry_max_delay)
    jitter = uniform(0, delay * 0.25)
    sleep(delay + jitter)

When a 429 response includes a ``Retry-After`` header with a numeric value in
seconds, that value replaces the computed delay (still capped at
``retry_max_delay``).

Configuration:

- ``max_retries``: Maximum retry attempts.  Default: 3.  Set to 0 to disable.
- ``retry_base_delay``: Base delay in seconds.  Default: 1.0.
- ``retry_max_delay``: Maximum delay cap in seconds.  Default: 60.0.

These can be set via CLI (``--max-retries``, ``--retry-base-delay``,
``--retry-max-delay``) or through the provider config dictionary.


The Coda Provider
^^^^^^^^^^^^^^^^^

The Coda provider (``qceval.providers.coda.CodaProvider``) sends each task to
the Coda Agents endpoint and reads the server-sent event stream returned by the
Build or Learn agent.

**Configuration:**

Provider settings are passed through CLI flags or the provider config
dictionary. When ``--coda-api-key`` is omitted, the CLI checks
``CODA_API_KEY`` and then ``.env`` in the current working directory.

- ``api_key``: The Coda API key. Required. Passed via ``--coda-api-key`` or
  resolved from ``CODA_API_KEY`` / ``.env`` by the CLI.
- ``agents_url``: The full Coda agents endpoint. Defaults to
  ``https://api.conductorquantum.com/v0/coda/agents``. Override with
  ``--coda-agents-url`` for a gateway or local endpoint.
- ``timeout``: HTTP request timeout in seconds. Defaults to 900 because Coda
  agent runs can include validation and repair work.
- ``mode``: Coda agent mode, either ``build`` or ``learn``. Defaults to
  ``build``.
- ``fast``: Whether to send ``fast: true`` in the request body.
- ``prefer_structured_response``: Whether structured response code gets priority
  when it defines the requested entry point.

Coda does not expose model selection through this API. The ``--model`` value is
only an output label for QCircuitEval results. If omitted, the CLI derives one
from the selected Coda mode: ``coda/build``, ``coda/build-fast``,
``coda/learn``, or ``coda/learn-fast``.

Coda also does not expose temperature. QCircuitEval never sends temperature to
Coda. If ``--temperature`` is provided with ``--provider coda``, the CLI prints
a warning and ignores the flag.

**Request format:**

The provider sends the direct Coda ``AgentsRequest`` body, without a wrapper:

.. code-block:: json

    {
        "messages": [{"role": "user", "content": "<prompt>"}],
        "mode": "build",
        "fast": false
    }

For feedback repair, the provider serializes ``ProviderRequest.messages`` as the
same ``messages`` array in order. Coda accepts only ``user`` and ``assistant``
roles, so ``system`` messages are rejected as provider errors.

The provider deliberately does not send ``thread_id`` by default. Each
QCircuitEval task is independent; sharing a remote thread could leak state
between tasks. Pass@K and feedback repair work through independent Coda calls
and explicit message history in ``ProviderRequest.messages``.

**Event parsing and code extraction:**

Coda returns a server-sent event stream. The provider parses ``data:`` events,
``event:`` plus ``data:`` pairs, keepalive comments, ``[DONE]`` sentinels, and
plain JSON fallback bodies. Token events, assistant messages, tool results, and
structured response fields are treated only as extraction sources for code Coda
has already emitted.

Generated-code extraction prefers emitted text that defines
``def <entry_point>``. By default, token and final-message sources are used
before structured response sources when both define the entry point. With
``--coda-prefer-structured-response``, structured response code fields are used
only when they define the entry point. The extracted source text is then passed
through the same markdown code extraction helper, using latest-block extraction
because Coda streams can contain an early draft followed by corrected code.

Malformed generated code is still returned to the evaluator. Missing entry
points, syntax errors, and runtime errors are benchmark outcomes, not provider
errors.

**Error handling:**

- If the API key is not configured, the provider returns an error without making
  a network request.
- Unsupported message roles return a provider error before any network request.
- HTTP errors are caught and reported with the status code and the first 500
  bytes of the response body.
- Terminal Coda ``error`` and ``cancelled`` events become provider errors.
- All other exceptions are caught and reported with the exception type and
  message.

**Retry behavior:**

The Coda provider retries transient HTTP errors with bounded exponential backoff
and jitter. Retryable status codes are 408, 429, 500, 502, 503, and 504.
Non-retryable codes (400, 401, 403, 404, 422) fail immediately.

Connection errors, DNS failures, and socket timeouts are also retried. Numeric
``Retry-After`` headers are honored and capped by ``retry_max_delay``.

Configuration:

- ``max_retries``: Maximum retry attempts. Default: 3. Set to 0 to disable.
- ``retry_base_delay``: Base delay in seconds. Default: 1.0.
- ``retry_max_delay``: Maximum delay cap in seconds. Default: 60.0.


The Provider Registry
---------------------

The :func:`~qceval.providers.registry.build_provider` function constructs a
provider instance from a name and a configuration dictionary:

.. code-block:: python

    from qceval.providers.registry import build_provider

    provider = build_provider(
        "coda",
        model="coda/build",
        config={"coda_api_key": "<key>"},
    )

The registry knows about three provider names: ``"smoke"``, ``"openrouter"``,
and ``"coda"``. Passing any other name raises ``ValueError``.

The ``config`` dictionary maps directly to provider constructor arguments. For
the smoke provider, the relevant key is ``smoke_mode``. For the OpenRouter
provider, the relevant keys are ``openrouter_api_key``,
``openrouter_base_url``, ``timeout``, ``temperature``, ``reasoning_effort``,
``reasoning_enabled``, ``max_retries``, ``retry_base_delay``,
``retry_max_delay``, ``openrouter_endpoint_tag``,
``openrouter_max_output_tokens``, ``openrouter_output_limit_source``,
``openrouter_endpoint_cap_status``, ``openrouter_output_token_parameter``,
``openrouter_route_revision``, and ``configuration_id``. When
``openrouter_endpoint_tag`` is set, omitted ``temperature`` stays unset rather
than defaulting to ``0.2``. For the Coda provider, the
relevant keys are ``coda_api_key``, ``coda_agents_url``, ``coda_mode``,
``coda_fast``, ``coda_prefer_structured_response``, ``timeout``,
``max_retries``, ``retry_base_delay``, and ``retry_max_delay``.


Writing a Custom Provider
--------------------------

To write a custom provider, create a class with a ``name`` attribute and a
``generate`` method:

.. code-block:: python

    from qceval.models import ProviderRequest, ProviderResponse

    class MyProvider:
        name = "my-provider"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            code = self._call_my_model(request.prompt, request.model)
            return ProviderResponse(
                code=code,
                model=request.model,
                metadata={"provider": self.name},
            )

        def _call_my_model(self, prompt: str, model: str | None) -> str:
            # Your model call here
            ...

You can use this provider directly with the
:class:`~qceval.core.runner.BenchmarkRunner`:

.. code-block:: python

    from qceval.core.bench import Adaptor
    from qceval.models import RunConfig
    from qceval.core.runner import BenchmarkRunner

    config = RunConfig(
        provider="my-provider",
        frameworks=("qiskit",),
        source_hint=None,
        model="my-model",
    )
    runner = BenchmarkRunner(
        config=config,
        provider=MyProvider(),
        adapter=Adaptor(),
    )
    payload = runner.run()

Note that custom providers are not automatically available through the CLI. The
CLI uses the provider registry, which only knows about ``smoke``,
``openrouter``, and ``coda``. To use a custom provider from the command line,
you would need to modify the registry or use the Python API directly.


Provider Contract Summary
--------------------------

A well-behaved provider should:

1. **Never raise exceptions from** ``generate``. Return a ``ProviderResponse``
   with an ``error`` string instead. The runner relies on this to record
   provider failures without crashing.

2. **Set** ``code`` **to a nonempty string on success.** The ``ok`` property
   requires ``error is None`` and a stripped-nonempty ``code``. Empty or
   whitespace-only code is treated as ``provider_failed`` and is not evaluated.

3. **Set** ``usage`` **when token data is available.** This is optional but
   valuable for cost tracking and analysis.

4. **Set** ``raw_response`` **when an API response exists.** This enables
   post-hoc debugging without re-running the benchmark.

5. **Set** ``model`` **to the actual model that produced the response.** This
   may differ from the requested model if the provider routes to a fallback or
   the API returns a different model name.

6. **Preserve order from** ``generate_many`` **if implemented.** The first
   response must correspond to the first request, even if provider calls finish
   out of order internally.
