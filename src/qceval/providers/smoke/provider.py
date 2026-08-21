"""Deterministic provider used for local smoke tests and examples."""

from __future__ import annotations

from collections.abc import Sequence

from qceval.models import ProviderRequest, ProviderResponse, TokenUsage
from qceval.providers.smoke.canonical import generated_canonical_code


class SmokeProvider:
    """Return deterministic local responses without calling remote models.

    ``SmokeProvider`` is useful for CI, docs, cache tests, and local validation.
    Canonical mode returns bundled canonical solutions when available.  Empty
    and error modes exercise failure paths without external credentials.

    Args:
        mode: One of ``"canonical"``, ``"empty"``, or ``"error"``.
        model: Model label recorded in provider responses.
        reasoning_effort: Optional named reasoning effort recorded in metadata.
        reasoning_enabled: Optional unnamed-reasoning flag recorded in metadata.
        configuration_id: Optional matrix configuration identity.

    Attributes:
        name: Stable provider name used in output and cache keys.
        mode: Response mode selected at construction time.
        model: Model label recorded in provider responses.
        reasoning_effort: Named reasoning effort, if configured.
        reasoning_enabled: Unnamed-reasoning flag, if configured.
        configuration_id: Matrix configuration identity, if configured.

    Raises:
        ValueError: If ``mode`` is unsupported.
    """

    name = "smoke"
    trusted_metadata = True

    def __init__(
        self,
        mode: str = "canonical",
        model: str = "smoke-canonical",
        *,
        reasoning_effort: str | None = None,
        reasoning_enabled: bool | None = None,
        configuration_id: str | None = None,
    ) -> None:
        if mode not in {"canonical", "empty", "error"}:
            raise ValueError("smoke mode must be canonical, empty, or error")
        self.mode = mode
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reasoning_enabled = reasoning_enabled
        self.configuration_id = configuration_id

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate deterministic response for one request.

        Args:
            request: Prompt and task metadata.

        Returns:
            Provider response containing canonical code, empty code, or a
            configured error depending on ``mode``.
        """
        if self.mode == "error":
            return ProviderResponse(code=None, model=self.model, error="smoke provider configured to fail")
        code = "" if self.mode == "empty" else self._code_for(request)
        metadata: dict[str, object] = {"provider": self.name, "mode": self.mode}
        if self.reasoning_effort is not None:
            metadata["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_enabled is not None:
            metadata["reasoning_enabled"] = self.reasoning_enabled
        if self.configuration_id is not None:
            metadata["configuration_id"] = self.configuration_id
        return ProviderResponse(
            code=code,
            model=self.model,
            metadata=metadata,
            usage=self._usage(request.prompt, code),
            raw_response={"mode": self.mode, "task_id": request.task_id},
        )

    def generate_many(self, requests: Sequence[ProviderRequest]) -> list[ProviderResponse]:
        """Generate deterministic responses for ordered requests.

        Args:
            requests: Ordered provider requests.

        Returns:
            Ordered provider responses matching ``requests``.
        """
        return [self.generate(request) for request in requests]

    def _code_for(self, request: ProviderRequest) -> str:
        canonical = request.metadata.get("canonical_solution")
        if isinstance(canonical, str) and canonical.strip():
            return canonical
        generated = generated_canonical_code(request)
        if generated is not None:
            return generated
        return (
            f"def {request.entry_point}(*args, **kwargs):\n"
            "    raise RuntimeError('smoke provider has no canonical solution for this task')\n"
        )

    @staticmethod
    def _usage(prompt: str, code: str) -> TokenUsage:
        prompt_tokens = len(prompt.split())
        completion_tokens = len(code.split())
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
