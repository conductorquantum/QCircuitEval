"""Provider protocol re-exports and generation concurrency helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from qceval.models import ProviderRequest, ProviderResponse
from qceval.typing import BatchProvider, Provider, is_batch_provider


def fan_out_generate(provider: Provider, requests: Sequence[ProviderRequest], workers: int) -> list[ProviderResponse]:
    """Generate responses concurrently with a single-request provider.

    Args:
        provider: Provider implementing ``generate``.
        requests: Ordered requests to generate.
        workers: Maximum worker thread count.

    Returns:
        Ordered responses matching ``requests``.
    """
    results: list[ProviderResponse | None] = [None] * len(requests)
    for index, response in fan_out_generate_stream(provider, requests, workers):
        results[index] = response
    return [_require_response(response) for response in results]


def fan_out_generate_stream(
    provider: Provider, requests: Sequence[ProviderRequest], workers: int
) -> Iterator[tuple[int, ProviderResponse]]:
    """Generate responses concurrently and yield each completion.

    Args:
        provider: Provider implementing ``generate``.
        requests: Ordered requests to generate.
        workers: Maximum worker thread count.

    Returns:
        Iterator over ``(request_index, response)`` pairs in completion order.

    Yields:
        Pairs of ``(request_index, response)`` in completion order.
    """
    if workers <= 1:
        for index, request in enumerate(requests):
            yield index, provider.generate(request)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {pool.submit(provider.generate, request): index for index, request in enumerate(requests)}
        for future in as_completed(future_to_index):
            yield future_to_index[future], future.result()


def _require_response(response: ProviderResponse | None) -> ProviderResponse:
    if response is None:
        raise RuntimeError("provider generation did not produce a response")
    return response


__all__ = ["BatchProvider", "Provider", "fan_out_generate", "fan_out_generate_stream", "is_batch_provider"]
