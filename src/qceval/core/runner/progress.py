"""Runner progress reporting helpers."""

from __future__ import annotations

import sys
from typing import Any

from tqdm import tqdm

from qceval.models import BenchmarkRecord, RunOptions


class ProgressMixin:
    """Progress output behavior shared by runner implementations."""

    options: RunOptions

    def _progress_bar(self, total: int) -> Any | None:
        if not self.options.progress:
            return None
        return tqdm(total=total, unit="task", dynamic_ncols=True, file=sys.stderr)

    def _close_progress(self, progress_bar: Any | None) -> None:
        if progress_bar is not None:
            progress_bar.close()

    def _emit_progress(
        self,
        record: BenchmarkRecord,
        completed: int,
        total: int,
        progress_bar: Any | None = None,
    ) -> None:
        if not self.options.progress:
            return
        message = f"[{completed}/{total}] {record.suite}:{record.framework}:{record.task_id} {record.status}"
        if progress_bar is not None:
            progress_bar.write(message, file=sys.stderr)
            progress_bar.update(1)
            return
        print(message, file=sys.stderr, flush=True)
