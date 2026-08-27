"""Parse Coda event streams and extract emitted code.

Public entry points live in the focused modules:

- ``event_parsing.parse_coda_events``
- ``event_extraction.extract_coda_generated_code``
- ``event_compact.compact_event``
- ``event_types.CodaEventStream`` / ``CodaCodeExtraction``
"""

from qceval.providers.coda.event_compact import compact_event
from qceval.providers.coda.event_extraction import extract_coda_generated_code
from qceval.providers.coda.event_parsing import parse_coda_events
from qceval.providers.coda.event_types import CodaCodeExtraction, CodaEventStream

__all__ = [
    "CodaCodeExtraction",
    "CodaEventStream",
    "compact_event",
    "extract_coda_generated_code",
    "parse_coda_events",
]
