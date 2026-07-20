"""Deterministic voice-session loop for Chapter 30.

The module models session semantics rather than audio quality: turn boundaries,
response-scoped playback, barge-in cancellation, exact-argument confirmation,
and stage latency.  Real ASR, model, and TTS adapters can sit behind the same
events without changing those invariants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from statistics import median
from typing import Iterable


class Phase(str, Enum):
    """Externally visible voice-session phases."""

    LISTENING = "listening"
    RESPONDING = "responding"


@dataclass(frozen=True)
class AudioFrame:
    """One timestamped input-energy observation."""

    at_ms: int
    energy: float


@dataclass(frozen=True)
class AudioChunk:
    """One output chunk owned by a particular response."""

    response_id: int
    sequence: int
    text: str


@dataclass(frozen=True)
class ToolProposal:
    """A typed effect proposal reconstructed from speech."""

    response_id: int
    tool: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class StageLatency:
    """Milliseconds spent on the first-audio critical path."""

    turn_end: int
    asr: int
    model: int
    tts: int
    network: int

    @property
    def sequential_ms(self) -> int:
        """Return latency when every stage waits for its predecessor."""

        return self.turn_end + self.asr + self.model + self.tts + self.network

    @property
    def overlapped_ms(self) -> int:
        """Return a conservative streaming-overlap estimate."""

        hidden_asr = min(self.asr, 90)
        hidden_tts = min(self.tts, 70)
        return self.sequential_ms - hidden_asr - hidden_tts


class VoiceSession:
    """Own turn state, interruptible output, and effect authorization.

    Args:
        silence_ms: Quiet duration that closes an input turn.
        speech_energy: Energy at or above this value counts as speech.

    Raises:
        ValueError: If either threshold is non-positive.
    """

    def __init__(self, silence_ms: int = 320, speech_energy: float = 0.35) -> None:
        if silence_ms <= 0 or speech_energy <= 0:
            raise ValueError("thresholds must be positive")
        self.silence_ms = silence_ms
        self.speech_energy = speech_energy
        self.phase = Phase.LISTENING
        self.active_response_id: int | None = None
        self.cancelled_response_ids: set[int] = set()
        self.output: list[AudioChunk] = []
        self.last_speech_ms: int | None = None
        self.turn_closed = False
        self.effects: list[dict[str, object]] = []
        self.events: list[str] = []

    def start_response(self) -> int:
        """Allocate a response identity and begin accepting its chunks."""

        next_id = 1 if self.active_response_id is None else self.active_response_id + 1
        self.active_response_id = next_id
        self.phase = Phase.RESPONDING
        self.events.append(f"response:{next_id}:started")
        return next_id

    def accept_chunk(self, chunk: AudioChunk) -> bool:
        """Queue a chunk only while its response still owns playback."""

        if chunk.response_id != self.active_response_id:
            self.events.append(f"response:{chunk.response_id}:late_chunk_rejected")
            return False
        if chunk.response_id in self.cancelled_response_ids:
            self.events.append(f"response:{chunk.response_id}:cancelled_chunk_rejected")
            return False
        self.output.append(chunk)
        return True

    def cancel_for_barge_in(self) -> None:
        """Cancel current generation and clear only its queued audio."""

        response_id = self.active_response_id
        if response_id is None:
            return
        self.cancelled_response_ids.add(response_id)
        self.output = [chunk for chunk in self.output if chunk.response_id != response_id]
        self.events.append(f"response:{response_id}:cancelled_and_cleared")
        self.phase = Phase.LISTENING

    def observe(self, frame: AudioFrame) -> bool:
        """Consume input energy and return whether a user turn just closed."""

        speaking = frame.energy >= self.speech_energy
        self.turn_closed = False
        if speaking:
            if self.phase is Phase.RESPONDING:
                self.cancel_for_barge_in()
            self.last_speech_ms = frame.at_ms
            return False
        if self.last_speech_ms is not None and frame.at_ms - self.last_speech_ms >= self.silence_ms:
            self.turn_closed = True
            self.last_speech_ms = None
            self.events.append(f"turn:{frame.at_ms}:closed")
        return self.turn_closed

    def propose_refund(self, order_id: str, amount: int) -> ToolProposal:
        """Create a typed proposal without causing an effect."""

        if self.active_response_id is None:
            raise RuntimeError("a response must own the proposal")
        return ToolProposal(
            response_id=self.active_response_id,
            tool="refund_order",
            arguments={"order_id": order_id, "amount": amount},
        )

    def confirm_and_execute(self, proposal: ToolProposal, confirmation: dict[str, object]) -> str:
        """Execute only when response identity and exact arguments still match."""

        if proposal.response_id != self.active_response_id:
            return "deny:superseded_response"
        if proposal.response_id in self.cancelled_response_ids:
            return "deny:cancelled_response"
        if confirmation != proposal.arguments:
            return "review:confirmation_mismatch"
        receipt = {"tool": proposal.tool, **proposal.arguments}
        self.effects.append(receipt)
        self.events.append(f"effect:{proposal.tool}:committed")
        return "allow"


def percentile(values: Iterable[int], fraction: float) -> int:
    """Return a nearest-rank percentile for a non-empty iterable."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("values cannot be empty")
    rank = max(1, int(len(ordered) * fraction + 0.999999))
    return ordered[min(rank - 1, len(ordered) - 1)]


def latency_fixture() -> list[StageLatency]:
    """Return twenty reproducible staged turns, in milliseconds."""

    rows: list[StageLatency] = []
    for index in range(20):
        rows.append(
            StageLatency(
                turn_end=520 + 20 * (index % 5),
                asr=115 + 8 * (index % 4),
                model=205 + 13 * (index % 6),
                tts=118 + 7 * (index % 3),
                network=62 + 6 * (index % 4),
            )
        )
    return rows


def latency_report(rows: Iterable[StageLatency] | None = None) -> dict[str, object]:
    """Summarize sequential and overlapped first-audio latency."""

    materialized = list(rows or latency_fixture())
    sequential = [row.sequential_ms for row in materialized]
    overlapped = [row.overlapped_ms for row in materialized]
    return {
        "turns": len(materialized),
        "sequential_p50_ms": int(median(sequential)),
        "sequential_p95_ms": percentile(sequential, 0.95),
        "overlapped_p50_ms": int(median(overlapped)),
        "overlapped_p95_ms": percentile(overlapped, 0.95),
        "subsecond_p50": median(overlapped) < 1000,
        "rows": [asdict(row) for row in materialized],
    }


def run_race_and_tool_fixture() -> dict[str, object]:
    """Exercise one barge-in race and one critical-entity mishearing."""

    session = VoiceSession()
    old_id = session.start_response()
    session.accept_chunk(AudioChunk(old_id, 0, "Your refund is"))
    session.observe(AudioFrame(100, 0.8))
    late_accepted = session.accept_chunk(AudioChunk(old_id, 1, "being processed"))

    new_id = session.start_response()
    proposal = session.propose_refund("order-7", 50)
    mismatch = session.confirm_and_execute(proposal, {"order_id": "order-7", "amount": 15})
    allowed = session.confirm_and_execute(proposal, {"order_id": "order-7", "amount": 50})
    return {
        "old_response_id": old_id,
        "new_response_id": new_id,
        "late_chunk_accepted": late_accepted,
        "queued_audio": [asdict(chunk) for chunk in session.output],
        "mismatched_confirmation": mismatch,
        "matched_confirmation": allowed,
        "effect_count": len(session.effects),
        "events": session.events,
    }


def build_report() -> dict[str, object]:
    """Return the complete deterministic Chapter 30 build report."""

    return {"latency": latency_report(), "race_and_tool": run_race_and_tool_fixture()}


if __name__ == "__main__":
    import json

    print(json.dumps(build_report(), indent=2))
