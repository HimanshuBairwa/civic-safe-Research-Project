"""Custom exception hierarchy for CIVIC-SAFE."""

from typing import Literal


class CivicSafeError(Exception):
    """Base exception for all CIVIC-SAFE errors."""


class KillCriterionTriggered(CivicSafeError):
    """Raised when a training kill criterion fires.

    Attributes:
        criterion_name: Identifier of the criterion that triggered.
        threshold: The configured threshold value.
        observed_value: The value that violated the threshold.
        direction: Whether the observed value was 'above' or 'below' the threshold.
    """

    def __init__(
        self,
        criterion_name: str,
        threshold: float,
        observed_value: float,
        direction: Literal["above", "below"],
    ) -> None:
        self.criterion_name: str = criterion_name
        self.threshold: float = threshold
        self.observed_value: float = observed_value
        self.direction: Literal["above", "below"] = direction
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            f"Kill criterion [{self.criterion_name}] triggered: "
            f"observed {self.observed_value} {self.direction} "
            f"threshold {self.threshold}"
        )


class NumericalInstabilityError(CivicSafeError):
    """Raised when a numerical operation produces an unstable result.

    Attributes:
        operation: Name of the operation that failed.
        value: The problematic value encountered.
    """

    def __init__(self, operation: str, value: float) -> None:
        self.operation: str = operation
        self.value: float = value
        super().__init__(
            f"Numerical instability in '{operation}': encountered value {value}"
        )


class DataValidationError(CivicSafeError):
    """Raised when input data fails validation.

    Attributes:
        field: Name of the field that failed validation.
        reason: Human-readable explanation of the failure.
    """

    def __init__(self, field: str, reason: str) -> None:
        self.field: str = field
        self.reason: str = reason
        super().__init__(f"Data validation failed for field '{field}': {reason}")


class CheckpointCorruptionError(CivicSafeError):
    """Raised when a checkpoint file fails integrity verification.

    Attributes:
        path: Filesystem path to the corrupted checkpoint.
    """

    def __init__(self, path: str) -> None:
        self.path: str = path
        super().__init__(f"Checkpoint corrupted or tampered with: {path}")
