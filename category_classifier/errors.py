"""Custom exceptions."""


class CategoryClassifierError(Exception):
    """Base exception for category classifier errors."""


class DataValidationError(CategoryClassifierError):
    """Raised when a dataset row or schema is invalid."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        joined = "\n".join(errors)
        super().__init__(f"Dataset validation failed:\n{joined}")


class ModelPackError(CategoryClassifierError):
    """Raised when a model pack cannot be loaded."""
