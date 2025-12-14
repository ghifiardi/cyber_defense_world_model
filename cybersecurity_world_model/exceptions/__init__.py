"""Custom exception classes for Cybersecurity World Model."""


class CybersecurityWorldModelError(Exception):
    """Base exception for all Cybersecurity World Model errors."""
    pass


class ConfigurationError(CybersecurityWorldModelError):
    """Raised when configuration is invalid or missing."""
    pass


class ModelError(CybersecurityWorldModelError):
    """Raised when model operations fail."""
    pass


class TrainingError(CybersecurityWorldModelError):
    """Raised when training operations fail."""
    pass


class PredictionError(CybersecurityWorldModelError):
    """Raised when prediction operations fail."""
    pass


class IntegrationError(CybersecurityWorldModelError):
    """Raised when external integrations fail."""
    pass


class DataError(CybersecurityWorldModelError):
    """Raised when data processing fails."""
    pass
