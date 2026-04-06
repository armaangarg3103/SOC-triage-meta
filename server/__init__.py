# server/__init__.py
# Exports the real runtime environment — NOT the legacy template file.
from server.environment import SOCAlertEnvironment

__all__ = ["SOCAlertEnvironment"]
