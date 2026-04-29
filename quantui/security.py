"""
QuantUI Security Module

Provides a catchable SecurityError exception for the local teaching interface.

Path-traversal hardening, SLURM resource limits, concurrent job enforcement,
and email/mail-event validation have been removed — they are only relevant
in a multi-user cluster environment. Local resource sanity checks live in
utils.session_can_handle() instead.
"""


class SecurityError(Exception):
    """Raised when a security constraint is violated."""
