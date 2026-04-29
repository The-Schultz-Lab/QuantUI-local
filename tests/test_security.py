"""
Tests for quantui.security

QuantUI security module contains only SecurityError.
Path-traversal hardening, resource-limit enforcement, concurrent-job
limits, and walltime validation are SLURM-cluster concerns removed from
the local version — see quantui.utils.session_can_handle() instead.
"""

import pytest

from quantui.security import SecurityError


class TestSecurityError:
    """SecurityError is the only public symbol in quantui.security."""

    def test_is_exception_subclass(self):
        """SecurityError must be catchable as a plain Exception."""
        assert issubclass(SecurityError, Exception)

    def test_raise_and_catch(self):
        """SecurityError can be raised and caught."""
        with pytest.raises(SecurityError):
            raise SecurityError("test constraint violated")

    def test_message_preserved(self):
        """The message passed to SecurityError is preserved."""
        msg = "path traversal detected"
        with pytest.raises(SecurityError, match=msg):
            raise SecurityError(msg)

    def test_catch_as_generic_exception(self):
        """SecurityError is catchable as a generic Exception."""
        caught = False
        try:
            raise SecurityError("boom")
        except Exception:
            caught = True
        assert caught

    def test_does_not_catch_base_exception(self):
        """SecurityError should not mask SystemExit/KeyboardInterrupt."""
        assert not issubclass(SecurityError, SystemExit)
        assert not issubclass(SecurityError, KeyboardInterrupt)
