"""Detector-neutral verified Base-run identity boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .stage_a import BaseRunIdentity


@dataclass(frozen=True, slots=True)
class _VerifiedBaseRunIdentityBinding:
    identity: BaseRunIdentity
    identity_values: tuple[tuple[str, str], ...]
    verify_source: Callable[[], None]


def _identity_values(identity: BaseRunIdentity) -> tuple[tuple[str, str], ...]:
    return tuple(identity.to_registry_dict().items())


@dataclass(frozen=True)
class VerifiedBaseRunIdentity:
    """A neutral identity returned by a provider-specific strict loader."""

    identity: BaseRunIdentity
    _verification_token: object

    def _verify_source_binding(self) -> _VerifiedBaseRunIdentityBinding:
        binding = self._verification_token
        if type(binding) is not _VerifiedBaseRunIdentityBinding:
            raise TypeError(
                "VerifiedBaseRunIdentity must come from a registered Base loader"
            )
        if binding.identity is not self.identity:
            raise TypeError("verified Base-run identity fields were replaced")
        if binding.identity_values != _identity_values(self.identity):
            raise TypeError("verified Base-run identity values were replaced")
        return binding

    def __post_init__(self) -> None:
        if not isinstance(self.identity, BaseRunIdentity):
            raise TypeError("identity must be a BaseRunIdentity")
        self._verify_source_binding()

    def verify_unchanged(self) -> None:
        """Invoke the provider's exact source-record verification."""

        self._verify_source_binding().verify_source()


def _bind_verified_base_run_identity(
    identity: BaseRunIdentity,
    verify_source: Callable[[], None],
) -> VerifiedBaseRunIdentity:
    if not isinstance(identity, BaseRunIdentity):
        raise TypeError("identity must be a BaseRunIdentity")
    if not callable(verify_source):
        raise TypeError("verify_source must be callable")
    binding = _VerifiedBaseRunIdentityBinding(
        identity,
        _identity_values(identity),
        verify_source,
    )
    result = VerifiedBaseRunIdentity(identity, binding)
    result.verify_unchanged()
    return result


__all__ = ["VerifiedBaseRunIdentity"]
