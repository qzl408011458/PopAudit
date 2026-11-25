"""Process-wide guard: forbid opening the reference split inside an HPO trial.

CPython imports ``sitecustomize`` automatically at interpreter start-up if it is
importable.  ``hpo.run`` prepends this directory to ``PYTHONPATH`` and sets
``SEMAPOP_HPO_FORBID_REF=1`` for every trial subprocess, so the assertion
"a trial never opens the reference population" is *enforced*, not merely
documented: any attempt raises immediately and fails the trial.

Deactivated (no patching at all) when the env var is unset, so ``finalize.py``
-- which legitimately reads the reference to report the val-vs-reference gap
AFTER the winner has been chosen -- is unaffected.
"""
from __future__ import annotations

import os

_FLAG = os.environ.get("SEMAPOP_HPO_FORBID_REF", "")

if _FLAG not in ("", "0"):
    import builtins
    import io

    FORBIDDEN = tuple(
        b.strip() for b in os.environ.get(
            "SEMAPOP_HPO_FORBID_BASENAMES",
            "referencedata_pers.csv,testdata_pers.csv",
        ).split(",") if b.strip()
    )

    class ReferenceLeak(RuntimeError):
        """Raised when trial code touches the reference / test split."""

    def _check(path) -> None:
        try:
            p = os.fspath(path)
        except TypeError:
            return                                    # fd or unusual object
        if isinstance(p, bytes):
            try:
                p = p.decode("utf-8", "ignore")
            except Exception:                         # noqa: BLE001
                return
        if not isinstance(p, str):
            return
        base = os.path.basename(p.rstrip("/"))
        if base in FORBIDDEN:
            raise ReferenceLeak(
                f"HPO fairness guard: trial code tried to open {p!r}. "
                "The objective is validation-only; the reference split may be "
                "read only by hpo/finalize.py after model selection."
            )

    _open, _io_open, _os_open = builtins.open, io.open, os.open

    def open(file, *a, **kw):                          # noqa: A001
        _check(file)
        return _open(file, *a, **kw)

    def _io_open_guarded(file, *a, **kw):
        _check(file)
        return _io_open(file, *a, **kw)

    def _os_open_guarded(path, *a, **kw):
        _check(path)
        return _os_open(path, *a, **kw)

    builtins.open = open
    io.open = _io_open_guarded
    os.open = _os_open_guarded
    builtins.__semapop_hpo_ref_guard__ = True
