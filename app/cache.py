"""Cache layer for the Streamlit dashboard.

Streamlit re-runs the script on every interaction, so we wrap heavy reads in
``st.cache_data`` to keep latency under a second on subsequent renders.

The @st.cache_data decorator must be applied at import time when running
inside Streamlit, but we also want the modules to import cleanly during unit
testing.  The ``_get_cache_decorator`` helper picks the right decorator
transparently.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

try:
    import streamlit as st  # noqa: F401  (presence test)

    def _get_cache_decorator() -> Callable[[Any], Any]:
        return st.cache_data(show_spinner=False)
except Exception:  # pragma: no cover -- exercised only when streamlit is missing
    def _get_cache_decorator() -> Callable[[Any], Any]:
        def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(fn)
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)

            return _wrapper

        return _identity


cache = _get_cache_decorator()
