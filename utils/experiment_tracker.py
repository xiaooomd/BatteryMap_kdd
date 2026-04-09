"""Unified experiment tracker abstraction backed by SwanLab."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional


class SwanLabTracker:
    """Fault-tolerant SwanLab tracker wrapper used by training scripts."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        mode: Optional[str] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._enabled = enabled
        self._logger = logger or print
        self._module = None
        self._run = None
        self._available = False

        if not self._enabled:
            return

        if self._is_disabled_by_env():
            self._log("SwanLab is disabled by environment variable. Running without experiment tracking.")
            return

        try:
            import swanlab  # type: ignore

            self._module = swanlab
            self._run = self._initialize_run(project=project, run_name=run_name, config=config or {}, mode=mode)
            self._available = True
        except Exception as exc:
            self._log(f"SwanLab initialization failed: {exc}. Proceeding without SwanLab.")

    @property
    def available(self) -> bool:
        return self._available

    def log(self, metrics: Dict[str, Any]) -> None:
        if not self._available:
            return
        try:
            if self._run is not None and hasattr(self._run, "log"):
                self._run.log(metrics)
            elif self._module is not None and hasattr(self._module, "log"):
                self._module.log(metrics)
        except Exception as exc:
            self._log(f"SwanLab metric logging failed: {exc}.")

    def finish(self) -> None:
        if not self._available:
            return
        try:
            if self._run is not None and hasattr(self._run, "finish"):
                self._run.finish()
            elif self._module is not None and hasattr(self._module, "finish"):
                self._module.finish()
        except Exception as exc:
            self._log(f"SwanLab finish failed: {exc}.")

    def _initialize_run(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        mode: Optional[str],
    ) -> Any:
        payload = {
            "project": project,
            "name": run_name,
            "config": config,
        }
        if mode:
            payload["mode"] = mode

        try:
            return self._module.init(**payload)
        except TypeError:
            # Keep backward compatibility with possible alternative arg names.
            fallback_payloads = [
                {"project": project, "experiment_name": run_name, "config": config},
                {"project": project, "run_name": run_name, "config": config},
                {"project": project, "name": run_name},
            ]
            if mode:
                for item in fallback_payloads:
                    item["mode"] = mode

            for item in fallback_payloads:
                try:
                    return self._module.init(**item)
                except TypeError:
                    continue

            raise

    def _is_disabled_by_env(self) -> bool:
        disable_flags = {
            "0",
            "false",
            "off",
            "disable",
            "disabled",
        }
        mode_value = os.getenv("SWANLAB_MODE", "").strip().lower()
        disable_value = os.getenv("SWANLAB_DISABLED", "").strip().lower()
        return mode_value in disable_flags or disable_value in disable_flags

    def _log(self, message: str) -> None:
        try:
            self._logger(message)
        except Exception:
            pass
