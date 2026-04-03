"""
PatchCascade SOC - OpenEnv Client
==================================

HTTP client for connecting to the PatchCascade SOC environment server.
Implements the OpenEnv HTTPEnvClient pattern for type-safe interactions.

Usage:
    async with PatchCascadeClient("http://localhost:8000") as client:
        obs = await client.reset(task_level="medium")
        result = await client.step(action)

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import httpx

from models import (
    ActionType,
    PatchCascadeAction,
    PatchCascadeObservation,
    PatchCascadeState,
    ServerNode,
    Vulnerability,
    Dependency,
    NetworkHealth,
    CriticalityTier,
    NodeState,
    SeverityLevel,
)


# =============================================================================
# TYPE VARIABLES FOR GENERIC CLIENT
# =============================================================================

ActionT = TypeVar("ActionT")
ObservationT = TypeVar("ObservationT")


# =============================================================================
# STEP RESULT (mirrors environment.py)
# =============================================================================

@dataclass
class StepResult:
    """
    Result from an environment step.
    
    Compatible with OpenEnv and Gymnasium conventions.
    """
    observation: PatchCascadeObservation
    reward: float
    done: bool
    truncated: bool
    info: dict
    
    def as_tuple(self) -> tuple:
        """Convert to standard (obs, reward, done, truncated, info) tuple."""
        return (self.observation, self.reward, self.done, self.truncated, self.info)


# =============================================================================
# BASE HTTP CLIENT (OpenEnv pattern)
# =============================================================================

class HTTPEnvClient(Generic[ActionT, ObservationT]):
    """
    Base HTTP client for OpenEnv environments.
    
    Subclasses must implement:
    - _step_payload(action): Convert action to JSON dict
    - _parse_observation(data): Parse JSON to observation type
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the HTTP client.
        
        Args:
            base_url: Base URL of the environment server (e.g., "http://localhost:8000").
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def __aenter__(self) -> "HTTPEnvClient[ActionT, ObservationT]":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get the HTTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        return self._client
    
    def _step_payload(self, action: ActionT) -> dict:
        """Convert action to JSON payload. Override in subclass."""
        raise NotImplementedError
    
    def _parse_observation(self, data: dict) -> ObservationT:
        """Parse JSON response to observation. Override in subclass."""
        raise NotImplementedError
    
    async def health(self) -> dict:
        """Check server health."""
        client = self._get_client()
        response = await client.get(f"{self._base_url}/health")
        response.raise_for_status()
        return response.json()


# =============================================================================
# PATCHCASCADE CLIENT
# =============================================================================

class PatchCascadeClient(HTTPEnvClient[PatchCascadeAction, PatchCascadeObservation]):
    """
    HTTP client for the PatchCascade SOC environment.
    
    Provides type-safe methods for interacting with the environment server.
    
    Usage:
        async with PatchCascadeClient("http://localhost:8000") as client:
            obs = await client.reset(task_level="medium")
            
            action = PatchCascadeAction(
                action_type=ActionType.APPLY_PATCH,
                target="web-server-01",
                cve_id="CVE-2024-1001"
            )
            result = await client.step(action)
    """
    
    def _step_payload(self, action: PatchCascadeAction) -> dict:
        """
        Convert PatchCascadeAction to JSON payload for /step endpoint.
        
        Args:
            action: The action to convert.
        
        Returns:
            JSON-serializable dict matching StepRequest schema.
        """
        return {
            "action_type": action.action_type.value,
            "target": action.target,
            "cve_id": action.cve_id,
            "reason": action.reason,
        }
    
    def _parse_observation(self, data: dict) -> PatchCascadeObservation:
        """
        Parse JSON response to PatchCascadeObservation.
        
        Handles nested model reconstruction for full type safety.
        
        Args:
            data: Raw JSON dict from server response.
        
        Returns:
            Fully typed PatchCascadeObservation.
        """
        # Parse nested nodes
        nodes = [
            ServerNode(
                hostname=n["hostname"],
                os=n["os"],
                tier=CriticalityTier(n["tier"]),
                state=NodeState(n["state"]),
                services=n.get("services", []),
                patch_turns_remaining=n.get("patch_turns_remaining", 0),
            )
            for n in data.get("nodes", [])
        ]
        
        # Parse nested vulnerabilities
        vulnerabilities = [
            Vulnerability(
                cve_id=v["cve_id"],
                severity=SeverityLevel(v["severity"]),
                cvss_score=v["cvss_score"],
                affected_hosts=v["affected_hosts"],
                description=v.get("description", ""),
                patch_available=v.get("patch_available", True),
                exploit_in_wild=v.get("exploit_in_wild", False),
            )
            for v in data.get("vulnerabilities", [])
        ]
        
        # Parse nested dependencies
        dependencies = [
            Dependency(
                node=d["node"],
                depends_on=d["depends_on"],
                dependency_type=d.get("dependency_type", "hard"),
                description=d.get("description", ""),
            )
            for d in data.get("dependencies", [])
        ]
        
        # Parse health metrics
        health_data = data.get("health", {})
        health = NetworkHealth(
            total_nodes=health_data.get("total_nodes", 0),
            nodes_online=health_data.get("nodes_online", 0),
            nodes_crashed=health_data.get("nodes_crashed", 0),
            nodes_patching=health_data.get("nodes_patching", 0),
            active_critical_vulns=health_data.get("active_critical_vulns", 0),
            active_high_vulns=health_data.get("active_high_vulns", 0),
            active_medium_vulns=health_data.get("active_medium_vulns", 0),
            active_low_vulns=health_data.get("active_low_vulns", 0),
            cumulative_downtime_penalty=health_data.get("cumulative_downtime_penalty", 0.0),
            cumulative_risk_penalty=health_data.get("cumulative_risk_penalty", 0.0),
            turn_number=health_data.get("turn_number", 0),
        )
        
        return PatchCascadeObservation(
            nodes=nodes,
            vulnerabilities=vulnerabilities,
            dependencies=dependencies,
            health=health,
            last_action_result=data.get("last_action_result"),
            messages=data.get("messages", []),
        )
    
    def _parse_result(self, response_data: dict) -> StepResult:
        """
        Parse full step response to StepResult.
        
        Args:
            response_data: JSON response from /step endpoint.
        
        Returns:
            Typed StepResult with parsed observation.
        """
        observation = self._parse_observation(response_data["observation"])
        
        return StepResult(
            observation=observation,
            reward=float(response_data["reward"]),
            done=bool(response_data["done"]),
            truncated=bool(response_data["truncated"]),
            info=response_data.get("info", {}),
        )
    
    async def reset(
        self,
        task_level: Literal["easy", "medium", "hard"] = "easy",
        seed: int | None = None,
    ) -> PatchCascadeObservation:
        """
        Reset the environment to a new episode.
        
        Args:
            task_level: Difficulty level.
            seed: Optional random seed.
        
        Returns:
            Initial observation.
        """
        client = self._get_client()
        
        payload = {"task_level": task_level}
        if seed is not None:
            payload["seed"] = seed
        
        response = await client.post(f"{self._base_url}/reset", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return self._parse_observation(data["observation"])
    
    async def step(self, action: PatchCascadeAction) -> StepResult:
        """
        Execute one step in the environment.
        
        Args:
            action: The action to execute.
        
        Returns:
            StepResult with observation, reward, done, truncated, info.
        """
        client = self._get_client()
        
        payload = self._step_payload(action)
        response = await client.post(f"{self._base_url}/step", json=payload)
        response.raise_for_status()
        
        return self._parse_result(response.json())
    
    async def get_observation(self) -> PatchCascadeObservation:
        """
        Get current observation without advancing state.
        
        Returns:
            Current observation.
        """
        client = self._get_client()
        
        response = await client.get(f"{self._base_url}/observation")
        response.raise_for_status()
        
        data = response.json()
        return self._parse_observation(data["observation"])
    
    async def get_state(self) -> dict:
        """
        Get internal state (for debugging/grading).
        
        Returns:
            Raw state dict.
        """
        client = self._get_client()
        
        response = await client.get(f"{self._base_url}/state")
        response.raise_for_status()
        
        return response.json()["state"]
    
    async def render(self) -> str:
        """
        Get human-readable state summary.
        
        Returns:
            Rendered state as string.
        """
        client = self._get_client()
        
        response = await client.get(f"{self._base_url}/render")
        response.raise_for_status()
        
        return response.json()["render"]
    
    async def get_action_schema(self) -> dict:
        """Get JSON schema for actions."""
        client = self._get_client()
        response = await client.get(f"{self._base_url}/schema/action")
        response.raise_for_status()
        return response.json()
    
    async def get_observation_schema(self) -> dict:
        """Get JSON schema for observations."""
        client = self._get_client()
        response = await client.get(f"{self._base_url}/schema/observation")
        response.raise_for_status()
        return response.json()


# =============================================================================
# SYNCHRONOUS WRAPPER (for non-async contexts)
# =============================================================================

class PatchCascadeClientSync:
    """
    Synchronous wrapper around PatchCascadeClient.
    
    For use in contexts where async is not available.
    
    Usage:
        client = PatchCascadeClientSync("http://localhost:8000")
        obs = client.reset(task_level="medium")
        result = client.step(action)
        client.close()
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """Initialize synchronous client."""
        self._base_url = base_url
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self) -> "PatchCascadeClientSync":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def reset(
        self,
        task_level: Literal["easy", "medium", "hard"] = "easy",
        seed: int | None = None,
    ) -> PatchCascadeObservation:
        """Reset environment synchronously."""
        payload = {"task_level": task_level}
        if seed is not None:
            payload["seed"] = seed
        
        response = self._client.post(f"{self._base_url}/reset", json=payload)
        response.raise_for_status()
        
        # Use simple dict access for sync version
        return PatchCascadeObservation.model_validate(response.json()["observation"])
    
    def step(self, action: PatchCascadeAction) -> StepResult:
        """Execute step synchronously."""
        payload = {
            "action_type": action.action_type.value,
            "target": action.target,
            "cve_id": action.cve_id,
            "reason": action.reason,
        }
        
        response = self._client.post(f"{self._base_url}/step", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return StepResult(
            observation=PatchCascadeObservation.model_validate(data["observation"]),
            reward=float(data["reward"]),
            done=bool(data["done"]),
            truncated=bool(data["truncated"]),
            info=data.get("info", {}),
        )


# =============================================================================
# LOCAL CLIENT (Direct environment access, no HTTP)
# =============================================================================

class PatchCascadeLocalClient:
    """
    Local client that directly wraps PatchCascadeEnv.
    
    Use this for local testing without starting a server.
    API mirrors PatchCascadeClient for easy swapping.
    
    Usage:
        client = PatchCascadeLocalClient()
        obs = client.reset(task_level="medium")
        result = client.step(action)
    """
    
    def __init__(self, seed: int | None = None):
        """Initialize local client with embedded environment."""
        from environment import PatchCascadeEnv
        self._env = PatchCascadeEnv(seed=seed)
    
    def reset(
        self,
        task_level: Literal["easy", "medium", "hard"] = "easy",
        seed: int | None = None,
    ) -> PatchCascadeObservation:
        """Reset environment."""
        return self._env.reset(task_level=task_level, seed=seed)
    
    def step(self, action: PatchCascadeAction) -> StepResult:
        """Execute one step."""
        result = self._env.step(action)
        # Convert environment.StepResult to client.StepResult (same structure)
        return StepResult(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            truncated=result.truncated,
            info=result.info,
        )
    
    def get_observation(self) -> PatchCascadeObservation:
        """Get current observation."""
        return self._env.get_observation()
    
    def render(self) -> str:
        """Render state."""
        return self._env.render()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "HTTPEnvClient",
    "PatchCascadeClient",
    "PatchCascadeClientSync",
    "PatchCascadeLocalClient",
    "StepResult",
]
