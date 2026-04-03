"""
PatchCascade SOC - OpenEnv Server
==================================

FastAPI-based server wrapping the PatchCascadeEnv for OpenEnv compliance.
Run with: uvicorn server:app --host 0.0.0.0 --port 8000

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import PatchCascadeEnv, StepResult
from models import (
    PatchCascadeAction,
    PatchCascadeObservation,
    PatchCascadeState,
    ActionType,
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class ResetRequest(BaseModel):
    """Request body for the /reset endpoint."""
    task_level: Literal["easy", "medium", "hard"] = "easy"
    seed: int | None = None


class StepRequest(BaseModel):
    """Request body for the /step endpoint."""
    action_type: str
    target: str = ""
    cve_id: str | None = None
    reason: str = ""


class StepResponse(BaseModel):
    """Response body from the /step endpoint."""
    observation: dict
    reward: float
    done: bool
    truncated: bool
    info: dict


class ObservationResponse(BaseModel):
    """Response body from the /observation endpoint."""
    observation: dict


class StateResponse(BaseModel):
    """Response body from the /state endpoint (debug only)."""
    state: dict


class HealthResponse(BaseModel):
    """Response body from the /health endpoint."""
    status: str
    environment: str
    version: str


# =============================================================================
# ENVIRONMENT WRAPPER (OpenEnv-style base class pattern)
# =============================================================================


class Environment:
    """
    Base Environment class following OpenEnv conventions.
    
    Subclasses should override reset(), step(), and the state property.
    This base class provides the interface contract.
    """
    
    def reset(self, task_level: str = "easy", seed: int | None = None) -> PatchCascadeObservation:
        """Reset the environment to initial state."""
        raise NotImplementedError
    
    def step(self, action: PatchCascadeAction) -> StepResult:
        """Execute one step in the environment."""
        raise NotImplementedError
    
    @property
    def state(self) -> PatchCascadeState:
        """Access the internal state (for debugging/grading)."""
        raise NotImplementedError
    
    def get_observation(self) -> PatchCascadeObservation:
        """Get current observation without advancing state."""
        raise NotImplementedError


class PatchCascadeEnvironment(Environment):
    """
    OpenEnv-compliant wrapper around PatchCascadeEnv.
    
    This class adapts our environment to the OpenEnv server interface,
    handling request parsing, action construction, and response formatting.
    """
    
    def __init__(self, seed: int | None = None):
        """Initialize the environment wrapper."""
        self._env = PatchCascadeEnv(seed=seed)
        self._initialized = False
    
    def reset(
        self,
        task_level: Literal["easy", "medium", "hard"] = "easy",
        seed: int | None = None,
    ) -> PatchCascadeObservation:
        """
        Reset the environment to a new episode.
        
        Args:
            task_level: Difficulty level ("easy", "medium", "hard").
            seed: Optional random seed for reproducibility.
        
        Returns:
            Initial observation for the agent.
        """
        obs = self._env.reset(task_level=task_level, seed=seed)
        self._initialized = True
        return obs
    
    def step(self, action: PatchCascadeAction) -> StepResult:
        """
        Execute one step in the environment.
        
        Args:
            action: The action to execute.
        
        Returns:
            StepResult containing observation, reward, done, truncated, info.
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._env.step(action)
    
    @property
    def state(self) -> PatchCascadeState:
        """Access internal state for debugging/grading."""
        return self._env.state
    
    def get_observation(self) -> PatchCascadeObservation:
        """Get current observation without advancing state."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._env.get_observation()
    
    def render(self) -> str:
        """Render human-readable state summary."""
        return self._env.render()


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Global environment instance
_env: PatchCascadeEnvironment | None = None


def get_env() -> PatchCascadeEnvironment:
    """Get or create the global environment instance."""
    global _env
    if _env is None:
        _env = PatchCascadeEnvironment()
    return _env


# Create FastAPI app
app = FastAPI(
    title="PatchCascade SOC Environment",
    description="OpenEnv-compliant RL environment for SOC vulnerability management simulation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="healthy",
        environment="patchcascade",
        version="1.0.0",
    )


@app.post("/reset", response_model=ObservationResponse)
async def reset_environment(request: ResetRequest) -> ObservationResponse:
    """
    Reset the environment to a new episode.
    
    Args:
        request: Contains task_level and optional seed.
    
    Returns:
        Initial observation for the agent.
    """
    try:
        env = get_env()
        obs = env.reset(task_level=request.task_level, seed=request.seed)
        return ObservationResponse(observation=obs.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/step", response_model=StepResponse)
async def step_environment(request: StepRequest) -> StepResponse:
    """
    Execute one step in the environment.
    
    Args:
        request: Contains action_type, target, and optional cve_id/reason.
    
    Returns:
        Step result with observation, reward, done, truncated, info.
    """
    try:
        env = get_env()
        
        # Parse action type
        try:
            action_type = ActionType(request.action_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action_type: {request.action_type}. "
                       f"Valid types: {[a.value for a in ActionType]}"
            )
        
        # Construct action
        action = PatchCascadeAction(
            action_type=action_type,
            target=request.target,
            cve_id=request.cve_id,
            reason=request.reason,
        )
        
        # Execute step
        result = env.step(action)
        
        return StepResponse(
            observation=result.observation.model_dump(),
            reward=result.reward,
            done=result.done,
            truncated=result.truncated,
            info=result.info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/observation", response_model=ObservationResponse)
async def get_observation() -> ObservationResponse:
    """
    Get the current observation without advancing state.
    
    Returns:
        Current observation.
    """
    try:
        env = get_env()
        obs = env.get_observation()
        return ObservationResponse(observation=obs.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/state", response_model=StateResponse)
async def get_state() -> StateResponse:
    """
    Get the internal state (for debugging/grading only).
    
    Returns:
        Full internal state.
    """
    try:
        env = get_env()
        return StateResponse(state=env.state.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/render")
async def render_environment() -> dict[str, str]:
    """
    Render a human-readable summary of the current state.
    
    Returns:
        Rendered state as text.
    """
    try:
        env = get_env()
        return {"render": env.render()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/schema/action")
async def get_action_schema() -> dict:
    """Get the JSON schema for PatchCascadeAction."""
    return PatchCascadeAction.model_json_schema()


@app.get("/schema/observation")
async def get_observation_schema() -> dict:
    """Get the JSON schema for PatchCascadeObservation."""
    return PatchCascadeObservation.model_json_schema()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
