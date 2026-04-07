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
from grader import (
    GRADERS,
    grade_episode as _grade_episode,
    list_graders,
    GraderResult,
)
from tasks import (
    TASK_REGISTRY,
    get_task,
    list_tasks,
    list_tasks_with_graders,
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class ResetRequest(BaseModel):
    """Request body for the /reset endpoint."""
    task_level: Literal["easy", "medium", "hard", "incident_response", "zero_day"] = "easy"
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
        task_level: Literal["easy", "medium", "hard", "incident_response", "zero_day"] = "easy",
        seed: int | None = None,
    ) -> PatchCascadeObservation:
        """Reset the environment to a new episode."""
        obs = self._env.reset(task_level=task_level, seed=seed)
        self._initialized = True
        return obs
    
    def step(self, action: PatchCascadeAction) -> StepResult:
        """Execute one step in the environment."""
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
# CORE API ENDPOINTS
# =============================================================================


@app.get("/")
async def root() -> dict:
    """Root endpoint with API info."""
    return {
        "name": "PatchCascade SOC",
        "version": "1.0.0",
        "description": "OpenEnv-compliant RL environment for vulnerability patch management",
        "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset",
            "step": "POST /step",
            "observation": "GET /observation",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "metadata": "GET /metadata",
            "grade": "POST /grade/{task_id}",
        },
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="healthy",
        environment="patchcascade",
        version="1.0.0",
    )


@app.post("/reset", response_model=ObservationResponse)
async def reset_environment(request: ResetRequest | None = None) -> ObservationResponse:
    """Reset the environment to a new episode."""
    try:
        env = get_env()
        if request is None:
            request = ResetRequest()
        obs = env.reset(task_level=request.task_level, seed=request.seed)
        return ObservationResponse(observation=obs.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/step", response_model=StepResponse)
async def step_environment(request: StepRequest) -> StepResponse:
    """Execute one step in the environment."""
    try:
        env = get_env()
        
        try:
            action_type = ActionType(request.action_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action_type: {request.action_type}. "
                       f"Valid types: {[a.value for a in ActionType]}"
            )
        
        action = PatchCascadeAction(
            action_type=action_type,
            target=request.target,
            cve_id=request.cve_id,
            reason=request.reason,
        )
        
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
    """Get the current observation without advancing state."""
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
    """Get the internal state (for debugging/grading only)."""
    try:
        env = get_env()
        return StateResponse(state=env.state.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/render")
async def render_environment() -> dict[str, str]:
    """Render a human-readable summary of the current state."""
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


@app.get("/schema")
async def get_schemas() -> dict:
    """Get combined schema for action, observation, and state."""
    return {
        "action": PatchCascadeAction.model_json_schema(),
        "observation": PatchCascadeObservation.model_json_schema(),
        "state": PatchCascadeState.model_json_schema(),
    }


# =============================================================================
# TASKS & GRADERS ENDPOINTS (Required for hackathon Phase 2 validation)
# =============================================================================


def _serialize_task(task: dict) -> dict:
    """Serialize a task for API responses, removing non-serializable fields."""
    return {
        "id": task["id"],
        "name": task["name"],
        "description": task["description"],
        "max_turns": task["max_turns"],
        "difficulty": task["difficulty"],
        "has_grader": task.get("has_grader", False),
        "grader": task.get("grader", {}),
        "success_criteria": task.get("success_criteria", {}),
    }


@app.get("/tasks")
async def get_tasks_endpoint() -> dict:
    """
    Get list of available tasks with their graders.

    This endpoint explicitly lists all tasks and their associated graders
    for hackathon validation compliance.
    """
    all_tasks = list_tasks()
    serialized = [_serialize_task(t) for t in all_tasks]
    tasks_with_graders = [t for t in serialized if t.get("has_grader", False)]
    return {
        "tasks": serialized,
        "count": len(serialized),
        "tasks_with_graders": len(tasks_with_graders),
        "graders_available": len(tasks_with_graders),
    }


@app.get("/tasks/{task_id}")
async def get_task_endpoint(task_id: str) -> dict:
    """Get details for a specific task including its grader."""
    try:
        task = get_task(task_id)
        return _serialize_task(task)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")


@app.get("/graders")
async def get_graders_endpoint() -> dict:
    """
    List all available graders.
    
    Returns grader metadata for each task, confirming that programmatic
    graders are registered and callable.
    """
    graders = list_graders()
    return {
        "graders": graders,
        "count": len(graders),
    }


@app.post("/grade/{task_id}")
async def grade_episode_endpoint(task_id: str, episode_data: dict) -> dict:
    """
    Grade an episode for a specific task using the programmatic grader.
    
    Args:
        task_id: The task identifier (easy, medium, hard)
        episode_data: Episode results including rewards, success status, etc.
    
    Returns:
        Grading results with normalized score (0.0-1.0).
    """
    try:
        result = _grade_episode(task_id, episode_data)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading error: {str(e)}")


@app.get("/metadata")
async def get_metadata() -> dict:
    """
    Get environment metadata including tasks with graders.
    
    Required by OpenEnv validator for metadata_endpoint check.
    Returns name, description, and task/grader information.
    """
    all_tasks = list_tasks()
    serialized_tasks = [_serialize_task(t) for t in all_tasks]
    tasks_with_graders = [t for t in serialized_tasks if t.get("has_grader", False)]
    graders = list_graders()
    
    return {
        "name": "patchcascade",
        "display_name": "PatchCascade SOC",
        "description": (
            "A Security Operations Center (SOC) simulation environment where an agent "
            "manages vulnerability patches across a network of interdependent servers. "
            "The agent must balance patching critical vulnerabilities (reducing risk) "
            "with keeping services online (reducing downtime), while avoiding cascade "
            "failures caused by dependency violations."
        ),
        "version": "1.0.0",
        "author": "Ayush Kumar & Ravi Prashant (PatchCascade SOC Team)",
        "license": "Apache-2.0",
        "repository": "https://github.com/Ayush-Kumar0207/PatchCascade-SOC",
        "tasks": serialized_tasks,
        "tasks_count": len(serialized_tasks),
        "tasks_with_graders": len(tasks_with_graders),
        "graders_count": len(graders),
        "graders": graders,
        "evaluation": {
            "default_task": "medium",
            "scoring_range": [0.0, 1.0],
            "primary_metric": "normalized_score",
        },
        "tags": [
            "openenv",
            "security",
            "network-management",
            "dependency-graph",
            "turn-based",
            "llm-agent",
        ],
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
