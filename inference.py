#!/usr/bin/env python3
"""
PatchCascade SOC - Inference Script
====================================

Baseline LLM evaluation script for the Meta PyTorch OpenEnv Hackathon.
Connects an LLM agent to the PatchCascade SOC environment and logs
results in the required standardized format.

Environment Variables:
    API_BASE_URL: LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME: Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN: HuggingFace API token (required)
    TASK_LEVEL: Environment difficulty (default: medium)
    ENV_SEED: Random seed for reproducibility (optional)

Output Format (STRICT - DO NOT MODIFY):
    [START] task={level} env=patchcascade model={model}
    [STEP] step={n} action={type}_{target} reward={r:.2f} done={true/false} error={msg|null}
    [END] success={true/false} steps={n} rewards={r1:.2f},{r2:.2f},...

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Literal

from openai import AsyncOpenAI

from client import PatchCascadeLocalClient, StepResult
from models import ActionType, PatchCascadeAction, PatchCascadeObservation


# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
TASK_LEVEL = os.environ.get("TASK_LEVEL", "medium")
ENV_SEED = os.environ.get("ENV_SEED")

# Maximum retries for LLM parsing errors
MAX_PARSE_RETRIES = 3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert Security Operations Center (SOC) engineer managing vulnerability patches across a network of interdependent servers.

Your task is to analyze the current network state and choose the optimal action to:
1. Patch critical vulnerabilities as quickly as possible (reduces risk penalty)
2. Minimize service downtime (reduces downtime penalty)
3. Avoid cascade failures by managing dependencies correctly

CRITICAL RULES:
- Tier 1 (CRITICAL) nodes MUST be SUSPENDED before patching
- If node A depends on node B, and B goes offline/suspended, A will CRASH unless A is also SUSPENDED first
- Patches take 1 turn to complete. After patching starts, the node returns to ONLINE automatically.
- Prioritize CRITICAL and HIGH severity vulnerabilities, especially if exploit_in_wild=true

AVAILABLE ACTIONS:
- scan_host: Get detailed info about a specific node (no state change)
- suspend_service: Gracefully take a node offline (required before patching Tier 1)
- apply_patch: Fix a vulnerability (specify cve_id). Node must be ONLINE (Tier 2-3) or SUSPENDED (Tier 1)
- resume_service: Bring a SUSPENDED or CRASHED node back ONLINE
- noop: Do nothing this turn

You must respond with ONLY a valid JSON object matching this exact schema:
{
    "action_type": "scan_host" | "suspend_service" | "apply_patch" | "resume_service" | "noop",
    "target": "<hostname or empty string for noop>",
    "cve_id": "<CVE-YYYY-NNNNN or null if not apply_patch>",
    "reason": "<brief explanation of your decision>"
}

Do NOT include any text before or after the JSON. Only output the JSON object."""


# =============================================================================
# LLM CLIENT
# =============================================================================

def create_llm_client() -> AsyncOpenAI:
    """Create the OpenAI-compatible async client."""
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    return AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )


async def get_llm_action(
    client: AsyncOpenAI,
    observation: PatchCascadeObservation,
    retry_count: int = 0,
) -> PatchCascadeAction:
    """
    Query the LLM to get the next action.
    
    Args:
        client: OpenAI-compatible async client.
        observation: Current environment observation.
        retry_count: Current retry attempt for parse errors.
    
    Returns:
        Parsed PatchCascadeAction from LLM response.
    
    Raises:
        ValueError: If LLM response cannot be parsed after retries.
    """
    # Prepare observation as JSON for LLM
    obs_json = observation.model_dump_json(indent=2)
    
    user_message = f"""Current network state:

{obs_json}

Analyze this state and respond with your action as a JSON object."""
    
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from LLM")
        
        # Strip any markdown code blocks if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON
        action_dict = json.loads(content)
        
        # Validate and construct action
        action_type = ActionType(action_dict["action_type"])
        target = action_dict.get("target", "")
        cve_id = action_dict.get("cve_id")
        reason = action_dict.get("reason", "")
        
        # Normalize empty cve_id
        if cve_id == "" or cve_id == "null":
            cve_id = None
        
        return PatchCascadeAction(
            action_type=action_type,
            target=target,
            cve_id=cve_id,
            reason=reason,
        )
    
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        if retry_count < MAX_PARSE_RETRIES:
            # Retry with backoff
            await asyncio.sleep(0.5 * (retry_count + 1))
            return await get_llm_action(client, observation, retry_count + 1)
        
        # Fallback to NOOP after max retries
        return PatchCascadeAction(
            action_type=ActionType.NOOP,
            target="",
            reason=f"Parse error fallback: {str(e)}",
        )


# =============================================================================
# OUTPUT FORMATTING (STRICT COMPLIANCE)
# =============================================================================

def print_start(task_level: str, model_name: str) -> None:
    """Print the [START] line. Exactly once at beginning."""
    print(f"[START] task={task_level} env=patchcascade model={model_name}", flush=True)


def print_step(step_num: int, action: PatchCascadeAction, reward: float, done: bool, error: str | None) -> None:
    """
    Print the [STEP] line. Exactly once per step.
    
    Format: [STEP]  step={n} action={json} reward={r:.2f} done={true/false} error={msg|null}
    NOTE: Two spaces after [STEP] per grading bot regex requirements.
    """
    action_str = action.model_dump_json(exclude_none=True).replace(" ", "")
    done_str = "true" if done else "false"
    error_str = error if error is not None else "null"
    print(f"[STEP]  step={step_num} action={action_str} reward={reward:.2f} done={done_str} error={error_str}", flush=True)


def print_end(success: bool, total_steps: int, rewards: list[float]) -> None:
    """
    Print the [END] line. Exactly once at termination.
    
    Format: [END]   success={true/false} steps={n} rewards={r1:.2f},{r2:.2f},...
    NOTE: Three spaces after [END] per grading bot regex requirements.
    """
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END]   success={success_str} steps={total_steps} rewards={rewards_str}", flush=True)


# =============================================================================
# MAIN INFERENCE LOOP
# =============================================================================

async def run_inference() -> None:
    """
    Main inference loop.
    
    1. Initialize environment and LLM client
    2. Reset environment
    3. Loop: get LLM action -> step environment -> log results
    4. Print final summary
    """
    # Validate configuration
    task_level: Literal["easy", "medium", "hard"]
    if TASK_LEVEL in ("easy", "medium", "hard"):
        task_level = TASK_LEVEL  # type: ignore
    else:
        print(f"ERROR: Invalid TASK_LEVEL: {TASK_LEVEL}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize clients
    llm_client = create_llm_client()
    env_seed = int(ENV_SEED) if ENV_SEED else None
    env_client = PatchCascadeLocalClient(seed=env_seed)
    
    # Print start marker (EXACTLY ONCE)
    print_start(task_level, MODEL_NAME)
    
    # Reset environment
    observation = env_client.reset(task_level=task_level, seed=env_seed)
    
    # Tracking variables
    step_num = 0
    rewards: list[float] = []
    done = False
    success = False
    
    # Main loop
    while not done:
        # Get action from LLM
        action = await get_llm_action(llm_client, observation)
        
        # Execute step
        result: StepResult = env_client.step(action)
        
        # Extract results
        observation = result.observation
        reward = result.reward
        done = result.done
        info = result.info
        
        # Track rewards
        rewards.append(reward)
        step_num += 1
        
        # Get error message if any
        error_msg: str | None = info.get("error") if not info.get("valid", True) else None
        
        # Print step marker (EXACTLY ONCE PER STEP)
        print_step(step_num, action, reward, done, error_msg)
        
        # Check for victory
        if done:
            # Success if all vulnerabilities patched (termination_reason would be "all_patched")
            # We can infer from observation: if no vulnerabilities remain, it's a success
            success = len(observation.vulnerabilities) == 0
    
    # Print end marker (EXACTLY ONCE)
    print_end(success, step_num, rewards)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Entry point for the inference script."""
    try:
        asyncio.run(run_inference())
    except KeyboardInterrupt:
        print("\nInference interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
