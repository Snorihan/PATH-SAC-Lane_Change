# CLAUDE.md

## Project
This repo is for a PATH autonomous lane-change reinforcement learning project.

We are building and debugging a lane-changing RL pipeline that connects:
- Aimsun Next 8.4 traffic simulation / plugin side
- Python RL training code
- A HighwayEnv-style custom environment / wrapper
- Soft Actor-Critic style continuous control for lane changing

## Research goal
Train an autonomous vehicle to perform safe and efficient lane changes in mixed traffic.

The project direction is similar to DRL lane-changing work that uses:
- continuous actions
- SAC
- surrounding-vehicle state inputs
- reward shaping around safety, efficiency, and completion
- naturalistic trajectory-inspired mixed traffic environments

Relevant paper summary:
- Uses SAC for continuous lane-changing control
- State includes relative longitudinal distances/velocities plus lateral state
- Action includes acceleration magnitude / direction-type control
- Reward emphasizes safety, efficiency, and lane-change completion
- Reports strong lane-change success in testing with human-trajectory-based mixed traffic setup :contentReference[oaicite:1]{index=1}

## Current codebase intent
The main practical goals are:
1. get the simulation / environment pipeline working reliably
2. verify file outputs, naming, and paths from the simulation side
3. ensure Python reads the correct generated data
4. debug custom env registration and wrappers
5. train and evaluate SAC-based lane-change behavior
6. keep changes modular and debuggable

## Important technical context
I am working across:
- Python
- Gymnasium / custom environment registration
- HighwayEnv-style wrappers
- SAC training code
- Aimsun plugin / C++ side
- XML / config / output-file path issues
- CSV / TXT trajectory data conversion and loading

## What I usually need help with
Claude should be ready to help with:
- tracing where simulation outputs are written
- finding file naming/path logic in C++ / XML / config
- checking whether Python loaders read the intended file
- debugging environment reset/step/reward logic
- debugging custom wrappers and registration
- checking reward shaping and termination logic
- making minimal, targeted edits instead of broad rewrites
- proposing concrete debugging steps with file/function names
- modifying reward functions and overall SAC pipeline
- I wish to be asked about intent and reasons behind design decisions

## Constraints / preferences
- Be direct and technical.
- Prefer small, inspectable edits.
- When debugging, first identify the exact file/function/control flow.
- When uncertain, inspect the code paths before suggesting architecture changes.
- Preserve existing project structure unless there is a strong reason to change it.
- Distinguish clearly between confirmed findings and hypotheses.

## Preferred debugging style
When investigating an issue:
1. identify the relevant files/functions
2. trace control flow
3. state what is confirmed
4. state what is still uncertain
5. propose the smallest next change or check

## Output style
When giving code help:
- reference exact files and functions
- explain why the issue occurs
- suggest a minimal patch
- mention side effects / assumptions