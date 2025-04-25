import logging
import os
import time
from typing import Annotated, Optional, Type

import requests
from agentcore.models import V1UserProfile
from fastapi import APIRouter, BackgroundTasks, Depends
from taskara import Task, TaskStatus
from taskara.server.models import V1Task, V1Tasks, V1TaskUpdate
from tenacity import retry, stop_after_attempt, wait_fixed

from surfkit.agent import TaskAgent
from surfkit.auth.transport import get_user_dependency
from surfkit.env import AGENTESEA_HUB_API_KEY_ENV
from surfkit.server.models import V1Agent, V1LearnTask, V1SolveTask
from surfkit.skill import Skill

DEBUG_ENV_VAR = os.getenv("DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if DEBUG_ENV_VAR else logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


def task_router(Agent: Type[TaskAgent]) -> APIRouter:
    """API router for a task agent.

    Args:
        Agent (Type[TaskAgent]): Task agent type.

    Returns:
        APIRouter: An APIRouter for the task agent.
    """

    api_router = APIRouter()

    @api_router.get("/")
    async def root():
        return {"message": f"{Agent.name()} in the shell"}

    @api_router.get("/health")
    async def health():
        return {"status": "ok"}

    @api_router.post("/v1/learn")
    async def learn_task(
        current_user: Annotated[V1UserProfile, Depends(get_user_dependency())],
        background_tasks: BackgroundTasks,
        learn_model: V1LearnTask,
    ):
        task_model = learn_model.task
        logger.info(
            f"learning task: {task_model.model_dump()} with user {current_user.model_dump()}"
        )

        found = Task.find(
            remote=task_model.remote,
            id=task_model.id,
            owner_id=task_model.owner_id,
            auth_token=task_model.auth_token,
        )
        if not found:
            raise Exception(f"Task {task_model.id} not found")

        logger.info(f"found task: {found[0].to_v1().model_dump()}")

        task = found[0]
        task.remote = task_model.remote  # type: ignore
        task.auth_token = task_model.auth_token  # type: ignore

        skill_id = None
        if task.skill:
            skill_id = task.skill
        elif "skill" in task.labels:
            skill_id = task.labels["skill"]
        elif "skill_id" in task.labels:
            skill_id = task.labels["skill_id"]
        else:
            raise ValueError("Task skill or skill label not set")

        logger.info(f"finding skill_id: {skill_id}")
        skills = Skill.find(
            id=skill_id, remote=task.remote, token=task_model.auth_token
        )
        if not skills:
            raise ValueError(f"Skill not found: {skill_id}")
        skill = skills[0]
        logger.info(f"skill: {skill.to_v1().model_dump()}")

        background_tasks.add_task(
            _learn_task, task, skill, current_user, learn_model.agent
        )

    def _learn_task(
        task: Task,
        skill: Skill,
        current_user: V1UserProfile,
        v1_agent: Optional[V1Agent] = None,
    ):
        if v1_agent:
            config = Agent.config_type().model_validate(v1_agent.config)
            agent = Agent.from_config(config=config)
        else:
            agent = Agent.default()

        print(f"agent: {agent}", flush=True)

        if not task.remote or not task.auth_token:
            raise ValueError("Task remote and auth token must be set")

        try:
            print(f"labeling task as training: {task.id}", flush=True)
            _label_task(
                task.remote, task.auth_token, task, "foo/train/status", "training"
            )
            print("labeled task as training", flush=True)
            agent.learn_task(task, skill)
            print(f"labeling task as finished: {task.id}", flush=True)
            _label_task(
                task.remote, task.auth_token, task, "foo/train/status", "finished"
            )
            print("labeled task as finished", flush=True)
        except Exception as e:
            logger.error(f"error learning task: {e}")
            print(f"labeling task as error: {task.id}", flush=True)
            _label_task(task.remote, task.auth_token, task, "foo/train/status", "error")
            _label_task(task.remote, task.auth_token, task, "foo/train/error", str(e))
            print("labeled task as error", flush=True)

    @api_router.post("/v1/tasks")
    async def solve_task(
        current_user: Annotated[V1UserProfile, Depends(get_user_dependency())],
        background_tasks: BackgroundTasks,
        task_model: V1SolveTask,
    ):
        logger.info(
            f"solving task: {task_model.model_dump()} with user {current_user.email}"
        )

        background_tasks.add_task(_solve_task, task_model, current_user)
        logger.info("created background task...")
        return

    def _solve_task(task_model: V1SolveTask, current_user: V1UserProfile):
        owner_id = task_model.task.owner_id
        if not owner_id:
            owner_id = "local"
        task = Task.from_v1(
            task_model.task, owner_id=owner_id, auth_token=task_model.task.auth_token
        )

        logger.info("Saving remote tasks status to running...")
        task.status = TaskStatus.IN_PROGRESS
        task.started = time.time()
        task.save()

        if task_model.task.device:
            logger.info(f"connecting to device {task_model.task.device.name}...")
            device = None
            for Device in Agent.supported_devices():
                if Device.type() == task_model.task.device.type:
                    logger.debug(f"found device: {task_model.task.device.model_dump()}")
                    api_key = task_model.task.auth_token
                    if api_key is None:
                        logger.info("No Api key/token on Task or in Auth")

                    try:
                        config = Device.connect_config_type()(
                            **{**task_model.task.device.config, "api_key": api_key}  # type: ignore
                        )
                        device = Device.connect(config=config)
                    except Exception as e:
                        err = f"error connecting to device: {e}"
                        task.error = err
                        task.status = TaskStatus.ERROR
                        task.save()
                        raise Exception(err)

            if not device:
                raise ValueError(
                    f"Device {task_model.task.device.name} provided in solve task, but not supported by agent"
                )

            logger.debug(f"connected to device: {device.__dict__}")
        else:
            raise ValueError("No device provided")

        logger.info("starting agent...")
        if task_model.agent:
            config = Agent.config_type().model_validate(task_model.agent.config)
            agent = Agent.from_config(config=config)
        else:
            agent = Agent.default()

        try:
            final_task = agent.solve_task(
                task=task, device=device, max_steps=task.max_steps
            )

        except Exception as e:
            logger.error(f"error running agent: {e}")

            task.refresh()
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed = time.time()
            task.save()
            task.post_message(
                "assistant", f"Failed to run task '{task.description}': {e}"
            )
            return

        finally:
            print(f"► task run ended '{task.id}'", flush=True)

        if final_task:
            final_task.refresh()
            final_task.completed = time.time()
            final_task.save()

    @api_router.get("/v1/tasks", response_model=V1Tasks)
    async def get_tasks(
        current_user: Annotated[V1UserProfile, Depends(get_user_dependency())],
    ):
        tasks = Task.find()
        return V1Tasks(tasks=[task.to_v1() for task in tasks])

    @api_router.get("/v1/tasks/{id}", response_model=V1Task)
    async def get_task(
        current_user: Annotated[V1UserProfile, Depends(get_user_dependency())], id: str
    ):
        tasks = Task.find(id=id)
        if not tasks:
            raise Exception(f"Task {id} not found")
        return tasks[0].to_v1()

    @api_router.put("/v1/tasks/{id}", response_model=V1Task)
    async def put_task(
        current_user: Annotated[V1UserProfile, Depends(get_user_dependency())],
        id: str,
        data: V1TaskUpdate,
    ):
        tasks = Task.find(id=id)
        if not tasks:
            raise Exception(f"Task {id} not found")
        task = tasks[0]
        if data.status:
            task.status = TaskStatus(data.status)
            logging.info("updated task status to: ", task.status)
        task.save()
        return task.to_v1()

    return api_router


@retry(stop=stop_after_attempt(10), wait=wait_fixed(10))
def get_remote_task(id: str, owner_id: str, server: str, auth_token: str) -> Task:
    HUB_API_KEY = os.environ.get(AGENTESEA_HUB_API_KEY_ENV)
    if not HUB_API_KEY:
        raise Exception(f"${AGENTESEA_HUB_API_KEY_ENV} not set")

    logger.debug(f"connecting to remote task: {id} key: {HUB_API_KEY}")
    try:
        tasks = Task.find(
            id=id,
            remote=server,
            owner_id=owner_id,
            auth_token=auth_token,
        )
        if not tasks:
            raise Exception(f"Task {id} not found")
        logger.debug(f"got remote task: {tasks[0].__dict__}")
        return tasks[0]
    except Exception as e:
        logger.error(f"error getting remote task: {e}")
        raise e


def _label_task(remote: str, token: str, task: Task, key: str, value: str) -> None:
    """Label a task as trained

    Args:
        task (Task): The task
    """
    update = V1TaskUpdate(
        set_labels={key: value},
    )
    resp = requests.put(
        f"{remote}/v1/tasks/{task.id}",
        json=update.model_dump(),
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
