import logging
import os
from typing import Annotated, Type
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from mllm import Router
from taskara import Task, TaskStatus
from taskara.server.models import V1Task, V1Tasks, V1TaskUpdate
from tenacity import retry, stop_after_attempt, wait_fixed

from surfkit.agent import TaskAgent
from surfkit.auth.transport import get_user_dependency
from surfkit.env import AGENTESEA_HUB_API_KEY_ENV
from surfkit.server.models import V1SolveTask, V1UserProfile

DEBUG_ENV_VAR = os.getenv("DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if DEBUG_ENV_VAR else logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


def task_router(Agent: Type[TaskAgent], mllm_router: Router) -> APIRouter:
    """API router for a task agent.

    Args:
        Agent (Type[TaskAgent]): Task agent type.
        mllm_router (Router): An MLLM router.

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

    @api_router.post("/v1/tasks")
    async def solve_task(
        current_user: Annotated[V1UserProfile, Depends(get_user_dependency())],
        background_tasks: BackgroundTasks,
        task_model: V1SolveTask,
    ):
        logger.info(f"solving task: {task_model.model_dump()}")
        try:
            # TODO: we need to find a way to do this earlier but get status back
            mllm_router.check_model()
        except Exception as e:
            logger.error(
                f"Cannot connect to LLM providers: {e} -- did you provide a valid key?"
            )
            raise HTTPException(
                status_code=500,
                detail=f"failed to conect to LLM providers: {e} -- did you provide a valid key?",
            )

        background_tasks.add_task(_solve_task, task_model)
        logger.info("created background task...")
        return

    def _solve_task(task_model: V1SolveTask):
        owner_id = task_model.task.owner_id
        if not owner_id:
            owner_id = "local"
        task = Task.from_v1(task_model.task, owner_id=owner_id)

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

                    config = Device.connect_config_type()(
                        **task_model.task.device.config  # type: ignore
                    )
                    device = Device.connect(config=config)

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
def get_remote_task(id: str, owner_id: str, server: str) -> Task:
    HUB_API_KEY = os.environ.get(AGENTESEA_HUB_API_KEY_ENV)
    if not HUB_API_KEY:
        raise Exception(f"${AGENTESEA_HUB_API_KEY_ENV} not set")

    logger.debug(f"connecting to remote task: {id} key: {HUB_API_KEY}")
    try:
        tasks = Task.find(
            id=id,
            remote=server,
            owner_id=owner_id,
        )
        if not tasks:
            raise Exception(f"Task {id} not found")
        logger.debug(f"got remote task: {tasks[0].__dict__}")
        return tasks[0]
    except Exception as e:
        logger.error(f"error getting remote task: {e}")
        raise e
