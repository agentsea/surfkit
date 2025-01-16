import json
import time
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import List, Optional

import requests
from mllm import Router
from shortuuid import uuid
from sqlalchemy import asc
from taskara import Task, TaskStatus
from threadmem import RoleThread

from surfkit.db.conn import WithDB
from surfkit.db.models import SkillRecord
from surfkit.server.models import UserTask, UserTasks, V1Skill, V1UpdateSkill


class SkillStatus(Enum):
    """Skill status"""

    COMPLETED = "completed"
    TRAINING = "training"
    NEEDS_DEFINITION = "needs_definition"
    CREATED = "created"
    FINISHED = "finished"
    CANCELED = "canceled"
    REVIEW = "review"


class Skill(WithDB):
    """An agent skill"""

    def __init__(
        self,
        description: Optional[str] = None,
        requirements: Optional[list[str]] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
        status: SkillStatus = SkillStatus.NEEDS_DEFINITION,
        agent_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        example_tasks: Optional[List[Task]]= None,
        min_demos: Optional[int] = None,
        demos_outstanding: Optional[int] = None,
        remote: Optional[str] = None,
    ):
        self.description = description or ""
        self.name = name
        self.generating_tasks = False
        if not name:
            self.name = self._get_name()
        self.status = status
        self.requirements = requirements or []
        self.tasks: List[Task] = []
        self.example_tasks: List[Task] = example_tasks or []
        self.owner_id = owner_id
        self.agent_type = agent_type
        if not self.agent_type:
            self.agent_type = "foo"
        self.min_demos = min_demos if min_demos is not None else 100
        self.demos_outstanding = (
            demos_outstanding if demos_outstanding is not None else 5
        )
        self.remote = remote
        self.threads: List[RoleThread] = []

        self.id = id or uuid()
        self.created = int(time.time())
        self.updated = int(time.time())

    def _get_name(self) -> str:
        router = Router(
            [
                "gemini/gemini-2.0-flash-exp",
                "anthropic/claude-3-5-sonnet-20240620",
                "gpt-4o",
            ]
        )
        print("generating Name")
        thread = RoleThread()
        thread.post(
            role="user",
            msg=f"Please generate a name for this skill description that is no longer than 5 words: '{self.description}'",
        )
        resp = router.chat(thread, model="gemini/gemini-2.0-flash-exp")
        print(
            "Get Name Chat response", asdict(resp), flush=True
        )  # TODO test pydantic dump
        return resp.msg.text

    def to_v1(self) -> V1Skill:
        if not hasattr(self, "remote"):
            self.remote = None
        return V1Skill(
            id=self.id,
            name=self.name,  # type: ignore
            description=self.description,
            requirements=self.requirements,
            agent_type=self.agent_type,  # type: ignore
            tasks=[task.to_v1() for task in self.tasks],
            threads=[thread.to_v1() for thread in self.threads],
            example_tasks=[task.to_v1() for task in self.example_tasks],
            status=self.status.value,
            generating_tasks=self.generating_tasks,
            min_demos=self.min_demos,
            demos_outstanding=self.demos_outstanding,
            owner_id=self.owner_id,
            created=self.created,
            updated=self.updated,
            remote=self.remote,
        )

    @classmethod
    def from_v1(cls, data: V1Skill, owner_id: Optional[str] = None) -> "Skill":
        skill_status = (
            SkillStatus(data.status) if data.status else SkillStatus.NEEDS_DEFINITION
        )
        out = cls.__new__(cls)
        out.id = data.id
        out.name = data.name
        out.description = data.description
        out.requirements = data.requirements
        out.agent_type = data.agent_type
        out.owner_id = owner_id
        out.tasks = [Task.find(id=task.id)[0] for task in data.tasks]
        out.example_tasks = [Task.find(id=task.id)[0] for task in data.example_tasks]
        out.threads = [RoleThread.find(id=thread.id)[0] for thread in data.threads]
        out.status = skill_status
        out.min_demos = data.min_demos
        out.demos_outstanding = data.demos_outstanding
        out.created = data.created
        out.updated = data.updated
        out.remote = data.remote
        return out

    def to_record(self) -> SkillRecord:
        return SkillRecord(
            id=self.id,
            owner_id=self.owner_id,
            name=self.name,
            description=self.description,
            requirements=json.dumps(self.requirements),
            agent_type=self.agent_type,
            threads=json.dumps([thread._id for thread in self.threads]),  # type: ignore
            tasks=json.dumps([task.id for task in self.tasks]),
            example_tasks=json.dumps([task.id for task in self.example_tasks]),
            generating_tasks=self.generating_tasks,
            status=self.status.value,
            min_demos=self.min_demos,
            demos_outstanding=self.demos_outstanding,
            created=self.created,
            updated=self.updated,
        )

    @classmethod
    def from_record(cls, record: SkillRecord) -> "Skill":
        thread_ids = json.loads(str(record.threads))
        threads = [RoleThread.find(id=thread_id)[0] for thread_id in thread_ids]
        example_task_ids = json.loads(str(record.example_tasks))
        task_ids = json.loads(str(record.tasks))
        example_tasks = Task.find_many_lite(example_task_ids)
        tasks = Task.find_many_lite(task_ids)
        valid_task_ids = []
        valid_example_task_ids = []

        if len(tasks) < len(task_ids):
            try:
                print(f"updating tasks for skill {record.id}", flush=True)
                task_map = {task.id: task for task in tasks}
                for task_id in task_ids:
                    if not task_map[task_id]:
                        print(f"Task {task_id} not found, removing from skill")
                        continue

                    valid_task_ids.append(task_id)

                record.tasks = json.dumps(valid_task_ids)  # type: ignore
                for db in cls.get_db():
                    db.merge(record)
                    db.commit()
                print(f"updated tasks for skill {record.id}", flush=True)
            except Exception as e:
                print(f"Error updating tasks for skill {record.id}: {e}", flush=True)

        if len(example_tasks) < len(example_task_ids):
            try:
                print(f"updating example_tasks for skill {record.id}", flush=True)
                example_task_map = {task.id: task for task in example_tasks}
                for example_task_id in example_task_ids:
                    if not example_task_map[example_task_id]:
                        print(f"Example Task {example_task_id} not found, removing from skill")
                        continue

                    valid_example_task_ids.append(example_task_id)

                record.example_tasks = json.dumps(valid_example_task_ids)  # type: ignore
                for db in cls.get_db():
                    db.merge(record)
                    db.commit()
                print(f"updated example_tasks for skill {record.id}", flush=True)
            except Exception as e:
                print(f"Error updating example_tasks for skill {record.id}: {e}", flush=True)

        requirements = json.loads(str(record.requirements))

        out = cls.__new__(cls)
        out.id = record.id
        out.name = record.name
        out.owner_id = record.owner_id
        out.description = record.description
        out.requirements = requirements
        out.agent_type = record.agent_type
        out.threads = threads
        out.tasks = tasks
        out.example_tasks = example_tasks
        out.generating_tasks = record.generating_tasks
        out.status = SkillStatus(record.status)
        out.min_demos = record.min_demos
        out.demos_outstanding = record.demos_outstanding
        out.created = record.created
        out.updated = record.updated
        out.remote = None
        return out

    def save(self):
        for db in self.get_db():
            record = self.to_record()
            db.merge(record)
            db.commit()

    @classmethod
    def find(cls, remote: Optional[str] = None, **kwargs) -> List["Skill"]:  # type: ignore
        print("running find for skills", flush=True)

        if remote:
            resp = requests.get(f"{remote}/v1/skills")
            skills = [cls.from_v1(skill) for skill in resp.json()]
            for key, value in kwargs.items():
                skills = [
                    skill for skill in skills if getattr(skill, key, None) == value
                ]

            for skill in skills:
                skill.remote = remote

            return skills

        for db in cls.get_db():
            records = (
                db.query(SkillRecord)
                .filter_by(**kwargs)
                .order_by(asc(SkillRecord.created))
                .all()
            )
            print(f"skills found in db {records}", flush=True)
            return [cls.from_record(record) for record in records]

        raise ValueError("no session")

    def update(self, data: V1UpdateSkill):
        print(f"updating skill {self.id} data: {data.model_dump_json()}", flush=True)
        if data.name:
            self.name = data.name
        if data.description:
            self.description = data.description
        if data.requirements:
            self.requirements = data.requirements
        if data.threads:
            self.threads = [
                RoleThread.find(id=thread_id)[0] for thread_id in data.threads
            ]
        if data.tasks:
            self.tasks = [Task.find(id=task_id)[0] for task_id in data.tasks]
        if data.example_tasks:
            self.example_tasks = [Task.find(id=task_id)[0] for task_id in data.example_tasks]
        if data.status:
            self.status = SkillStatus(data.status)
        if data.min_demos:
            self.min_demos = data.min_demos
        if data.demos_outstanding:
            self.demos_outstanding = data.demos_outstanding

        self.save()

    def refresh(self):
        """
        Refresh the object state from the database.
        """
        found = self.find(id=self.id)
        if not found:
            raise ValueError("Skill not found")

        new = found[0]
        self.name = new.name
        self.description = new.description
        self.requirements = new.requirements
        self.threads = new.threads
        self.tasks = new.tasks
        self.example_tasks = new.example_tasks
        self.created = new.created
        self.updated = new.updated
        self.owner_id = new.owner_id
        self.agent_type = new.agent_type
        self.generating_tasks = new.generating_tasks
        self.status = new.status
        self.min_demos = new.min_demos
        self.demos_outstanding = new.demos_outstanding
        return self

    def set_generating_tasks(self, input: bool):
        if self.generating_tasks != input:
            self.generating_tasks = input
            self.save()

    def get_task_descriptions(self, limit: Optional[int] = None):
        maxLimit = len(self.tasks)
        limit = limit if limit and limit < maxLimit else maxLimit
        return { "tasks": [task.description for task in self.tasks[-limit:]]}

    def generate_tasks(
        self,
        n_permutations: int = 1,
        assigned_to: Optional[str] = None,
        assigned_type: Optional[str] = None,
    ) -> List[Task]:
        self.set_generating_tasks(True)
        router = Router(
            [
                "gemini/gemini-2.0-flash-exp",
                "anthropic/claude-3-5-sonnet-20240620",
                "gpt-4o",
            ]
        )
        current_date = datetime.now().strftime("%B %d, %Y")
        example_task_descriptions = [task.description for task in self.example_tasks]
        example_str = str(
            "For example, if the skill is 'search for stays on airbnb' "
            "and a requirement is 'find stays within a travel window' then a task "
            "might be 'Find the most popular available stays on Airbnb between October 12th to October 14th' "
        )
        example_schema = '{"tasks": ["Find stays from october 2nd to 3rd", "Find stays from January 15th-17th"]}'
        if self.example_tasks:
            example_str = str(
                f"Some examples of tasks for this skill are: '{json.dumps(example_task_descriptions)}'"
            )
            example_schema = str('{"tasks": ' f'{json.dumps(example_task_descriptions)}' '}' )
        old_task_str = ""
        old_tasks = self.get_task_descriptions(limit=15000)
        if old_tasks:
            old_task_str = str(
                "Please do not create any tasks identical to these tasks that have already been created: "
                f"{old_tasks}"
            )
        if len(self.requirements) > 0:
            print(
                f"Generating tasks for skill: '{self.description}', skill ID: {self.id} with requirements: {self.requirements}",
                flush=True,
            )
            thread = RoleThread(owner_id=self.owner_id) # TODO is this gonna keep one thread? I don't see a need for a new thread every time
            result: List[Task] = []
            
            for n in range(n_permutations):
                print(
                    f"task generation interation: {n} for skill ID {self.id}",
                    flush=True
                    )
               
                prompt = (
                    f"Given the agent skill '{self.description}', and the "
                    f"configurable requirements that the agent skill encompasses '{json.dumps(self.requirements)}', "
                    "Please generate a task that a user could take which will excercise this skill, "
                    "our goal is to train and get good at using a skill "
                    f"Today's date is {current_date}. "
                    f"{example_str} "
                    f"Please return a raw json object that looks like the following example: "
                    f'{example_schema} '
                    f"{old_task_str}"
                )
                print(f"prompt: {prompt}", flush=True)
                thread.post("user", prompt)
                response = router.chat(
                    thread, model="gemini/gemini-2.0-flash-exp", expect=UserTasks
                )
                print(f"thread {thread}, response: {response}", flush=True)
                if not response.parsed:
                    self.set_generating_tasks(False)
                    raise ValueError(f"unable to parse response: {response}")

                print(
                    f"Generated tasks: {response.parsed.model_dump_json()} for skill ID {self.id}",
                    flush=True,
                )

                gen_tasks = response.parsed.tasks
                if not gen_tasks:
                    self.set_generating_tasks(False)
                    raise ValueError(f"no tasks generated for skill ID {self.id}")
                gen_tasks = gen_tasks[:1] # take only one as we are doing this one at a time

                if not self.owner_id:
                    self.set_generating_tasks(False)
                    raise ValueError(
                        f"Owner ID must be set on skill ID {self.id} to generate tasks"
                    )

                for task in gen_tasks:
                    tsk = Task(
                        task,
                        owner_id=self.owner_id,
                        review_requirements=[  # TODO commenting this out for now since we are only doing user tasks
                            # ReviewRequirement(
                            #     number_required=1, users=[self.owner_id]
                            # )  # TODO: make this configurable
                        ],
                        assigned_to=assigned_to if assigned_to else self.owner_id,
                        assigned_type=assigned_type if assigned_type else "user",
                        labels={"skill": self.id},
                    )
                    tsk.status = TaskStatus.IN_QUEUE
                    self.tasks.append(tsk)
                    tsk.save()
                    print(
                        f"task saved for skill ID: {self.id}",
                        tsk.to_v1().model_dump_json(),
                        flush=True,
                    )
                    result.append(tsk)
                self.save() # need to save for every iteration as we want tasks to incrementally show up
            self.generating_tasks = False
            self.save()

            return result

        else:
            print(f"Generating tasks for skill: {self.description}", flush=True)
            prompt = (
                f"Given the agent skill '{self.description}' "
                "Please generate a task that a agent could do which will excercise this skill, "
                "our goal is to test whether the agent can perform the skill "
                f"Today's date is {current_date}. "
                f"{example_str} "
                f"Please return a raw json object that looks like the following example: "
                f'{example_schema} '
                f"{old_task_str} "
            )
        thread = RoleThread(owner_id=self.owner_id)
        thread.post("user", prompt)

        response = router.chat(
            thread, model="gemini/gemini-2.0-flash-exp", expect=UserTask
        )

        if not response.parsed:
            raise ValueError(f"unable to parse response: {response}")

        if not self.owner_id:
            raise ValueError("Owner ID must be set on story to generate tasks")

        task = Task(
            response.parsed.task,
            owner_id=self.owner_id,
            review_requirements=[  # TODO commenting this out for now since we are only doing user tasks
                # ReviewRequirement(
                #     number_required=1, users=[self.owner_id]
                # )  # TODO: make this configurable
            ],
            assigned_to=assigned_to if assigned_to else self.owner_id,
            assigned_type=assigned_type if assigned_type else "user",
            labels={"skill": self.id},
        )
        task.status = TaskStatus.IN_QUEUE
        self.tasks.append(task)
        task.save()
        print("task saved", task.to_v1().model_dump_json(), flush=True)
        self.generating_tasks = False
        self.save()
        print(f"Generated task: {task.id}", flush=True)
        return [task]

    def delete(self, owner_id: str):
        for db in self.get_db():
            record = (
                db.query(SkillRecord).filter_by(id=self.id, owner_id=owner_id).first()
            )
            db.delete(record)
            db.commit()
