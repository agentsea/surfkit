from collections import defaultdict
import json
import time
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from mllm import Router
from shortuuid import uuid
from sqlalchemy import Integer, asc, func, case, cast, and_, distinct, over
from sqlalchemy.orm import joinedload
from taskara import ReviewRequirement, Task, TaskStatus
from threadmem import RoleThread
from skillpacks.db.models import ActionRecord, EpisodeRecord, ReviewRecord, action_reviews
from taskara.db.conn import get_db as get_task_DB
from taskara.db.models import TaskRecord, LabelRecord, task_label_association
from surfkit.db.conn import WithDB
from surfkit.db.models import SkillRecord
from surfkit.server.models import (
    SkillsWithGenTasks,
    UserTask,
    UserTasks,
    V1Skill,
    V1UpdateSkill,
)


class SkillStatus(Enum):
    """Skill status"""

    COMPLETED = "completed"
    TRAINING = "training"
    AGENT_TRAINING = "agent_training"  # state change for when generated tasks start assigning to agents
    AGENT_REVIEW = "agent_review"
    DEMO = "demo"
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
        example_tasks: Optional[list[str]] = None,
        min_demos: Optional[int] = None,
        demos_outstanding: Optional[int] = None,
        demo_queue_size: Optional[int] = None,
        remote: Optional[str] = None,
        kvs: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
        max_steps_agent: Optional[int] = None,
        review_requirements: Optional[list[ReviewRequirement]] = None,
    ):
        self.description = description or ""
        self.name = name
        self.generating_tasks = False
        if not name:
            self.name = self._get_name()
        self.status = status
        self.requirements = requirements or []
        self.tasks: List[Task] = []
        self.example_tasks = example_tasks or []
        self.owner_id = owner_id
        self.agent_type = agent_type
        self.max_steps = max_steps_agent if max_steps_agent is not None else 30
        self.review_requirements = review_requirements or []
        if not self.agent_type:
            self.agent_type = "foo"
        self.min_demos = min_demos if min_demos is not None else 100
        self.demos_outstanding = (
            demos_outstanding if demos_outstanding is not None else 3
        )
        self.demo_queue_size = demo_queue_size if demo_queue_size is not None else 5
        self.remote = remote
        self.threads: List[RoleThread] = []
        self.kvs = kvs or {}
        self.id = id or uuid()
        self.created = int(time.time())
        self.updated = int(time.time())
        self.token = token

    def _get_name(self) -> str:
        router = Router(
            [
                "mistral/mistral-medium-latest",
                "mistral/mistral-small-latest",
                "mistral/mistral-large-latest",
            ]
        )
        print("generating Name")
        thread = RoleThread()
        thread.post(
            role="user",
            msg=f"Please generate a name for this skill description that is no longer than 5 words, lowercase and hyphenated as a single word, if there is a specific tool involved like 'airbnb' make sure to include that e.g. 'search-for-stays-on-airbnb': '{self.description}'",
        )
        resp = router.chat(thread, model="mistral/mistral-small-latest")
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
            max_steps=self.max_steps,
            review_requirements=(
                [review.to_v1() for review in self.review_requirements]
                if self.review_requirements
                else []
            ),
            agent_type=self.agent_type,  # type: ignore
            tasks=[task.to_v1() for task in self.tasks],
            threads=[thread.to_v1() for thread in self.threads],
            example_tasks=self.example_tasks,
            status=self.status.value,
            generating_tasks=(
                self.generating_tasks if hasattr(self, "generating_tasks") else False
            ),
            min_demos=self.min_demos,
            demo_queue_size=self.demo_queue_size,
            demos_outstanding=self.demos_outstanding,
            owner_id=self.owner_id,
            created=self.created,
            updated=self.updated,
            remote=self.remote,
            kvs=self.kvs,
        )

    @classmethod
    def from_v1(
        cls,
        data: V1Skill,
        owner_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        remote: Optional[str] = None,
    ) -> "Skill":
        skill_status = (
            SkillStatus(data.status) if data.status else SkillStatus.NEEDS_DEFINITION
        )
        out = cls.__new__(cls)
        out.id = data.id
        out.name = data.name
        out.description = data.description
        out.requirements = data.requirements
        out.max_steps = data.max_steps
        out.review_requirements = (
            [ReviewRequirement.from_v1(r) for r in data.review_requirements]
            if data.review_requirements
            else []
        )
        out.agent_type = data.agent_type
        out.owner_id = owner_id
        owners = None
        if not out.owner_id:
            out.owner_id = data.owner_id
        if out.owner_id:
            owners = [out.owner_id]

        if not remote:
            remote = data.remote

        out.tasks = []
        for task in data.tasks:
            found = Task.find(
                id=task.id,
                remote=remote,
                auth_token=auth_token,
                owners=owners,
                owner_id=out.owner_id,
            )
            if found:
                out.tasks.append(found[0])
            else:
                print(
                    f"Task {task.id} not found when searching with owners {owners} and remote {remote} and auth_token {auth_token}",
                    flush=True,
                )

        out.example_tasks = data.example_tasks
        out.threads = []  # TODO: fix if needed
        out.status = skill_status
        out.min_demos = data.min_demos
        out.demos_outstanding = data.demos_outstanding
        out.demo_queue_size = data.demo_queue_size
        out.generating_tasks = data.generating_tasks
        out.created = data.created
        out.updated = data.updated
        out.remote = data.remote
        out.kvs = data.kvs
        return out

    def to_record(self) -> SkillRecord:
        return SkillRecord(
            id=self.id,
            owner_id=self.owner_id,
            name=self.name,
            description=self.description,
            requirements=json.dumps(self.requirements),
            max_steps=self.max_steps,
            review_requirements=json.dumps(self.review_requirements),
            agent_type=self.agent_type,
            threads=json.dumps([thread._id for thread in self.threads]),  # type: ignore
            tasks=json.dumps([task.id for task in self.tasks]),
            example_tasks=json.dumps(self.example_tasks),
            generating_tasks=self.generating_tasks,
            status=self.status.value,
            min_demos=self.min_demos,
            demos_outstanding=self.demos_outstanding,
            demo_queue_size=self.demo_queue_size,
            kvs=json.dumps(self.kvs),
            created=self.created,
            updated=int(time.time()),
        )

    @classmethod
    def from_record(cls, record: SkillRecord) -> "Skill":
        start_time = time.time()
        # We aren't using threads right now
        # thread_ids = json.loads(str(record.threads))
        threads = [] # [RoleThread.find(id=thread_id)[0] for thread_id in thread_ids]
        tasks = []
        task_ids = json.loads(str(record.tasks))

        if task_ids:
            tasks = Task.find_many_lite(task_ids)
            valid_task_ids = []

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
                    print(
                        f"Error updating tasks for skill {record.id}: {e}", flush=True
                    )
        print(
            f"tasks found for skill {record.id} time lapsed: {(time.time() - start_time):.4f}"
        )
        example_tasks = json.loads(str(record.example_tasks))

        requirements = json.loads(str(record.requirements))

        out = cls.__new__(cls)
        out.id = record.id
        out.name = record.name
        out.owner_id = record.owner_id
        out.description = record.description
        out.requirements = requirements
        out.max_steps = record.max_steps
        out.review_requirements = (
            json.loads(str(record.review_requirements))
            if record.review_requirements is not None
            else []
        )
        out.agent_type = record.agent_type
        out.threads = threads
        out.tasks = tasks
        out.example_tasks = example_tasks
        out.generating_tasks = record.generating_tasks
        out.status = SkillStatus(record.status)
        out.min_demos = record.min_demos
        out.demos_outstanding = record.demos_outstanding
        out.demo_queue_size = record.demo_queue_size
        out.kvs = json.loads(str(record.kvs)) if record.kvs else {}  # type: ignore
        out.created = record.created
        out.updated = record.updated
        out.remote = None
        print(
            f"record composed for skill {record.id} time lapsed: {(time.time() - start_time):.4f}"
        )
        return out


    @classmethod
    def from_record_with_tasks(cls, record: SkillRecord, tasks: List[Task]) -> "Skill":
        start_time = time.time()
        # We aren't using threads right now
        # thread_ids = json.loads(str(record.threads))
        threads = [] # [RoleThread.find(id=thread_id)[0] for thread_id in thread_ids]
        example_tasks = json.loads(str(record.example_tasks))

        requirements = json.loads(str(record.requirements))

        out = cls.__new__(cls)
        out.id = record.id
        out.name = record.name
        out.owner_id = record.owner_id
        out.description = record.description
        out.requirements = requirements
        out.max_steps = record.max_steps
        out.review_requirements = (
            json.loads(str(record.review_requirements))
            if record.review_requirements is not None
            else []
        )
        out.agent_type = record.agent_type
        out.threads = threads
        out.tasks = tasks
        out.example_tasks = example_tasks
        out.generating_tasks = record.generating_tasks
        out.status = SkillStatus(record.status)
        out.min_demos = record.min_demos
        out.demos_outstanding = record.demos_outstanding
        out.demo_queue_size = record.demo_queue_size
        out.kvs = json.loads(str(record.kvs)) if record.kvs else {}  # type: ignore
        out.created = record.created
        out.updated = record.updated
        out.remote = None
        print(
            f"record composed for skill {record.id} time lapsed: {(time.time() - start_time):.4f}"
        )
        return out

    def save(self):
        """
        Save the current state of the Skill either locally or via the remote API.

        For remote saves:
        - If the skill exists remotely, update it via a PUT request.
        - If the skill does not exist remotely, create it via a POST request.

        For local saves, perform the normal database merge/commit.
        """
        if self.remote:
            skill_url = f"{self.remote}/v1/skills/{self.id}"
            payload = self.to_v1().model_dump()  # Adjust serialization as needed
            try:
                # Check if the skill exists remotely.
                get_resp = requests.get(
                    skill_url, headers={"Authorization": f"Bearer {self.token}"}
                )
                if get_resp.status_code == 404:
                    # The skill does not exist remotely, so create it using POST.
                    create_url = f"{self.remote}/v1/skills"
                    post_resp = requests.post(create_url, json=payload)
                    post_resp.raise_for_status()
                    print(f"Skill {self.id} created remotely", flush=True)
                else:
                    # If found, update the remote record using PUT.
                    put_resp = requests.put(skill_url, json=payload)
                    put_resp.raise_for_status()
                    print(f"Skill {self.id} updated remotely", flush=True)
            except requests.RequestException as e:
                print(f"Error saving skill {self.id} on remote: {e}", flush=True)
            return
        else:
            for db in self.get_db():
                record = self.to_record()
                db.merge(record)
                db.commit()

    @classmethod
    def find(
        cls,
        remote: Optional[str] = None,
        owners: Optional[List[str]] = None,
        token: Optional[str] = None,
        **kwargs,  # type: ignore
    ) -> List["Skill"]:
        print("running find for skills", flush=True)
        start_time = time.time()
        if remote:
            # Prepare query parameters
            params = dict(kwargs)
            if owners:
                # Pass owners as multiple query parameters
                for owner in owners:
                    params.setdefault("owners", []).append(owner)

            print(f"Query params for remote request: {params}", flush=True)

            try:
                resp = requests.get(
                    f"{remote}/v1/skills",
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"Error fetching skills from remote: {e}", flush=True)
                return []

            skills_json = resp.json()
            skills = [
                cls.from_v1(
                    V1Skill.model_validate(skill_data),
                    auth_token=token,
                    remote=remote,
                )
                for skill_data in skills_json["skills"]
            ]

            # Set remote attribute for each skill
            for skill in skills:
                skill.remote = remote

            return skills

        else:
            out = []
            for db in cls.get_db():
                query = db.query(SkillRecord)

                # Apply owners filter if provided
                if owners:
                    query = query.filter(SkillRecord.owner_id.in_(owners))

                # Apply additional filters from kwargs
                for key, value in kwargs.items():
                    column_attr = getattr(SkillRecord, key)
                    if isinstance(value, (list, tuple)):  # Support multiple values
                        query = query.filter(column_attr.in_(value))
                    else:
                        query = query.filter(column_attr == value)

                records = query.order_by(asc(SkillRecord.created)).all()
                print(
                    f"skills found in db {records} time lapsed: {(time.time() - start_time):.4f}",
                    flush=True,
                )
                out.extend([cls.from_record(record) for record in records])
                print(
                    f"skills from_record ran time lapsed: {(time.time() - start_time):.4f}",
                    flush=True,
                )
            return out

    @classmethod
    def find_many(
        cls,
        owners: Optional[List[str]] = None,
        token: Optional[str] = None,
        **kwargs,  # type: ignore
    ) -> List["Skill"]:
        print("running find for skills", flush=True)
        start_time = time.time()
        out = []
        for db in cls.get_db():
            query = db.query(SkillRecord)

            # Apply owners filter if provided
            if owners:
                query = query.filter(SkillRecord.owner_id.in_(owners))

            # Apply additional filters from kwargs
            for key, value in kwargs.items():
                column_attr = getattr(SkillRecord, key)
                if isinstance(value, (list, tuple)):  # Support multiple values
                    query = query.filter(column_attr.in_(value))
                else:
                    query = query.filter(column_attr == value)

            records = query.order_by(asc(SkillRecord.created)).all()
            print(
                f"skills found in db {records} time lapsed: {(time.time() - start_time):.4f}",
                flush=True,
            )
            skill_ids = [str(record.id) for record in records]
            tasks = Task.find_many_lite(skill_ids=skill_ids)

            task_map: defaultdict[str, list[Task]] = defaultdict(list)
            for task in tasks:
                task_map[task.skill].append(task) # type: ignore
            out.extend([cls.from_record_with_tasks(record, task_map[str(record.id)]) for record in records])
            print(
                f"skills from_record ran time lapsed: {(time.time() - start_time):.4f}",
                flush=True,
            )
        return out

    def update(self, data: V1UpdateSkill):
        """
        Update the skill's properties based on the provided data.
        After updating the in-memory attributes, the method calls save() to persist
        changes locally or remotely.
        """
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
            self.example_tasks = data.example_tasks
        if data.status:
            self.status = SkillStatus(data.status)
        if data.max_steps:
            self.max_steps = data.max_steps
        if data.review_requirements:
            self.review_requirements = [
                ReviewRequirement.from_v1(r) for r in data.review_requirements
            ]
        if data.min_demos is not None:
            self.min_demos = data.min_demos
        if data.demos_outstanding is not None:
            self.demos_outstanding = data.demos_outstanding
        if data.demo_queue_size is not None:
            self.demo_queue_size = data.demo_queue_size
        if data.kvs is not None:
            self.kvs = data.kvs

        # Save the updated skill, either locally or remotely.
        self.save()

    def set_key(self, key: str, value: str):
        """
        Sets the given key to the specified value.
        If a remote is set, delegate this operation to the remote API.
        """
        if self.remote:
            url = f"{self.remote}/v1/skills/{self.id}/keys"
            payload = {"key": key, "value": value}
            try:
                resp = requests.post(
                    url, json=payload, headers={"Authorization": f"Bearer {self.token}"}
                )
                resp.raise_for_status()
                print(
                    f"Successfully set key '{key}' on remote for skill {self.id}",
                    flush=True,
                )
            except requests.RequestException as e:
                print(
                    f"Error setting key on remote for skill {self.id}: {e}", flush=True
                )
            return
        else:
            self.kvs[key] = value
            self.save()

    def get_key(self, key: str) -> Optional[str]:
        """
        Retrieves the value for the given key.
        If a remote is set, retrieve the value via the remote API.
        """
        if self.remote:
            url = f"{self.remote}/v1/skills/{self.id}/keys/{key}"
            try:
                resp = requests.get(
                    url, headers={"Authorization": f"Bearer {self.token}"}
                )
                resp.raise_for_status()
                data = resp.json()
                value = data.get("value")
                print(
                    f"Successfully retrieved key '{key}' from remote for skill {self.id}",
                    flush=True,
                )
                return value
            except requests.RequestException as e:
                print(
                    f"Error retrieving key on remote for skill {self.id}: {e}",
                    flush=True,
                )
                return None
        else:
            return self.kvs.get(key)

    def delete_key(self, key: str):
        """
        Deletes the given key.
        If a remote is set, perform the deletion via the remote API.
        """
        if self.remote:
            url = f"{self.remote}/v1/skills/{self.id}/keys/{key}"
            try:
                resp = requests.delete(url)
                resp.raise_for_status()
                print(
                    f"Successfully deleted key '{key}' on remote for skill {self.id}",
                    flush=True,
                )
            except requests.RequestException as e:
                print(
                    f"Error deleting key on remote for skill {self.id}: {e}", flush=True
                )
            return
        else:
            if key in self.kvs:
                del self.kvs[key]
                self.save()

    def refresh(self):
        """
        Refresh the object state from the database or remote API.
        """
        if self.remote:
            url = f"{self.remote}/v1/skills/{self.id}"
            try:
                resp = requests.get(
                    url, headers={"Authorization": f"Bearer {self.token}"}
                )
                resp.raise_for_status()
                data = resp.json()
                # Assume that the response data can be used to instantiate a V1Skill.
                # You may need to adjust this based on your actual API response format.
                v1skill = V1Skill(**data)
                new = Skill.from_v1(v1skill, owner_id=self.owner_id)
            except requests.RequestException as e:
                raise ValueError(f"Error refreshing skill from remote: {e}")
        else:
            found = self.find(id=self.id)
            if not found:
                raise ValueError("Skill not found")
            new = found[0]

        # Update the current object's fields with the new data.
        self.name = new.name
        self.description = new.description
        self.requirements = new.requirements
        self.max_steps = new.max_steps
        self.review_requirements = new.review_requirements
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
        self.kvs = new.kvs

        return self

    def set_generating_tasks(self, input: bool):
        if self.generating_tasks != input:
            self.generating_tasks = input
            self.save()

    def get_task_descriptions(self, limit: Optional[int] = None):
        maxLimit = len(self.tasks)
        limit = limit if limit and limit < maxLimit else maxLimit
        return {"tasks": [task.description for task in self.tasks[-limit:]]}

    def generate_tasks(
        self,
        n_permutations: int = 1,
        assigned_to: Optional[str] = None,
        assigned_type: Optional[str] = None,
    ) -> List[Task]:
        self.set_generating_tasks(True)
        task_assigned_to = assigned_to or self.owner_id
        if assigned_to is None and assigned_type != "user":
            task_assigned_to = None

        router = Router(
            [
                "mistral/mistral-medium-latest",
                "mistral/mistral-small-latest",
                "mistral/mistral-large-latest",
            ]
        )
        current_date = datetime.now().strftime("%B %d, %Y")
        example_str = str(
            "For example, if the skill is 'search for stays on airbnb' "
            "and a requirement is 'find stays within a travel window' then a task "
            "might be 'Find the most popular available stays on Airbnb between October 12th to October 14th' "
        )
        example_schema = '{"tasks": ["Find stays from october 2nd to 3rd", "Find stays from January 15th-17th"]}'
        if self.example_tasks:
            example_str = str(
                f"Some examples of tasks for this skill are: '{json.dumps(self.example_tasks)}'"
            )
            example_schema = str('{"tasks": ' f"{json.dumps(self.example_tasks)}" "}")
        if len(self.requirements) > 0:
            print(
                f"Generating tasks for skill: '{self.description}', skill ID: {self.id} with requirements: {self.requirements}",
                flush=True,
            )

            thread = RoleThread(
                owner_id=self.owner_id
            )  # TODO is this gonna keep one thread? I don't see a need for a new thread every time
            result: List[Task] = []

            for n in range(n_permutations):
                print(
                    f"task generation interation: {n} for skill ID {self.id}",
                    flush=True,
                )

                old_task_str = ""
                old_tasks = self.get_task_descriptions(limit=15000)
                if old_tasks:
                    old_task_str = str(
                        "Please do not create any tasks identical to these tasks that have already been created: "
                        f"{old_tasks}"
                    )

                prompt = (
                    f"Given the agent skill '{self.description}', and the "
                    f"configurable requirements that the agent skill encompasses '{json.dumps(self.requirements)}', "
                    "Please generate a task that a user could take which will excercise this skill, "
                    "our goal is to train and get good at using a skill "
                    f"Today's date is {current_date}. "
                    f"{example_str} "
                    f"Please return a raw json object that looks like the following example: "
                    f"{example_schema} "
                    f"{old_task_str}"
                    "Please ensure the task parameters are varied. If there are dates or numbers please vary them a little bit."
                )
                print(f"prompt: {prompt}", flush=True)
                thread.post("user", prompt)
                response = router.chat(
                    thread, model="mistral/mistral-small-latest", expect=UserTasks
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
                gen_tasks = gen_tasks[
                    :1
                ]  # take only one as we are doing this one at a time

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
                        max_steps=self.max_steps,
                        assigned_to=task_assigned_to,
                        assigned_type=assigned_type if assigned_type else "user",
                        labels={"skill": self.id},
                        skill=self.id,
                        created_by="agenttutor",
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
                self.save()  # need to save for every iteration as we want tasks to incrementally show up
            self.generating_tasks = False
            # self.kvs["agenttutor_msg"] = {"msg": "test information message, baby shark do do do do", "timestamp": time.time()}
            # self.kvs["alert"] = {"msg": "test alert message, the lion sleeps in the jungle tonight", "timestamp": time.time()}
            self.save()

            return result

        else:
            print(f"Generating tasks for skill: {self.description}", flush=True)
            old_task_str = ""
            old_tasks = self.get_task_descriptions(limit=15000)
            if old_tasks:
                old_task_str = str(
                    "Please do not create any tasks identical to these tasks that have already been created: "
                    f"{old_tasks}"
                )
            prompt = (
                f"Given the agent skill '{self.description}' "
                "Please generate a task that a agent could do which will excercise this skill, "
                "our goal is to test whether the agent can perform the skill "
                f"Today's date is {current_date}. "
                f"{example_str} "
                f"Please return a raw json object that looks like the following example: "
                f"{example_schema} "
                f"{old_task_str} "
                "Please ensure the task parameters are varied. If there are dates or numbers please vary them a little bit."
            )
        thread = RoleThread(owner_id=self.owner_id)
        thread.post("user", prompt)

        response = router.chat(
            thread, model="mistral/mistral-small-latest", expect=UserTask
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
            max_steps=self.max_steps,
            assigned_to=task_assigned_to,
            assigned_type=assigned_type if assigned_type else "user",
            labels={"skill": self.id},
            skill=self.id,
            created_by="agenttutor",
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

    @classmethod
    def find_skills_for_task_gen(cls) -> list[SkillsWithGenTasks]:
        skill_records = []
        for skill_session in cls.get_db():

            # Query all skills needing tasks
            skill_records = (
                skill_session.query(SkillRecord.id, SkillRecord.demo_queue_size)
                .filter(
                    SkillRecord.status.in_(
                        [SkillStatus.TRAINING.value, SkillStatus.DEMO.value]
                    ),
                    SkillRecord.generating_tasks == False,  # noqa: E712
                )
                .all()
            )
            skill_session.close()

        # Return early if no matching skills
        if not skill_records:
            return []

        # Create a dict of skill_id -> (demos_outstanding, min_demos)
        skill_map = {
            row[0]: {  # row[0] is skill_id
                "demo_queue_size": row[1],  # row[1] is demos_outstanding
            }
            for row in skill_records
        }
        skill_ids = list(skill_map.keys())
        direct_rows = []
        labeled_rows = []
        for task_session in get_task_DB():
            # Query A: Direct skill references
            direct_rows = (
                task_session.query(
                    TaskRecord.skill.label("skill_id"),
                    func.count().label("count"),
                )
                .filter(
                    TaskRecord.status == (TaskStatus.IN_QUEUE.value),
                    TaskRecord.skill.in_(skill_ids),
                )
                .group_by(TaskRecord.skill)
                .all()
            )

            # Query B: Labeled tasks only, excluding tasks that already have a direct TaskRecord.skill
            labeled_rows = (
                task_session.query(
                    LabelRecord.value.label("skill_id"),
                    func.count().label("count"),
                )
                .join(
                    TaskRecord.labels.and_(LabelRecord.key == "skill")
                    .and_(TaskRecord.skill.is_(None))
                    .and_(TaskRecord.status == (TaskStatus.IN_QUEUE.value))
                )
                .filter(
                    LabelRecord.value.in_(skill_ids),
                )
                .group_by(LabelRecord.value)
                .all()
            )

            task_session.close()

        # Combine all counts into a single dict. If a skill never appears, it remains 0.
        in_queue_counts = defaultdict(int)

        # direct_rows + labeled_rows => e.g. [("skillA", 2), ("skillC", 1), ...]
        for sid, count_value in direct_rows + labeled_rows:
            in_queue_counts[sid] += count_value

        # Now iterate over all skill IDs to catch zero counts
        results = []
        for sid in skill_ids:
            min_demos = skill_map[sid]["demo_queue_size"]
            count_value = in_queue_counts[sid]  # defaults to 0 if sid never occurred
            if count_value < min_demos:
                results.append(
                    SkillsWithGenTasks(
                        skill_id=sid,
                        in_queue_count=count_value,
                        tasks_needed=min_demos - count_value,
                    )
                )

        return results
    
    @classmethod
    def stop_failing_agent_tasks(cls, timestamp: float | None = None) -> list[str]:
        if timestamp is None:
            timestamp = time.time() - 86400  # Compute dynamically at runtime

        direct_rows = []
        for task_session in get_task_DB():
            # Find skills with failing tasks
            direct_rows = (
                task_session.query(
                    TaskRecord.skill.label("skill_id"),
                    func.count().label("task_count"),
                ).join(
                    TaskRecord.labels
                    .and_(TaskRecord.completed > timestamp)
                    .and_(LabelRecord.key == "can_review")
                    .and_(LabelRecord.value == "false")
                    .and_(
                        TaskRecord.status.in_([
                            TaskStatus.ERROR.value, 
                            TaskStatus.FAILED.value, 
                            TaskStatus.TIMED_OUT.value
                        ])
                    )
                )
                .group_by(TaskRecord.skill)
                .all()
            )

            task_session.close()

        skills_with_failure_conditions = []
        print(f'stop_failing_agent_tasks got skills list {direct_rows}', flush=True)
        for skill_id, task_count in direct_rows:
            found = cls.find(id=skill_id)
            if not found:
                print(f'stop_failing_agent_tasks: ERROR skill: {skill_id} not found #slack-alerts', flush=True)
                continue
            skill = found[0]

            # Only process skills in training
            if skill.status != SkillStatus.TRAINING:
                continue

            # Skip if tasks are currently being generated, we don't want to overwrite anything
            if skill.generating_tasks: 
                continue

            # Get failure limit from skill's key-value store (defaults to 3)
            allowed_consecutive_fails = 3
            if 'fail_limit' in skill.kvs:
                print(f'fail limit is {skill.kvs["fail_limit"]}', flush=True)
                try:
                    allowed_consecutive_fails = int(skill.kvs['fail_limit'])  # Ensure conversion
                except ValueError:
                    print(f"Invalid fail_limit for skill {skill.id}, using default 3", flush=True)

            # Only proceed if enough tasks exist
            if len(skill.tasks) < allowed_consecutive_fails:
                continue

            # Sort tasks by completion time (descending)
            sorted_tasks = sorted(
                skill.tasks,
                key=lambda t: t.completed or 0,  # Handle None safely
                reverse=True
            )

            last_tasks = sorted_tasks[:allowed_consecutive_fails]
            # Check if all last `allowed_consecutive_fails` tasks match the failure conditions
            all_failed = all(
                (task.completed or 0) > 1 and  # Completed after timestamp
                task.assigned_type != "user" and       # Assigned to agent
                task.labels.get("can_review") == "false" and  # Labeled as unreviewable
                task.status in [TaskStatus.FAILED, TaskStatus.ERROR, TaskStatus.TIMED_OUT]
                for task in last_tasks
            )

            if all_failed:
                print(f"Skill {skill.id} has {allowed_consecutive_fails} consecutive failing tasks.", flush=True)
                print(f"Failing tasks: {[task.id for task in last_tasks]}", flush=True)
                
                # Update skill state
                skill.status = SkillStatus.DEMO
                skill.min_demos += 1
                skill.kvs['last_agent_stop_from_failure'] = time.time()
                skill.save()
                skills_with_failure_conditions.append(skill_id)

        return skills_with_failure_conditions
    
    @classmethod
    def find_skills_for_agent_task_gen(cls) -> list[SkillsWithGenTasks]:
        skill_records = []
        for skill_session in cls.get_db():

            # Query all skills needing tasks
            skill_records = (
                skill_session.query(SkillRecord.id, SkillRecord.kvs)
                .filter(
                    SkillRecord.status.in_(
                        [SkillStatus.TRAINING.value]
                    ),
                    SkillRecord.generating_tasks == False,  # noqa: E712
                )
                .all()
            )
            skill_session.close()

        if not skill_records:
            return []

        # Create a dict of skill_id -> (kvs)
        skill_map = {
            row[0]: {  # row[0] is skill_id
                "kvs": json.loads(str(row[1])) if row[1] else {},  # row[1] is kvs
            }
            for row in skill_records
        }
        skill_ids = list(skill_map.keys())
        incomplete_agent_Tasks = []
        for task_session in get_task_DB():
            # tasks that need a review or are in progress
            incomplete_agent_Tasks = (
                task_session.query(
                    TaskRecord.skill.label("skill_id"),
                    func.count().label("count"),
                )
                .filter(
                    TaskRecord.assigned_type != 'user',
                    TaskRecord.reviews == '[]',  # Only include tasks with an empty reviews field
                    TaskRecord.status.in_([
                        TaskStatus.IN_QUEUE.value,
                        TaskStatus.TIMED_OUT.value,
                        TaskStatus.WAITING.value,
                        TaskStatus.IN_PROGRESS.value,
                        TaskStatus.FAILED.value,
                        TaskStatus.ERROR.value,
                        TaskStatus.REVIEW.value,
                    ]),
                )
                .outerjoin(
                    task_label_association,
                    TaskRecord.id == task_label_association.c.task_id
                )
                .outerjoin(
                    LabelRecord,
                    and_(
                        task_label_association.c.label_id == LabelRecord.id,
                        LabelRecord.key == 'can_review',
                        LabelRecord.value == 'false'
                    )
                )
                .filter(LabelRecord.id.is_(None))  # Exclude tasks with the "can_review" label set to 'false'
                .group_by(TaskRecord.skill)
                .all()
            )
            task_session.close()

        # Combine all counts into a single dict. If a skill never appears, it remains 0.
        in_queue_counts = defaultdict(int)

        # direct_rows + labeled_rows => e.g. [("skillA", 2), ("skillC", 1), ...]
        for sid, count_value in incomplete_agent_Tasks:
            in_queue_counts[sid] += count_value

        # Now iterate over all skill IDs to catch zero counts
        results = []
        for sid in skill_ids:
            agent_task_queue_size = 5
            if 'agent_task_queue_size' in skill_map[sid]['kvs']:
                print(f'fail limit is {skill_map[sid]["kvs"]["agent_task_queue_size"]}', flush=True)
                try:
                    agent_task_queue_size = int(skill_map[sid]['kvs']['agent_task_queue_size'])  # Ensure conversion
                except ValueError:
                    print(f"Invalid agent_task_queue_size for skill {skill_map[sid]}, using default 3", flush=True)
                
            count_value = in_queue_counts[sid]  # defaults to 0 if sid never occurred
            if count_value < agent_task_queue_size:
                results.append(
                    SkillsWithGenTasks(
                        skill_id=sid,
                        in_queue_count=count_value,
                        tasks_needed=agent_task_queue_size - count_value,
                    )
                )

        return results

    def calc_metrics(self, start_time: float = 0.0, end_time: float|None = None):
        end = end_time if end_time else time.time()
        new_skill_metrics = []
        new_skill_metrics = []
        for metrics_session in get_task_DB():
            # Optimized Task reviews subquery
            subquery_task_reviews = (
                metrics_session.query(
                    TaskRecord.id.label("task_id"),
                    TaskRecord.skill.label("skill_id"),
                    (func.max(TaskRecord.completed) - func.max(TaskRecord.started)).label("task_completion_time"),
                    func.max(case((ReviewRecord.approved, 1), else_=0)).label("task_review_approved"),
                )
                .join(
                    ReviewRecord,
                    and_(
                        ReviewRecord.resource_type == "task",
                        TaskRecord.id == ReviewRecord.resource_id,
                        TaskRecord.skill == self.id,
                        TaskRecord.assigned_type != "user",
                        TaskRecord.reviews != "[]",
                        TaskRecord.completed.between(start_time, end)
                    ),
                )
                .group_by(TaskRecord.id, TaskRecord.skill)
                .subquery()
            )

            # Subquery: For each action, compute "is_approved" = 1 if any review is True, else 0
            action_approvals_subq = (
                metrics_session.query(
                    action_reviews.c.action_id.label("action_id"),
                    func.max(case((ReviewRecord.approved, 1), else_=0)).label("is_approved"),
                )
                .select_from(TaskRecord)
                .join(
                    EpisodeRecord,
                    and_(
                        TaskRecord.episode_id == EpisodeRecord.id,
                        TaskRecord.skill == self.id,  # earliest filtering by skill
                        TaskRecord.assigned_type != "user",
                        TaskRecord.reviews != "[]",
                        TaskRecord.completed.between(start_time, end)
                    ),
                )
                .join(
                    ActionRecord,
                    ActionRecord.episode_id == EpisodeRecord.id
                )
                .join(
                    action_reviews,
                    action_reviews.c.action_id == ActionRecord.id,
                )
                .join(
                    ReviewRecord,
                    ReviewRecord.id == action_reviews.c.review_id,
                )
                .group_by(action_reviews.c.action_id)
                .subquery()
            )

            # Define a window specification for calculating the time difference
            # 'partition_by' = group by each task, 'order_by' = sequence by the action's created time
            # The time-difference expression
            time_diff_expr = (
                ActionRecord.created - func.lag(ActionRecord.created).over(partition_by=TaskRecord.id, order_by=ActionRecord.created)
            )
            # calc action time diff using lag function
            subquery_action_approval_flags = (
                metrics_session.query(
                    TaskRecord.id.label("task_id"),
                    ActionRecord.id.label("action_id"),
                    ActionRecord.created.label("action_created"),
                    time_diff_expr.label("time_diff"),
                    action_approvals_subq.c.is_approved.label("is_approved"),
                )
                .select_from(action_approvals_subq)
                .join(
                    ActionRecord, action_approvals_subq.c.action_id == ActionRecord.id
                )
                .join(
                    EpisodeRecord, ActionRecord.episode_id == EpisodeRecord.id
                )
                .join(
                    TaskRecord, TaskRecord.episode_id == EpisodeRecord.id
                )
                .subquery()
            )
            # Aggregate to task level
            subquery_action_reviews = (
                metrics_session.query(
                    subquery_action_approval_flags.c.task_id,
                    func.count(subquery_action_approval_flags.c.action_id).label("action_count"),
                    func.sum(subquery_action_approval_flags.c.is_approved).label("approved_count"),
                    func.avg(subquery_action_approval_flags.c.time_diff).label("avg_actions_time"),
                )
                .group_by(subquery_action_approval_flags.c.task_id)
                .subquery()
            )

             # Final aggregation joining both subqueries by task_id
            query_result = metrics_session.query(
                subquery_task_reviews.c.skill_id,
                subquery_task_reviews.c.task_id,
                func.count(subquery_task_reviews.c.task_id).label("task_count"),
                func.sum(subquery_task_reviews.c.task_review_approved).label("task_reviews_approved_count"),
                func.avg(subquery_task_reviews.c.task_completion_time).label("avg_task_completion_time"),
                func.sum(subquery_action_reviews.c.action_count).label("action_count"),
                func.sum(subquery_action_reviews.c.approved_count).label("approved_count"),
                func.avg(subquery_action_reviews.c.avg_actions_time).label("avg_action_completion_time")
            ).join(
                subquery_action_reviews,
                subquery_action_reviews.c.task_id == subquery_task_reviews.c.task_id,
            ).group_by(
                subquery_task_reviews.c.skill_id,
                subquery_task_reviews.c.task_id
            ).one_or_none()

            new_skill_metrics.append(query_result)

        return new_skill_metrics