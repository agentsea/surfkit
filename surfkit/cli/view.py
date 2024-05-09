# from __future__ import annotations
import urllib.parse
import time
import os
from typing import Optional
import atexit
import random
import webbrowser

import docker
from docker.models.containers import Container
from agentdesk.vm.base import DesktopVM
from agentdesk.util import (
    get_docker_host,
    check_command_availability,
    find_open_port,
)
from agentdesk.key import SSHKeyPair
from agentdesk.proxy import ensure_ssh_proxy, cleanup_proxy

from surfkit.runtime.agent.base import AgentInstance


UI_IMG = "us-central1-docker.pkg.dev/agentsea-dev/guisuirfer/surfkit-ui:634820941cbbba4b3cd51149b25d0a4c8d1a35f4"


def view(desk: DesktopVM, agent: AgentInstance, background: bool = False) -> None:
    """Opens the desktop in a browser window"""

    desk_port = 6080
    if desk.requires_proxy:
        keys = SSHKeyPair.find(name=desk.key_pair_name)
        if not keys:
            raise ValueError(
                f"No key pair found with name {desk.key_pair_name} and is required for this desktop"
            )
        key_pair = keys[0]

        desk_port = find_open_port(6080, 7080)
        if not desk_port:
            raise ValueError("Could not find an open port for the desktop proxy")
        proxy_pid = ensure_ssh_proxy(
            desk_port,
            6080,
            desk.ssh_port,
            "agentsea",
            desk.addr,
            key_pair.decrypt_private_key(key_pair.private_key),
        )
        atexit.register(cleanup_proxy, proxy_pid)

    agent_port = agent.port
    agent_proxy_pid = None
    if agent.runtime.requires_proxy():
        agent_port = find_open_port(9090, 10090)
        print(f"proxying agent to port {agent_port}...")
        if not agent_port:
            raise ValueError("Could not find an open port for the agent proxy")

        agent_proxy_pid = agent.proxy(agent_port)

    check_command_availability("docker")

    host = get_docker_host()
    os.environ["DOCKER_HOST"] = host
    client = docker.from_env()

    host_port = None
    ui_container: Optional[Container] = None

    for container in client.containers.list():
        if container.image.tags[0] == UI_IMG:  # type: ignore
            print("found running UI container")
            # Retrieve the host port for the existing container
            host_port = container.attrs["NetworkSettings"]["Ports"]["3000/tcp"][0][  # type: ignore
                "HostPort"
            ]
            ui_container = container  # type: ignore
            break

    if not ui_container:
        print("creating UI container...")
        host_port = random.randint(1024, 65535)
        ui_container = client.containers.run(  # type: ignore
            UI_IMG,
            ports={"3000/tcp": host_port},
            detach=True,
        )
        print("waiting for UI container to start...")
        time.sleep(10)

    encoded_agent_addr = urllib.parse.quote(f"http://localhost:{agent_port}")
    encoded_vnc_addr = urllib.parse.quote(f"ws://localhost:{desk_port}")

    # Construct the URL with the encoded parameters
    url = f"http://localhost:{host_port}/?agentAddr={encoded_agent_addr}&vncAddr={encoded_vnc_addr}"
    webbrowser.open(url)

    if background:
        return

    def onexit():
        nonlocal proxy_pid, agent_proxy_pid
        print("Cleaning up resources...")

        # Check if the UI container still exists and stop/remove it if so
        if ui_container:
            try:
                container_status = client.containers.get(ui_container.id).status  # type: ignore
                if container_status in ["running", "paused"]:
                    print("stopping UI container...")
                    ui_container.stop()
                    print("removing UI container...")
                    ui_container.remove()
            except docker.errors.NotFound:  # type: ignore
                print("UI container already stopped/removed.")

        # Stop the SSH proxy if required and not already stopped
        if desk.requires_proxy and proxy_pid:
            try:
                print("stopping ssh proxy...")
                cleanup_proxy(proxy_pid)
            except Exception as e:
                print(f"Error stopping SSH proxy: {e}")
            finally:
                proxy_pid = None  # Ensure we don't try to stop it again

        # Stop the agent proxy if required and not already stopped
        if agent.runtime.requires_proxy() and agent_proxy_pid:
            try:
                print("stopping agent proxy...")
                cleanup_proxy(agent_proxy_pid)
            except Exception as e:
                print(f"Error stopping agent proxy: {e}")
            finally:
                agent_proxy_pid = None

    atexit.register(onexit)
    try:
        while True:
            print(f"proxying desktop vnc '{desk.name}' to localhost:6080...")
            time.sleep(20)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, exiting...")
        onexit()
