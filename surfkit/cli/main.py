from urllib.parse import urljoin
import os

import typer
import webbrowser

from surfkit.config import GlobalConfig, AGENTSEA_HUB_URL, HUB_URL

art = """
 _______                ___  __  __  __  __   
|     __|.--.--..----..'  _||  |/  ||__||  |_ 
|__     ||  |  ||   _||   _||     < |  ||   _|
|_______||_____||__|  |__|  |__|\__||__||____|
"""

app = typer.Typer()

# Sub-command groups
create = typer.Typer(help="Create an agent or device")
list_group = typer.Typer(help="List resources")
get = typer.Typer(help="Get resources")
view = typer.Typer(help="View resources")

app.add_typer(create, name="create")
app.add_typer(list_group, name="list")
app.add_typer(get, name="get")
app.add_typer(view, name="view")


# Callback for showing help
def show_help(ctx: typer.Context, command_group: str):
    if ctx.invoked_subcommand is None:
        if command_group == "root":
            typer.echo(art)
        typer.echo(ctx.get_help())
        raise typer.Exit()


# Root command callback
@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    show_help(ctx, "root")


# 'create' command group callback
@create.callback(invoke_without_command=True)
def create_default(ctx: typer.Context):
    show_help(ctx, "create")


# 'list' command group callback
@list_group.callback(invoke_without_command=True)
def list_default(ctx: typer.Context):
    show_help(ctx, "list")


# 'get' command group callback
@get.callback(invoke_without_command=True)
def get_default(ctx: typer.Context):
    show_help(ctx, "get")


# 'create' sub-commands
@create.command("devices")
def create_app(name: str, type: str = "desktop"):
    typer.echo(f"Creating device: {name}")


@create.command("agents")
def create_agent(name: str, type: str = "SurfDino"):
    typer.echo(f"Creating agent: {name}")


# 'list' sub-commands


@list_group.command("agents")
def list_agents():
    typer.echo("Listing agents")


@list_group.command("devices")
def list_devices():
    typer.echo("Listing devices")


@list_group.command("types")
def list_types():
    typer.echo("Listing desktops")


# 'get' sub-commands
@get.command("agent")
def get_agent(name: str):
    typer.echo(f"Getting agent: {name}")


@get.command("device")
def get_device(name: str):
    typer.echo(f"Getting device: {name}")


@get.command("type")
def get_type(name: str):
    typer.echo(f"Getting type: {name}")


# Other commands
@app.command(help="Login to the hub")
def login():
    url = urljoin(HUB_URL, "cli-login")
    typer.echo(f"\nVisit {url} to get an API key\n")

    webbrowser.open(url)
    api_key = typer.prompt("Enter your API key", hide_input=True)

    config = GlobalConfig.read()
    config.api_key = api_key
    config.write()

    typer.echo("\nLogin successful!")


@app.command(help="Deploy an agent")
def deploy():
    typer.echo("Deploying...")


@app.command(help="Publish an agent")
def publish():
    typer.echo("Publishing...")


@app.command(help="Initialize an agent repo")
def init():
    typer.echo("Initializing...")


@app.command(help="Use an agent to solve a task")
def solve(description: str):
    typer.echo(f"Solving task {description}...")


if __name__ == "__main__":
    app()
