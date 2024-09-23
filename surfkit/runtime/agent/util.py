import random
import string

from docker.api.client import APIClient
from docker.errors import APIError
from namesgenerator import get_random_name
from tqdm import tqdm

from surfkit.types import AgentType


def instance_name(type: AgentType) -> str:
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    random_name = get_random_name("-")
    if not random_name:
        raise ValueError("Could not generate a random name")
    name_parts = random_name.split("-")
    if len(name_parts) != 2:
        raise ValueError("Could not generate a random name with 2 parts")
    name_only = name_parts[1]

    return f"{type.name.lower()}-{name_only.lower()}-{random_string.lower()}"


def pull_image(img: str, api_client: APIClient):
    """
    Pulls a Docker image with progress bars for each layer.

    Args:
        img (str): The Docker image to pull.
        api_client (APIClient): The Docker API client.
    """

    print(f"Pulling Docker image '{img}'...")

    progress_bars = {}
    layers = {}

    try:
        for line in api_client.pull(img, stream=True, decode=True):
            if "id" in line and "progressDetail" in line:
                layer_id = line["id"]
                progress_detail = line["progressDetail"]
                current = progress_detail.get("current", 0)
                total = progress_detail.get("total", 0)

                if total:
                    if layer_id not in layers:
                        progress_bars[layer_id] = tqdm(
                            total=total,
                            desc=f"Layer {layer_id}",
                            leave=False,
                            ncols=100,
                        )
                        layers[layer_id] = 0

                    layers[layer_id] = current
                    progress_bars[layer_id].n = current
                    progress_bars[layer_id].refresh()
            elif "status" in line and "id" in line:
                print(f"Status update for {line['id']}: {line['status']}")
            elif "error" in line:
                raise APIError(line["error"])

    except APIError as e:
        print(f"Error pulling Docker image: {e.explanation}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        # Close all progress bars
        for bar in progress_bars.values():
            bar.n = bar.total  # Ensure the progress bar is full before closing
            bar.refresh()
            bar.close()

        print("")
