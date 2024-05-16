import os

import requests

from .server.models import V1UserProfile


class Hub:
    """The Agentsea Hub"""

    def __init__(self, hub_url: str = "https://hub.agentsea.ai") -> None:
        self.hub_url = os.getenv("AGENTSEA_HUB_URL", hub_url)

    def get_api_key(self, token: str) -> str:
        """Get the api key from the hub"""

        hub_key_url = f"{self.hub_url}/v1/users/me/keys"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(hub_key_url, headers=headers)
        response.raise_for_status()
        key_data = response.json()

        return key_data["key"]

    def get_user_info(self, token: str) -> V1UserProfile:
        """Get user info from the hub"""

        hub_user_url = f"{self.hub_url}/v1/users/me"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(hub_user_url, headers=headers)
        response.raise_for_status()
        user_data = response.json()

        return V1UserProfile(**user_data)
