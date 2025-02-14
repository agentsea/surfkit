import requests
from agentcore.models import V1UserProfile

from surfkit.config import AGENTSEA_AUTH_URL


def get_user_info(token: str) -> V1UserProfile:
    response = requests.get(
        f"{AGENTSEA_AUTH_URL}/v1/users/me", headers={"Authorization": f"Bearer {token}"}
    )
    return V1UserProfile.model_validate(response.json())
