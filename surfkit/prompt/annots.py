from orign.models import (
    ChatRequest,
    ContentItem,
    ImageUrlContent,
    MessageItem,
    Prompt,
)


def create_description_text(action: str, with_image: bool = False) -> str:
    """
    Create a text prompt describing what's happened between two images.
    """
    prompt_text = (
        f"The first image is the before image, and the second image is the after\n"
        f"image of a GUI interaction. The action that occurred is: {action}. "
        "Can you give a task description for what was accomplished?\n"
        "The goal would be for an agent to look at the first image and the task "
        "description which would result in the second image, for example "
        '"click on login button" would be a good description, or '
        '"move mouse to be over user icon", or "type text \'good fellas\'"\n'
    )
    if with_image:
        prompt_text += " <image><image>"
    return prompt_text


def create_reason_text(
    action: str, task_description: str, with_image: bool = False
) -> str:
    """
    Create a text prompt describing the reasoning chain needed to connect an action
    and a desired task outcome.
    """
    prompt_text = (
        f"The first image is the before image, and the second image is the after\n"
        f"image of a GUI interaction. The action that occurred is: {action}. "
        "Can you give a reasoning chain for what the user would need to think\n"
        "through in order to take the correct action with respect to the task? "
        f"The current task is: {task_description}\n"
    )
    if with_image:
        prompt_text += " <image><image>"
    return prompt_text


def create_validation_text(
    action: str, task_description: str, with_image: bool = False
) -> str:
    """
    Create a text prompt asking the LLM to validate whether the action completed
    successfully for a given task.
    """
    prompt_text = (
        f"The first image is the before image, and the second image is the after\n"
        f"image of a GUI interaction. The action that occurred is: {action}. "
        "Considering the task we want to accomplish,\n"
        "please give me the reason why this action completed successfully or not. "
        f"The current task is: {task_description}\n"
    )
    if with_image:
        prompt_text += " <image><image>"
    return prompt_text


def create_swift_description_prompt(
    image1: str,
    image2: str,
    action: str,
    answer: str,
) -> dict:
    """
    Create a text prompt and attach images in a separate field.
    This might be used by a Swift client or a different consumer that
    expects separate 'messages' and 'images' structure.
    """
    prompt_text = create_description_text(action, with_image=True)
    return {
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "images": [image1, image2],
    }


def create_swift_reason_prompt(
    image1: str,
    image2: str,
    action: str,
    task_description: str,
    answer: str,
) -> dict:
    prompt_text = create_reason_text(action, task_description, with_image=True)
    return {
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "images": [image1, image2],
    }


def create_swift_validation_prompt(
    image1: str,
    image2: str,
    action: str,
    task_description: str,
    answer: str,
) -> dict:
    prompt_text = create_validation_text(action, task_description, with_image=True)
    return {
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "images": [image1, image2],
    }


def create_orign_description_prompt(
    image1: str,
    image2: str,
    action: str,
) -> ChatRequest:
    """
    Create a single batch task for the OpenAI API, returning a ChatRequest
    describing the difference between two images based on an action.
    """
    prompt_text = create_description_text(action, with_image=False)
    return ChatRequest(
        prompt=Prompt(
            messages=[
                MessageItem(
                    role="user",
                    content=[
                        ContentItem(type="text", text=prompt_text),
                        ContentItem(
                            type="image_url", image_url=ImageUrlContent(url=image1)
                        ),
                        ContentItem(
                            type="image_url", image_url=ImageUrlContent(url=image2)
                        ),
                    ],
                )
            ]
        ),
        max_tokens=500,
    )


def create_orign_reason_prompt(
    image1: str,
    image2: str,
    action: str,
    task_description: str,
) -> ChatRequest:
    """
    Create a single batch task for the OpenAI API, returning a ChatRequest
    that requests a reasoning chain tying the action and the task.
    """
    prompt_text = create_reason_text(action, task_description, with_image=False)
    return ChatRequest(
        prompt=Prompt(
            messages=[
                MessageItem(
                    role="user",
                    content=[
                        ContentItem(type="text", text=prompt_text),
                        ContentItem(
                            type="image_url", image_url=ImageUrlContent(url=image1)
                        ),
                        ContentItem(
                            type="image_url", image_url=ImageUrlContent(url=image2)
                        ),
                    ],
                )
            ]
        ),
        max_tokens=500,
    )


def create_orign_validation_prompt(
    image1: str,
    image2: str,
    action: str,
    task_description: str,
) -> ChatRequest:
    """
    Create a single batch task for the OpenAI API, returning a ChatRequest
    that asks for validation on whether the action matched the desired task outcome.
    """
    prompt_text = create_validation_text(action, task_description, with_image=False)
    return ChatRequest(
        prompt=Prompt(
            messages=[
                MessageItem(
                    role="user",
                    content=[
                        ContentItem(type="text", text=prompt_text),
                        ContentItem(
                            type="image_url", image_url=ImageUrlContent(url=image1)
                        ),
                        ContentItem(
                            type="image_url", image_url=ImageUrlContent(url=image2)
                        ),
                    ],
                )
            ]
        ),
        max_tokens=500,
    )
