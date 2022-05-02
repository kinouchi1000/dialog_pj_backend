import logging


class InferenceClient:
    def __init__(self) -> None:
        pass

    def get_reply(self, comment: str) -> str:
        # TODO get from inference server
        return f"You sent {comment} right?"
