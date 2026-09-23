from enum import StrEnum

from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role = Field(..., description="Role identifying who sent this message.")
    content: str = Field(..., description="Content of the message ")

    @classmethod
    def _msg(cls, role: Role, content: str) -> Message:
        return Message(role=role, content=content)

    @classmethod
    def user_msg(cls, content: str) -> Message:
        return cls._msg(role=Role.USER, content=content)

    @classmethod
    def system_msg(cls, content: str) -> Message:
        return cls._msg(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_msg(cls, content: str) -> Message:
        return cls._msg(role=Role.ASSISTANT, content=content)
