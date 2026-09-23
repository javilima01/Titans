from src.llm.models import Qwen35Wrapper
from src.llm.schemas.messages import Message

if __name__ == "__main__":
    model = Qwen35Wrapper()

    msg = Message.user_msg(content=input("Type your message: "))
    answer = model.generate(msgs=msg)
    print(answer)
