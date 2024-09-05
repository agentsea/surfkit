from abc import ABC, abstractmethod


class Teacher(ABC):

    @abstractmethod
    def teach(self, *args, **kwargs):
        pass


class LLMTeach(Teacher):
    pass
