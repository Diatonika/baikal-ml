from abc import ABC, abstractmethod


class WindowStrategy(ABC):
    @abstractmethod
    def length(self) -> int: ...

    @abstractmethod
    def window(self, index: int) -> int: ...
