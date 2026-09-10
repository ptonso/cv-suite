from dataclasses import dataclass


@dataclass
class Logger:
    verbose: int

    def info(self, msg: str) -> None:
        if self.verbose >= 1:
            print(msg)

    def debug(self, msg: str) -> None:
        if self.verbose >= 2:
            print(msg)

    def error(self, msg: str) -> None:
        print(msg)
