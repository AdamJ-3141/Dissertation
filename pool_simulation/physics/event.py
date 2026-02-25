from dataclasses import dataclass, field

@dataclass(order=True)
class Event:
    t: float
    kind: str = field(compare=False) 
    i: int = field(compare=False)
    j: int | tuple[str, int] | None = field(default=None, compare=False)
    version_i: int = field(default=-1, compare=False)
    version_j: int = field(default=-1, compare=False)