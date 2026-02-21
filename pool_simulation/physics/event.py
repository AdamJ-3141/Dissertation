class Event:
    def __init__(self,t: float , kind: str, i: int, j: int | None):
        self.t = t
        self.kind = kind
        self.i = i
        self.j = j

    def __lt__(self, other):
        return self.t < other.t
