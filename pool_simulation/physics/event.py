class Event:
    def __init__(self, t: float, kind: str, i: int, j: int | None, version_i: int, version_j: int | None = None):
        self.t = t
        self.kind = kind
        self.i = i
        self.j = j
        self.version_i = version_i
        self.version_j = version_j

    def __lt__(self, other):
        return self.t < other.t

    def __str__(self):
        return f"{self.kind} {self.i} {self.j} at t={self.t}"