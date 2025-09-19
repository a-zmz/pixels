from typing import Iterable

class SelectedUnits(dict[str, list[int]]):
    name: str
    """
    A mapping from stream_id to lists of unit IDs.
    Behaves like a dict in every way, except that when represented as a string,
    it can return a `name` if set. This allows named instances to be cached to file.
    """

    def __init__(self, *args, name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if name is not None:
            self.name = name

    def __repr__(self) -> str:
        if hasattr(self, "name"):
            return self.name
        return dict.__repr__(self)

    # Convenience helpers
    def add(self, stream_id: str, unit_id: int) -> None:
        self.setdefault(stream_id, []).append(unit_id)

    def extend(self, stream_id: str, unit_ids: Iterable[int]) -> None:
        self.setdefault(stream_id, []).extend(unit_ids)

    def flat(self) -> list[int]:
        # If you sometimes need a flat list of all unit IDs
        out: list[int] = []
        for ids in self.values():
            out.extend(ids)
        return out
