class SelectedUnits(list):
    name: str
    """
    This is the return type of Behaviour.select_units, which is a list in every way
    except that when represented as a string, it can return a name, if a `name`
    attribute has been set on it. This allows methods that have had `units` passed to be
    cached to file.
    """
    def __repr__(self):
        if hasattr(self, "name"):
            return self.name
        return list.__repr__(self)
