"""Private immutable storage for completed native contact telemetry lists.

These values carry no validity or model authority. Native capture and completed
state guards still validate every solve; public readers materialize fresh lists.
"""


class _FrozenList(tuple):
    __slots__ = ()

    def __new__(cls, values):
        values = tuple(values)
        if any(type(value) not in (str, int, _FrozenList) for value in values):
            raise ValueError('private telemetry requires plain immutable leaves')
        return super().__new__(cls, values)

    def __init_subclass__(cls, **kwargs):
        raise TypeError('private telemetry cannot be subclassed')

    def __deepcopy__(self, memo):
        return self

    def as_list(self):
        return [value.as_list() if type(value) is _FrozenList else value for value in self]


def _freeze_contacts(sample):
    """Freeze only the two lists from validated native capture, once per solve."""
    sample['contact_pairs'] = _FrozenList(_FrozenList(pair) for pair in sample['contact_pairs'])
    sample['contact_constraint_addresses'] = _FrozenList(sample['contact_constraint_addresses'])


def _public_contacts(sample):
    """Thaw private fields in an already detached reply; preserve legacy types."""
    for field in ('contact_pairs', 'contact_constraint_addresses'):
        value = sample.get(field)
        if type(value) is _FrozenList:
            sample[field] = value.as_list()
    return sample
