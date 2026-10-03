"""Bounded conservative row packing of measured XY bounds; no robot/SDK I/O.

This establishes geometric capacity only. The caller must validate every
actual carry/release/home path, and later observe the physical placement.
"""
from itertools import permutations
import math

import numpy as np


def pack_rows(bounds, objects, *, margin, separation):
    """Return interior layouts for up to six opaque, measured body identities.

    A layout separates every pair along one axis. This is conservative: a
    refusal does not prove general two-dimensional packing is impossible.
    Fixed objects retain their exact observed bounds. Margin applies only to
    future placements; fixed bodies need exterior containment and separation.
    Spare space is distributed between free footprints and their boundaries.
    """
    bounds = np.asarray(bounds, float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or (bounds[1] <= bounds[0]).any() or not math.isfinite(margin) or margin <= 0
            or not math.isfinite(separation) or separation <= 0):
        raise ValueError('invalid placement capacity bounds or margins')
    if not objects or len(objects) > 6:
        raise ValueError('row capacity requires one to six measured bodies')
    inventory = []
    for obj in objects:
        lo, hi = np.asarray(obj['lower'], float), np.asarray(obj['upper'], float)
        if (type(obj['id']) is not int or lo.shape != (2,) or hi.shape != (2,)
                or not np.isfinite([lo, hi]).all() or (hi <= lo).any()
                or type(obj['fixed']) is not bool):
            raise ValueError('invalid measured placement footprint')
        if obj['fixed'] and ((lo < bounds[0]).any() or (hi > bounds[1]).any()):
            return []
        inventory.append({'id': obj['id'], 'lower': lo, 'upper': hi,
                          'size': hi-lo, 'fixed': obj['fixed']})
    if len({o['id'] for o in inventory}) != len(inventory):
        raise ValueError('duplicate placement body identity')
    inventory.sort(key=lambda o: (-float(np.prod(o['size'])), o['id']))
    layouts, seen = [], set()
    for axis in (0, 1):
        cross = 1-axis
        inner_lo, inner_hi = bounds[0]+margin, bounds[1]-margin
        for order in permutations(inventory):
            proposed, free, cursor = {}, [], float(inner_lo[axis])

            def fill(end):
                nonlocal cursor
                if not free:
                    return True
                # A fixed body may be closer to the exterior than the
                # planning margin. It cannot enlarge a free body's interval.
                cursor = max(cursor, float(inner_lo[axis]))
                end = min(end, float(inner_hi[axis]))
                total = sum(o['size'][axis] for o in free)+separation*(len(free)-1)
                spare = end-cursor-total
                if spare < 0:
                    return False
                extra = spare/(len(free)+1)
                cursor += extra
                for obj in free:
                    if obj['size'][cross] > inner_hi[cross]-inner_lo[cross]:
                        return False
                    lo = np.empty(2)
                    lo[axis] = cursor
                    lo[cross] = (bounds[0, cross]+bounds[1, cross]-obj['size'][cross])/2
                    hi = lo+obj['size']
                    proposed[obj['id']] = [lo.tolist(), hi.tolist()]
                    cursor = float(hi[axis]+separation+extra)
                free.clear()
                return True

            valid = True
            for obj in order:
                if not obj['fixed']:
                    free.append(obj)
                    continue
                if not fill(float(obj['lower'][axis]-separation)):
                    valid = False
                    break
                proposed[obj['id']] = [obj['lower'].tolist(), obj['upper'].tolist()]
                cursor = float(obj['upper'][axis]+separation)
            if not valid or not fill(float(inner_hi[axis])):
                continue
            # Adjacent fixed objects can legitimately use less than the
            # planned border margin, but may never overlap one another.
            for left, right in zip(order, order[1:]):
                if proposed[right['id']][0][axis]-proposed[left['id']][1][axis] < separation:
                    valid = False
            if not valid:
                continue
            key = tuple((n, *v[0], *v[1]) for n, v in sorted(proposed.items()))
            if key in seen:
                continue
            seen.add(key)
            free_margins = [min(*(np.asarray(proposed[o['id']][0])-bounds[0]),
                                *(bounds[1]-np.asarray(proposed[o['id']][1])))
                            for o in inventory if not o['fixed']]
            layouts.append({'axis': axis, 'footprints': proposed,
                            'minimum_planned_border_m': min(free_margins) if free_margins else None})
    layouts.sort(key=lambda p: (-(p['minimum_planned_border_m'] or 0.), p['axis'],
                               tuple((k, *v[0]) for k, v in sorted(p['footprints'].items()))))
    return layouts
