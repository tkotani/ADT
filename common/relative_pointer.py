"""
v25: Relative pointer conversion utilities.

Converts absolute pointer values (from_id, to_id) to relative offsets
in the token array, and back.

Convention:
  ADD step placing atom k (1-indexed):
    offset = k - from_id   (offset=1 → previous atom, offset=k-1 → first atom)

  LINK after atom k:
    offset_from = k - lid_from + 1  (offset=1 → just-placed atom k)
    offset_to   = k - lid_to + 1

Reverse (generation):
  ADD: from_id = current_atom_count + 1 - offset
  LINK: lid = current_atom_count - offset + 1
"""

import numpy as np
from adt_tokenizer import ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK, END

N_SLOTS = 7


def absolute_to_relative(arr):
    """Convert absolute pointer values to relative offsets in-place.
    arr: 1D int array of tokens [action, from, atom/to, r, hp0, hp1, hp2] x N_steps.
    Returns modified arr (also modifies in-place).
    """
    arr = arr.copy()
    n_steps = len(arr) // N_SLOTS
    atom_count = 0  # atoms placed so far

    for s in range(n_steps):
        base = s * N_SLOTS
        action = int(arr[base])

        if action == END:
            break

        if action in (ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD):
            atom_count += 1  # this step places a new atom
            from_id = int(arr[base + 1])
            if from_id > 0:  # not NULL_POINTER
                offset = atom_count - from_id
                assert offset >= 1, f"step {s}: atom_count={atom_count} from_id={from_id} offset={offset}"
                arr[base + 1] = offset
            # slot 2 is atom_type for ADD, not a pointer

        elif action == LINK:
            lid_from = int(arr[base + 1])
            lid_to = int(arr[base + 2])
            if lid_from > 0:
                offset_from = atom_count - lid_from + 1
                assert offset_from >= 1, f"step {s}: atom_count={atom_count} lid_from={lid_from}"
                arr[base + 1] = offset_from
            if lid_to > 0:
                offset_to = atom_count - lid_to + 1
                assert offset_to >= 1, f"step {s}: atom_count={atom_count} lid_to={lid_to}"
                arr[base + 2] = offset_to

    return arr


def relative_to_absolute_from(offset, current_atom_count, is_link=False):
    """Convert relative offset back to absolute from_id during generation.
    current_atom_count: atoms placed so far (before this step for ADD, after for LINK).
    """
    if is_link:
        return current_atom_count - offset + 1
    else:
        return current_atom_count + 1 - offset


def relative_to_absolute_to(offset, current_atom_count):
    """Convert relative offset back to absolute to_id for LINK."""
    return current_atom_count - offset + 1


# ============================================================
# Test
# ============================================================

def test_roundtrip():
    """Test absolute → relative → verify consistency."""
    # Simulate a simple token sequence
    # Atom 1 (ADD_INIT): from=0 (null)
    # Atom 2 (ADD_CHAIN): from=1 (atom 1)
    # Atom 3 (ADD_ANGLE): from=2 (atom 2)
    # Atom 4 (ADD): from=1 (atom 1)
    # LINK: from=3, to=1
    # Atom 5 (ADD): from=4 (atom 4)

    tokens = [
        ADD_INIT, 0, 6, 0, 0, 0, 0,      # atom 1, from=null
        ADD_CHAIN, 1, 6, 50, 0, 0, 0,     # atom 2, from=1
        ADD_ANGLE, 2, 6, 50, 7, 8, 0,     # atom 3, from=2
        ADD, 1, 6, 50, 4, 8, 3,           # atom 4, from=1
        LINK, 3, 1, 50, 4, 8, 3,          # link from=3, to=1
        ADD, 4, 7, 50, 4, 8, 3,           # atom 5, from=4
    ]
    arr = np.array(tokens, dtype=np.int64)
    rel = absolute_to_relative(arr)

    print("Absolute → Relative:")
    for s in range(6):
        base = s * N_SLOTS
        action = int(arr[base])
        act_name = {ADD_INIT: "INIT", ADD_CHAIN: "CHAIN", ADD_ANGLE: "ANGLE",
                    ADD: "ADD", LINK: "LINK"}.get(action, "?")
        orig_from = int(tokens[base + 1])
        new_from = int(rel[base + 1])
        if action == LINK:
            orig_to = int(tokens[base + 2])
            new_to = int(rel[base + 2])
            print(f"  step {s} {act_name:5s}: from {orig_from}→{new_from}, to {orig_to}→{new_to}")
        else:
            print(f"  step {s} {act_name:5s}: from {orig_from}→{new_from}")

    # Verify reverse
    atom_count = 0
    for s in range(6):
        base = s * N_SLOTS
        action = int(arr[base])
        if action in (ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD):
            atom_count += 1
            if int(rel[base + 1]) > 0:
                recovered = relative_to_absolute_from(int(rel[base + 1]), atom_count - 1)
                assert recovered == int(tokens[base + 1]), \
                    f"step {s}: expected {tokens[base+1]}, got {recovered}"
        elif action == LINK:
            recovered_from = relative_to_absolute_from(int(rel[base + 1]), atom_count, is_link=True)
            recovered_to = relative_to_absolute_to(int(rel[base + 2]), atom_count)
            assert recovered_from == int(tokens[base + 1]), \
                f"step {s} LINK from: expected {tokens[base+1]}, got {recovered_from}"
            assert recovered_to == int(tokens[base + 2]), \
                f"step {s} LINK to: expected {tokens[base+2]}, got {recovered_to}"

    print("\nRoundtrip: ALL PASSED")


if __name__ == "__main__":
    test_roundtrip()
