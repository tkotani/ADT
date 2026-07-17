"""klrl_control.py -- pure (torch-free) helpers for KL-RLVR, extracted so they can be unit-tested
without a GPU:

  * CODE_VERSION / check_ckpt_compat(): the ckpt architecture-version stamp + compat check.
  * initial_target(): the §15 size-target init. The hidden default sigma_off0=2 made a HOLD start
    at sigma=goal+2 (=7 for goal 5); forward-KL then forced the policy to widen into the size tails
    -> less-stable molecules -> VR drop (diagnosed 2026-06-23). Default off0 is now 0.

Imported by rl_train_klrl.py. (The old external control_dir state machine -- newest_command_file /
apply_command -- was removed 2026-07-06: all run params are now fixed at launch via argv, never
mutated mid-run.)
"""


# --- ckpt architecture version: a date-time version number stamped on save, consistency-checked on
#     resume AND generation, so a ckpt is never silently loaded by code that builds a DIFFERENT arch. ---
#
# CODE_VERSION is an ARCHITECTURE identity stamped into every ckpt; check_ckpt_compat refuses to load a
# ckpt whose stamp differs. It is a v<author><date-time> tag fixed when the architecture was started
# (author so a stranger reading the ckpt knows WHO/what); bump (new date-time) ONLY on an arch change
# that makes old ckpts un-loadable.
#   vtakao202606231610 = the emb_offset (counter-less) architecture: parent pointer stored AND embedded
#                        as a relative offset (emb_offset), translation-invariant, unbounded atom count.
#                        Model + trainer live in ~/ADT/Drugs/vtakao202606231610/.
#   legacy (UNSTAMPED) = the old ABSOLUTE architecture (emb_pointer, capped atom count): E142,
#                        ~/ADT/common, geom. check_ckpt_compat treats unstamped ckpts as legacy.
# A legacy state_dict (emb_pointer.*) and an emb_offset one (emb_offset.*) have different keys, so even a
# bare load_state_dict(strict=True) rejects a cross-arch load -- this stamp just makes that failure EARLY
# and human-readable instead of a torch key-mismatch traceback.
CODE_VERSION = "vtakao202606231610"   # emb_offset arch. Bump (new date-time) on arch change.


def check_ckpt_compat(ckpt, code_version=CODE_VERSION, allow_mismatch=False):
    """Raise unless the ckpt's stamped architecture version matches this code's. Returns the ckpt's
    version (or None if unstamped). Call on RESUME (training) and before GENERATION.

      * unstamped (legacy, pre-versioning) ckpt -> returns None and is ALLOWED: load_state_dict is the
        backstop, so a true arch mismatch still fails there with a clear key error.
      * stamped but DIFFERENT version -> RuntimeError, unless allow_mismatch=True (caller's risk, e.g.
        a no-op code edit that did not touch the architecture).
    """
    ver = ckpt.get("arch_version")
    if ver is not None and ver != code_version and not allow_mismatch:
        raise RuntimeError(
            f"ckpt arch_version={ver!r} != this code's {code_version!r}. This ckpt was produced by a "
            f"different architecture version; load it with the matching code, or re-train. "
            f"Override with allow_mismatch=True only if you are certain the weights are compatible.")
    return ver


def initial_target(mu_star, sigma_star, mu_init=-1.0, sigma_init=-1.0, mu_off0=0.0, sigma_off0=0.0):
    """§15 initial size target (cur_mu, cur_sigma).

    mu_init / sigma_init >= 0 override explicitly; otherwise start at goal + off0.
    REGRESSION GUARD: with the default off0=0 a HOLD starts AT the goal sigma (not goal+2). A POSITIVE
    sigma offset makes the target WIDER than the policy's natural sigma, and forward-KL(P_tgt||P_hat)
    is mass-covering, so it pushes the policy to put mass in the size tails = widen = destabilize.
    """
    cur_mu = mu_init if mu_init >= 0 else mu_star + mu_off0
    cur_sigma = sigma_init if sigma_init >= 0 else sigma_star + sigma_off0
    return float(cur_mu), float(cur_sigma)


def check_gen_compat(ckpt, max_steps_per_mol=None, size_nmax=None, max_offset=None, allow_mismatch=False):
    """Verify generation-time settings passed as args against the ckpt's embedded "gen" block.

    A ckpt saved by rl_train_klrl.py (2026-07-06+) carries gen={max_steps_per_mol, size_nmax, max_offset}
    -- the bounds it was trained/generated with. For each arg that is NOT None, compare it to the stored
    value; on mismatch raise RuntimeError (or, with allow_mismatch=True, keep the passed value and let the
    caller warn). A ckpt with NO gen block (pretrain / pre-2026-07-06) returns {} and is allowed -- there
    is nothing to check against. Returns the resolved gen dict: the stored values, with any arg that was
    None filled from the ckpt (so a caller can `g = check_gen_compat(...); max_steps = g.get(...)`).
    """
    gen = ckpt.get("gen") or {}
    if not gen:
        return {}
    passed = {"max_steps_per_mol": max_steps_per_mol, "size_nmax": size_nmax, "max_offset": max_offset}
    mism = [f"{k}: arg={v} != ckpt={gen[k]}"
            for k, v in passed.items()
            if v is not None and gen.get(k) is not None and v != gen[k]]
    if mism and not allow_mismatch:
        raise RuntimeError(
            "generation args disagree with the ckpt's trained gen-params: " + "; ".join(mism)
            + ". Pass the matching values, or allow_mismatch=True to override.")
    resolved = dict(gen)
    for k, v in passed.items():   # an explicitly-passed value wins (allow_mismatch path / override)
        if v is not None:
            resolved[k] = v
    return resolved
