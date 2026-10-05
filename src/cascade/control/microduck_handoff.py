"""Opt-in standing handoff between a motion policy and the standing policy.

While the commanded body twist is nonzero the motion (velocity) policy is
evaluated; whenever it is exactly zero (no lease, a completed distance, a latched
stop, a zero-twist balance request) the standing policy is evaluated instead.
The selector presents the stepper with the single-policy interface and chooses
the active network per attempt from the command the stepper is about to observe,
so a retry after a command invalidation re-selects with the latest intent. The
previous action carried into an observation is always the last committed raw
action, whichever network produced it: a switch never fabricates history. It
changes which network is evaluated and nothing else: cadence, the targets
transform, limits, stop semantics, support checks and verification are untouched.

This is a software candidate for the measured post-completion drift of the
velocity policies under a zero twist; selecting it is not locomotion admission.
"""
from __future__ import annotations

import numpy as np

HANDOFF_PROFILES = ('stand-on-zero-twist-v1',)
HANDOFF_RULE = ('standing policy whenever the commanded twist is exactly zero; motion policy otherwise; '
                'the previous action is the last committed raw action across switches; nothing else changes')
_REQUIRED = ('previous_action', 'preview', 'targets', 'commit', 'reset', 'sha256', 'target_contract')


def _check_policy(name, policy):
    missing = [attr for attr in _REQUIRED if not hasattr(policy, attr)]
    if missing:
        raise ValueError(f'{name} policy lacks {missing}')
    digest = getattr(policy, 'sha256', None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f'{name} policy must expose its admitted SHA-256')
    contract = getattr(policy, 'target_contract', None)
    if not isinstance(contract, dict) or contract.get('policy_sha256') != digest:
        raise ValueError(f'{name} policy target contract must be bound to its SHA-256')
    return policy


class StandingHandoff:
    """Two admitted policies behind the stepper's single-policy interface."""

    def __init__(self, motion, standing, *, profile='stand-on-zero-twist-v1'):
        if profile not in HANDOFF_PROFILES:
            raise ValueError('unknown standing handoff profile')
        self._motion, self._standing = _check_policy('motion', motion), _check_policy('standing', standing)
        if motion.sha256 == standing.sha256:
            raise ValueError('standing handoff requires two different policies')
        own = {k: v for k, v in motion.target_contract.items() if k != 'policy_sha256'}
        other = {k: v for k, v in standing.target_contract.items() if k != 'policy_sha256'}
        if own != other:
            raise ValueError('motion and standing policies must share the same target transform')
        self.profile = profile
        self._policies = {'motion': motion, 'standing': standing}
        self._active = None          # role selected for the current attempt
        self._last_committer = None  # role whose action was committed last
        self._previous = np.zeros(14, dtype=np.float32)
        self.selections = self.switches = 0
        self.commits = {'motion': 0, 'standing': 0}
        self.last_switch = None

    # --- identity -----------------------------------------------------------------
    @property
    def sha256(self):
        """The commanded behaviour's digest; the standing digest is published separately."""
        return self._motion.sha256

    @property
    def standing_sha256(self):
        return self._standing.sha256

    @property
    def target_contract(self):
        return dict(self._motion.target_contract)

    @property
    def action_scale(self):
        return getattr(self._motion, 'action_scale', None)

    def contract(self):
        return {'profile': self.profile, 'rule': HANDOFF_RULE,
                'motion_policy_sha256': self._motion.sha256, 'standing_policy_sha256': self._standing.sha256}

    # --- stepper interface --------------------------------------------------------
    @property
    def active_role(self):
        return self._active

    @property
    def previous_action(self) -> np.ndarray:
        return self._previous.copy()

    @property
    def committed_targets(self):
        if self._last_committer is None:
            return None
        return self._policies[self._last_committer].committed_targets

    def select(self, command):
        """Choose the network for the observation the stepper is about to build."""
        values = np.asarray(command)
        if values.shape != (13,) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError('command must be a finite numeric vector of shape (13,)')
        role = 'standing' if not np.any(values[:3]) else 'motion'
        self._active = role
        self.selections += 1
        return {'policy_role': role, 'policy_sha256': self._policies[role].sha256,
                'switch_pending': self._last_committer not in (None, role)}

    def _policy(self):
        if self._active is None:
            raise RuntimeError('select(command) must precede preview/targets/commit')
        return self._policies[self._active]

    def preview(self, obs) -> np.ndarray:
        return self._policy().preview(obs)

    def targets(self, action) -> np.ndarray:
        return self._policy().targets(action)

    def infer(self, obs) -> np.ndarray:
        action = self.preview(obs)
        self.commit(action)
        return action

    def commit(self, action):
        role, policy = self._active, self._policy()
        if self._last_committer not in (None, role):
            self.switches += 1
            self.last_switch = {'from': self._last_committer, 'to': role,
                                'commit_index': sum(self.commits.values())}
        policy.commit(action)
        self._previous = np.asarray(policy.previous_action, dtype=np.float32).copy()
        self._last_committer = role
        self.commits[role] += 1

    def reset(self):
        for policy in self._policies.values():
            policy.reset()
        self._previous = np.zeros(14, dtype=np.float32)
        self._active = self._last_committer = None

    def telemetry(self) -> dict:
        return {'profile': self.profile, 'active_role': self._active, 'last_committer': self._last_committer,
                'selections': self.selections, 'switches': self.switches, 'commits': dict(self.commits),
                'last_switch': None if self.last_switch is None else dict(self.last_switch),
                'motion_policy_sha256': self._motion.sha256, 'standing_policy_sha256': self._standing.sha256}
