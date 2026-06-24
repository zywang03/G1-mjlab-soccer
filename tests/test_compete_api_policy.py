"""Tests for competition API policy client reset behavior."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import requests
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_compete():
  path = REPO_ROOT / "scripts" / "compete.py"
  spec = importlib.util.spec_from_file_location("compete", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class _FailingResponse:
  def raise_for_status(self):
    raise requests.HTTPError("server error")


class ApiPolicyTest(unittest.TestCase):
  def test_reset_warns_when_remote_reset_fails(self):
    compete = _load_compete()

    with mock.patch.object(compete.requests, "post", return_value=_FailingResponse()):
      policy = object.__new__(compete.ApiPolicy)
      policy._url = "http://example.invalid"
      policy._timeout = 0.01

      output = io.StringIO()
      with contextlib.redirect_stdout(output):
        policy.reset()

    self.assertIn("[WARN] API policy reset failed", output.getvalue())
    self.assertIn("http://example.invalid/reset", output.getvalue())

  def test_combined_policy_resets_when_viewer_env_auto_resets(self):
    compete = _load_compete()
    captured_prev_actions = []

    class CountingPolicy:
      def __init__(self, value: float):
        self.value = value
        self.reset_count = 0

      def __call__(self, _raw_state):
        return torch.full((1, 29), self.value)

      def reset(self):
        self.reset_count += 1

    def fake_build_raw_state(_env_base, prev_action_s, prev_action_g):
      captured_prev_actions.append((prev_action_s.clone(), prev_action_g.clone()))
      return {}

    env_base = SimpleNamespace(episode_length_buf=torch.tensor([1]))
    shooter = CountingPolicy(1.0)
    goalkeeper = CountingPolicy(2.0)
    combined = compete.CombinedPolicy(shooter, goalkeeper, env_base, "cpu")

    with mock.patch.object(compete, "_build_raw_state", side_effect=fake_build_raw_state):
      combined({})
      env_base.episode_length_buf[0] = 0
      combined({})

    self.assertEqual(shooter.reset_count, 1)
    self.assertEqual(goalkeeper.reset_count, 1)
    self.assertTrue(torch.count_nonzero(captured_prev_actions[1][0]).item() == 0)
    self.assertTrue(torch.count_nonzero(captured_prev_actions[1][1]).item() == 0)


if __name__ == "__main__":
  unittest.main()
