# Copyright 2024 GuYueHome
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for pure utility functions in traffic_light_detector_node.

These tests do not require a running ROS 2 environment and exercise the
helper functions that underpin the detection logic, as well as verifying
that the always-publish (UNKNOWN) semantics are preserved.
"""

import numpy as np
import pytest

from originbot_traffic_light._utils import (
    DebounceFilter,
    clamp_roi,
    mask_ratio,
    order_corners,
    ratio_to_roi,
)


class TestClampRoi:
    """Tests for the clamp_roi helper."""

    def test_no_clamp_needed(self):
        x, y, w, h = clamp_roi(10, 20, 100, 80, 640, 480)
        assert (x, y, w, h) == (10, 20, 100, 80)

    def test_clamp_negative_x(self):
        x, y, w, h = clamp_roi(-5, 0, 50, 50, 640, 480)
        assert x == 0

    def test_clamp_negative_y(self):
        x, y, w, h = clamp_roi(0, -10, 50, 50, 640, 480)
        assert y == 0

    def test_clamp_width_beyond_right_edge(self):
        x, y, w, h = clamp_roi(600, 0, 200, 50, 640, 480)
        assert x + w <= 640

    def test_clamp_height_beyond_bottom_edge(self):
        x, y, w, h = clamp_roi(0, 450, 50, 200, 640, 480)
        assert y + h <= 480

    def test_minimum_width_one(self):
        _, _, w, _ = clamp_roi(0, 0, 0, 50, 640, 480)
        assert w >= 1

    def test_minimum_height_one(self):
        _, _, _, h = clamp_roi(0, 0, 50, 0, 640, 480)
        assert h >= 1


class TestMaskRatio:
    """Tests for the mask_ratio helper."""

    def test_all_zero_mask_returns_zero(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert mask_ratio(mask) == pytest.approx(0.0)

    def test_all_nonzero_mask_returns_one(self):
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        assert mask_ratio(mask) == pytest.approx(1.0)

    def test_half_nonzero_returns_half(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :] = 255
        assert mask_ratio(mask) == pytest.approx(0.5)


class TestOrderCorners:
    """Tests for the order_corners helper."""

    def test_axis_aligned_rectangle(self):
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        ordered = order_corners(pts)
        tl, tr, br, bl = ordered
        assert tl[0] < tr[0], 'top-left x should be left of top-right x'
        assert bl[0] < br[0], 'bottom-left x should be left of bottom-right x'
        assert tl[1] < bl[1], 'top-left y should be above bottom-left y'

    def test_output_shape(self):
        pts = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
        ordered = order_corners(pts)
        assert ordered.shape == (4, 2)

    def test_shuffled_input_gives_same_result(self):
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        shuffled = pts[[2, 0, 3, 1]]
        np.testing.assert_array_almost_equal(
            order_corners(pts), order_corners(shuffled)
        )


class TestRatioToRoi:
    """Tests for the ratio_to_roi helper."""

    def test_valid_ratios_produce_correct_pixels(self):
        # 50 % width starting at x=0.2, 25 % height starting at y=0.3
        result = ratio_to_roi(0.2, 0.3, 0.5, 0.25, 640, 480)
        assert result is not None
        x, y, w, h = result
        assert x == 128     # 0.2 * 640
        assert y == 144     # 0.3 * 480
        assert w == 320     # 0.5 * 640
        assert h == 120     # 0.25 * 480

    def test_top_left_origin(self):
        result = ratio_to_roi(0.0, 0.0, 0.5, 0.5, 640, 480)
        assert result is not None
        x, y, w, h = result
        assert x == 0
        assert y == 0
        assert w == 320
        assert h == 240

    def test_full_frame(self):
        result = ratio_to_roi(0.0, 0.0, 1.0, 1.0, 640, 480)
        assert result is not None
        x, y, w, h = result
        assert x == 0
        assert y == 0
        assert w == 640
        assert h == 480

    def test_result_is_clamped_within_image(self):
        # Large x_ratio + w_ratio would overflow; clamp_roi should fix it
        result = ratio_to_roi(0.9, 0.9, 0.5, 0.5, 640, 480)
        assert result is not None
        x, y, w, h = result
        assert x + w <= 640
        assert y + h <= 480

    def test_zero_w_ratio_returns_none(self):
        assert ratio_to_roi(0.0, 0.0, 0.0, 0.5, 640, 480) is None

    def test_zero_h_ratio_returns_none(self):
        assert ratio_to_roi(0.0, 0.0, 0.5, 0.0, 640, 480) is None

    def test_w_ratio_above_one_returns_none(self):
        assert ratio_to_roi(0.0, 0.0, 1.5, 0.5, 640, 480) is None

    def test_h_ratio_above_one_returns_none(self):
        assert ratio_to_roi(0.0, 0.0, 0.5, 1.1, 640, 480) is None

    def test_negative_x_ratio_returns_none(self):
        assert ratio_to_roi(-0.1, 0.0, 0.5, 0.5, 640, 480) is None

    def test_x_ratio_above_one_returns_none(self):
        assert ratio_to_roi(1.1, 0.0, 0.5, 0.5, 640, 480) is None

    def test_different_resolutions_same_ratios(self):
        # ROI ratios should scale proportionally with frame size
        r1 = ratio_to_roi(0.2, 0.3, 0.4, 0.5, 640, 480)
        r2 = ratio_to_roi(0.2, 0.3, 0.4, 0.5, 1280, 720)
        assert r1 is not None and r2 is not None
        # x and w scale by 2 in width dimension
        assert r2[0] == pytest.approx(r1[0] * 2, abs=2)
        assert r2[2] == pytest.approx(r1[2] * 2, abs=2)


class TestDebounceFilter:
    """Tests for the DebounceFilter temporal stabiliser."""

    # State constants matching TrafficLight.msg values
    UNKNOWN = 0
    STOP = 1
    LEFT = 2
    STRAIGHT = 3
    RIGHT = 4

    def _make_filter(self, frames=3, debounce_unknown=True):
        return DebounceFilter(
            debounce_frames=frames,
            debounce_unknown=debounce_unknown,
            unknown_state=self.UNKNOWN,
        )

    def test_initial_state_is_unknown(self):
        f = self._make_filter()
        assert f._stable_state == self.UNKNOWN

    def test_invalid_debounce_frames_raises(self):
        with pytest.raises(ValueError):
            DebounceFilter(debounce_frames=0)
        with pytest.raises(ValueError):
            DebounceFilter(debounce_frames=-1)

    def test_no_switch_before_streak_complete(self):
        f = self._make_filter(frames=3)
        # Feed STOP for 2 frames – not enough to commit
        for _ in range(2):
            state, _ = f.update(self.STOP, 0.9)
        assert state == self.UNKNOWN

    def test_switches_after_n_consecutive_frames(self):
        f = self._make_filter(frames=3)
        for _ in range(3):
            state, conf = f.update(self.STOP, 0.9)
        assert state == self.STOP
        assert conf == pytest.approx(0.9)

    def test_streak_resets_on_stable_frame(self):
        f = self._make_filter(frames=3)
        # Two STOP frames (streak=2), then one UNKNOWN (matches stable=UNKNOWN)
        f.update(self.STOP, 0.9)
        f.update(self.STOP, 0.9)
        state, _ = f.update(self.UNKNOWN, 0.0)
        assert state == self.UNKNOWN  # streak reset; still UNKNOWN
        # Next STOP should start fresh streak – still not committed after 2 more
        f.update(self.STOP, 0.9)
        state, _ = f.update(self.STOP, 0.9)
        assert state == self.UNKNOWN

    def test_streak_resets_on_candidate_change(self):
        f = self._make_filter(frames=3)
        # Two STOP frames, then switch to LEFT – streak should restart
        f.update(self.STOP, 0.9)
        f.update(self.STOP, 0.9)
        # Different candidate (LEFT) – streak resets to 1
        state, _ = f.update(self.LEFT, 0.8)
        assert state == self.UNKNOWN
        # One more LEFT (streak=2) – still not committed
        state, _ = f.update(self.LEFT, 0.8)
        assert state == self.UNKNOWN
        # Third LEFT (streak=3) – commits
        state, conf = f.update(self.LEFT, 0.8)
        assert state == self.LEFT
        assert conf == pytest.approx(0.8)

    def test_debounce_frames_one_is_immediate(self):
        f = self._make_filter(frames=1)
        state, conf = f.update(self.STOP, 0.7)
        assert state == self.STOP
        assert conf == pytest.approx(0.7)

    def test_confidence_held_during_streak(self):
        f = self._make_filter(frames=3)
        # Commit STOP with confidence 0.9
        for _ in range(3):
            state, conf = f.update(self.STOP, 0.9)
        assert state == self.STOP
        # Now try to move to LEFT; confidence should stay 0.9 until committed
        f.update(self.LEFT, 0.5)
        _, conf = f.update(self.LEFT, 0.6)
        assert conf == pytest.approx(0.9)  # held from last commit

    def test_debounce_unknown_false_bypasses_streak(self):
        f = self._make_filter(frames=3, debounce_unknown=False)
        # First commit STOP
        for _ in range(3):
            f.update(self.STOP, 0.9)
        # Single UNKNOWN frame should bypass debounce
        state, conf = f.update(self.UNKNOWN, 0.0)
        assert state == self.UNKNOWN
        assert conf == pytest.approx(0.0)

    def test_debounce_unknown_true_requires_streak_for_unknown(self):
        f = self._make_filter(frames=3, debounce_unknown=True)
        # Commit STOP first
        for _ in range(3):
            f.update(self.STOP, 0.9)
        # Only one UNKNOWN frame – not enough to commit
        state, _ = f.update(self.UNKNOWN, 0.0)
        assert state == self.STOP

    def test_debounce_frames_one_disables_filtering(self):
        """Verifies debounce_frames=1 commits every state change immediately."""
        f = self._make_filter(frames=1)
        # Each state change should be immediately committed
        state, _ = f.update(self.LEFT, 0.8)
        assert state == self.LEFT
        state, _ = f.update(self.RIGHT, 0.7)
        assert state == self.RIGHT
        state, _ = f.update(self.UNKNOWN, 0.0)
        assert state == self.UNKNOWN
