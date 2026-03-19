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

"""Pure, ROS-independent utility functions for the traffic-light detector."""

import numpy as np


def clamp_roi(x, y, w, h, img_w, img_h):
    """Clamp an ROI so that it lies fully within the image bounds."""
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    w = max(1, min(int(w), img_w - x))
    h = max(1, min(int(h), img_h - y))
    return x, y, w, h


def ratio_to_roi(x_ratio, y_ratio, w_ratio, h_ratio, img_w, img_h):
    """Convert normalised ROI parameters to absolute pixel coordinates.

    Valid range: ``x_ratio`` and ``y_ratio`` must be in ``[0, 1]``;
    ``w_ratio`` and ``h_ratio`` must be in ``(0, 1]``.

    Returns a clamped ``(x, y, w, h)`` tuple on success, or ``None`` when any
    ratio falls outside its valid range.
    """
    if not (
        0.0 <= x_ratio <= 1.0
        and 0.0 <= y_ratio <= 1.0
        and 0.0 < w_ratio <= 1.0
        and 0.0 < h_ratio <= 1.0
    ):
        return None
    x = int(x_ratio * img_w)
    y = int(y_ratio * img_h)
    w = max(1, int(w_ratio * img_w))
    h = max(1, int(h_ratio * img_h))
    return clamp_roi(x, y, w, h, img_w, img_h)


def mask_ratio(mask):
    """Return the fraction of non-zero pixels in *mask*."""
    return float(np.count_nonzero(mask)) / float(mask.size)


def order_corners(pts):
    """Return four corner points in (top-left, top-right, bottom-right, bottom-left) order.

    Sorting strategy: split the four points into the two with smaller y (top pair)
    and the two with larger y (bottom pair), then order each pair left-to-right.
    This is robust to perspective skew and rotated cards.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    # Sort by y coordinate; top two points have smaller y
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]
    # Within each pair, sort left-to-right by x
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


class DebounceFilter:
    """Temporal state stabiliser that requires N consecutive frames before switching.

    Parameters
    ----------
    debounce_frames : int
        Number of consecutive frames a new candidate must be seen before the
        stable state is updated.  ``1`` is equivalent to no debounce.
    debounce_unknown : bool
        When *True* (default) the UNKNOWN state is also subject to debounce.
        When *False* transitions **to** UNKNOWN bypass the streak requirement
        and take effect immediately.
    unknown_state : int
        Integer value that represents the UNKNOWN state in the message type
        (``TrafficLight.UNKNOWN == 0``).
    """

    def __init__(self, debounce_frames=3, debounce_unknown=True, unknown_state=0):
        """Initialise the filter with the given parameters."""
        if int(debounce_frames) < 1:
            raise ValueError(f'debounce_frames must be >= 1, got {debounce_frames}')
        self.debounce_frames = int(debounce_frames)
        self.debounce_unknown = bool(debounce_unknown)
        self._unknown_state = unknown_state

        self._stable_state = unknown_state
        self._stable_confidence = 0.0
        self._candidate_state = unknown_state
        self._candidate_streak = 0

    def update(self, state, confidence):
        """Feed one frame's candidate state and return the stabilised result.

        Parameters
        ----------
        state : int
            Per-frame detected state.
        confidence : float
            Per-frame detection confidence.

        Returns
        -------
        tuple[int, float]
            ``(stable_state, stable_confidence)`` after applying debounce.

        Notes
        -----
        Confidence policy: the published confidence is updated only when the
        stable state commits to a new value.  While a transition is being held
        by the streak requirement the previously committed confidence is kept.
        """
        if state == self._stable_state:
            # Candidate matches stable state – reset the streak counter so
            # any in-progress transition is cancelled.
            self._candidate_streak = 0
            self._candidate_state = state
        else:
            # Bypass debounce for UNKNOWN when debounce_unknown is False
            bypass = (not self.debounce_unknown and state == self._unknown_state)

            if bypass:
                self._stable_state = state
                self._stable_confidence = confidence
                self._candidate_streak = 0
                self._candidate_state = state
            else:
                if state == self._candidate_state:
                    self._candidate_streak += 1
                else:
                    # New candidate; start a fresh streak
                    self._candidate_state = state
                    self._candidate_streak = 1

                if self._candidate_streak >= self.debounce_frames:
                    self._stable_state = state
                    self._stable_confidence = confidence
                    self._candidate_streak = 0

        return self._stable_state, self._stable_confidence
