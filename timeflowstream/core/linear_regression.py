from typing import Optional
from timeflowstream.core.types import LinearSegment

class StreamingLinearRegression:
    """
    Streaming O(1) Auto Linear Regression using online running moments.
    Fits y(t) = a + b * (t - t_start) with bounded absolute error.
    """
    def __init__(self, channel_id: int = 0, tolerance: float = 0.05, min_window: int = 4, max_window: int = 256):
        self.channel_id = channel_id
        self.tolerance = tolerance
        self.min_window = min_window
        self.max_window = max_window
        self.reset()

    def reset(self):
        self.n = 0
        self.t_start = 0.0
        self.t_vals = []
        self.y_vals = []
        self.sum_t = 0.0
        self.sum_y = 0.0
        self.sum_t2 = 0.0
        self.sum_ty = 0.0

    def _fit(self, sum_t, sum_y, sum_t2, sum_ty, n):
        if n < 2:
            return (sum_y / max(n, 1)), 0.0
        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-12:
            return (sum_y / n), 0.0
        slope = (n * sum_ty - sum_t * sum_y) / denom
        intercept = (sum_y - slope * sum_t) / n
        return intercept, slope

    def push(self, t: float, y: float) -> Optional[LinearSegment]:
        if self.n == 0:
            self.t_start = t
            self.n = 1
            self.t_vals = [0.0]
            self.y_vals = [y]
            self.sum_t = 0.0
            self.sum_y = y
            self.sum_t2 = 0.0
            self.sum_ty = 0.0
            return None

        rel_t = t - self.t_start
        new_n = self.n + 1
        new_sum_t = self.sum_t + rel_t
        new_sum_y = self.sum_y + y
        new_sum_t2 = self.sum_t2 + rel_t * rel_t
        new_sum_ty = self.sum_ty + rel_t * y

        intercept, slope = self._fit(new_sum_t, new_sum_y, new_sum_t2, new_sum_ty, new_n)

        # Check maximum residual against tolerance
        exceeds_tolerance = False
        if new_n >= self.min_window:
            for rt, ry in zip(self.t_vals + [rel_t], self.y_vals + [y]):
                pred = intercept + slope * rt
                if abs(ry - pred) > self.tolerance:
                    exceeds_tolerance = True
                    break

        if exceeds_tolerance or new_n > self.max_window:
            cur_intercept, cur_slope = self._fit(self.sum_t, self.sum_y, self.sum_t2, self.sum_ty, self.n)
            segment = LinearSegment(
                channel_id=self.channel_id,
                t_start=self.t_start,
                length=self.n,
                intercept=cur_intercept,
                slope=cur_slope,
                mse=0.0
            )
            # Start new window with current sample
            self.t_start = t
            self.n = 1
            self.t_vals = [0.0]
            self.y_vals = [y]
            self.sum_t = 0.0
            self.sum_y = y
            self.sum_t2 = 0.0
            self.sum_ty = 0.0
            return segment

        # Accept sample into current window
        self.n = new_n
        self.sum_t = new_sum_t
        self.sum_y = new_sum_y
        self.sum_t2 = new_sum_t2
        self.sum_ty = new_sum_ty
        self.t_vals.append(rel_t)
        self.y_vals.append(y)
        return None

    def flush(self) -> Optional[LinearSegment]:
        if self.n == 0:
            return None
        intercept, slope = self._fit(self.sum_t, self.sum_y, self.sum_t2, self.sum_ty, self.n)
        segment = LinearSegment(
            channel_id=self.channel_id,
            t_start=self.t_start,
            length=self.n,
            intercept=intercept,
            slope=slope,
            mse=0.0
        )
        self.reset()
        return segment
