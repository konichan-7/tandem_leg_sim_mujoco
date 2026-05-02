from pathlib import Path

import numpy as np


class Plotter:
    def __init__(
        self,
        state_names: tuple[str, ...],
        print_interval: float = 0.05,
        auto_open: bool = True,
    ) -> None:
        self.state_names = state_names
        self.print_interval = print_interval
        self.auto_open = auto_open
        self.time: list[float] = []
        self.expected: list[np.ndarray] = []
        self.feedback: list[np.ndarray] = []
        self.last_print = 0.0

    def record(
        self,
        time: float,
        expected: np.ndarray,
        feedback: np.ndarray,
    ) -> None:
        expected = np.asarray(expected, dtype=float).reshape(-1)
        feedback = np.asarray(feedback, dtype=float).reshape(-1)

        self.time.append(time)
        self.expected.append(expected.copy())
        self.feedback.append(feedback.copy())

        if time - self.last_print >= self.print_interval:
            self.last_print = time
            self.print_state(time, expected, feedback)

    def print_state(
        self,
        time: float,
        expected: np.ndarray,
        feedback: np.ndarray,
    ) -> None:
        values = " ".join(
            f"{name}: exp={exp:+.2f} fb={fb:+.2f}"
            for name, exp, fb in zip(self.state_names, expected, feedback)
        )
        print(f"t={time:.3f} {values}")

    def render(self) -> None:
        if not self.time:
            return None

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("plotly is not installed, skip state plot rendering")
            return None

        expected = np.vstack(self.expected)
        feedback = np.vstack(self.feedback)
        fig = make_subplots(
            rows=len(self.state_names),
            cols=1,
            shared_xaxes=True,
            subplot_titles=self.state_names,
        )

        for row, name in enumerate(self.state_names, start=1):
            col = row - 1
            fig.add_trace(
                go.Scatter(
                    x=self.time,
                    y=expected[:, col],
                    mode="lines",
                    name=f"{name}_expected",
                    line={"dash": "dash"},
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.time,
                    y=feedback[:, col],
                    mode="lines",
                    name=f"{name}_feedback",
                ),
                row=row,
                col=1,
            )

        fig.update_layout(
            height=260 * len(self.state_names),
            title="Demo LQR State Tracking",
        )
        fig.update_xaxes(title_text="time [s]", row=len(self.state_names), col=1)

        return None
