import unittest

from main import create_controller

TRANSITIONS = {
    "idle": {"crouch"},
    "crouch": {"thrust"},
    "thrust": {"flight", "retract"},
    "retract": {"flight"},
    "flight": {"landing"},
    "landing": {"flight", "idle"},
}
PLAN = ((500, False), (400, True), (2100, False))


class JumpPhaseMachineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.controllers = {
            name: create_controller(controller=name, terrain="flat")
            for name in ("lqr", "whole_body_lqr")
        }

    def phase_sequences(self) -> dict[str, list[str]]:
        traces = {}
        for name, controller in self.controllers.items():
            controller.reset()
            phases = []
            for count, held in PLAN:
                for _ in range(count):
                    controller.command(0.0, 0.0, 0.0, jump=held)
                    controller.step()
                    if not phases or phases[-1] != controller.jump.phase:
                        phases.append(controller.jump.phase)
            traces[name] = phases
        return traces

    def test_phase_sequence_is_legal_and_shared(self) -> None:
        traces = self.phase_sequences()
        for name, phases in traces.items():
            if phases[0] != "idle" or phases[-1] != "idle":
                raise ValueError(f"{name}: jump did not start and end idle: {phases}")
            for before, after in zip(phases, phases[1:]):
                if after not in TRANSITIONS[before]:
                    raise ValueError(f"{name}: illegal transition {before}->{after}")
            if "flight" not in phases:
                raise ValueError(f"{name}: jump never left the ground: {phases}")
        if traces["lqr"] != traces["whole_body_lqr"]:
            raise ValueError(f"Phase sequences diverged between branches: {traces}")


if __name__ == "__main__":
    unittest.main()
