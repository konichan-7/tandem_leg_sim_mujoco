from mjlab.rl.runner import MjlabOnPolicyRunner


class WheelLegRunner(MjlabOnPolicyRunner):
    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        directory, filename, _ = self._get_export_paths(path)
        self.export_policy_to_onnx(str(directory), filename)
