from __future__ import annotations

from collections.abc import Callable
import ctypes
import sys
import time
from typing import Protocol

import glfw
import mujoco
import mujoco.viewer


class Controller(Protocol):
    model: mujoco.MjModel
    data: mujoco.MjData
    paused: bool

    def command(
        self,
        linear_direction: float,
        yaw_direction: float,
        leg_length_direction: float,
        spin: bool = False,
        jump: bool = False,
    ) -> None: ...

    def toggle_pause(self) -> None: ...

    def step(self) -> None: ...


class MacKeyboard:
    keycodes = {
        "1": 18,
        "2": 19,
        "3": 20,
        "4": 21,
        "5": 23,
        "6": 22,
    }

    def __init__(self) -> None:
        self.core_graphics = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        self.core_graphics.CGEventSourceKeyState.argtypes = [
            ctypes.c_int32,
            ctypes.c_uint16,
        ]
        self.core_graphics.CGEventSourceKeyState.restype = ctypes.c_bool

    def command_direction(
        self,
        commands: dict[str, tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        linear_direction = 0.0
        yaw_direction = 0.0
        leg_length_direction = 0.0
        for key, (linear, yaw, leg_length) in commands.items():
            if key in self.keycodes and self.core_graphics.CGEventSourceKeyState(
                0,
                self.keycodes[key],
            ):
                linear_direction += linear
                yaw_direction += yaw
                leg_length_direction += leg_length
        return linear_direction, yaw_direction, leg_length_direction

    def spin_held(self) -> bool:
        return any(self.core_graphics.CGEventSourceKeyState(0, key) for key in (56, 60))

    def jump_held(self) -> bool:
        return self.core_graphics.CGEventSourceKeyState(0, 49)


class GlfwViewer:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        pause_callback: Callable[[], None],
        title: str,
    ) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")

        self.model = model
        self.data = data
        self.pause_callback = pause_callback
        self.window = glfw.create_window(1280, 720, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            model,
            mujoco.mjtFontScale.mjFONTSCALE_150,
        )
        mujoco.mjv_defaultFreeCamera(model, self.camera)
        self.pressed_keys: set[int] = set()
        self.cursor_x, self.cursor_y = glfw.get_cursor_pos(self.window)

        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_window_focus_callback(self.window, self._focus_callback)

    def __enter__(self) -> "GlfwViewer":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        glfw.make_context_current(self.window)
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()

    def _key_callback(
        self,
        window: ctypes._Pointer[glfw._GLFWwindow],
        key: int,
        scancode: int,
        action: int,
        mods: int,
    ) -> None:
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_P and action == glfw.PRESS:
            self.pause_callback()
        elif action == glfw.PRESS:
            self.pressed_keys.add(key)
        elif action == glfw.RELEASE:
            self.pressed_keys.discard(key)

    def _cursor_callback(
        self,
        window: ctypes._Pointer[glfw._GLFWwindow],
        xpos: float,
        ypos: float,
    ) -> None:
        dx = xpos - self.cursor_x
        dy = ypos - self.cursor_y
        self.cursor_x = xpos
        self.cursor_y = ypos

        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not (left or middle or right):
            return

        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if right:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif left:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        _, height = glfw.get_window_size(window)
        if height == 0:
            return
        mujoco.mjv_moveCamera(
            self.model,
            action,
            dx / height,
            dy / height,
            self.camera,
        )

    def _scroll_callback(
        self,
        window: ctypes._Pointer[glfw._GLFWwindow],
        xoffset: float,
        yoffset: float,
    ) -> None:
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.camera,
        )

    def _focus_callback(
        self, window: ctypes._Pointer[glfw._GLFWwindow], focused: int
    ) -> None:
        if not focused:
            self.pressed_keys.clear()

    def command_direction(
        self,
        commands: dict[str, tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        linear_direction = 0.0
        yaw_direction = 0.0
        leg_length_direction = 0.0
        for key, (linear, yaw, leg_length) in commands.items():
            if ord(key) in self.pressed_keys:
                linear_direction += linear
                yaw_direction += yaw
                leg_length_direction += leg_length
        return linear_direction, yaw_direction, leg_length_direction

    def is_running(self) -> bool:
        return not glfw.window_should_close(self.window)

    def spin_held(self) -> bool:
        return bool(self.pressed_keys & {glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT})

    def jump_held(self) -> bool:
        return glfw.KEY_SPACE in self.pressed_keys

    def poll_events(self) -> None:
        glfw.poll_events()

    def render(self) -> None:
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)


def run_glfw(
    controller: Controller,
    commands: dict[str, tuple[float, float, float]],
    fps: float,
    title: str,
) -> None:
    frame_period = 1.0 / fps
    wall_start = time.perf_counter()
    simulation_start = controller.data.time

    with GlfwViewer(
        controller.model,
        controller.data,
        controller.toggle_pause,
        title,
    ) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()
            viewer.poll_events()
            controller.command(
                *viewer.command_direction(commands),
                spin=viewer.spin_held(),
                jump=viewer.jump_held(),
            )

            if controller.paused:
                wall_start = frame_start
                simulation_start = controller.data.time
            else:
                target_time = simulation_start + frame_start - wall_start
                while controller.data.time < target_time and not controller.paused:
                    controller.step()

            viewer.render()
            sleep_time = frame_period - (time.perf_counter() - frame_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def run_mujoco_viewer(
    controller: Controller,
    commands: dict[str, tuple[float, float, float]],
    fps: float,
) -> None:
    keyboard = MacKeyboard()
    frame_period = 1.0 / fps
    wall_start = time.perf_counter()
    simulation_start = controller.data.time

    def key_callback(keycode: int) -> None:
        if keycode == glfw.KEY_P:
            controller.toggle_pause()

    with mujoco.viewer.launch_passive(
        controller.model,
        controller.data,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(controller.model, viewer.cam)
        while viewer.is_running():
            frame_start = time.perf_counter()
            controller.command(
                *keyboard.command_direction(commands),
                spin=keyboard.spin_held(),
                jump=keyboard.jump_held(),
            )

            if controller.paused:
                wall_start = frame_start
                simulation_start = controller.data.time
            else:
                target_time = simulation_start + frame_start - wall_start
                while controller.data.time < target_time and not controller.paused:
                    controller.step()

            viewer.sync()
            sleep_time = frame_period - (time.perf_counter() - frame_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def run_interactive(
    controller: Controller,
    commands: dict[str, tuple[float, float, float]],
    title: str,
    fps: float = 60.0,
) -> None:
    if sys.platform == "darwin":
        run_mujoco_viewer(controller, commands, fps)
    else:
        run_glfw(controller, commands, fps, title)
