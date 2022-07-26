# Copyright 2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Viewer application module."""

import collections

from dm_control import _render

from dm_envs.myviewer import gui
from dm_envs.myviewer import renderer
from dm_envs.myviewer import runtime
from dm_envs.myviewer import user_input
from dm_envs.myviewer import util
from dm_envs.myviewer import viewer
from dm_envs.myviewer import views

_DOUBLE_BUFFERING = (user_input.KEY_F5)
_PAUSE = user_input.KEY_SPACE
_RESTART = user_input.KEY_BACKSPACE
_ADVANCE_SIMULATION = user_input.KEY_RIGHT
_SPEED_UP_TIME = user_input.KEY_EQUAL
_SLOW_DOWN_TIME = user_input.KEY_MINUS
_HELP = user_input.KEY_F1
_STATUS = user_input.KEY_F2

_MAX_FRONTBUFFER_SIZE = 2048
_MISSING_STATUS_ENTRY = '--'
_RUNTIME_STOPPED_LABEL = 'EPISODE TERMINATED - hit backspace to restart'
_STATUS_LABEL = 'Status'
_TIME_LABEL = 'Time'
_CPU_LABEL = 'CPU'
_FPS_LABEL = 'FPS'
_CAMERA_LABEL = 'Camera'
_PAUSED_LABEL = 'Paused'
_ERROR_LABEL = 'Error'


class Help(views.ColumnTextModel):
    """Contains the description of input map employed in the application."""

    def __init__(self):
        """Instance initializer."""
        self._value = [
            ['Help', 'F1'],
            ['Info', 'F2'],
            ['Stereo', 'F5'],
            ['Frame', 'F6'],
            ['Label', 'F7'],
            ['--------------', ''],
            ['Pause', 'Space'],
            ['Reset', 'BackSpace'],
            ['Autoscale', 'Ctrl A'],
            ['Geoms', '0 - 4'],
            ['Sites', 'Shift 0 - 4'],
            ['Speed Up', '='],
            ['Slow Down', '-'],
            ['Switch Cam', '[ ]'],
            ['--------------', ''],
            ['Translate', 'R drag'],
            ['Rotate', 'L drag'],
            ['Zoom', 'Scroll'],
            ['Select', 'L dblclick'],
            ['Center', 'R dblclick'],
            ['Track', 'Ctrl R dblclick / Esc'],
            ['Perturb', 'Ctrl [Shift] L/R drag'],
        ]

    def get_columns(self):
        """Returns the text to display in two columns."""
        return self._value


class Status(views.ColumnTextModel):
    """Monitors and returns the status of the application."""

    def __init__(self, time_multiplier, pause, frame_timer):
        """Instance initializer.

        Args:
          time_multiplier: Instance of util.TimeMultiplier.
          pause: An observable pause subject, instance of util.ObservableFlag.
          frame_timer: A Timer instance counting duration of frames.
        """
        self._runtime = None
        self._time_multiplier = time_multiplier
        self._camera = None
        self._pause = pause
        self._frame_timer = frame_timer
        self._fps_counter = util.Integrator()
        self._cpu_counter = util.Integrator()

        self._value = collections.OrderedDict([
            (_STATUS_LABEL, _MISSING_STATUS_ENTRY),
            (_TIME_LABEL, _MISSING_STATUS_ENTRY),
            (_CPU_LABEL, _MISSING_STATUS_ENTRY),
            (_FPS_LABEL, _MISSING_STATUS_ENTRY),
            (_CAMERA_LABEL, _MISSING_STATUS_ENTRY),
            (_PAUSED_LABEL, _MISSING_STATUS_ENTRY),
            (_ERROR_LABEL, _MISSING_STATUS_ENTRY),
        ])

    def set_camera(self, camera):
        """Updates the active camera instance.

        Args:
          camera: Instance of renderer.SceneCamera.
        """
        self._camera = camera

    def set_runtime(self, instance):
        """Updates the active runtime instance.

        Args:
          instance: Instance of runtime.Base.
        """
        if self._runtime:
            self._runtime.on_error -= self._on_error
            self._runtime.on_episode_begin -= self._clear_error
        self._runtime = instance
        if self._runtime:
            self._runtime.on_error += self._on_error
            self._runtime.on_episode_begin += self._clear_error

    def get_columns(self):
        """Returns the text to display in two columns."""
        if self._frame_timer.measured_time > 0:
            self._fps_counter.value = 1. / self._frame_timer.measured_time
        self._value[_FPS_LABEL] = '{0:.1f}'.format(self._fps_counter.value)

        if self._runtime:
            if self._runtime.state == runtime.State.STOPPED:
                self._value[_STATUS_LABEL] = _RUNTIME_STOPPED_LABEL
            else:
                self._value[_STATUS_LABEL] = str(self._runtime.state)

            self._cpu_counter.value = self._runtime.simulation_time

            self._value[_TIME_LABEL] = '{0:.1f} ({1}x)'.format(
                self._runtime.get_time(), str(self._time_multiplier))
            self._value[_CPU_LABEL] = '{0:.2f}ms'.format(
                self._cpu_counter.value * 1000.0)
        else:
            self._value[_STATUS_LABEL] = _MISSING_STATUS_ENTRY
            self._value[_TIME_LABEL] = _MISSING_STATUS_ENTRY
            self._value[_CPU_LABEL] = _MISSING_STATUS_ENTRY

        if self._camera:
            self._value[_CAMERA_LABEL] = self._camera.name
        else:
            self._value[_CAMERA_LABEL] = _MISSING_STATUS_ENTRY

        self._value[_PAUSED_LABEL] = str(self._pause.value)

        return list(self._value.items())  # For Python 2/3 compatibility.

    def _clear_error(self):
        self._value[_ERROR_LABEL] = _MISSING_STATUS_ENTRY

    def _on_error(self, error_msg):
        self._value[_ERROR_LABEL] = error_msg


class ReloadParams(collections.namedtuple(
    'RefreshParams', ['zoom_to_scene'])):
    """Parameters of a reload request."""


class Application:
    """Viewer application."""

    def __init__(self, physics, title='Explorer', width=1024, height=768):
        """Instance initializer."""
        self._physics = physics
        self._render_surface = None
        self._renderer = renderer.NullRenderer()
        self._viewport = renderer.Viewport(width, height)
        self._window = gui.RenderWindow(width, height, title)

        self._pause_subject = util.ObservableFlag(True)
        self._time_multiplier = util.TimeMultiplier(1.)
        self._frame_timer = util.Timer()
        self._viewer = viewer.Viewer(
            self._viewport, self._window.mouse, self._window.keyboard)
        self._viewer_layout = views.ViewportLayout()
        self._status = Status(
            self._time_multiplier, self._pause_subject, self._frame_timer)

        status_view_toggle = self._build_view_toggle(
            views.ColumnTextView(self._status), views.PanelLocation.BOTTOM_LEFT)
        help_view_toggle = self._build_view_toggle(
            views.ColumnTextView(Help()), views.PanelLocation.TOP_RIGHT)
        status_view_toggle()

        self._input_map = user_input.InputMap(self._window.mouse, self._window.keyboard)
        self._input_map.bind(help_view_toggle, _HELP)
        self._input_map.bind(status_view_toggle, _STATUS)

        self._viewer.deinitialize()
        self._status.set_camera(None)
        self._render_surface = _render.Renderer(max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
        self._renderer = renderer.OffScreenRenderer(self._physics.model, self._render_surface)
        self._renderer.components += self._viewer_layout
        self._viewer.initialize(self._physics, self._renderer, touchpad=False)
        self._status.set_camera(self._viewer.camera)

    def _build_view_toggle(self, view, location):
        def toggle():
            if view in self._viewer_layout:
                self._viewer_layout.remove(view)
            else:
                self._viewer_layout.add(view, location)

        return toggle

    def _tick(self):
        self._viewer.render()

    def launch(self):
        def tick():
            self._viewport.set_size(*self._window.shape)
            self._tick()
            return self._renderer.pixels

        self._window.event_loop(tick_func=tick)
        self._window.close()
