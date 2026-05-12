"""
missiontools.cesium._viewer
===========================
pywebview-based 3D globe viewer for missiontools CZML data.
"""

from __future__ import annotations

import os
import tempfile
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..ground_station import GroundStation
    from ..aoi import AoI
    from ..spacecraft import Spacecraft


class _QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        pass


def _start_http_server(serve_dir: str) -> HTTPServer:
    server = HTTPServer(
        ("127.0.0.1", 0), lambda *a, **k: _QuietHandler(*a, directory=serve_dir, **k)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class CesiumViewer:
    """Collect missiontools objects and display them on an interactive 3D globe.

    Parameters
    ----------
    title : str
        Window title.

    Examples
    --------
    ::

        from missiontools.cesium import CesiumViewer
        viewer = CesiumViewer()
        viewer.add_spacecraft(sc, t_start, t_end)
        viewer.add_ground_station(gs)
        viewer.add_aoi(aoi)
        viewer.show()
    """

    def __init__(self, title: str = "missiontools 3D Viewer") -> None:
        self._title = title
        self._packets: list = []
        self._sensor_packets: list[dict] = []
        self._has_preamble = False
        self._t_start: np.datetime64 | None = None
        self._t_end: np.datetime64 | None = None
        self._sc_counter = 0
        self._gs_counter = 0
        self._aoi_counter = 0
        self._model_files: dict[str, str] = {}

    def _ensure_preamble(self) -> None:
        if self._has_preamble:
            return
        from ._czml import _build_preamble

        t0 = self._t_start
        t1 = self._t_end
        if t0 is None or t1 is None:
            import datetime as _dt

            now = np.datetime64(
                _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), "s"
            )
            t0 = t0 or now
            t1 = t1 or (now + np.timedelta64(86_400, "s"))
        self._packets.insert(0, _build_preamble(t0, t1))
        self._has_preamble = True

    def _update_time_range(
        self,
        t_start: np.datetime64,
        t_end: np.datetime64,
    ) -> None:
        if self._t_start is None or t_start < self._t_start:
            self._t_start = t_start
        if self._t_end is None or t_end > self._t_end:
            self._t_end = t_end

        if self._has_preamble:
            from ._czml import _build_preamble

            self._packets[0] = _build_preamble(self._t_start, self._t_end)

    def add_spacecraft(
        self,
        spacecraft: Spacecraft,
        t_start: np.datetime64,
        t_end: np.datetime64,
        step: np.timedelta64 = np.timedelta64(30, "s"),
        *,
        color: tuple[int, int, int, int] = (0, 200, 255, 255),
        path_color: tuple[int, int, int, int] | None = None,
        label: str | None = None,
        show_sensors: bool = False,
        sensor_length: float | None = None,
        show_model: bool = True,
        model: str | None = None,
        scale: float = 1.0,
    ) -> None:
        """Add a spacecraft orbit to the visualization.

        Parameters
        ----------
        spacecraft : Spacecraft
        t_start, t_end : np.datetime64
            Propagation time window.
        step : np.timedelta64
            Propagation step size (default 30 s).
        color : tuple[int, int, int, int]
            RGBA colour for the spacecraft icon.
        path_color : tuple[int, int, int, int] | None
            RGBA colour for the orbit trail.
        label : str | None
            Display label.
        show_sensors : bool
            If ``True``, render the field of view of each attached sensor
            as a translucent 3D volume.  Requires the *cesium-sensor-volumes*
            plugin bundled with the viewer.  Defaults to ``False``.
        sensor_length : float | None
            Sensor volume radius in metres (distance from spacecraft to
            the far end of the cone/pyramid).  Defaults to the perigee
            distance ``a * (1 - e)``.
        show_model : bool
            If ``True`` (default), render a 3D model at the spacecraft
            position.  When ``False``, a coloured point is shown instead.
        model : str | None
            Path to a glTF model file (``.glb`` or ``.gltf``).  When
            ``None`` and *show_model* is ``True``, a default bevelled
            cube with embossed face labels is rendered.
        scale : float
            Uniform scale factor for the 3D model.  Default 1.0.
        """
        from ._czml import build_spacecraft_packets

        self._sc_counter += 1
        packet_id = f"sc-{self._sc_counter}"

        # Resolve model path: if a bare filename is passed, look in the
        # bundled _static/Models directory before falling back to CWD/abs.
        resolved_model = model
        if resolved_model is not None:
            if not os.path.isabs(resolved_model) and not os.path.isfile(resolved_model):
                builtin_models = os.path.join(
                    os.path.dirname(__file__), "_static", "Models"
                )
                candidate = os.path.join(builtin_models, resolved_model)
                if os.path.isfile(candidate):
                    resolved_model = candidate

        packets, model_path = build_spacecraft_packets(
            spacecraft,
            t_start,
            t_end,
            step,
            color=color,
            path_color=path_color,
            label=label,
            packet_id=packet_id,
            show_model=show_model,
            model=resolved_model,
            scale=scale,
        )
        self._update_time_range(t_start, t_end)
        self._packets.extend(packets)

        if model_path is not None:
            self._model_files[packet_id] = model_path
            if model is not None:
                base_dir = os.path.dirname(model_path)
                bin_name = os.path.splitext(os.path.basename(model_path))[0]
                bin_path = os.path.join(base_dir, f"{bin_name}.bin")
                tex_candidates = [
                    os.path.join(base_dir, f"{bin_name}.png"),
                    os.path.join(base_dir, f"{bin_name}.jpg"),
                    os.path.join(base_dir, f"{bin_name}_0.png"),
                    os.path.join(base_dir, f"{bin_name}_0.jpg"),
                ]
                for tex_path in tex_candidates:
                    if os.path.isfile(tex_path):
                        self._model_files[
                            f"{packet_id}_tex_{os.path.basename(tex_path)}"
                        ] = tex_path

        if show_sensors and packets:
            from ._czml import build_sensor_packets

            length = sensor_length
            if length is None:
                length = spacecraft.a * (1.0 - spacecraft.e)

            state = spacecraft.propagate(t_start, t_end, step)
            epoch = state["t"][0]
            self._sensor_packets.extend(
                build_sensor_packets(
                    spacecraft,
                    state["r"],
                    state["v"],
                    state["t"],
                    epoch,
                    packet_id,
                    color,
                    length,
                )
            )

    def add_ground_station(
        self,
        ground_station: GroundStation,
        *,
        color: tuple[int, int, int, int] = (255, 220, 50, 255),
        label: str | None = None,
    ) -> None:
        """Add a ground station to the visualization.

        Parameters
        ----------
        ground_station : GroundStation
        color : tuple[int, int, int, int]
            RGBA colour for the station icon.
        label : str | None
            Display label.
        """
        from ._czml import build_groundstation_packet

        self._gs_counter += 1
        packet_id = f"gs-{self._gs_counter}"
        packet = build_groundstation_packet(
            ground_station,
            color=color,
            label=label,
            packet_id=packet_id,
        )
        self._packets.append(packet)

    def add_aoi(
        self,
        aoi: AoI,
        *,
        color: tuple[int, int, int, int] = (255, 80, 80, 80),
        label: str | None = None,
    ) -> None:
        """Add an area of interest to the visualization.

        Parameters
        ----------
        aoi : AoI
        color : tuple[int, int, int, int]
            RGBA fill colour.
        label : str | None
            Display label.
        """
        from ._czml import build_aoi_packets

        self._aoi_counter += 1
        packet_id = f"aoi-{self._aoi_counter}"
        packets = build_aoi_packets(
            aoi,
            color=color,
            label=label,
            packet_id=packet_id,
        )
        self._packets.extend(packets)

    def to_czml(self) -> str:
        """Serialize all added objects to a CZML JSON string.

        Returns
        -------
        str
            CZML JSON document.
        """
        self._ensure_preamble()
        from ._czml import build_czml_document

        extra = self._sensor_packets or None
        return build_czml_document(self._packets, extra_packets=extra)

    def show(self, block: bool = True) -> None:
        """Open the pywebview window and render the scene.

        Parameters
        ----------
        block : bool
            If ``True`` (default), blocks until the window is closed.
            If ``False``, returns immediately after starting the GUI loop
            in a background thread.
        """
        try:
            import webview
        except ImportError as exc:
            raise ImportError(
                "pywebview is required for the 3D viewer. "
                "Install it with:  pip install missiontools[cesium]"
            ) from exc

        self._ensure_preamble()
        czml_json = self.to_czml()

        template_dir = os.path.join(os.path.dirname(__file__), "_templates")
        html_path = os.path.join(template_dir, "viewer.html")

        with open(html_path) as f:
            html = f.read()

        html = html.replace(
            "var MISSIONTOOLS_CZML = null;",
            f"var MISSIONTOOLS_CZML = {czml_json};",
        )

        static_dir = os.path.join(os.path.dirname(__file__), "_static")

        tmp_dir = tempfile.mkdtemp(prefix="missiontools_cesium_")
        tmp_html = os.path.join(tmp_dir, "index.html")
        with open(tmp_html, "w") as f:
            f.write(html)

        static_cesium = os.path.join(static_dir, "Cesium")
        if os.path.isdir(static_cesium):
            import shutil

            shutil.copytree(
                static_cesium, os.path.join(tmp_dir, "Cesium"), dirs_exist_ok=True
            )

        if self._model_files:
            models_dir = os.path.join(tmp_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            for key, src_path in self._model_files.items():
                dest = os.path.join(models_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dest)
                if key.endswith("_tex_"):
                    continue
                base = os.path.splitext(src_path)[0]
                bin_src = base + ".bin"
                if os.path.isfile(bin_src):
                    shutil.copy2(
                        bin_src, os.path.join(models_dir, os.path.basename(bin_src))
                    )
                for tex_src in [
                    base + "_0.png",
                    base + "_0.jpg",
                    base + ".png",
                    base + ".jpg",
                ]:
                    if os.path.isfile(tex_src):
                        shutil.copy2(
                            tex_src, os.path.join(models_dir, os.path.basename(tex_src))
                        )

        server = _start_http_server(tmp_dir)
        port = server.server_address[1]
        url = f"http://localhost:{port}/index.html"
        print(f"[missiontools] Cesium viewer at {url}")

        webview.create_window(
            self._title,
            url,
            width=1400,
            height=900,
            text_select=True,
        )

        if block:
            webview.start()
            server.shutdown()
        else:
            webview.start(threaded=True)
