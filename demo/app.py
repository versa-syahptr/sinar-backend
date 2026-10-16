"""SINAR step-by-step demo: detection -> tracking -> centroid matrix -> classification.

    uv run streamlit run demo/app.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # before sinar pulls in tensorflow

import importlib.util
from functools import partial
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import records
from records import VID_DIR, DemoRun, WindowRec

# recording needs the full sinar stack (pipeline.py); the hosted viewer only plays recordings
CAN_RECORD = importlib.util.find_spec("sinar") is not None
STEPS = ["1 Detection", "2 Tracking", "3 Centroids → matrix", "4 Classification"]
STEP_ICONS = [":material/center_focus_strong:", ":material/route:", ":material/grid_on:", ":material/gavel:"]

THUMB_STRIDE = 15  # greedy makes a window per start frame; thumbnails show every 15th

st.set_page_config(page_title="SINAR demo", page_icon=":material/two_wheeler:", layout="wide")
st.logo(str(Path(__file__).parent / "logo.svg"))


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Loading YOLO + CNN (first OpenVINO compile is slow)…")
def get_sinar(yolo: str, cnn: str, device: str):
    import sinar.logger
    sinar.logger.set_level("warning")
    import pipeline
    return pipeline.load_sinar(yolo, cnn, device)


@st.cache_data(max_entries=8, show_spinner="Loading recording…")
def load_run(path: str, mtime: float) -> DemoRun:
    return DemoRun.load(Path(path))


def render(rec: DemoRun):
    import pipeline
    bar = st.progress(0.0, text="Rendering videos…")
    pipeline.render_videos(rec, progress=lambda i, n: bar.progress(i / max(n, 1), text=f"Rendering frame {i}/{n}"))
    bar.empty()


# ---------- state ----------
st.session_state.setdefault("frame", 0)
st.session_state.setdefault("step", STEPS[0])


def reset_frame():
    st.session_state.frame = 0
    st.session_state.pop("win", None)


def go_step(delta: int):
    i = STEPS.index(st.session_state.step or STEPS[0])
    st.session_state.step = STEPS[min(max(i + delta, 0), len(STEPS) - 1)]


def jump_to_window(win: WindowRec, step: str = STEPS[2]):
    st.session_state.win = win.index
    st.session_state.frame = win.start
    st.session_state.step = step


def shift_window(delta: int):
    i = min(max(st.session_state.win + delta, 0), len(rec.windows) - 1)
    jump_to_window(rec.windows[i])


def set_frame_from(key: str):
    st.session_state.frame = st.session_state[key]


# ---------- sidebar ----------
with st.sidebar:
    st.header("SINAR demo")
    clips = sorted(p for p in VID_DIR.glob("*.mp4") if not records.is_rendered(p))
    if not clips:
        st.error(f"No clips in `{VID_DIR}`. Set `SINAR_DEMO_VID_DIR` to the folder with the clips and recordings.")
        st.stop()
    clip = st.selectbox("Clip", clips, format_func=lambda p: p.name, on_change=reset_frame)
    if CAN_RECORD:
        import pipeline
        with st.expander("Models", icon=":material/tune:"):
            yolo = st.text_input("YOLO", pipeline.DEFAULT_YOLO)
            cnn = st.text_input("Classifier", pipeline.DEFAULT_CNN)
            device = st.text_input("Device", "intel:gpu")

    pkl = records.cache_path(clip)
    rec = None
    if pkl.exists():
        try:
            rec = load_run(str(pkl), pkl.stat().st_mtime)
        except ValueError:  # recording from an older version of the demo
            pass
    if rec is None:
        st.warning("This clip has not been processed yet (or was recorded by an older version).",
                   icon=":material/hourglass_empty:")
        if not CAN_RECORD:
            st.caption("Record it where SINAR is installed (`uv run python demo/pipeline.py vid/*.mp4`) "
                       "and copy the `.demo.pkl`, `.detect.mp4` and `.track.mp4` files here.")
        elif st.button("Process", icon=":material/play_arrow:", type="primary", width="stretch"):
            s = get_sinar(yolo, cnn, device)
            bar = st.progress(0.0, text="Processing…")
            rec = pipeline.run(s, clip, progress=lambda i, n: bar.progress(min(i / max(n, 1), 1.0),
                                                                           text=f"Frame {i}/{n}"))
            rec.save(pkl)
            bar.empty()
            render(rec)
            st.rerun()
        st.stop()

    rec.clip = str(clip)  # recordings store the clip path they were made with; use where it is now
    vi = rec.video_info
    last = len(rec.frames) - 1
    st.session_state.frame = min(st.session_state.frame, last)

    frame_idx = st.slider("Frame", 0, last, key="frame",
                          help="Shared by every step, so all pages show the same moment.")
    st.caption(f"t = {frame_idx / vi.fps:.2f} s · {vi.width}×{vi.height} @ {vi.fps} fps · "
               f"{len(rec.windows)} windows")
    with st.container(horizontal=True):
        st.button("Prev", icon=":material/chevron_left:", on_click=go_step, args=(-1,), width="stretch")
        st.button("Next", icon=":material/chevron_right:", on_click=go_step, args=(1,), width="stretch")

    st.divider()
    st.markdown("**· SINAR ·**", text_alignment="center")
    st.caption("**_Sistem Inteligen Anti Kejahatan Jalan Raya_**", text_alignment="center")
    st.caption(
        ":material/description: Implementation of the paper  \n"
        "> *Detecting Motorcycle Crime Gangs in CCTV Video Footage Using YOLOv9 and CNN*  \n\n"
        "ICICyTA 2024 · [doi:10.1109/ICICYTA64807.2024.10913459]"
        "(https://doi.org/10.1109/ICICYTA64807.2024.10913459)"
    )
    st.link_button("GitHub", url="https://github.com/versa-syahptr/sinar-backend", icon=":material/code:", width="stretch")

step = st.segmented_control("Step", STEPS, key="step", required=True, label_visibility="collapsed",
                            format_func=lambda s: f"{STEP_ICONS[STEPS.index(s)]} {s}", width="stretch")
fr = rec.frames[frame_idx]

# selected window: kept while the frame stays inside it, else the thumbnail window at the frame
win = None
if rec.windows:
    w = st.session_state.get("win")
    if w is None or w >= len(rec.windows) or not rec.windows[w].start <= frame_idx <= rec.windows[w].end:
        w = min(frame_idx - frame_idx % THUMB_STRIDE, len(rec.windows) - 1)
    st.session_state.win = w
    win = rec.windows[w]


# ---------- helpers ----------
def video_player(kind: str, alt_text: str):
    path = records.video_paths(clip)[kind]
    if not path.exists():
        st.warning("The annotated video for this clip has not been rendered yet.", icon=":material/movie:")
        if CAN_RECORD and st.button("Render videos", icon=":material/movie:", type="primary"):
            render(rec)
            st.rerun()
        return
    st.video(path, start_time=int(frame_idx / vi.fps), muted=True, alt=alt_text)
    st.caption("Starts at the sidebar frame (rounded down to the second). The overlay shows the frame number.")


def heatmap_data(win: WindowRec) -> pd.DataFrame:
    n = win.cnn_input.shape[0]
    return pd.DataFrame([(win.index, r, c, float(win.cnn_input[r, c])) for r in range(n) for c in range(n)],
                        columns=["window", "row", "col", "value"])


def heatmap(data: pd.DataFrame, size: int, legend: bool = True) -> alt.Chart:
    """30x30 CNN input as an exactly square heatmap of `size` px."""
    return alt.Chart(data).mark_rect().encode(
        x=alt.X("col:O", axis=None),
        y=alt.Y("row:O", axis=None),
        color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                        legend=alt.Legend(title=None) if legend else None),
        tooltip=["window", "row", "col", alt.Tooltip("value:Q", format=".3f")],
    ).properties(width=size, height=size)


def picked_window(key: str) -> WindowRec | None:
    points = st.session_state[key].selection.get("pick", [])
    return rec.windows[int(points[0]["window"])] if points else None


def cursor(x: int, field: str = "frame") -> alt.Chart:
    return alt.Chart(pd.DataFrame({field: [x]})).mark_rule(color="red", strokeDash=[4, 3]).encode(x=f"{field}:Q")


# ---------- steps ----------
if step == STEPS[0]:
    st.subheader("Detection — YOLOv9m, single class `motor`")
    left, right = st.columns([3, 2])
    with left:
        video_player("detect", "Clip with YOLO detection boxes and confidences")
    with right:
        st.metric("Motorcycles in this frame", len(fr.conf))
        if len(fr.conf):
            st.caption(f"Confidence {fr.conf.min():.2f} – {fr.conf.max():.2f}")
        counts = pd.DataFrame({"frame": [f.frame_idx for f in rec.frames],
                               "motorcycles": [len(f.conf) for f in rec.frames]})
        chart = alt.Chart(counts).mark_line().encode(x="frame:Q", y="motorcycles:Q") + cursor(frame_idx)
        st.altair_chart(chart, alt="Motorcycles detected per frame")
        st.info("Every frame goes through the detector on its own: boxes and confidences only, "
                "no notion of which motorcycle is which yet.", icon=":material/info:")

elif step == STEPS[1]:
    st.subheader("Tracking — ByteTrack keeps one ID per motorcycle")
    left, right = st.columns([3, 2])
    with left:
        video_player("track", "Clip with ByteTrack IDs and trails")
    with right:
        st.metric("Tracked IDs in this frame", 0 if fr.ids is None else len(fr.ids))
        life = pd.DataFrame([(f.frame_idx, int(t)) for f in rec.frames if f.ids is not None for t in f.ids],
                            columns=["frame", "id"])
        if life.empty:
            st.warning("No tracks in this clip.")
        else:
            spans = life.groupby("id")["frame"].agg(start="min", end="max").reset_index()
            chart = alt.Chart(spans).mark_bar(height=6).encode(
                x=alt.X("start:Q", title="frame"), x2="end:Q",
                y=alt.Y("id:O", title="track ID", sort="ascending"),
                tooltip=["id", "start", "end"],
            ) + cursor(frame_idx, "start")
            st.altair_chart(chart, alt="Lifespan of every track ID across the clip")
        st.info("Same frame as step 1, but each box now has an identity that persists across frames; "
                f"trails show the last {records.TRAIL} frames.", icon=":material/info:")

elif step == STEPS[2]:
    st.subheader("Centroids → matrix — the classifier's input")
    if win is None:
        st.warning("The clip is too short to fill a single 30-row window.")
        st.stop()
    st.caption(f"Greedy mode: a 30-row window starts at **every** frame ({len(rec.windows)} windows, "
               f"heavily overlapping). The grid shows every {THUMB_STRIDE}th; ◀ ▶ step one frame.")

    # carousel: every THUMB_STRIDE-th window's 30x30 input, click one to open it
    shown = rec.windows[::THUMB_STRIDE]
    thumbs = pd.concat([heatmap_data(w) for w in shown])
    thumbs["label"] = thumbs["window"].map({w.index: f"f{w.start} · {w.prob:.2f}" for w in shown})
    thumbs["current"] = thumbs["window"] == win.index - win.index % THUMB_STRIDE
    pick = alt.selection_point(fields=["window"], name="pick")
    carousel = heatmap(thumbs, 64, legend=False).encode(
        opacity=alt.condition("datum.current", alt.value(1), alt.value(0.35)),
    ).add_params(pick).facet(
        facet=alt.Facet("label:N", title=None, sort=alt.SortField("window"),
                        header=alt.Header(labelFontSize=10)),
        columns=20, spacing=8,
    )

    def on_carousel_pick():
        if (w := picked_window("carousel")) is not None:
            jump_to_window(w)

    st.markdown("**Windows** — start frame · P(gang); click one to open it")
    st.altair_chart(carousel, key="carousel", on_select=on_carousel_pick, selection_mode="pick",
                    width="content", alt="The 30 by 30 classifier input of every 15th window; the current one is highlighted")

    with st.container(horizontal=True, vertical_alignment="center"):
        st.button("−1 frame", icon=":material/chevron_left:", on_click=shift_window, args=(-1,), help="Window one frame earlier")
        st.markdown(f"**Window {win.index}** · frames {win.start}–{win.end} · p = {win.prob:.3f} "
                    + (":red-badge[gang]" if win.is_gang else ":blue-badge[normal]"))
        st.button("+1 frame", icon=":material/chevron_right:", on_click=shift_window, args=(1,), help="Window one frame later")
    in_window = win.start <= frame_idx <= win.end
    row = max((i for i, f in enumerate(win.sampled_frames) if f <= frame_idx), default=None) if in_window else None

    # row-by-row slider over this window's sampled frames, kept in step with the shared frame
    row_key = f"row_{win.index}"
    st.session_state[row_key] = win.sampled_frames[row if row is not None else 0]
    st.select_slider("Matrix row", win.sampled_frames, key=row_key, on_change=set_frame_from, args=(row_key,),
                     format_func=lambda f: f"row {win.sampled_frames.index(f)} · frame {f}")
    st.caption("Every 5th frame of the window is a row · rows = time, column pairs = one motorcycle's (x, y) · "
               "zero = motorcycle absent")
    marks = pd.DataFrame({"frame": win.sampled_frames})
    strip = alt.Chart(marks).mark_tick(thickness=2).encode(
        x=alt.X("frame:Q", scale=alt.Scale(domain=[0, last])), tooltip=["frame"],
    ).properties(height=50) + cursor(frame_idx)
    st.altair_chart(strip, alt="The frames sampled into this window, along the clip")

    table, plots = st.columns([3, 2])
    with table:
        m = win.matrix.copy()
        m.index = pd.Index(win.sampled_frames + [None] * (len(m) - len(win.sampled_frames)), name="frame")
        styled = m.style.format("{:.3f}")
        if row is not None:
            styled = styled.apply(lambda r: ["background-color: rgba(255, 75, 75, 0.25)"] * len(r)
                                  if r.name == win.sampled_frames[row] else [""] * len(r), axis=1)
        st.markdown(f"**Raw window matrix** — {m.shape[0]}×{m.shape[1]}"
                    + (f", {m.shape[0] - len(win.sampled_frames)} empty rows past the clip end"
                       if len(win.sampled_frames) < m.shape[0] else "")
                    + (f", current row {row}" if row is not None else ""))
        st.dataframe(styled, height=420, alt=f"Window {win.index} centroid matrix")

    with plots:
        long = []
        for c in range(0, win.matrix.shape[1], 2):
            tid = win.matrix.columns[c][1:]
            for r, (x, y) in enumerate(win.matrix.iloc[:, c:c + 2].values):
                if x or y:
                    long.append({"id": tid, "row": r, "x": x, "y": y})
        long = pd.DataFrame(long)
        if not long.empty:
            traj = alt.Chart(long).mark_line(point=True).encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[1, 0])),  # image coordinates, y down
                color=alt.Color("id:N", title="ID"), order="row:Q", tooltip=["id", "row", "x", "y"],
            ).properties(height=260, title="Normalized trajectories in this window")
            st.altair_chart(traj, alt="Trajectories of each motorcycle in the window")

        n = win.cnn_input.shape[0]
        heat = pd.DataFrame([(r, c, float(win.cnn_input[r, c])) for r in range(n) for c in range(n)],
                            columns=["row", "col", "value"])
        hm = alt.Chart(heat).mark_rect().encode(
            x=alt.X("col:O", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("row:O", axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1])),
            tooltip=["row", "col", "value"],
        ).properties(height=260, title="30×30 CNN input (zero-padded / cut at 15 tracks)")
        if row is not None:
            hm += alt.Chart(pd.DataFrame({"row": [row]})).mark_rect(
                fill=None, stroke="red", strokeWidth=2).encode(y="row:O")
        st.altair_chart(hm, alt="Heatmap of the 30 by 30 matrix fed to the CNN")

elif step == STEPS[3]:
    st.subheader("Classification — CNN1-set-D49, ~7k parameters")
    if win is None:
        st.warning("The clip is too short to fill a single 30-row window.")
        st.stop()
    n_gang = sum(w.is_gang for w in rec.windows)
    with st.container(horizontal=True):
        st.metric("Verdict", "GANG" if rec.is_gang else "normal", border=True,
                  help="Greedy mode: mean P(gang) over all windows ≥ threshold")
        st.metric("Mean P(gang)", f"{rec.mean_prob:.3f}", border=True)
        st.metric("Windows above threshold", f"{n_gang} / {len(rec.windows)}", border=True)
        st.metric("Threshold", f"{rec.threshold:.2f}", border=True)

    curve = pd.DataFrame({"window": [w.index for w in rec.windows], "start": [w.start for w in rec.windows],
                          "prob": [w.prob for w in rec.windows]})
    pick = alt.selection_point(fields=["window"], name="pick", on="click", nearest=True)
    line = alt.Chart(curve).mark_line(point=alt.OverlayMarkDef(size=12)).encode(
        x=alt.X("start:Q", title="window start frame", scale=alt.Scale(domain=[0, last])),
        y=alt.Y("prob:Q", scale=alt.Scale(domain=[0, 1]), title="P(gang)"),
        tooltip=["window", "start", alt.Tooltip("prob:Q", format=".3f")],
    ).add_params(pick)
    rules = alt.Chart(pd.DataFrame({"y": [rec.threshold, rec.mean_prob], "what": ["threshold", "mean"]})).mark_rule(
        strokeDash=[4, 3]).encode(y="y:Q", color=alt.Color("what:N", title=None,
                                                         scale=alt.Scale(range=["#888", "#e4572e"])))
    sel = alt.Chart(pd.DataFrame({"start": [win.start]})).mark_rule(color="red").encode(x="start:Q")

    def on_curve_pick():
        if (w := picked_window("curve")) is not None:
            jump_to_window(w)

    st.altair_chart((line + rules + sel).properties(height=220), key="curve", on_select=on_curve_pick,
                    selection_mode="pick", alt="Gang probability of every window over its start frame")
    st.caption("One point per window start frame. Click the curve to open that window in step 3.")

    def on_thumb_pick(key: str):
        if (w := picked_window(key)) is not None:
            jump_to_window(w)

    widths = [1.3, 0.3, 5, 2]
    head = st.columns(widths)
    head[0].caption("30×30 input")
    head[2].caption(f"P(gang) · dashed line = threshold {rec.threshold:.2f}")
    head[3].caption(f"Window (every {THUMB_STRIDE}th)")
    rows = st.container(height=520)
    for w in rec.windows[::THUMB_STRIDE]:
        thumb, arrow, bar, verdict = rows.columns(widths, vertical_alignment="center")
        with thumb:
            key = f"thumb_{w.index}"
            st.altair_chart(heatmap(heatmap_data(w), 90, legend=False).add_params(
                alt.selection_point(fields=["window"], name="pick")),
                key=key, on_select=partial(on_thumb_pick, key), selection_mode="pick", width="content",
                alt=f"Window {w.index} 30 by 30 classifier input")
        arrow.markdown(":material/arrow_forward:")
        with bar:
            prob = pd.DataFrame({"prob": [w.prob], "verdict": ["gang" if w.is_gang else "normal"]})
            chart = alt.Chart(prob).mark_bar(height=28).encode(
                x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                        axis=alt.Axis(format=".1f", tickCount=5)),
                color=alt.Color("verdict:N", legend=None,
                                scale=alt.Scale(domain=["gang", "normal"], range=["#e4572e", "#4c78a8"])),
                tooltip=[alt.Tooltip("prob:Q", format=".3f"), "verdict"],
            ) + alt.Chart(pd.DataFrame({"t": [rec.threshold]})).mark_rule(strokeDash=[4, 3]).encode(x="t:Q")
            st.altair_chart(chart.properties(height=60), alt=f"Window {w.index} gang probability {w.prob:.3f}")
        with verdict:
            st.markdown(f"**p = {w.prob:.3f}**")
            if w.is_gang:
                st.badge("gang", color="red")
            else:
                st.badge("normal", color="blue")
            st.caption(f"frames {w.start}–{w.end} · {w.start / vi.fps:.1f}–{w.end / vi.fps:.1f} s")
    st.caption("Click a vector to open that window in step 3.")
