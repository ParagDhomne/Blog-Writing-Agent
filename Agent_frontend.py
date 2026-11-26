from __future__ import annotations

import json
import re
import traceback
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

try:
    from Agent_backend import OUTPUT_DIR, app, build_initial_state, safe_slug
except Exception as import_error:  # e.g. missing OPENAI_API_KEY or package
    st.error(f"Could not load the backend: {import_error}")
    st.info("Check your `.env` (OPENAI_API_KEY, TAVILY_API_KEY) and installed packages, then reload.")
    st.stop()

BASE_DIR = OUTPUT_DIR  # markdown files and their relative `images/...` links live here

st.session_state.setdefault("logs", [])
st.session_state.setdefault("last_out", None)


def log(message: str) -> None:
    st.session_state["logs"].append(message)


# -----------------------------[ Markdown renderer that supports local images ]-----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str, base_dir: Path) -> Optional[Path]:
    """Resolve a markdown image path against the blog's folder; refuse anything outside it."""
    base = base_dir.resolve()
    candidate = Path(src.strip())
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(base)
    except (ValueError, OSError):
        return None
    return resolved


def referenced_images(md: str, base_dir: Path) -> List[Path]:
    """Local images that this specific blog actually references (and that exist on disk)."""
    found: List[Path] = []
    for match in _MD_IMG_RE.finditer(md):
        src = match.group("src").strip().strip('"').strip("'")
        if src.startswith(("http://", "https://")):
            continue
        path = _resolve_image_path(src, base_dir)
        if path is not None and path.is_file() and path not in found:
            found.append(path)
    return found


def render_markdown_with_local_images(md: str, base_dir: Path) -> None:
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for match in matches:
        before = md[last : match.start()]
        if before:
            parts.append(("md", before))

        alt = (match.group("alt") or "").strip()
        src = match.group("src").strip().strip('"').strip("'")
        parts.append(("img", f"{alt}|||{src}"))
        last = match.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    # Turn a following *caption* line into the image caption.
    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "img":
            alt, src = payload.split("|||", 1)
            caption = None

            if i + 1 < len(parts) and parts[i + 1][0] == "md":
                nxt = parts[i + 1][1].lstrip()
                lines = nxt.splitlines()
                if lines:
                    caption_match = _CAPTION_LINE_RE.match(lines[0].strip())
                    if caption_match:
                        caption = caption_match.group("cap").strip()
                        rest = "\n".join(lines[1:])
                        parts[i + 1] = ("md", rest)

            if src.startswith(("http://", "https://")):
                st.image(src, caption=caption or (alt or None), width="stretch")
            else:
                image_path = _resolve_image_path(src, base_dir)
                if image_path is not None and image_path.is_file():
                    st.image(str(image_path), caption=caption or (alt or None), width="stretch")
                else:
                    st.warning(f"Image not found: `{src}`")
            i += 1
        else:
            if payload.strip():
                st.markdown(payload, unsafe_allow_html=False)
            i += 1


# ---------------------------------------------[ Downloads ]---------------------------------------------
def _build_zip(
    files: List[Path],
    base_dir: Path,
    md_text: Optional[str] = None,
    md_filename: Optional[str] = None,
) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if md_text is not None and md_filename:
            zf.writestr(md_filename, md_text.encode("utf-8"))
        for path in files:
            # Keep paths relative to the blog folder so `images/x.png` links keep working.
            zf.write(path, arcname=str(path.relative_to(base_dir.resolve())))
    return buf.getvalue()


def bundle_zip(md_text: str, md_filename: str, images: List[Path], base_dir: Path) -> bytes:
    return _build_zip(images, base_dir, md_text=md_text, md_filename=md_filename)


def images_zip(images: List[Path], base_dir: Path) -> bytes:
    return _build_zip(images, base_dir)


# ----------------------------------------------[ Past blogs ]-----------------------------------------------
def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def list_past_blogs() -> List[Path]:
    if not BASE_DIR.exists():
        return []
    files = [p for p in BASE_DIR.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


@st.cache_data(show_spinner=False)
def _cached_title(path_str: str, mtime: float) -> str:  # mtime is part of the cache key
    path = Path(path_str)
    return extract_title_from_md(path.read_text(encoding="utf-8", errors="replace"), path.stem)


def read_md_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------[ Streaming ]---------------------------------------------
def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Run the graph ONCE. Yields ("updates", payload) per step, then ("final", final_state).
    Errors propagate to the caller instead of being swallowed (and the graph is never re-run).
    """
    final_state: Optional[Dict[str, Any]] = None
    for mode, chunk in graph_app.stream(inputs, stream_mode=["updates", "values"]):
        if mode == "updates":
            yield ("updates", chunk)
        elif mode == "values":
            final_state = chunk

    if final_state is None:
        raise RuntimeError("The graph finished without producing any output.")
    yield ("final", final_state)


def _plan_to_dict(plan: Any) -> Optional[Dict[str, Any]]:
    if plan is None:
        return None
    if isinstance(plan, dict):
        return plan
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    return None


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if not isinstance(step_payload, dict):
        return current_state

    for _node_name, update in step_payload.items():
        if not isinstance(update, dict):
            continue
        for key, value in update.items():
            if key == "sections" and isinstance(value, list):
                # `sections` is an additive reducer in the graph; mirror that here.
                current_state["sections"] = list(current_state.get("sections", [])) + value
            else:
                current_state[key] = value
    return current_state


# --------------------------------------------------[ UI ]--------------------------------------------------
st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area(
        "Topic",
        height=120,
        key="topic",
        placeholder="e.g. Self-attention in the Transformer architecture",
    )
    as_of = st.date_input("As-of date", value=date.today())
    recency_days = st.number_input(
        "Recency window (days)",
        min_value=1,
        max_value=365,
        value=7,
        step=1,
        help="How far back 'recent' reaches for news-style topics.",
    )
    run_btn = st.button("🚀 Generate Blog", type="primary")

    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found yet.")
        selected_path: Optional[Path] = None
    else:
        labels: List[str] = []
        by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            label = f"{_cached_title(str(p), p.stat().st_mtime)}  ·  {p.name}"
            labels.append(label)
            by_label[label] = p

        selected_label = st.radio("Select a blog to load", options=labels, index=0, label_visibility="collapsed")
        selected_path = by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_path:
                md_text = read_md_file(selected_path)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.session_state["logs"] = [f"Loaded blog: {selected_path}"]

tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs = build_initial_state(topic.strip(), as_of.isoformat(), int(recency_days))
    st.session_state["logs"] = []

    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node: Optional[str] = None

    try:
        for kind, payload in try_stream(app, inputs):
            if kind == "updates":
                if isinstance(payload, dict) and payload:
                    last_node = next(iter(payload.keys()))
                current_state = extract_latest_state(current_state, payload)

                plan_dict = _plan_to_dict(current_state.get("plan"))
                summary = {
                    "node": last_node,
                    "mode": current_state.get("mode"),
                    "needs_research": current_state.get("needs_research"),
                    "queries": (current_state.get("queries") or [])[:5],
                    "evidence_count": len(current_state.get("evidence") or []),
                    "tasks": len(plan_dict["tasks"]) if plan_dict and "tasks" in plan_dict else None,
                    "sections_done": len(current_state.get("sections") or []),
                }
                progress_area.json(summary)
                log(f"[updates] {json.dumps(payload, default=str)[:1200]}")

            elif kind == "final":
                st.session_state["last_out"] = payload
                status.update(label="✅ Done", state="complete", expanded=False)
                log("[final] received final state")
    except Exception as error:
        status.update(label="❌ Failed", state="error", expanded=True)
        st.error(f"Run failed: {type(error).__name__}: {error}")
        log(traceback.format_exc())

out = st.session_state.get("last_out")

if out:
    plan_dict = _plan_to_dict(out.get("plan"))
    final_md = out.get("final") or ""
    blog_title = (plan_dict or {}).get("blog_title") or extract_title_from_md(final_md, "blog")

    # --- Plan tab ---
    with tab_plan:
        st.subheader("Plan")
        if not plan_dict:
            st.info("No plan found in output.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.write("**Title:**", plan_dict.get("blog_title"))
                st.write("**Audience:**", plan_dict.get("audience"))
                st.write("**Tone:**", plan_dict.get("tone"))
                st.write("**Blog kind:**", plan_dict.get("blog_kind"))
            with c2:
                st.write("**Constraints:**")
                st.write(plan_dict.get("constraints", []))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags", [])),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, width="stretch", hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (closed_book mode or no results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.subheader("Markdown Preview")
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md, BASE_DIR)

            md_filename = f"{safe_slug(blog_title)}.md"
            images = referenced_images(final_md, BASE_DIR)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                )
            with col2:
                st.download_button(
                    "📦 Download Bundle (MD + images)",
                    data=bundle_zip(final_md, md_filename, images, BASE_DIR),
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                )

    # --- Images tab ---
    with tab_images:
        st.subheader("Images")
        specs = out.get("image_specs") or []
        images = referenced_images(final_md, BASE_DIR)

        if not specs and not images:
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if images:
                st.write("**Images used in this blog:**")
                for img in images:
                    st.image(str(img), caption=img.name, width="stretch")

                st.download_button(
                    "⬇️ Download Images (zip)",
                    data=images_zip(images, BASE_DIR),
                    file_name="images.zip",
                    mime="application/zip",
                )
else:
    with tab_plan:
        st.info("Enter a topic and click **Generate Blog**, or load a past blog from the sidebar.")

# --- Logs tab (always available, so errors are visible even when a run fails) ---
with tab_logs:
    st.subheader("Logs")
    st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
