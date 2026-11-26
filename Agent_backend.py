from __future__ import annotations

import base64
import operator
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import OpenAI
from pydantic import BaseModel, Field

# Prefer the maintained `langchain-tavily` package; fall back to the old community tool.
try:
    from langchain_tavily import TavilySearch

    _NEW_TAVILY = True
except ImportError:  # pragma: no cover
    from langchain_community.tools.tavily_search import TavilySearchResults

    _NEW_TAVILY = False

load_dotenv()

# ---------------------------------------------[ Config ]------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2.6-sunburst")

MAX_QUERIES = 10          # max web-search queries per run
RESULTS_PER_QUERY = 6     # Tavily results per query
SNIPPET_CHARS = 400       # snippet length kept per evidence item
MAX_IMAGES = 3            # max generated images per blog
MAX_PARALLEL = 5          # threads for search / image generation

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
IMAGES_DIR = OUTPUT_DIR / "images"


def safe_slug(title: str) -> str:
    """Windows-safe file name stem for a blog title (shared with the frontend)."""
    name = re.sub(r'[<>:"/\\|?*]', "", title).strip()
    name = re.sub(r"\s+", "_", name).strip(" ._")
    return name or "blog"


# --------------------------------------------[ Schemas ]------------------------------------------------
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: list[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
        ...,
        ge=120,
        le=550,
        description="Target word count for this section (120–550).",
    )

    tags: list[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str = Field(..., description="Who this blog is for.")
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp).")
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: list[str] = Field(default_factory=list)
    tasks: list[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: list[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(default="", description="Leave empty; assigned in code (e.g. [[IMAGE_1]]).")
    after_heading: str = Field(
        ...,
        description="EXACT heading line of the section the image belongs in, e.g. '## How Attention Works'.",
    )
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    images: list[ImageSpec] = Field(default_factory=list)


# -----------------------------------------------[ State ]-----------------------------------------------------
class State(TypedDict):
    topic: str
    as_of: str            # ISO date used for "latest / this week" style topics
    recency_days: int     # how far back "recent" reaches

    # routing / research
    mode: str
    needs_research: bool
    queries: list[str]
    evidence: list[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[list[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer / images
    merged_md: str
    md_with_placeholders: str
    image_specs: list[dict]

    final: str


def build_initial_state(topic: str, as_of: Optional[str] = None, recency_days: int = 7) -> dict:
    """Single place that defines the graph's input (used by the frontend and the CLI)."""
    return {
        "topic": topic.strip(),
        "as_of": as_of or date.today().isoformat(),
        "recency_days": int(recency_days),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }


def _as_of_date(state: dict) -> date:
    try:
        return date.fromisoformat(state.get("as_of") or "")
    except ValueError:
        return date.today()


# ---------------------------------------------[ LLM ]------------------------------------------------------
llm = ChatOpenAI(model=LLM_MODEL, max_retries=3)

# -------------------------------------------[ Router (decide upfront) ]------------------------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- Today's date and a recency window are provided. If the user asked for "last week/this week/latest",
  reflect that constraint IN THE QUERIES (use the real month/year from today's date).
"""


def router_node(state: State) -> dict:
    as_of = _as_of_date(state)
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Today's date: {as_of.isoformat()}\n"
                    f"Recency window (days): {state.get('recency_days', 7)}"
                )
            ),
        ]
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries[:MAX_QUERIES],
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# --------------------------------------------------[ Research (Tavily) ]-----------------------------------------------------------
def _normalize_results(raw) -> list[dict]:
    """Accept both the new (dict with 'results') and old (list) Tavily payloads."""
    if isinstance(raw, dict):
        raw = raw.get("results", [])
    if not isinstance(raw, list):
        return []

    normalized: list[dict] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        url = r.get("url") or ""
        if not url:
            continue
        published = r.get("published_date") or r.get("published_at")
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": url,
                "snippet": (r.get("content") or r.get("snippet") or "")[:SNIPPET_CHARS],
                "published_at": str(published) if published else None,
                "source": r.get("source"),
            }
        )
    return normalized


def _tavily_search(query: str, max_results: int = RESULTS_PER_QUERY, start_date: Optional[str] = None) -> list[dict]:
    """Run one search. Never raises: a failed query just returns no results."""
    try:
        if _NEW_TAVILY:
            tool = TavilySearch(max_results=max_results)
            payload: dict = {"query": query}
            if start_date:
                payload["start_date"] = start_date
            try:
                raw = tool.invoke(payload)
            except Exception:
                if not start_date:
                    raise
                raw = tool.invoke({"query": query})  # retry without the date filter
        else:
            raw = TavilySearchResults(max_results=max_results).invoke({"query": query})
    except Exception as error:
        print(f"[research] query failed: {query!r} ({error})")
        return []

    return _normalize_results(raw)


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url, copied EXACTLY from the raw results.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short and faithful to the raw text. Do not add facts that are not in the snippet.
- Deduplicate by URL.
"""


def _raw_to_evidence(items: list[dict]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            title=r.get("title") or r["url"],
            url=r["url"],
            published_at=r.get("published_at"),
            snippet=r.get("snippet"),
            source=r.get("source"),
        )
        for r in items
    ]


def research_node(state: State) -> dict:
    queries = [q.strip() for q in (state.get("queries") or []) if q and q.strip()][:MAX_QUERIES]
    if not queries:
        return {"evidence": []}

    # For volatile topics, only look at the recent window.
    start_date = None
    if state.get("mode") == "open_book":
        start_date = (_as_of_date(state) - timedelta(days=int(state.get("recency_days", 7)))).isoformat()

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        batches = list(pool.map(lambda q: _tavily_search(q, start_date=start_date), queries))

    # Deduplicate by URL *before* spending tokens on the extractor.
    raw_by_url: dict[str, dict] = {}
    for batch in batches:
        for r in batch:
            raw_by_url.setdefault(r["url"], r)
    raw_results = list(raw_by_url.values())

    if not raw_results:
        return {"evidence": []}

    evidence: list[EvidenceItem] = []
    try:
        pack = llm.with_structured_output(EvidencePack).invoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(content=f"Raw results:\n{raw_results}"),
            ]
        )
        seen: set[str] = set()
        for e in pack.evidence:
            raw = raw_by_url.get(e.url)
            if raw is None or e.url in seen:  # drops URLs the model invented
                continue
            seen.add(e.url)
            if not e.snippet:
                e.snippet = raw["snippet"]
            evidence.append(e)
    except Exception as error:
        print(f"[research] evidence extraction failed, using raw results: {error}")

    if not evidence:
        evidence = _raw_to_evidence(raw_results[:12])

    return {"evidence": evidence}


# ---------------------------------------------------[ Orchestrator (Plan) ]-----------------------------------------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"Today's date: {_as_of_date(state).isoformat()}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    # Don't trust LLM-assigned ids: they decide section order and must be unique.
    for index, task in enumerate(plan.tasks, start=1):
        task.id = index

    return {"plan": plan}


# ---------------------------------------------------------[ Fanout ]--------------------------------------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


# --------------------------------------------------------[ Worker (write one section) ]---------------------------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Cite each claim individually, right after the claim it supports. Do NOT group citations at the end of a list or paragraph.
- Never state a specific number, benchmark score, parameter count, price or date unless it appears in the Evidence snippets.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}\n  {(e.snippet or '')[:300]}"
            for e in evidence[:20]
        )

    try:
        section_md = llm.invoke(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog title: {plan.blog_title}\n"
                        f"Audience: {plan.audience}\n"
                        f"Tone: {plan.tone}\n"
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Constraints: {plan.constraints}\n"
                        f"Topic: {topic}\n"
                        f"Mode: {mode}\n\n"
                        f"Section title: {task.title}\n"
                        f"Goal: {task.goal}\n"
                        f"Target words: {task.target_words}\n"
                        f"Tags: {task.tags}\n"
                        f"requires_research: {task.requires_research}\n"
                        f"requires_citations: {task.requires_citations}\n"
                        f"requires_code: {task.requires_code}\n"
                        f"Bullets:{bullets_text}\n\n"
                        f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                    )
                ),
            ]
        ).content.strip()
    except Exception as error:
        # One failed section must not sink the whole blog.
        print(f"[worker] section {task.id} failed: {error}")
        section_md = f"## {task.title}\n\n> This section could not be generated ({type(error).__name__})."

    return {"sections": [(task.id, section_md)]}


# ----------------------[ Reducer with images (merge content -> decide images -> generate and place images) ]----------------------
def merge_content(state: State) -> dict:
    plan = state["plan"]

    ordered_sections = [markdown for _, markdown in sorted(state["sections"], key=lambda item: item[0])]
    body = "\n\n".join(ordered_sections).strip()
    return {"merged_md": f"# {plan.blog_title}\n\n{body}\n"}


DECIDE_IMAGES_SYSTEM = """
You are an expert technical editor.
Decide whether up to 3 images would materially improve this blog.
Prefer useful technical diagrams over decorative images.
Do NOT rewrite or return the blog. For each image, return only:
- after_heading: the EXACT heading line (e.g. '## Section Title') of the section the image belongs in
- filename, alt, caption, prompt (and optionally size/quality)
Leave 'placeholder' empty.
If no images are needed, return images=[].
"""


def _norm_heading(line: str) -> str:
    return re.sub(r"^#+\s*", "", line.strip()).strip().lower()


def _insert_placeholders(md: str, specs: list[ImageSpec]) -> str:
    """Insert each placeholder after the first block of its target section (done in code, not by the LLM)."""
    lines = md.split("\n")
    inserts: list[tuple[int, str]] = []

    for spec in specs:
        target = _norm_heading(spec.after_heading)
        heading_idx = next(
            (i for i, line in enumerate(lines) if line.lstrip().startswith("#") and _norm_heading(line) == target),
            None,
        )
        if heading_idx is None:
            continue

        j = heading_idx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        if j < len(lines) and lines[j].lstrip().startswith("```"):
            j += 1
            while j < len(lines) and not lines[j].lstrip().startswith("```"):
                j += 1
            j += 1
        else:
            while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith("#"):
                j += 1

        inserts.append((min(j, len(lines)), spec.placeholder))

    for position, placeholder in sorted(inserts, reverse=True):
        lines[position:position] = ["", placeholder, ""]

    return "\n".join(lines)


def decide_images(state: State) -> dict:
    plan = state["plan"]

    try:
        image_plan = llm.with_structured_output(GlobalImagePlan).invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog title: {plan.blog_title}\n"
                        f"Topic: {state['topic']}\n\n"
                        f"{state['merged_md']}"
                    )
                ),
            ]
        )
        specs = image_plan.images[:MAX_IMAGES]
    except Exception as error:
        print(f"[images] planning failed, continuing without images: {error}")
        specs = []

    for number, spec in enumerate(specs, start=1):
        spec.placeholder = f"[[IMAGE_{number}]]"

    md = _insert_placeholders(state["merged_md"], specs)
    specs = [spec for spec in specs if spec.placeholder in md]  # drop images that found no home

    return {
        "md_with_placeholders": md,
        "image_specs": [spec.model_dump() for spec in specs],
    }


@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    return OpenAI()  # Reads OPENAI_API_KEY from the environment.


def _openai_generate_image_bytes(spec: dict) -> bytes:
    response = _openai_client().images.generate(
        model=IMAGE_MODEL,
        prompt=spec["prompt"],
        size=spec.get("size", "1024x1024"),
        quality=spec.get("quality", "medium"),
    )

    encoded = response.data[0].b64_json
    if not encoded:
        raise RuntimeError("OpenAI returned no image data.")

    return base64.b64decode(encoded)


def _safe_filename(name: str) -> str:
    # Keep only the filename, remove Windows-invalid characters and path parts.
    name = Path(name).name
    name = re.sub(r'[<>:"/\\|?*]', "_", name).strip(" .")
    if not name:
        name = "image.png"
    if not name.lower().endswith(".png"):
        name += ".png"
    return name


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = (state.get("image_specs") or [])[:MAX_IMAGES]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Unique file names, then generate all images in parallel.
    used: set[str] = set()
    jobs: list[tuple[dict, str]] = []
    for spec in image_specs:
        filename = _safe_filename(spec["filename"])
        stem, counter = Path(filename).stem, 1
        while filename in used:
            counter += 1
            filename = f"{stem}_{counter}.png"
        used.add(filename)
        jobs.append((spec, filename))

    def render(job: tuple[dict, str]) -> tuple[dict, str, bool]:
        spec, filename = job
        try:
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            (IMAGES_DIR / filename).write_bytes(_openai_generate_image_bytes(spec))
            return spec, filename, True
        except Exception as error:
            print(f"[images] generation failed for {filename}: {error}")
            return spec, filename, False

    results: list[tuple[dict, str, bool]] = []
    if jobs:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            results = list(pool.map(render, jobs))

    for spec, filename, ok in results:
        if ok:
            replacement = f"![{spec['alt']}](images/{filename})\n\n*{spec['caption']}*"
        else:
            replacement = f"> Image unavailable: {spec.get('caption', '')}"
        md = md.replace(spec["placeholder"], replacement)

    # Remove any placeholders that were not replaced.
    md = re.sub(r"\[\[IMAGE_\d+\]\]", "", md)

    blog_path = OUTPUT_DIR / f"{safe_slug(plan.blog_title)}.md"
    char_count = blog_path.write_text(md, encoding="utf-8")
    print(f"Saved {char_count} characters to: {blog_path.resolve()}")

    return {"final": md}


# Build reducer subgraph
reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)

reducer_subgraph = reducer_graph.compile()


# -----------------------------------------------------[ Building Graph ]----------------------------------------------------------
g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator",
    },
)
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()


# -------------------------------------------------------------[ Runner ]---------------------------------------------------------
def run(topic: str, as_of: Optional[str] = None, recency_days: int = 7) -> dict:
    return app.invoke(build_initial_state(topic, as_of, recency_days))


if __name__ == "__main__":
    # CLI usage:  python Agent_backend.py "Self Attention in Transformer Architecture"
    cli_topic = " ".join(sys.argv[1:]).strip() or input("Topic: ").strip()
    if cli_topic:
        run(cli_topic)
