"""B48: the architecture's "what is NOT here yet" and the ROADMAP capability
table describe the code as it is, not as it was on 2026-09-10.

Doc-consistency only; no runtime is built. Three kinds of check:

* the ROADMAP "Remaining capabilities" table names every capability once, and
  the one "Whole-body humanoids" row keeps every fact of the two rows it
  replaced (the B30 landed note, the H2 embodiment and episodes, B29/B29b);
* the ARCHITECTURE section "ROS2, humanoids, and what is NOT here yet" cites
  only paths/anchors that exist, points to the module of every capability
  that exists, names an evidence level for every claim, and its "Not here"
  list names no capability whose module exists;
* README and the other current-status docs carry none of the stale claims
  ("no mobile base", "humanoids = one arm", "design, not code", "to build")
  unless struck through, and the H2 gate status agrees across docs.

Every checker also runs on a synthetic snippet with a known violation, so a
checker that silently stops finding anything fails here too.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "cascade"
DOCS = REPO / "docs"
SECTION = "## ROS2, humanoids, and what is NOT here yet"

# A capability named in the "Not here" list is a false claim once its module
# exists; every such module must instead be cited by the section's claims.
PRESENT_CAPABILITIES = (
    (r"(?i)\bmobile[ -]bases?\b", "src/cascade/control/mobile_base.py"),
    (r"\bMobileBase\b", "src/cascade/control/mobile_base.py"),
    (r"\bMobileRig\b", "src/cascade/control/mobile_rig.py"),
    (r"\bSafeBase\b", "src/cascade/safety/base_harness.py"),
    (r"`(?:walk_velocity|walk_distance|turn|stop_navigation)`", "src/cascade/skills/mobile_runtime.py"),
    (r"(?i)(?<![\w/])navigation(?!\w)", "src/cascade/spatial/navigation.py"),
    (r"`go_to`", "src/cascade/spatial/navigation.py"),
    (r"(?i)\bplanar (?:route|grid)", "src/cascade/spatial/grid.py"),
    (r"(?i)\bself-locali[sz]ation\b", "src/cascade/spatial/cuvslam.py"),
    (r"(?i)\bcuVSLAM\b", "src/cascade/spatial/cuvslam.py"),
    (r"(?i)(?<![\w/])odometry\b", "src/cascade/spatial/cuvslam_uncertainty.py"),
    (r"(?i)\bhumanoid (?:locomotion|walking)\b|\bwalking humanoids?\b", "src/cascade/sim/h2_physx.py"),
    (r"(?i)\bwhole-body (?:contract|composition)\b", "src/cascade/robotics/whole_body.py"),
)

# Stale claims refuted by a module that now exists; allowed only inside ~~...~~.
STALE_CLAIMS = (
    (r"(?i)\b(?:is|has) no mobile base\b|not here: a mobile base", "src/cascade/control/mobile_base.py"),
    (r"(?i)humanoids today = one arm", "src/cascade/sim/h2_physx.py"),
    (r"(?i)nothing publishes a twist, consumes odometry", "src/cascade/spatial/cuvslam_uncertainty.py"),
    (r"(?i)\*\*design, not code\*\*|\bnot code yet\b", "src/cascade/control/mobile_base.py"),
    (r"(?i)bridge \(to build\)|owner architecture \(to build", "scripts/isaac_h2_bridge.py"),
)
CURRENT_DOCS = ("README.md", "docs/ARCHITECTURE.md", "docs/ROADMAP.md", "docs/ROBOT_MODULARITY.md",
                "docs/MOBILITY_AND_NAVIGATION_DESIGN.md", "docs/HUMANOID_H2.md", "docs/SPATIAL_PROVIDERS.md")
NO_GATE_PASSED = re.compile(r"(?i)\bno (?:admission )?gate (?:below )?(?:is |has )?(?:been )?passed\b")
EVIDENCE = re.compile(r"(?i)\bmeasured\b|\bmock-only\b|\bunverified\s+on\s+hardware\b|\bCPU\s+tests?\b"
                      r"|\bsoftware\s+only\b|\bsimulation\s+only\b|\bnot\s+admitted\b|\bno\s+physical\s+admission\b")
NOT_HERE = re.compile(r"^\*\*(?:Still )?[Nn]ot here\b", re.M)
CITED_PATH = re.compile(r"`([\w.*-]+(?:/[\w.*-]+)*(?:\.(?:py|ya?ml|json|md)|/))(?:::\w+)?`")
DOC_LINK = re.compile(r"\]\(([\w./-]+\.md)(?:#([\w-]+))?\)")

WHOLE_BODY_FACTS = (
    # former row 1: the B30 landed note
    "~~Separate domain controllers must not compete for the same command endpoint.~~ "
    "**landed 2026-10-08** (B30, mock-only)",
    "`exclusive` coordination by default (`concurrent` opt-in)",
    "measured by 43 software tests incl. golden digests of every earlier robot profile",
    "Still open: physical admission of any mounted composition (measured mount calibration, independent "
    "base-pose source, moving-frame arm limits), the whole-body controller, balance, "
    "self/environment collision and H2 arm physics",
    # both rows: the H2 embodiment and its first episodes
    "Embodiment chosen on 7 October 2026: [Unitree H2, PhysX first](HUMANOID_H2.md)",
    "the reference loop walks 0.63 m in 2 s on the internal 6.2 build",
    "the independent verifier confirming 3 of 5 `walk_velocity` commands",
    "Newton route tracked through newton-assets PR #53",
    "Candidate revision 2 the same day (settle budget 8 s wall, rest thresholds unchanged)",
    "yaw oscillation 0.69 → 0.32 rad/s over 3.75 s sim",
    # former row 2: B29 and B29b
    "~~next revision is control-side (ramp the turn rate before the goal), not a bigger budget~~ "
    "**landed 2026-10-07 in software (B29, candidate revision 3)**",
    "0.8 rad 8/8 with the ramp vs 1/6 without, but 1.0 rad 1/5 vs 5/6",
    "~~next control revision open~~ **landed 2026-10-08 in software (B29b, candidate revision 4)**",
    "75/76 command sequences identical",
    "Still open: the owner A/B (revision 4 vs 3 vs 2), the 1.0 rad translation margin "
    "(0.166–0.227 m vs 0.20 m in both arms), physical admission.",
)


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    start = text.index(heading + "\n")
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end < 0 else text[start:end]


def _unstruck(text: str) -> str:
    return re.sub(r"~~.*?~~", "", text, flags=re.S)


def _capability_rows(roadmap: str) -> list[tuple[str, str]]:
    table = _section(roadmap, "## Remaining capabilities")
    rows = []
    for line in table.splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| Capability |"):
            continue
        rows.append((line[2:].split(" | ", 1)[0].strip(), line))
    return rows


def _duplicates(names) -> list[str]:
    seen, dup = set(), []
    for name in names:
        if name in seen and name not in dup:
            dup.append(name)
        seen.add(name)
    return dup


def _split_not_here(section: str) -> tuple[str, str]:
    """(claims before the "Not here" paragraph, the "Not here" paragraph to the next bold one)."""
    match = NOT_HERE.search(section)
    if match is None:
        raise AssertionError("the section has no **Not here** paragraph")
    rest = section[match.start():]
    following = re.search(r"^\*\*", rest[2:], re.M)
    block = rest if following is None else rest[:following.start() + 2]
    return section[:match.start()] + rest[len(block):], block


def _absent_but_present(block: str, root: Path = REPO) -> list[tuple[str, str]]:
    found = []
    for pattern, module in PRESENT_CAPABILITIES:
        if not (root / module).exists():
            continue
        found.extend((m.group(0), module) for m in re.finditer(pattern, block))
    return found


def _uncited_modules(claims: str, root: Path = REPO) -> list[str]:
    missing = []
    for _, module in PRESENT_CAPABILITIES:
        cited = module.removeprefix("src/cascade/")
        if (root / module).exists() and cited not in claims and module not in missing:
            missing.append(module)
    return missing


def _claims_without_evidence(section: str) -> list[str]:
    paragraphs = re.split(r"\n(?=\*\*)", section)
    return [p.splitlines()[0] for p in paragraphs
            if p.startswith("**") and not NOT_HERE.match(p) and not EVIDENCE.search(p)]


def _path_exists(path: str) -> bool:
    for base in (REPO, SRC, DOCS):
        candidate = str(base / path)
        if glob.glob(candidate) if "*" in path else Path(candidate).exists():
            return True
    return False


def _missing_paths(text: str) -> list[str]:
    return [m.group(1) for m in CITED_PATH.finditer(text) if not _path_exists(m.group(1))]


def _slug(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def _broken_links(text: str, docs: Path = DOCS) -> list[str]:
    broken = []
    for match in DOC_LINK.finditer(text):
        target, anchor = docs / match.group(1), match.group(2)
        if not target.exists():
            broken.append(match.group(0))
        elif anchor is not None:
            slugs = {_slug(h) for h in re.findall(r"^#+ (.+)$", target.read_text(encoding="utf-8"), re.M)}
            if anchor not in slugs:
                broken.append(match.group(0))
    return broken


def _stale_claims(text: str, root: Path = REPO) -> list[tuple[str, str]]:
    live = _unstruck(text)
    found = []
    for pattern, module in STALE_CLAIMS:
        if (root / module).exists():
            found.extend((m.group(0), module) for m in re.finditer(pattern, live))
    return found


def _passed_gates(h2_doc: str) -> list[str]:
    table = _section(h2_doc, next(line for line in h2_doc.splitlines() if line.startswith("## Admission gates")))
    return [line.split(" | ")[0].lstrip("| ").strip() for line in table.splitlines()
            if line.startswith("| ") and re.search(r"\| \*\*passed\b", line)]


def _whole_body_row_problems(row: str) -> list[str]:
    problems = [f"missing: {fact}" for fact in WHOLE_BODY_FACTS if fact not in row]
    b30, h2 = "(B30, mock-only)", "Embodiment chosen on 7 October 2026"
    if b30 in row and h2 in row and row.index(b30) > row.index(h2):
        problems.append("the B30 note follows the H2 history it does not close")
    return problems


# --- premises: the modules the guards key on exist (pass on main by design) ---

def test_premise_every_guarded_module_exists():
    modules = {m for _, m in PRESENT_CAPABILITIES} | {m for _, m in STALE_CLAIMS}
    assert sorted(m for m in modules if not (REPO / m).exists()) == []


def test_premise_ros2_mobile_interfaces_are_still_absent_from_the_code():
    # What the "Not here" list keeps claiming is checked against the code.
    absent = re.compile(r"geometry_msgs|nav_msgs|nav2_msgs|NavigateToPose|cmd_vel|OccupancyGrid"
                        r"|go_to_pixel|go_to_object|where_am_i|ros2_base|unitree_base|mujoco_base")
    hits = [str(p.relative_to(REPO)) for p in sorted((REPO / "src").rglob("*.py"))
            + sorted((REPO / "scripts").glob("*.py")) if absent.search(p.read_text(encoding="utf-8"))]
    assert hits == [], "a mobile interface the ARCHITECTURE 'Not here' list names now exists: update it"
    runtime = (SRC / "apps" / "mobile_runtime.py").read_text(encoding="utf-8")
    make_base = runtime[runtime.index("def make_base("):runtime.index("\ndef ", runtime.index("def make_base("))]
    assert re.findall(r'profile\["type"\] == "(\w+)"', make_base) == ["mock", "isaac"]


# --- checker self-tests: each checker finds a known violation ---

def test_checkers_find_known_violations_in_synthetic_docs(tmp_path):
    table = ("## Remaining capabilities\n\n| Capability | Remaining |\n| --- | --- |\n"
             "| Locomotion | a |\n| Whole-body humanoids | b |\n| Whole-body humanoids | c |\n\n## Next\n| Locomotion | x |\n")
    assert [n for n, _ in _capability_rows(table)] == ["Locomotion", "Whole-body humanoids", "Whole-body humanoids"]
    assert _duplicates(["a", "b", "a", "a", "c"]) == ["a"]
    section = (f"{SECTION}\n\nIntro.\n\n**ROS2 today = arms.** `control/ros2_arm.py`. **Unverified on hardware.**\n\n"
               "**Mobile bases.** It exists, trust us.\n\n**Not here: a mobile base, navigation, `go_to`.**\n"
               "- `stop_navigation` is not navigation in MOBILITY_AND_NAVIGATION_DESIGN.md, nor `nav_msgs/Odometry`\n\n"
               "**After.** measured `control/no_such_module.py`\n")
    claims, block = _split_not_here(section)
    assert block.startswith("**Not here") and "**After.**" not in block and "**After.**" in claims
    assert "**Mobile bases.**" in claims and "Not here" not in claims
    names = [name for name, _ in _absent_but_present(block)]
    assert names == ["mobile base", "`stop_navigation`", "navigation", "navigation", "`go_to`"]
    assert _absent_but_present(block, root=tmp_path) == []
    assert "src/cascade/control/mobile_base.py" in _uncited_modules(claims)
    assert _uncited_modules(claims, root=tmp_path) == []
    assert _uncited_modules("`control/mobile_base.py`" + "".join(
        m.removeprefix("src/cascade/") for _, m in PRESENT_CAPABILITIES)) == []
    assert _claims_without_evidence(section) == ["**Mobile bases.** It exists, trust us."]
    assert _claims_without_evidence("**Arm.** A wrapped **Unverified on\nhardware.**\n\n**Base.** CPU\ntests.") == []
    assert _missing_paths(section) == ["control/no_such_module.py"]
    assert _missing_paths("`tests/test_mobile_*.py` `configs/bases/` `tests/test_nope_*.py` `configs/nope/`") == [
        "tests/test_nope_*.py", "configs/nope/"]
    (tmp_path / "A.md").write_text("# Title\n\n## Live result (8 October 2026)\n", encoding="utf-8")
    links = "[a](A.md) [b](A.md#live-result-8-october-2026) [c](A.md#nope) [d](B.md)"
    assert _broken_links(links, docs=tmp_path) == ["](A.md#nope)", "](B.md)"]
    stale = "There ~~is no mobile base~~ here. Humanoids today = one arm. **design, not code**"
    assert [c for c, _ in _stale_claims(stale)] == ["Humanoids today = one arm", "**design, not code**"]
    assert _stale_claims(stale, root=tmp_path) == []
    gates = ("## Admission gates (x)\n\n| gate | status |\n| --- | --- |\n| 1 Binding | **passed in simulation** |\n"
             "| 2 Frames | **partially shown** |\n| 3 Limits | open |\n\n## Next\n| 9 X | **passed** |\n")
    assert _passed_gates(gates) == ["1 Binding"]
    assert NO_GATE_PASSED.search("Status: candidate, no admission gate passed.")
    assert NO_GATE_PASSED.search("no gate below is passed yet")
    assert not NO_GATE_PASSED.search("gate 1 (binding) passed in simulation")
    facts = list(WHOLE_BODY_FACTS)
    assert _whole_body_row_problems(" ".join(facts)) == []
    assert _whole_body_row_problems(" ".join(facts[:-1])) == [f"missing: {facts[-1]}"]
    assert _whole_body_row_problems(" ".join(facts[4:] + facts[:4])) == [
        "the B30 note follows the H2 history it does not close"]


# --- the docs follow the code ---

def test_remaining_capabilities_table_names_each_capability_once():
    names = [name for name, _ in _capability_rows(_read("docs/ROADMAP.md"))]
    assert len(names) >= 10
    assert _duplicates(names) == []


def test_whole_body_row_keeps_every_fact_of_both_former_rows():
    rows = [row for name, row in _capability_rows(_read("docs/ROADMAP.md")) if name == "Whole-body humanoids"]
    assert len(rows) == 1
    assert _whole_body_row_problems(rows[0]) == []


def test_not_here_list_names_no_capability_whose_module_exists():
    _, block = _split_not_here(_section(_read("docs/ARCHITECTURE.md"), SECTION))
    assert _absent_but_present(block) == []


def test_section_points_to_the_module_of_every_present_capability():
    claims, _ = _split_not_here(_section(_read("docs/ARCHITECTURE.md"), SECTION))
    assert _uncited_modules(claims) == []


def test_every_claim_in_the_section_names_its_evidence_level():
    section = _section(_read("docs/ARCHITECTURE.md"), SECTION)
    assert _claims_without_evidence(section) == []


def test_section_cites_only_paths_and_anchors_that_exist():
    section = _section(_read("docs/ARCHITECTURE.md"), SECTION)
    assert _missing_paths(section) == []
    assert _broken_links(section) == []


def test_current_docs_carry_no_stale_mobility_or_humanoid_claim():
    found = {doc: _stale_claims(_read(doc)) for doc in CURRENT_DOCS}
    assert {doc: claims for doc, claims in found.items() if claims} == {}


def test_h2_gate_status_agrees_across_docs():
    passed = _passed_gates(_read("docs/HUMANOID_H2.md"))
    assert passed, "premise: the H2 gate table records a passed gate"
    contradicting = {doc: NO_GATE_PASSED.findall(_unstruck(_read(doc)))
                     for doc in ("README.md", "docs/HUMANOID_H2.md", "docs/ROBOT_MODULARITY.md")}
    assert {doc: hits for doc, hits in contradicting.items() if hits} == {}
