from json import loads
from pathlib import Path
from httpx import get
from evals.schemas import (
    DB_PATH,
    Choice,
    Domain,
    Principle,
    Question,
    Relevance,
    Scenario,
    TaskStatement,
    create_database,
    session_for,
)


SOURCE_URL = "https://architecturetoautonomy.com/assets/Tools/whetstone/whetstone.html"

CCA_F_DOMAINS = {
    1: "agentic_architecture",
    2: "tool_design_mcp",
    3: "claude_code_workflows",
    4: "prompt_engineering",
    5: "context_management",
}

EXPECTED_COUNTS = {
    "domains": 5,
    "scenarios": 6,
    "task_statements": 30,
    "principles": 34,
    "questions": 163,
}


def fetch_page(url: str = SOURCE_URL) -> str:
    response = get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    return response.text


def extract_object(html: str, name: str) -> dict | list:
    marker = f"const {name} = "
    start = html.find(marker)
    if start < 0:
        raise ValueError(f"Could not find {name} in the page")

    start += len(marker)
    while html[start].isspace():
        start += 1

    opener = html[start]
    closer = {"{": "}", "[": "]"}[opener]
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(html)):
        char = html[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return loads(html[start : index + 1])

    raise ValueError(f"Could not parse {name}")


def parse_bank(html: str) -> dict:
    return {
        "domains": extract_object(html, "DOMAINS"),
        "scenarios": extract_object(html, "SCENARIOS"),
        "principles": extract_object(html, "PRINCIPLES"),
        "task_statements": extract_object(html, "TS"),
        "questions": extract_object(html, "ITEMS"),
    }


def validate(bank: dict) -> None:
    counts = {
        "domains": len(bank["domains"]),
        "scenarios": len(bank["scenarios"]),
        "task_statements": len(bank["task_statements"]),
        "principles": len(bank["principles"]),
        "questions": len(bank["questions"]),
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected bank counts: {counts}")

    domain_ids = {int(domain_id) for domain_id in bank["domains"]}
    scenario_ids = {int(scenario_id) for scenario_id in bank["scenarios"]}
    task_codes = set(bank["task_statements"])
    principle_codes = set(bank["principles"])

    if domain_ids != set(CCA_F_DOMAINS):
        raise ValueError(f"Unexpected domain ids: {sorted(domain_ids)}")

    for code in task_codes:
        domain_id = int(code.split(".")[0])
        if domain_id not in domain_ids:
            raise ValueError(f"Task {code} has no domain")

    for question in bank["questions"]:
        if question["ts"] not in task_codes:
            raise ValueError(
                f"{question['id']} references missing task {question['ts']}"
            )
        if question["p"] not in principle_codes:
            raise ValueError(
                f"{question['id']} references missing principle {question['p']}"
            )
        if (
            int(question["d"]) not in domain_ids
            or int(question["s"]) not in scenario_ids
        ):
            raise ValueError(
                f"{question['id']} references a missing domain or scenario"
            )
        if int(question["ts"].split(".")[0]) != int(question["d"]):
            raise ValueError(f"{question['id']} task does not match its domain")

        correct = set(question["c"])
        if not correct or any(
            index < 0 or index >= len(question["o"]) for index in correct
        ):
            raise ValueError(
                f"{question['id']} has a correct choice outside its options"
            )


def clear_bank(session) -> None:
    session.query(Choice).delete()
    session.query(Relevance).delete()
    session.query(Question).delete()
    session.query(Principle).delete()
    session.query(TaskStatement).delete()
    session.query(Scenario).delete()
    session.query(Domain).delete()


def insert_bank(session, bank: dict) -> None:
    for domain_id, domain in bank["domains"].items():
        session.add(
            Domain(
                id=int(domain_id),
                name=domain["name"],
                weight=domain["weight"],
                cca_f_domain=CCA_F_DOMAINS[int(domain_id)],
            )
        )

    for scenario_id, scenario in bank["scenarios"].items():
        session.add(
            Scenario(
                id=int(scenario_id),
                title=scenario["title"],
                description=scenario["desc"],
            )
        )

    for code, title in bank["task_statements"].items():
        session.add(
            TaskStatement(
                code=code,
                domain_id=int(code.split(".")[0]),
                title=title,
            )
        )

    for code, principle in bank["principles"].items():
        session.add(
            Principle(
                code=code,
                domain_id=principle["d"],
                rule=principle["r"],
                trap=principle["t"],
            )
        )

    session.flush()

    for item in bank["questions"]:
        correct = set(item["c"])
        session.add(
            Question(
                id=item["id"],
                scenario_id=item["s"],
                domain_id=item["d"],
                task_code=item["ts"],
                principle_code=item["p"],
                stem=item["q"],
                explanation=item["e"],
                multi_response=bool(item.get("m")),
                from_exam_guide=item.get("src") == "guide",
                contrast_note=item.get("cf"),
            )
        )
        for position, body in enumerate(item["o"]):
            session.add(
                Choice(
                    question_id=item["id"],
                    position=position,
                    body=body,
                    is_correct=position in correct,
                )
            )


def load(path: Path = DB_PATH) -> None:
    create_database(path)
    bank = parse_bank(fetch_page())
    validate(bank)

    with session_for(path) as session:
        clear_bank(session)
        insert_bank(session, bank)
        session.commit()
        print(
            "Loaded "
            f"{session.query(Question).count()} questions, "
            f"{session.query(Choice).count()} choices"
        )


if __name__ == "__main__":
    load()
