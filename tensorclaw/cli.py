from __future__ import annotations

import argparse
from pathlib import Path

from .tui.app import TensorClawApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorclaw",
        description="TensorClaw: agent-first research harness TUI.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run apply/loop logic without executing experiment commands",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path.cwd().resolve()
    app = TensorClawApp(project_root=project_root, dry_run=bool(args.dry_run))
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
