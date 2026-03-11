"""
Pipeline runner — executes all three simulation scripts in order:
  1. binary_body_main.py  (2-body validation, figs 1–4)
  2. scaling.py           (mass-transfer fraction scaling study)
  3. three_body_validation.py (hierarchical triple validation, fig 5)

All output files are written to the `output/` subdirectory.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    _RICH = True
    console = Console()
except Exception:
    _RICH = False

    class _Plain:
        def print(self, *a, **k):
            print(*a)
        def rule(self, *a, **k):
            print("-" * 70)

    console = _Plain()
    Panel = Table = Rule = None


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _stage(n, title):
    if _RICH:
        console.print(Rule(f"[bold cyan]Stage {n} — {title}[/]", style="cyan"))
    else:
        print(f"\n{'=' * 70}\nStage {n} — {title}\n{'=' * 70}")


def _done(label, elapsed):
    if _RICH:
        console.print(f"  [green]done[/] ({elapsed:.1f}s)\n")
    else:
        print(f"  done ({elapsed:.1f}s)\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if _RICH and Panel is not None:
        console.print(Panel.fit(
            f"[bold]Binary Mass-Transfer Framework — Full Pipeline[/]\n"
            f"Output directory: [cyan]{OUTPUT_DIR}[/]",
            border_style="cyan",
        ))
    else:
        print("=" * 70)
        print("Binary Mass-Transfer Framework — Full Pipeline")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    timings = {}

    # ── Stage 1: binary_body_main ────────────────────────────────────────────
    _stage(1, "2-body validation (figs 1–4)")
    from binary_body_main import generate_all_results
    t0 = time.perf_counter()
    generate_all_results(output_dir=OUTPUT_DIR, use_cache=False)
    timings["binary_body_main"] = time.perf_counter() - t0
    _done("binary_body_main", timings["binary_body_main"])

    # ── Stage 2: scaling ─────────────────────────────────────────────────────
    _stage(2, "Mass-transfer fraction scaling study")
    from scaling import main as scaling_main
    t0 = time.perf_counter()
    scaling_main(output_dir=OUTPUT_DIR)
    timings["scaling"] = time.perf_counter() - t0
    _done("scaling", timings["scaling"])

    # ── Stage 3: three_body_validation ───────────────────────────────────────
    _stage(3, "Three-body (hierarchical triple) validation (fig 5)")
    from three_body_validation import main as tb_main
    t0 = time.perf_counter()
    tb_main(output_dir=OUTPUT_DIR)
    timings["three_body_validation"] = time.perf_counter() - t0
    _done("three_body_validation", timings["three_body_validation"])

    # ── Summary ──────────────────────────────────────────────────────────────
    total = sum(timings.values())
    if _RICH and Table is not None:
        t = Table(title="Pipeline Summary")
        t.add_column("Stage", justify="left")
        t.add_column("Time (s)", justify="right")
        t.add_row("1. binary_body_main", f"{timings['binary_body_main']:.1f}")
        t.add_row("2. scaling",           f"{timings['scaling']:.1f}")
        t.add_row("3. three_body_validation", f"{timings['three_body_validation']:.1f}")
        t.add_section()
        t.add_row("[bold]Total[/]", f"[bold]{total:.1f}[/]")
        console.print(t)

        files = [
            "fig1.jpg", "fig2.jpg", "fig3.jpg", "fig4.jpg",
            "results_fixed.json",
            "scaling_results.csv",
            "fig5.jpg",
            "threebody_summary.json", "threebody_table.csv", "threebody_table.tex",
        ]
        file_table = Table(title=f"Output files → {OUTPUT_DIR}")
        file_table.add_column("File", justify="left")
        file_table.add_column("Size", justify="right")
        for fname in files:
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                size_str = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
            else:
                size_str = "[red]missing[/]"
            file_table.add_row(fname, size_str)
        console.print(file_table)

        console.print(Panel.fit("[bold green]Pipeline complete.[/]", border_style="green"))
    else:
        print(f"\nTotal time: {total:.1f}s")
        print(f"All outputs written to: {OUTPUT_DIR}")
        print("Pipeline complete.")


if __name__ == "__main__":
    main()
