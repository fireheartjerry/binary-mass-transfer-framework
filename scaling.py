"""
R1.6 Response: Mass-transfer fraction scaling study.

Runs the unclosed mapping at multiple mass-transfer fractions (1%, 2%, 5%, 10%, 15%)
for all three mass ratios (q0 = 0.5, 1.0, 2.0) to show how orbital drift
scales with total transferred mass.

Includes analytical verification: since the unclosed mapping yields a_f/a_0 = 1
exactly, the deviation from the conservative prediction is fully determined by
|1 - (a_f/a_0)_cons| / (a_f/a_0)_cons. This formula should predict every entry
in the table to within simulation precision, as claimed in the R1 response letter.

Output: a clean table + CSV for inclusion in the paper/response letter.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binary_body_main import run_simulation

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    _RICH = True
    console = Console()
except Exception:
    _RICH = False

    class _Plain:
        def print(self, *a, **k):
            print(*a)

    console = _Plain()
    Panel = Table = Progress = None


mass_fractions = [0.01, 0.02, 0.05, 0.10, 0.15]
mass_ratios = [0.5, 1.0, 2.0]
q_labels = {0.5: "A", 1.0: "B", 2.0: "C"}

# n_events scaling: keeps dm per event approximately constant across fractions.
# Note: the unclosed analytical result L_f/L_0 = (1-f)(1+f*q0) and a_f/a_0 = 1
# are both independent of step size (Section 5.4), so this scaling does not
# affect the unclosed error. It is retained for physical consistency.
base_events_at_15pct = 500


def _run_one(job):
    """Top-level picklable worker — runs one simulation and returns only scalars."""
    r = run_simulation(
        q0=job["q0"],
        mass_transfer_fraction=job["fraction"],
        n_transfer_steps=job["n_events"],
        enforce_angular_momentum=job["enforce"],
    )
    return {
        "fraction": job["fraction"],
        "q0": job["q0"],
        "enforce": job["enforce"],
        "n_events": job["n_events"],
        "error_percent": r["error_percent"],
        "L_ratio": r["L_ratio"],
        "a_ratio_analytical": r["a_ratio_analytical"],
    }


def _mk_progress():
    if not _RICH:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=True,
    )


def _print_full_table(results):
    if _RICH and Table is not None:
        table = Table(title="Scaling Study: Unclosed Drift vs. Transfer Fraction")
        columns = [
            ("f", "right"),
            ("q0", "right"),
            ("Case", "left"),
            ("Events", "right"),
            ("Err(unc) %", "right"),
            ("Pred.err %", "right"),
            ("Δerr", "right"),
            ("L/L0", "right"),
            ("L_ana", "right"),
            ("ΔL", "right"),
        ]
        for label, justify in columns:
            table.add_column(label, justify=justify)
        for row in results:
            table.add_row(
                f"{row['f_pct']:.0f}%",
                f"{row['q0']:.1f}",
                row["case"],
                f"{row['n_events']}",
                f"{row['err_unclosed']:.2f}",
                f"{row['predicted_error']:.2f}",
                f"{row['error_residual']:.2e}",
                f"{row['L_ratio']:.4f}",
                f"{row['L_analytical']:.4f}",
                f"{row['L_residual']:.2e}",
            )
        console.print(table)
        return

    print("\n" + "=" * 110)
    print("TABLE: Orbital deviation (unclosed) vs. mass-transfer fraction, with analytical verification")
    print("=" * 110)
    print(f"{'f':<8} {'q0':<6} {'Case':<6} {'Events':<8} {'Err(unc)':<12} {'Pred.err':<12} {'Δerr':<12} {'L/L0':<10} {'L_ana':<10} {'ΔL'}")
    print("-" * 110)
    for row in results:
        print(
            f"{row['f_pct']:>4.0f}%   {row['q0']:<6.1f} {row['case']:<6} {row['n_events']:<8d} "
            f"{row['err_unclosed']:<12.2f} {row['predicted_error']:<12.2f} {row['error_residual']:<12.2e} "
            f"{row['L_ratio']:<10.4f} {row['L_analytical']:<10.4f} {row['L_residual']:.2e}"
        )


def _print_condensed_table(results):
    if _RICH and Table is not None:
        table = Table(title="Condensed Table: Error (%) Without Closure")
        for label in ("f", "q0=0.5", "q0=1.0", "q0=2.0"):
            table.add_column(label, justify="right")
        for fraction in mass_fractions:
            row = [r for r in results if round(r["f_pct"], 4) == round(fraction * 100, 4)]
            vals = {r["q0"]: r["err_unclosed"] for r in row}
            table.add_row(
                f"{fraction * 100:.0f}%",
                f"{vals[0.5]:.2f}",
                f"{vals[1.0]:.2f}",
                f"{vals[2.0]:.2f}",
            )
        console.print(table)
        return

    print("\n\n" + "=" * 75)
    print("CONDENSED TABLE (for paper): Error(%) without closure")
    print("=" * 75)
    print(f"{'f':<12} {'q0=0.5':<14} {'q0=1.0':<14} {'q0=2.0':<14}")
    print("-" * 55)
    for fraction in mass_fractions:
        row = [r for r in results if round(r["f_pct"], 4) == round(fraction * 100, 4)]
        vals = {r["q0"]: r["err_unclosed"] for r in row}
        print(f"{fraction*100:>5.0f}%       {vals[0.5]:<14.2f} {vals[1.0]:<14.2f} {vals[2.0]:<14.2f}")


def _print_verification_summary(results):
    max_err_res = max(r["error_residual"] for r in results)
    max_l_res = max(r["L_residual"] for r in results)
    verified = max_err_res < 0.05 and max_l_res < 1e-3

    if _RICH and Table is not None:
        table = Table(title="Verification Summary")
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")
        table.add_row("Max |err_numerical - err_predicted|", f"{max_err_res:.2e} %")
        table.add_row("Max |L/L0_numerical - L/L0_analytical|", f"{max_l_res:.2e}")
        console.print(table)
        message = (
            "[bold green]All residuals within simulation precision. Reviewer claim verified.[/]"
            if verified
            else "[bold red]Residuals exceed expected simulation precision. Investigate.[/]"
        )
        console.print(Panel.fit(message, border_style="green" if verified else "red"))
        return

    print("\n\n" + "=" * 75)
    print("VERIFICATION SUMMARY: Analytical vs. numerical agreement")
    print("(Both residuals should be at or below ~1e-2 for all rows)")
    print("=" * 75)
    print(f"Max |err_numerical - err_predicted| across all cases: {max_err_res:.2e} %")
    print(f"Max |L/L0_numerical - L/L0_analytical| across all cases: {max_l_res:.2e}")
    if verified:
        print("All residuals within simulation precision. Reviewer claim verified.")
    else:
        print("WARNING: residuals exceed expected simulation precision. Investigate.")


def _write_csv(results, csv_path):
    with open(csv_path, "w") as fout:
        fout.write(
            "f_transfer_pct,q0,case,n_events,error_unclosed_pct,error_closed_pct,"
            "L_ratio,predicted_error_pct,error_residual_pct,L_analytical,L_residual\n"
        )
        for row in results:
            fout.write(
                f"{row['f_pct']:.0f},{row['q0']:.1f},{row['case']},{row['n_events']},"
                f"{row['err_unclosed']:.4f},{row['err_closed']:.4f},"
                f"{row['L_ratio']:.6f},{row['predicted_error']:.4f},"
                f"{row['error_residual']:.2e},{row['L_analytical']:.6f},{row['L_residual']:.2e}\n"
            )


def _assemble_results(raw):
    """Build the ordered results list from the flat dict of completed sim outputs."""
    results = []
    for fraction in mass_fractions:
        n_events = max(10, int(base_events_at_15pct * fraction / 0.15))
        for q0 in mass_ratios:
            r        = raw[(fraction, q0, False)]
            r_closed = raw[(fraction, q0, True)]

            a_ana_cons    = r["a_ratio_analytical"]
            predicted_error = abs(1.0 - a_ana_cons) / a_ana_cons * 100
            l_analytical  = (1 - fraction) * (1 + fraction * q0)
            error_residual = abs(r["error_percent"] - predicted_error)
            l_residual    = abs(r["L_ratio"] - l_analytical)

            results.append({
                "f_pct":          fraction * 100,
                "q0":             q0,
                "case":           q_labels[q0],
                "n_events":       n_events,
                "err_unclosed":   r["error_percent"],
                "err_closed":     r_closed["error_percent"],
                "L_ratio":        r["L_ratio"],
                "predicted_error": predicted_error,
                "error_residual": error_residual,
                "L_analytical":   l_analytical,
                "L_residual":     l_residual,
            })
    return results


def main(output_dir="."):
    title = "Mass-Transfer Fraction Scaling Study"
    subtitle = "Unclosed drift scaling across transfer fractions with analytical verification."
    if _RICH and Panel is not None:
        console.print(Panel.fit(f"[bold]{title}[/]\n{subtitle}", border_style="cyan"))
    else:
        print("=" * 78 + f"\n{title}\n{subtitle}\n" + "=" * 78)

    # Build the full job list (5 fractions × 3 q0 × 2 enforce = 30 simulations)
    jobs = []
    for fraction in mass_fractions:
        n_events = max(10, int(base_events_at_15pct * fraction / 0.15))
        for q0 in mass_ratios:
            for enforce in (False, True):
                jobs.append({"fraction": fraction, "q0": q0, "n_events": n_events, "enforce": enforce})

    n_workers = min(os.cpu_count() or 4, len(jobs))
    raw = {}  # (fraction, q0, enforce) -> sim scalars

    prog = _mk_progress()
    if prog is None:
        for job in jobs:
            out = _run_one(job)
            raw[(out["fraction"], out["q0"], out["enforce"])] = out
            print(f"  f={out['fraction']*100:.0f}% q0={out['q0']} enforce={out['enforce']} "
                  f"err={out['error_percent']:.2f}% L/L0={out['L_ratio']:.4f}")
    else:
        label = f"Running {len(jobs)} simulations across {n_workers} workers"
        with prog:
            task = prog.add_task(label, total=len(jobs))
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_run_one, job): job for job in jobs}
                for future in as_completed(futures):
                    out = future.result()
                    raw[(out["fraction"], out["q0"], out["enforce"])] = out
                    prog.advance(task)

    results = _assemble_results(raw)

    _print_full_table(results)
    _print_condensed_table(results)
    _print_verification_summary(results)

    csv_path = os.path.join(output_dir, "scaling_results.csv")
    _write_csv(results, csv_path)
    console.print(f"[green]CSV saved to:[/] {csv_path}") if _RICH else print(f"\nCSV saved to: {csv_path}")

    if _RICH and Panel is not None:
        console.print(Panel.fit("[bold green]Scaling study complete.[/]", border_style="green"))
    else:
        print("Done.")


if __name__ == "__main__":
    main()
