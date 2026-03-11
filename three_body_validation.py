import numpy as np
import json
import csv
import matplotlib.pyplot as plt

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

G = 6.67430e-11
M_SUN = 1.9891e30
AU = 1.496e11


class Body:
    def __init__(self, mass, pos, vel):
        self.mass = float(mass)
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)


def pairAccelerations(bodies):
    n = len(bodies)
    acc = [np.zeros(2) for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = bodies[j].pos - bodies[i].pos
            dist = np.linalg.norm(r_vec)
            r_hat = r_vec / dist
            acc_i = G * bodies[j].mass / dist**2 * r_hat
            acc_j = -G * bodies[i].mass / dist**2 * r_hat
            acc[i] += acc_i
            acc[j] += acc_j
    return acc


def velocityVerletStep(bodies, dt):
    acc0 = pairAccelerations(bodies)
    for body, a0 in zip(bodies, acc0):
        body.pos += body.vel * dt + 0.5 * a0 * dt**2
    acc1 = pairAccelerations(bodies)
    for body, a0, a1 in zip(bodies, acc0, acc1):
        body.vel += 0.5 * (a0 + a1) * dt


def shiftToCOMFrame(bodies):
    total_mass = sum(body.mass for body in bodies)
    com_pos = sum(body.mass * body.pos for body in bodies) / total_mass
    com_vel = sum(body.mass * body.vel for body in bodies) / total_mass
    for body in bodies:
        body.pos -= com_pos
        body.vel -= com_vel


def decomposeVelocity(v, r):
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-30:
        return np.zeros(2), v.copy()
    r_hat = r / r_norm
    v_rad = np.dot(v, r_hat) * r_hat
    v_tan = v - v_rad
    return v_rad, v_tan


def innerPairCOM(b1, b2):
    total_mass = b1.mass + b2.mass
    com_pos = (b1.mass * b1.pos + b2.mass * b2.pos) / total_mass
    com_vel = (b1.mass * b1.vel + b2.mass * b2.vel) / total_mass
    return com_pos, com_vel


def cross2d_z(a, b):
    # z-component of a 3D cross product for vectors constrained to the xy-plane.
    return a[0] * b[1] - a[1] * b[0]


def innerPairAngularMomentum(bodies):
    b1 = bodies[0]
    b2 = bodies[1]
    com_pos, com_vel = innerPairCOM(b1, b2)

    r1 = b1.pos - com_pos
    r2 = b2.pos - com_pos
    v1 = b1.vel - com_vel
    v2 = b2.vel - com_vel

    L = b1.mass * cross2d_z(r1, v1) + b2.mass * cross2d_z(r2, v2)
    return float(L)


def innerPairOsculatingSemimajorAxis(bodies):
    b1 = bodies[0]
    b2 = bodies[1]

    r_vec = b2.pos - b1.pos
    v_vec = b2.vel - b1.vel
    r = np.linalg.norm(r_vec)
    v2 = float(np.dot(v_vec, v_vec))
    total_mass = b1.mass + b2.mass

    eps = 0.5 * v2 - G * total_mass / r
    if eps >= 0:
        return np.nan

    a = -G * total_mass / (2 * eps)
    if not np.isfinite(a) or a <= 0 or a > 100 * AU:
        return np.nan
    return float(a)


def setupHierarchicalTriple(m1, m2, m3, a_in, a_out):
    m12 = m1 + m2

    r1_inner = np.array([-a_in * m2 / m12, 0.0])
    r2_inner = np.array([ a_in * m1 / m12, 0.0])

    v_rel_in = np.sqrt(G * m12 / a_in)
    v1_inner = np.array([0.0,  v_rel_in * m2 / m12])
    v2_inner = np.array([0.0, -v_rel_in * m1 / m12])

    m_total = m12 + m3
    r12_outer = np.array([-a_out * m3 / m_total, 0.0])
    r3_outer = np.array([ a_out * m12 / m_total, 0.0])

    v_rel_out = np.sqrt(G * m_total / a_out)
    v12_outer = np.array([0.0,  v_rel_out * m3 / m_total])
    v3_outer = np.array([0.0, -v_rel_out * m12 / m_total])

    b1 = Body(m1, r12_outer + r1_inner, v12_outer + v1_inner)
    b2 = Body(m2, r12_outer + r2_inner, v12_outer + v2_inner)
    b3 = Body(m3, r3_outer, v3_outer)

    bodies = [b1, b2, b3]
    shiftToCOMFrame(bodies)
    return bodies


def enforceInnerPairClosure(bodies, L_target):
    b1 = bodies[0]
    b2 = bodies[1]

    com_pos, com_vel = innerPairCOM(b1, b2)

    r1 = b1.pos - com_pos
    r2 = b2.pos - com_pos
    v1 = b1.vel - com_vel
    v2 = b2.vel - com_vel

    v1_rad, v1_tan = decomposeVelocity(v1, r1)
    v2_rad, v2_tan = decomposeVelocity(v2, r2)

    L_current = b1.mass * cross2d_z(r1, v1_tan) + b2.mass * cross2d_z(r2, v2_tan)
    L_current = float(L_current)

    if abs(L_current) < 1e-30:
        return

    scale = L_target / L_current

    b1.vel = com_vel + v1_rad + v1_tan * scale
    b2.vel = com_vel + v2_rad + v2_tan * scale


def transferMassInnerPair(bodies, delta_m, enforce_closure):
    L_before = innerPairAngularMomentum(bodies)

    b1 = bodies[0]
    b2 = bodies[1]

    b1.mass -= delta_m
    b2.mass += delta_m

    shiftToCOMFrame(bodies)

    if enforce_closure:
        enforceInnerPairClosure(bodies, L_before)
        shiftToCOMFrame(bodies)

    L_after = innerPairAngularMomentum(bodies)
    relative_jump = abs(L_after - L_before) / abs(L_before)
    return relative_jump


def cleanPlotSeries(y):
    y = np.array(y, dtype=float)
    x = np.arange(len(y))

    finite = np.isfinite(y)
    if finite.sum() < 5:
        return y

    y_interp = np.interp(x, x[finite], y[finite])

    median = np.median(y_interp)
    mad = np.median(np.abs(y_interp - median)) + 1e-12
    good = np.abs(y_interp - median) < 8 * mad

    y_clean = np.interp(x, x[good], y_interp[good])
    return y_clean


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


def runThreeBodyCase(case_label, m3_solar, a3_au, enforce_closure, q0=2.0, m2_solar=1.0,
                     a_in_au=1.0, transfer_fraction=0.15, n_orbits=50,
                     steps_per_orbit=4000, n_transfer_events=500,
                     progress=None, task_id=None):
    m1 = q0 * m2_solar * M_SUN
    m2 = m2_solar * M_SUN
    m3 = m3_solar * M_SUN
    a_in = a_in_au * AU
    a_out = a3_au * AU

    bodies = setupHierarchicalTriple(m1, m2, m3, a_in, a_out)

    a0 = innerPairOsculatingSemimajorAxis(bodies)
    T0 = 2 * np.pi * np.sqrt(a_in**3 / (G * (m1 + m2)))
    dt = T0 / steps_per_orbit
    n_steps = int(n_orbits * T0 / dt)

    total_transfer_mass = transfer_fraction * m1
    delta_m = total_transfer_mass / n_transfer_events
    transfer_interval_steps = max(1, n_steps // n_transfer_events)

    times = []
    a_ratio_series = []
    jump_series = []

    sample_every = max(1, steps_per_orbit // 8)
    transfer_count = 0

    for step in range(n_steps):
        if step % sample_every == 0:
            t_orbits = step * dt / T0
            a_now = innerPairOsculatingSemimajorAxis(bodies)
            if np.isfinite(a_now):
                a_ratio_series.append(a_now / a0)
            else:
                a_ratio_series.append(np.nan)
            times.append(t_orbits)

        if progress is not None and task_id is not None:
            progress.advance(task_id)

        velocityVerletStep(bodies, dt)

        target_transfers = int((step + 1) / transfer_interval_steps)
        while transfer_count < min(target_transfers, n_transfer_events):
            jump = transferMassInnerPair(bodies, delta_m, enforce_closure)
            jump_series.append(jump)
            transfer_count += 1

    a_final = innerPairOsculatingSemimajorAxis(bodies)

    result = {
        "case": case_label,
        "m3_msun": m3_solar,
        "a3_au": a3_au,
        "closed": enforce_closure,
        "times": np.array(times),
        "a_ratio_series_raw": np.array(a_ratio_series),
        "a_ratio_series_plot": cleanPlotSeries(a_ratio_series),
        "a_ratio_final": float(a_final / a0),
        "mean_jump": float(np.mean(jump_series)),
    }
    return result


def saveOutputs(closed_D, unclosed_D, closed_E, unclosed_E):
    summary = {
        "D": {
            "description": "Weak perturber",
            "m3_msun": 0.3,
            "a3_au": 8.0,
            "a_ratio_closed": closed_D["a_ratio_final"],
            "a_ratio_unclosed": unclosed_D["a_ratio_final"],
            "mean_jump_closed": closed_D["mean_jump"],
            "mean_jump_unclosed": unclosed_D["mean_jump"],
        },
        "E": {
            "description": "Stronger perturber",
            "m3_msun": 0.8,
            "a3_au": 5.0,
            "a_ratio_closed": closed_E["a_ratio_final"],
            "a_ratio_unclosed": unclosed_E["a_ratio_final"],
            "mean_jump_closed": closed_E["mean_jump"],
            "mean_jump_unclosed": unclosed_E["mean_jump"],
        },
    }

    with open("threebody_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("threebody_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Case",
            "m3 (M_sun)",
            "a3 (AU)",
            "(a_f/a_0)_closed",
            "(a_f/a_0)_unclosed",
            "<|ΔL_pair|>/L0 closed",
            "<|ΔL_pair|>/L0 unclosed",
        ])
        writer.writerow([
            "D",
            0.3,
            8.0,
            closed_D["a_ratio_final"],
            unclosed_D["a_ratio_final"],
            closed_D["mean_jump"],
            unclosed_D["mean_jump"],
        ])
        writer.writerow([
            "E",
            0.8,
            5.0,
            closed_E["a_ratio_final"],
            unclosed_E["a_ratio_final"],
            closed_E["mean_jump"],
            unclosed_E["mean_jump"],
        ])

    with open("threebody_table.tex", "w") as f:
        f.write("\\begin{tabular}{@{}lcccccc@{}}\n")
        f.write("Case & $m_3$ ($M_\\odot$) & $a_3$ (AU) & $(a_f/a_0)_{\\rm closed}$ & $(a_f/a_0)_{\\rm unclosed}$ & $\\langle |\\Delta L_{\\rm pair}|/L_0 \\rangle_{\\rm closed}$ & $\\langle |\\Delta L_{\\rm pair}|/L_0 \\rangle_{\\rm unclosed}$ \\\\\n")
        f.write("\\midrule\n")
        f.write(f"D & 0.3 & 8.0 & {closed_D['a_ratio_final']:.4f} & {unclosed_D['a_ratio_final']:.4f} & ${closed_D['mean_jump']:.2e}$ & ${unclosed_D['mean_jump']:.2e}$ \\\\\n")
        f.write(f"E & 0.8 & 5.0 & {closed_E['a_ratio_final']:.4f} & {unclosed_E['a_ratio_final']:.4f} & ${closed_E['mean_jump']:.2e}$ & ${unclosed_E['mean_jump']:.2e}$ \\\\\n")
        f.write("\\end{tabular}\n")


def makeFigure(closed_D, unclosed_D, closed_E, unclosed_E):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.linewidth": 1.0,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=200, sharey=True)

    isolated_baseline = 0.8190

    panel_data = [
        (
            axes[0],
            closed_D,
            unclosed_D,
            r"D. Weak perturber: $m_3 = 0.3\,M_\odot$, $a_3 = 8.0$ AU"
        ),
        (
            axes[1],
            closed_E,
            unclosed_E,
            r"E. Stronger perturber: $m_3 = 0.8\,M_\odot$, $a_3 = 5.0$ AU"
        ),
    ]

    for ax, closed_data, unclosed_data, title in panel_data:
        plot_stride = 2
        t_plot = closed_data["times"][::plot_stride]

        ax.plot(
            t_plot,
            closed_data["a_ratio_series_plot"][::plot_stride],
            color="#0173B2",
            linewidth=1.6,
            label="Closed"
        )
        ax.plot(
            t_plot,
            unclosed_data["a_ratio_series_plot"][::plot_stride],
            color="#CC3311",
            linewidth=1.6,
            linestyle="--",
            label="Unclosed"
        )
        ax.axhline(
            isolated_baseline,
            color="#555555",
            linewidth=1.2,
            linestyle=":",
            label="Isolated 2-body baseline"
        )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Time [inner-binary orbital periods, $T_0$]")
        ax.set_xlim(0, 50)
        ax.set_ylim(0.79, 1.01)

    axes[0].set_ylabel(r"Inner-binary osculating semi-major axis, $a_{\rm osc}(t)/a_0$")
    axes[1].legend(loc="lower left", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    fig.savefig("fig5.jpg", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    title = "Three-Body Validation (Hierarchical Triple)"
    subtitle = "Weak & stronger perturbers; closed vs unclosed mass transfer."
    if _RICH and Panel is not None:
        console.print(Panel.fit(f"[bold]{title}[/]\n{subtitle}", border_style="cyan"))
    else:
        print("=" * 70 + f"\n{title}\n{subtitle}\n" + "=" * 70)

    # n_orbits=50, steps_per_orbit=4000 → 200 000 steps per run
    n_steps_each = 50 * 4000

    prog = _mk_progress()
    if prog is None:
        print("Case D | closed")
        closed_D = runThreeBodyCase("D", 0.3, 8.0, True)
        print("Case D | unclosed")
        unclosed_D = runThreeBodyCase("D", 0.3, 8.0, False)
        print("Case E | closed")
        closed_E = runThreeBodyCase("E", 0.8, 5.0, True)
        print("Case E | unclosed")
        unclosed_E = runThreeBodyCase("E", 0.8, 5.0, False)
    else:
        with prog:
            tDc = prog.add_task("Case D (m3=0.3, a3=8 AU) | closed",   total=n_steps_each)
            closed_D   = runThreeBodyCase("D", 0.3, 8.0, True,  progress=prog, task_id=tDc)
            prog.update(tDc, completed=n_steps_each)

            tDu = prog.add_task("Case D (m3=0.3, a3=8 AU) | unclosed", total=n_steps_each)
            unclosed_D = runThreeBodyCase("D", 0.3, 8.0, False, progress=prog, task_id=tDu)
            prog.update(tDu, completed=n_steps_each)

            tEc = prog.add_task("Case E (m3=0.8, a3=5 AU) | closed",   total=n_steps_each)
            closed_E   = runThreeBodyCase("E", 0.8, 5.0, True,  progress=prog, task_id=tEc)
            prog.update(tEc, completed=n_steps_each)

            tEu = prog.add_task("Case E (m3=0.8, a3=5 AU) | unclosed", total=n_steps_each)
            unclosed_E = runThreeBodyCase("E", 0.8, 5.0, False, progress=prog, task_id=tEu)
            prog.update(tEu, completed=n_steps_each)

    if _RICH and Table is not None:
        t = Table(title="Three-Body Results")
        for col in ("Case", "m3 (M_sun)", "a3 (AU)", "(af/a0) closed", "(af/a0) unclosed", "<|dL|/L0> closed", "<|dL|/L0> unclosed"):
            t.add_column(col, justify="right" if col != "Case" else "left")
        t.add_row("D", "0.3", "8.0",
                  f"{closed_D['a_ratio_final']:.4f}", f"{unclosed_D['a_ratio_final']:.4f}",
                  f"{closed_D['mean_jump']:.2e}",     f"{unclosed_D['mean_jump']:.2e}")
        t.add_row("E", "0.8", "5.0",
                  f"{closed_E['a_ratio_final']:.4f}", f"{unclosed_E['a_ratio_final']:.4f}",
                  f"{closed_E['mean_jump']:.2e}",     f"{unclosed_E['mean_jump']:.2e}")
        console.print(t)
    else:
        print(f"D: closed={closed_D['a_ratio_final']:.4f}, unclosed={unclosed_D['a_ratio_final']:.4f}")
        print(f"E: closed={closed_E['a_ratio_final']:.4f}, unclosed={unclosed_E['a_ratio_final']:.4f}")

    console.print("\n[bold]Saving outputs...[/]") if _RICH else print("\nSaving outputs...")
    saveOutputs(closed_D, unclosed_D, closed_E, unclosed_E)

    console.print("[bold]Generating figure...[/]") if _RICH else print("Generating figure...")
    makeFigure(closed_D, unclosed_D, closed_E, unclosed_E)

    if _RICH and Panel is not None:
        console.print(Panel.fit("[bold green]Three-body validation complete.[/]", border_style="green"))
    else:
        print("Three-body validation complete.")


if __name__ == "__main__":
    main()