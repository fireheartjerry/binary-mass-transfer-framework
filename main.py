import json, os, warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

COLORS = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "purple": "#7E1E9C",
    "green": "#029E73",
    "red": "#CC3311",
    "gray": "#555555",
}


@dataclass
class Body:
    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):
        self.mass = float(self.mass)
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)


class BinarySystem:
    def __init__(self, donor: Body, accretor: Body):
        self.donor, self.accretor = donor, accretor

    def _com(self) -> Tuple[np.ndarray, np.ndarray]:
        M = self.donor.mass + self.accretor.mass
        R = (self.donor.mass * self.donor.position + self.accretor.mass * self.accretor.position) / M
        V = (self.donor.mass * self.donor.velocity + self.accretor.mass * self.accretor.velocity) / M
        return R, V

    def shift_to_com_frame(self):
        R, V = self._com()
        self.donor.position -= R
        self.accretor.position -= R
        self.donor.velocity -= V
        self.accretor.velocity -= V

    def compute_separation(self) -> float:
        return float(np.linalg.norm(self.accretor.position - self.donor.position))

    def compute_linear_momentum(self) -> np.ndarray:
        return self.donor.mass * self.donor.velocity + self.accretor.mass * self.accretor.velocity

    def compute_angular_momentum(self) -> float:
        return float(
            self.donor.mass * np.cross(self.donor.position, self.donor.velocity)
            + self.accretor.mass * np.cross(self.accretor.position, self.accretor.velocity)
        )

    def compute_energy(self) -> float:
        r = self.compute_separation()
        ke = 0.5 * self.donor.mass * float(self.donor.velocity @ self.donor.velocity)
        ke += 0.5 * self.accretor.mass * float(self.accretor.velocity @ self.accretor.velocity)
        pe = -G * self.donor.mass * self.accretor.mass / r
        return float(ke + pe)

    def compute_osculating_semimajor_axis(self) -> float:
        r = self.compute_separation()
        M = self.donor.mass + self.accretor.mass
        vrel = self.accretor.velocity - self.donor.velocity
        eps = 0.5 * float(vrel @ vrel) - G * M / r
        return float(-G * M / (2 * eps))

    def compute_gravitational_acceleration(self) -> Tuple[np.ndarray, np.ndarray]:
        rvec = self.accretor.position - self.donor.position
        r = np.linalg.norm(rvec)
        rhat = rvec / r
        F = G * self.donor.mass * self.accretor.mass / r**2
        return (F / self.donor.mass) * rhat, -(F / self.accretor.mass) * rhat

    def velocity_verlet_step(self, dt: float):
        a1, a2 = self.compute_gravitational_acceleration()
        self.donor.position += self.donor.velocity * dt + 0.5 * a1 * dt**2
        self.accretor.position += self.accretor.velocity * dt + 0.5 * a2 * dt**2
        a1n, a2n = self.compute_gravitational_acceleration()
        self.donor.velocity += 0.5 * (a1 + a1n) * dt
        self.accretor.velocity += 0.5 * (a2 + a2n) * dt

    @staticmethod
    def _decompose_perp(v: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rn = np.linalg.norm(r)
        if rn < 1e-30:
            return np.zeros(2), v.copy()
        rhat = r / rn
        vr = float(v @ rhat) * rhat
        return vr, v - vr

    def _enforce_L_perp(self, L_target: float):
        v1r, v1p = self._decompose_perp(self.donor.velocity, self.donor.position)
        v2r, v2p = self._decompose_perp(self.accretor.velocity, self.accretor.position)
        Lcur = float(
            self.donor.mass * np.cross(self.donor.position, v1p)
            + self.accretor.mass * np.cross(self.accretor.position, v2p)
        )
        if abs(Lcur) < 1e-30:
            return
        s = L_target / Lcur
        self.donor.velocity = v1r + v1p * s
        self.accretor.velocity = v2r + v2p * s

    def transfer_mass(self, dm: float, enforce_angular_momentum: bool = True, L_target: float | None = None):
        if dm <= 0:
            return
        self.donor.mass -= dm
        self.accretor.mass += dm
        self.shift_to_com_frame()
        if enforce_angular_momentum and L_target is not None:
            self._enforce_L_perp(L_target)
            self.shift_to_com_frame()


def setup_circular_orbit(M1: float, M2: float, a0: float) -> BinarySystem:
    M = M1 + M2
    r1, r2 = -a0 * M2 / M, a0 * M1 / M
    v = np.sqrt(G * M / a0)
    v1, v2 = v * M2 / M, -v * M1 / M
    sys = BinarySystem(Body(M1, [r1, 0.0], [0.0, v1]), Body(M2, [r2, 0.0], [0.0, v2]))
    assert np.linalg.norm(sys.compute_linear_momentum()) / (M * v) < 1e-14
    return sys


def compute_orbital_period(M_total: float, a: float) -> float:
    return float(2 * np.pi * np.sqrt(a**3 / (G * M_total)))


def analytical_final_separation(M1_i, M2_i, M1_f, M2_f) -> float:
    return float((M1_i * M2_i / (M1_f * M2_f)) ** 2)


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


def run_simulation(
    q0: float,
    M2: float = 1.0 * M_SUN,
    a0: float = 1.0 * AU,
    mass_transfer_fraction: float = 0.15,
    n_orbits: float = 50,
    steps_per_orbit: int = 1000,
    n_transfer_steps: int = 500,
    enforce_angular_momentum: bool = True,
    progress=None,
    task_id=None,
) -> Dict:
    M1, Mtot = q0 * M2, q0 * M2 + M2
    sys = setup_circular_orbit(M1, M2, a0)
    v0 = np.sqrt(G * Mtot / a0)
    P_scale = Mtot * v0

    L0, E0, a0_osc = sys.compute_angular_momentum(), sys.compute_energy(), sys.compute_osculating_semimajor_axis()
    T0 = compute_orbital_period(Mtot, a0)
    dt = T0 / steps_per_orbit
    n_steps = int(n_orbits * T0 / dt)

    total_dm = mass_transfer_fraction * M1
    dm = total_dm / n_transfer_steps
    interval = max(1, n_steps // n_transfer_steps)
    sample_every = max(1, steps_per_orbit // 10)

    times, a_series, L_series, E_series = [], [], [], []
    transfer_count = 0

    for step in range(n_steps):
        if step % sample_every == 0:
            times.append(step * dt / T0)
            a_series.append(sys.compute_osculating_semimajor_axis() / a0_osc)
            L_series.append(sys.compute_angular_momentum() / L0)
            E_series.append(sys.compute_energy() / abs(E0))
        if progress is not None and task_id is not None:
            progress.advance(task_id, 1)
        sys.velocity_verlet_step(dt)
        if transfer_count < n_transfer_steps:
            target = int((step + 1) / interval)
            while transfer_count < target and transfer_count < n_transfer_steps:
                sys.transfer_mass(dm, enforce_angular_momentum, L0)
                transfer_count += 1

    a_final = sys.compute_osculating_semimajor_axis()
    L_final = sys.compute_angular_momentum()
    P_rel = float(np.linalg.norm(sys.compute_linear_momentum()) / P_scale)

    M1f, M2f = M1 - total_dm, M2 + total_dm
    a_ana = analytical_final_separation(M1, M2, M1f, M2f)
    a_num = float(a_final / a0_osc)
    err = abs(a_num - a_ana) / a_ana * 100

    return dict(
        q0=q0,
        enforce=enforce_angular_momentum,
        times=np.asarray(times),
        separations=np.asarray(a_series),
        angular_momenta=np.asarray(L_series),
        energies=np.asarray(E_series),
        a_ratio_numerical=a_num,
        a_ratio_analytical=a_ana,
        error_percent=err,
        L_ratio=float(L_final / L0),
        P_final_relative=P_rel,
    )


def _save_cache(path: str, enforced, non_enforced):
    def pack(r):
        d = dict(r)
        for k in ("times", "separations", "angular_momenta", "energies"):
            d[k] = d[k].tolist()
        return d

    with open(path, "w") as f:
        json.dump({"enforced": [pack(r) for r in enforced], "non_enforced": [pack(r) for r in non_enforced]}, f, indent=2)


def _load_cache(path: str):
    with open(path, "r") as f:
        d = json.load(f)
    for grp in ("enforced", "non_enforced"):
        for r in d.get(grp, []):
            for k in ("times", "separations", "angular_momenta", "energies"):
                r[k] = np.asarray(r[k])
    return d.get("enforced", []), d.get("non_enforced", [])


def _print_tables(enf, non, mass_ratios, case_names, behaviors):
    if _RICH and Table is not None:
        t1 = Table(title="Table 1: Validation (15% transfer, L enforced)")
        for c in ("Case", "q0", "(af/a0)_num", "(af/a0)_ana", "Error (%)", "Behavior"):
            t1.add_column(c, justify="right" if c not in ("Case", "Behavior") else "left")
        for r, name, beh in zip(enf, case_names, behaviors):
            t1.add_row(name, f"{r['q0']:.1f}", f"{r['a_ratio_numerical']:.4f}", f"{r['a_ratio_analytical']:.4f}", f"{r['error_percent']:.4f}", beh)
        console.print(t1)

        t2 = Table(title="Table 2: Enforced vs Non-enforced")
        for c in ("Case", "q0", "Err(enf) %", "Err(non-enf) %", "L/L0(non-enf)", "|P|/P_scale"):
            t2.add_column(c, justify="right" if c != "Case" else "left")
        for i, name in enumerate(case_names):
            rE, rN = enf[i], non[i]
            t2.add_row(name, f"{mass_ratios[i]:.1f}", f"{rE['error_percent']:.4f}", f"{rN['error_percent']:.2f}", f"{rN['L_ratio']:.4f}", f"{rN['P_final_relative']:.2e}")
        console.print(t2)
        return

    print("\n" + "=" * 70)
    print("TABLE 1: Validation results for 15% mass transfer (L enforced)")
    print("=" * 70)
    print(f"{'Case':<6} {'q0':<6} {'(af/a0)_num':<14} {'(af/a0)_ana':<14} {'Error':<12} {'Behavior'}")
    print("-" * 70)
    for r, name, beh in zip(enf, case_names, behaviors):
        print(f"{name:<6} {r['q0']:<6.1f} {r['a_ratio_numerical']:<14.4f} {r['a_ratio_analytical']:<14.4f} {r['error_percent']:<12.4f}% {beh}")

    print("\n" + "=" * 70)
    print("TABLE 2: Enforced vs Non-enforced comparison")
    print("=" * 70)
    print(f"{'Case':<6} {'q0':<6} {'Err(enf)':<12} {'Err(non-enf)':<14} {'L/L0(non-enf)':<14} {'|P|/P_scale'}")
    print("-" * 70)
    for i, name in enumerate(case_names):
        rE, rN = enf[i], non[i]
        print(f"{name:<6} {mass_ratios[i]:<6.1f} {rE['error_percent']:<12.4f}% {rN['error_percent']:<14.2f}% {rN['L_ratio']:<14.4f} {rN['P_final_relative']:.2e}")


def generate_all_results(output_dir: str = ".", use_cache: bool = True):
    mass_ratios = [0.5, 1.0, 2.0]
    case_names = ["A", "B", "C"]
    behaviors = ["Expansion", "Weak expansion", "Contraction"]
    cache = os.path.join(output_dir, "results_fixed.json")

    title = "Conservative Mass Transfer Simulation (V5)"
    subtitle = "COM frame enforced; optional tangential rescale to close angular momentum."
    if _RICH and Panel is not None:
        console.print(Panel.fit(f"[bold]{title}[/]\n{subtitle}", border_style="cyan"))
    else:
        print("=" * 70 + f"\n{title}\n{subtitle}\n" + "=" * 70)

    enforced, non_enforced = [], []
    if use_cache and os.path.exists(cache):
        try:
            enforced, non_enforced = _load_cache(cache)
            if len(enforced) == len(non_enforced) == 3:
                console.print(f"[green]Loaded cache:[/] {cache}") if _RICH else print(f"Loaded cache: {cache}")
        except Exception as e:
            enforced, non_enforced = [], []
            console.print(f"[yellow]Cache read failed[/]: {e}. Regenerating.") if _RICH else print(f"Cache read failed: {e}. Regenerating.")

    if not enforced:
        n_steps = 50 * 1000
        prog = _mk_progress()
        if prog is None:
            for q0, name in zip(mass_ratios, case_names):
                print(f"Case {name} (q0={q0})")
                rE = run_simulation(q0, enforce_angular_momentum=True)
                rN = run_simulation(q0, enforce_angular_momentum=False)
                enforced.append(rE)
                non_enforced.append(rN)
                print(f"  enforced:      L/L0={rE['L_ratio']:.6f}, |P|/P={rE['P_final_relative']:.2e}")
                print(f"  non-enforced:  L/L0={rN['L_ratio']:.6f}, |P|/P={rN['P_final_relative']:.2e}")
        else:
            with prog:
                for q0, name in zip(mass_ratios, case_names):
                    tE = prog.add_task(f"Case {name} (q0={q0}) | enforced", total=n_steps)
                    rE = run_simulation(q0, enforce_angular_momentum=True, progress=prog, task_id=tE)
                    prog.update(tE, completed=n_steps)

                    tN = prog.add_task(f"Case {name} (q0={q0}) | non-enforced", total=n_steps)
                    rN = run_simulation(q0, enforce_angular_momentum=False, progress=prog, task_id=tN)
                    prog.update(tN, completed=n_steps)

                    enforced.append(rE)
                    non_enforced.append(rN)

        if use_cache:
            try:
                _save_cache(cache, enforced, non_enforced)
                console.print(f"[green]Saved cache:[/] {cache}") if _RICH else print(f"Saved cache: {cache}")
            except Exception as e:
                console.print(f"[yellow]Cache save failed[/]: {e}") if _RICH else print(f"Cache save failed: {e}")

    _print_tables(enforced, non_enforced, mass_ratios, case_names, behaviors)

    console.print("\n[bold]Generating figures...[/]") if _RICH else print("\nGenerating figures...")
    colors = [COLORS["blue"], COLORS["orange"], COLORS["purple"]]
    labels = [r"$q_0 = 0.5$ (expansion)", r"$q_0 = 1.0$ (weak expansion)", r"$q_0 = 2.0$ (contraction)"]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for r, c, lab in zip(enforced, colors, labels):
        ax.plot(r["times"], r["separations"], color=c, label=lab, linewidth=2.2, alpha=0.9)
    ax.set(xlabel=r"Time [orbital periods, $T_0$]", ylabel=r"Normalized Semi-major Axis, $a_{\rm osc}(t) / a_0$", title="Orbital Evolution Under Conservative Mass Transfer", xlim=(0, 50), ylim=(0.82, 1.32))
    ax.legend(loc="best", framealpha=0.98)
    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/figure1_osculating_sma.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, (axL, axE) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(3)
    mean_err, max_err = [], []
    for r in enforced:
        dL = np.maximum(np.abs(r["angular_momenta"] - 1.0), 1e-17)
        mean_err.append(float(np.mean(dL)))
        max_err.append(float(np.max(dL)))
    axL.bar(x, mean_err, 0.5, color=colors, edgecolor="black", linewidth=1.2, alpha=0.8, label="Mean error")
    axL.scatter(x, max_err, s=100, color="black", marker="v", zorder=5, label="Max error")
    axL.axhline(y=2.2e-16, color=COLORS["gray"], linestyle="--", linewidth=2, label=r"Machine $\epsilon$")
    axL.set(yscale="log", xlabel=r"Initial Mass Ratio, $q_0$", ylabel=r"Relative Angular Momentum Error, $|\Delta L / L_0|$", title="Angular Momentum Conservation", ylim=(1e-17, 1e-13))
    axL.set_xticks(x, ["0.5", "1.0", "2.0"])
    axL.legend(loc="upper right", framealpha=0.98)
    axL.minorticks_on()
    axL.text(0.5, 0.15, "All cases conserve $L$ to\nmachine precision", transform=axL.transAxes, ha="center", fontsize=11, color=COLORS["green"], fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=COLORS["green"], linewidth=1.5, alpha=0.95))

    for r, c, lab in zip(enforced, colors, labels):
        dE = r["energies"] - r["energies"][0]
        axE.plot(r["times"], dE, color=c, label=lab, linewidth=2.2, alpha=0.9)
    axE.set(xlabel=r"Time [orbital periods, $T_0$]", ylabel=r"Relative Energy Change, $\Delta E / |E_0|$", title="Energy Evolution", xlim=(0, 50))
    axE.legend(loc="best", framealpha=0.98)
    axE.minorticks_on()
    fig.tight_layout(pad=2.0)
    fig.savefig(f"{output_dir}/figure2_conservation.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    width = 0.38
    num = [r["a_ratio_numerical"] for r in enforced]
    ana = [r["a_ratio_analytical"] for r in enforced]
    err = [r["error_percent"] for r in enforced]
    ax.bar(x - width / 2, num, width, label="Numerical Simulation", color=COLORS["blue"], edgecolor="black", linewidth=1.2, alpha=0.85)
    ax.bar(x + width / 2, ana, width, label="Analytical Prediction", color=COLORS["orange"], edgecolor="black", linewidth=1.2, alpha=0.85, hatch="//")
    for i, (n, a, e) in enumerate(zip(num, ana, err)):
        ax.annotate(f"Δ = {e:.4f}%", xy=(i, max(n, a) + 0.04), ha="center", fontsize=10, color=COLORS["green"], fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=COLORS["green"], linewidth=1.5, alpha=0.9))
    ax.set(xlabel=r"Initial Mass Ratio, $q_0 = M_1/M_2$", ylabel=r"Final Semi-major Axis Ratio, $a_f/a_0$", title="Numerical Validation Against Analytical Predictions", ylim=(0, 1.45))
    ax.set_xticks(x, ["0.5", "1.0", "2.0"])
    ax.legend(loc="upper left", framealpha=0.98)
    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/figure3_comparison.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    idx = 2
    rE, rN = enforced[idx], non_enforced[idx]
    a_ana = rE["a_ratio_analytical"]
    ax.axhline(y=a_ana, color=COLORS["green"], linewidth=2.5, label=f"Analytical prediction: {a_ana:.4f}", zorder=1)
    ax.axhspan(a_ana * 0.999, a_ana * 1.001, alpha=0.15, color=COLORS["green"], zorder=0)
    ax.plot(rE["times"], rE["separations"], color=COLORS["blue"], linewidth=2.8, label="With $L$ enforcement", alpha=0.95, zorder=2)
    ax.plot(rN["times"], rN["separations"], color=COLORS["red"], linewidth=2.8, linestyle="--", label="Without $L$ enforcement", alpha=0.95, zorder=3)
    ax.annotate(f"Error: {rE['error_percent']:.4f}%", xy=(46, rE["separations"][-1] + 0.012), fontsize=10, color=COLORS["blue"], fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=COLORS["blue"], linewidth=1.5, alpha=0.95))
    ax.annotate(f"Error: {rN['error_percent']:.2f}%", xy=(46, rN["separations"][-1] - 0.022), fontsize=10, color=COLORS["red"], fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=COLORS["red"], linewidth=1.5, alpha=0.95))
    ax.set(xlabel=r"Time [orbital periods, $T_0$]", ylabel=r"Normalized Semi-major Axis, $a_{\rm osc}(t) / a_0$", title=r"Impact of Angular Momentum Enforcement ($q_0 = 2.0$, 15% Mass Transfer)", xlim=(0, 50))
    y_min = min(float(rE["separations"].min()), float(rN["separations"].min()), a_ana) - 0.05
    y_max = max(float(rE["separations"].max()), float(rN["separations"].max()), a_ana) + 0.05
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper right", framealpha=0.98, fontsize=11)
    ax.minorticks_on()
    ax.text(0.028, 0.30, "Without enforcement, discrete mass transfer\ncauses angular momentum to drift at each step.\nThis accumulated drift, not integrator error,\ndrives the deviation from the analytical line.", transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF8DC", edgecolor=COLORS["gray"], linewidth=1.5, alpha=0.95), linespacing=1.5)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/figure4_failure_modes.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

    if _RICH and Panel is not None:
        console.print(Panel.fit("[bold green]All simulations and figures complete.[/]", border_style="green"))
    else:
        print("All simulations and figures complete.")
    return dict(enforced=enforced, non_enforced=non_enforced, mass_ratios=mass_ratios, case_names=case_names)


if __name__ == "__main__":
    generate_all_results(output_dir=".", use_cache=True)
