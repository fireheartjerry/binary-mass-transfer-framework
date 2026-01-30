
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass
from typing import Dict, Tuple
import json
import os
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console

# Initialize rich console for better output
console = Console()

# =============================================================================
# Plotting Configuration
# =============================================================================
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['xtick.major.width'] = 1.2
matplotlib.rcParams['ytick.major.width'] = 1.2
matplotlib.rcParams['lines.linewidth'] = 2.0

# =============================================================================
# Physical Constants (SI units)
# =============================================================================
G = 6.67430e-11          # Gravitational constant [m³ kg⁻¹ s⁻²]
M_SUN = 1.9891e30        # Solar mass [kg]
AU = 1.496e11            # Astronomical unit [m]
YEAR = 365.25 * 24 * 3600  # Year [s]

# =============================================================================
# Body Class
# =============================================================================
@dataclass
class Body:
    """Point mass with position and velocity in 3D (z=0 for planar orbits)."""
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    
    def __post_init__(self):
        self.mass = float(self.mass)
        pos = np.array(self.position, dtype=float)
        vel = np.array(self.velocity, dtype=float)
        if len(pos) == 2:
            pos = np.array([pos[0], pos[1], 0.0])
        if len(vel) == 2:
            vel = np.array([vel[0], vel[1], 0.0])
        self.position = pos
        self.velocity = vel

# =============================================================================
# Binary System Class
# =============================================================================
class BinarySystem:
    """
    Binary star system simulator with conservative mass transfer.
    
    Uses Velocity Verlet integration for orbit evolution.
    Angular momentum conservation is ENFORCED by velocity rescaling
    after each mass transfer step.
    """
    
    def __init__(self, donor: Body, accretor: Body):
        self.donor = donor      # M₁ (loses mass)
        self.accretor = accretor  # M₂ (gains mass)
    
    def compute_separation(self) -> float:
        """Compute instantaneous separation r = |r₂ - r₁|."""
        return np.linalg.norm(self.accretor.position - self.donor.position)
    
    def compute_total_mass(self) -> float:
        """Compute total mass M = M₁ + M₂."""
        return self.donor.mass + self.accretor.mass
    
    def compute_relative_velocity(self) -> np.ndarray:
        """Compute relative velocity v_rel = v₂ - v₁."""
        return self.accretor.velocity - self.donor.velocity
    
    def compute_osculating_semimajor_axis(self) -> float:
        """
        Compute osculating semi-major axis from instantaneous orbital energy.
        
        For the two-body problem in the center-of-mass frame:
            ε = (1/2)|v_rel|² - GM/r    (specific orbital energy)
            a_osc = -GM / (2ε)
        
        This gives a smooth orbital element even if the orbit has e > 0.
        """
        r = self.compute_separation()
        v_rel = self.compute_relative_velocity()
        v_rel_mag = np.linalg.norm(v_rel)
        M_total = self.compute_total_mass()
        
        # Specific orbital energy (per unit reduced mass)
        epsilon = 0.5 * v_rel_mag**2 - G * M_total / r
        
        # Semi-major axis from vis-viva
        if epsilon < 0:  # Bound orbit
            a_osc = -G * M_total / (2 * epsilon)
        else:
            a_osc = np.inf  # Unbound
        
        return a_osc
    
    def compute_angular_momentum(self) -> float:
        """
        Compute total orbital angular momentum (z-component).
        L = M₁(r₁ × v₁) + M₂(r₂ × v₂)
        """
        L1_vec = self.donor.mass * np.cross(self.donor.position, self.donor.velocity)
        L2_vec = self.accretor.mass * np.cross(self.accretor.position, self.accretor.velocity)
        return (L1_vec + L2_vec)[2]
    
    def compute_energy(self) -> float:
        """Compute total mechanical energy E = KE + PE."""
        r = self.compute_separation()
        KE1 = 0.5 * self.donor.mass * np.dot(self.donor.velocity, self.donor.velocity)
        KE2 = 0.5 * self.accretor.mass * np.dot(self.accretor.velocity, self.accretor.velocity)
        PE = -G * self.donor.mass * self.accretor.mass / r
        return KE1 + KE2 + PE
    
    def compute_gravitational_acceleration(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gravitational accelerations on both bodies."""
        r_vec = self.accretor.position - self.donor.position
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r
        
        F = G * self.donor.mass * self.accretor.mass / r**2
        a1 = (F / self.donor.mass) * r_hat
        a2 = -(F / self.accretor.mass) * r_hat
        
        return a1, a2
    
    def transfer_mass_conservative(self, dm: float) -> float:
        """
        Conservative mass transfer with ENFORCED angular momentum conservation.
        
        After changing masses, we rescale velocities to maintain L = const.
        This constructs a controlled numerical experiment consistent with
        the classical assumption dL/dt = 0.
        
        Returns:
            Actual mass transferred (may be less than dm near end)
        """
        if dm <= 0:
            return 0.0
        
        # Don't transfer more than available
        actual_dm = min(dm, self.donor.mass * 0.99)  # Keep 1% minimum
        if actual_dm <= 0:
            return 0.0
        
        # Record L before transfer
        L_before = self.compute_angular_momentum()
        
        # Update masses
        self.donor.mass -= actual_dm
        self.accretor.mass += actual_dm
        
        # Compute L after mass change (same velocities)
        L_after = self.compute_angular_momentum()
        
        # Rescale velocities to restore L
        if abs(L_after) > 1e-30:
            scale = L_before / L_after
            self.donor.velocity *= scale
            self.accretor.velocity *= scale
        
        return actual_dm
    
    def velocity_verlet_step(self, dt: float):
        """Perform one Velocity Verlet integration step."""
        a1_old, a2_old = self.compute_gravitational_acceleration()
        
        self.donor.position += self.donor.velocity * dt + 0.5 * a1_old * dt**2
        self.accretor.position += self.accretor.velocity * dt + 0.5 * a2_old * dt**2
        
        a1_new, a2_new = self.compute_gravitational_acceleration()
        
        self.donor.velocity += 0.5 * (a1_old + a1_new) * dt
        self.accretor.velocity += 0.5 * (a2_old + a2_new) * dt


# =============================================================================
# Initial Conditions
# =============================================================================
def setup_circular_orbit(M1: float, M2: float, a0: float) -> BinarySystem:
    """Set up binary in circular orbit (center-of-mass frame)."""
    M_total = M1 + M2
    
    r1 = -a0 * M2 / M_total
    r2 = a0 * M1 / M_total
    
    v_orbital = np.sqrt(G * M_total / a0)
    v1 = v_orbital * M2 / M_total
    v2 = v_orbital * M1 / M_total
    
    donor = Body(mass=M1, position=np.array([r1, 0.0]), velocity=np.array([0.0, v1]))
    accretor = Body(mass=M2, position=np.array([r2, 0.0]), velocity=np.array([0.0, -v2]))
    
    return BinarySystem(donor, accretor)


# =============================================================================
# Analytical Predictions
# =============================================================================
def analytical_final_separation(M1_0: float, M2_0: float, a0: float, f: float) -> float:
    """
    Analytical prediction for conservative mass transfer.
    
    From L = μ√(GMa) = const and M = const:
        a_f/a_0 = (M₁₀·M₂₀ / M₁_f·M₂_f)²
    """
    M1_f = M1_0 * (1 - f)
    M2_f = M2_0 + f * M1_0
    ratio = (M1_0 * M2_0) / (M1_f * M2_f)
    return a0 * ratio**2


# =============================================================================
# Data Persistence
# =============================================================================
def save_results_to_json(results: Dict, filename: str = 'simulation_results.json'):
    """
    Save simulation results to JSON file for caching.
    Converts numpy arrays to lists for JSON serialization.
    """
    serializable_results = {}
    
    for case_name, result in results.items():
        case_data = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                case_data[key] = value.tolist()
            else:
                case_data[key] = value
        serializable_results[case_name] = case_data
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Saving results...", total=None)
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Saved results to [cyan]{filename}[/cyan]")


def load_results_from_json(filename: str = 'simulation_results.json') -> Dict:
    """
    Load simulation results from cached JSON file.
    Converts lists back to numpy arrays.
    """
    if not os.path.exists(filename):
        return None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Loading cached results...", total=None)
        with open(filename, 'r') as f:
            data = json.load(f)
        progress.update(task, completed=True)
    
    results = {}
    for case_name, case_data in data.items():
        result = {}
        for key, value in case_data.items():
            if isinstance(value, list) and key in [
                'times', 'a_osculating', 'separations', 
                'angular_momenta', 'energies', 'mass_ratios'
            ]:
                result[key] = np.array(value)
            else:
                result[key] = value
        results[case_name] = result
    
    console.print(f"[green]✓[/green] Loaded results from [cyan]{filename}[/cyan]")
    return results


# =============================================================================
# Simulation Runner
# =============================================================================
def run_simulation(M1_0: float, M2_0: float, a0: float,
                   mass_transfer_fraction: float,
                   n_orbits: int = 50,
                   steps_per_orbit: int = 1000,
                   sample_interval: int = 100,
                   progress: Progress = None,
                   task_id = None) -> Dict:
    """
    Run conservative mass transfer simulation.
    
    Returns dictionary with simulation results including osculating semi-major axis.
    """
    system = setup_circular_orbit(M1_0, M2_0, a0)
    
    M_total = M1_0 + M2_0
    T_orbit_initial = 2 * np.pi * np.sqrt(a0**3 / (G * M_total))
    
    total_time = n_orbits * T_orbit_initial
    dt = T_orbit_initial / steps_per_orbit
    n_steps = int(total_time / dt)
    
    total_mass_to_transfer = mass_transfer_fraction * M1_0
    dm_per_step = total_mass_to_transfer / n_steps
    
    # Storage
    times = []
    a_osculating = []  # Osculating semi-major axis (smooth!)
    separations = []   # Instantaneous separation (for comparison)
    angular_momenta = []
    energies = []
    mass_ratios = []
    
    # Initial values
    L0 = system.compute_angular_momentum()
    E0 = system.compute_energy()
    a0_actual = system.compute_osculating_semimajor_axis()
    
    # Track actual mass transferred
    total_mass_transferred = 0.0
    
    # Main loop with progress bar
    if progress is not None and task_id is not None:
        for step in range(n_steps):
            if step % sample_interval == 0:
                times.append(step * dt)
                a_osculating.append(system.compute_osculating_semimajor_axis())
                separations.append(system.compute_separation())
                angular_momenta.append(system.compute_angular_momentum())
                energies.append(system.compute_energy())
                mass_ratios.append(system.donor.mass / system.accretor.mass)
            
            system.velocity_verlet_step(dt)
            
            dm_actual = system.transfer_mass_conservative(dm_per_step)
            total_mass_transferred += dm_actual
            
            progress.update(task_id, advance=1)
    else:
        # Fallback: run without progress bar
        for step in range(n_steps):
            if step % sample_interval == 0:
                times.append(step * dt)
                a_osculating.append(system.compute_osculating_semimajor_axis())
                separations.append(system.compute_separation())
                angular_momenta.append(system.compute_angular_momentum())
                energies.append(system.compute_energy())
                mass_ratios.append(system.donor.mass / system.accretor.mass)
            
            system.velocity_verlet_step(dt)
            
            dm_actual = system.transfer_mass_conservative(dm_per_step)
            total_mass_transferred += dm_actual
    
    # Final recording
    times.append(n_steps * dt)
    a_osculating.append(system.compute_osculating_semimajor_axis())
    separations.append(system.compute_separation())
    angular_momenta.append(system.compute_angular_momentum())
    energies.append(system.compute_energy())
    mass_ratios.append(system.donor.mass / system.accretor.mass)
    
    # Convert to arrays
    times = np.array(times)
    a_osculating = np.array(a_osculating)
    separations = np.array(separations)
    angular_momenta = np.array(angular_momenta)
    energies = np.array(energies)
    
    # Effective mass transfer fraction (actual)
    f_effective = total_mass_transferred / M1_0
    
    # Analytical prediction using effective f
    a_analytical = analytical_final_separation(M1_0, M2_0, a0, f_effective)
    
    return {
        'times': times,
        'a_osculating': a_osculating,
        'separations': separations,
        'angular_momenta': angular_momenta,
        'energies': energies,
        'mass_ratios': np.array(mass_ratios),
        'a0': a0_actual,
        'L0': L0,
        'E0': E0,
        'a_final_numerical': a_osculating[-1],
        'a_final_analytical': a_analytical,
        'L_final': angular_momenta[-1],
        'E_final': energies[-1],
        'q0': M1_0 / M2_0,
        'f_requested': mass_transfer_fraction,
        'f_effective': f_effective,
        'T_orbit_initial': T_orbit_initial
    }


# =============================================================================
# Validation Test
# =============================================================================
def run_validation_test(verbose: bool = True) -> bool:
    """Validate integrator conserves E and L without mass transfer."""
    if verbose:
        console.rule("[bold cyan]VALIDATION: Integrator test (no mass transfer)[/bold cyan]")
    
    M1 = 1.0 * M_SUN
    M2 = 1.0 * M_SUN
    a0 = 1.0 * AU
    
    system = setup_circular_orbit(M1, M2, a0)
    
    E0 = system.compute_energy()
    L0 = system.compute_angular_momentum()
    
    T_orbit = 2 * np.pi * np.sqrt(a0**3 / (G * (M1 + M2)))
    dt = T_orbit / 1000
    n_steps = int(10 * T_orbit / dt)
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        transient=True
    ) as progress:
        task = progress.add_task("Validating integrator...", total=n_steps)
        
        for _ in range(n_steps):
            system.velocity_verlet_step(dt)
            progress.update(task, advance=1)
    
    E_final = system.compute_energy()
    L_final = system.compute_angular_momentum()
    
    dE = abs(E_final - E0) / abs(E0)
    dL = abs(L_final - L0) / abs(L0)
    
    if verbose:
        console.print(f"  Energy drift:    [yellow]{dE:.2e}[/yellow]")
        console.print(f"  L drift:         [yellow]{dL:.2e}[/yellow]")
    
    passed = dE < 1e-8 and dL < 1e-8
    if verbose:
        status = "[green]✓ PASSED[/green]" if passed else "[red]✗ FAILED[/red]"
        console.print(f"  Status:          {status}")
    
    return passed


# =============================================================================
# Main Cases
# =============================================================================
def run_all_cases(verbose: bool = True) -> Dict:
    """
    Run three cases demonstrating classical orbital response.
    
    Case A: q₀ = 0.5 → orbit expands
    Case B: q₀ = 1.0 → weak expansion (q drops below 1 immediately)
    Case C: q₀ = 2.0 → orbit contracts
    """
    if verbose:
        console.rule("[bold cyan]CONSERVATIVE MASS TRANSFER SIMULATIONS[/bold cyan]")
    
    M2 = 1.0 * M_SUN
    a0 = 1.0 * AU
    f = 0.15
    
    cases = {
        'A': {'M1': 0.5 * M_SUN, 'label': r'$q_0 = 0.5$ (expansion)'},
        'B': {'M1': 1.0 * M_SUN, 'label': r'$q_0 = 1.0$ (weak expansion)'},
        'C': {'M1': 2.0 * M_SUN, 'label': r'$q_0 = 2.0$ (contraction)'},
    }
    
    results = {}
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Running cases...", total=len(cases))
        
        for case_name, params in cases.items():
            if verbose:
                console.print(f"\n[bold]Case {case_name}:[/bold] {params['label']}")
            
            result = run_simulation(
                M1_0=params['M1'],
                M2_0=M2,
                a0=a0,
                mass_transfer_fraction=f,
                n_orbits=50,
                steps_per_orbit=1000,
                sample_interval=50,
                progress=progress,
                task_id=task
            )
            result['label'] = params['label']
            results[case_name] = result
            
            if verbose:
                a_num = result['a_final_numerical']
                a_ana = result['a_final_analytical']
                dL = (result['L_final'] - result['L0']) / result['L0']
                dE = (result['E_final'] - result['E0']) / abs(result['E0'])
                
                console.print(f"  q₀:              [cyan]{result['q0']:.2f}[/cyan]")
                console.print(f"  f_eff:           [cyan]{result['f_effective']:.4f}[/cyan]")
                console.print(f"  a_osc/a₀ (num):  [cyan]{a_num/result['a0']:.6f}[/cyan]")
                console.print(f"  a/a₀ (theory):   [cyan]{a_ana/result['a0']:.6f}[/cyan]")
                console.print(f"  Agreement:       [green]{100*abs(a_num-a_ana)/a_ana:.3f}%[/green]")
                console.print(f"  ΔL/L₀:           [yellow]{dL:.2e}[/yellow] (enforced)")
                console.print(f"  ΔE/|E₀|:         [yellow]{dE:.4f}[/yellow]")
            
            progress.update(task, advance=1)
    
    return results


# =============================================================================
# Figure Generation
# =============================================================================
def generate_figure1_osculating(results: Dict, save_path: str = None):
    """
    Figure 1: Osculating semi-major axis evolution.
    Smooth curves without phase-dependent oscillations.
    Professional styling for research papers.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Generating Figure 1...", total=None)
        
        fig, ax = plt.subplots(figsize=(10, 6.5))
        
        # Professional color palette (colorblind-friendly)
        colors = {'A': '#0173B2', 'B': '#DE8F05', 'C': '#CC78BC'}
        linestyles = {'A': '-', 'B': '-', 'C': '-'}
        
        for case_name in ['A', 'B', 'C']:
            result = results[case_name]
            t_norm = result['times'] / result['T_orbit_initial']
            a_norm = result['a_osculating'] / result['a0']
            
            ax.plot(t_norm, a_norm, color=colors[case_name], linestyle=linestyles[case_name],
                    linewidth=2.5, label=result['label'], marker=None)
        
        ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.set_xlabel(r'Time [$T_0$]', fontsize=13, labelpad=8)
        ax.set_ylabel(r'$a_{\rm osc}(t) / a_0$', fontsize=13, labelpad=8)
        ax.set_title('Orbital Evolution Under Conservative Mass Transfer', fontsize=15, pad=15)
        ax.legend(loc='best', fontsize=11, frameon=True, shadow=False, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, which='major')
        ax.set_axisbelow(True)
        
        # Improve spine visibility
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
        
        progress.update(task, completed=True)
    
    return fig


def generate_figure2_conservation(results: Dict, save_path: str = None):
    """
    Figure 2: Conservation diagnostics.
    Shows ΔL/L₀ residual (should be machine precision) and ΔE/|E₀|.
    Professional styling for research papers.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Generating Figure 2...", total=None)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        colors = {'A': '#0173B2', 'B': '#DE8F05', 'C': '#CC78BC'}
        linestyles = {'A': '-', 'B': '-', 'C': '-'}
        
        # Left: Angular momentum residual
        ax1 = axes[0]
        for case_name in ['A', 'B', 'C']:
            result = results[case_name]
            t_norm = result['times'] / result['T_orbit_initial']
            dL = np.abs((result['angular_momenta'] - result['L0']) / result['L0'])
            # Avoid log(0) by setting floor
            dL = np.maximum(dL, 1e-16)
            ax1.semilogy(t_norm, dL, color=colors[case_name], linestyle=linestyles[case_name],
                         linewidth=1.5, label=result['label'])
        
        ax1.set_xlabel(r'Time [$T_0$]', fontsize=13, labelpad=8)
        ax1.set_ylabel(r'$|\Delta L / L_0|$', fontsize=13, labelpad=8)
        ax1.set_title('Angular Momentum Conservation', fontsize=13, pad=10)
        ax1.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
        ax1.grid(True, alpha=0.25, which='both', linestyle='-', linewidth=0.5)
        ax1.set_axisbelow(True)
        ax1.set_ylim(1e-16, 1e-10)
        for spine in ax1.spines.values():
            spine.set_linewidth(1.2)
        
        # Right: Energy evolution
        ax2 = axes[1]
        for case_name in ['A', 'B', 'C']:
            result = results[case_name]
            t_norm = result['times'] / result['T_orbit_initial']
            dE = (result['energies'] - result['E0']) / np.abs(result['E0'])
            ax2.plot(t_norm, dE, color=colors[case_name], linestyle=linestyles[case_name],
                     linewidth=1.5, label=result['label'])
        
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
        ax2.set_xlabel(r'Time [$T_0$]', fontsize=13, labelpad=8)
        ax2.set_ylabel(r'$\Delta E / |E_0|$', fontsize=13, labelpad=8)
        ax2.set_title('Energy Evolution', fontsize=13, pad=10)
        ax2.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
        ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax2.set_axisbelow(True)
        for spine in ax2.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
        
        progress.update(task, completed=True)
    
    return fig


def generate_figure3_comparison(results: Dict, save_path: str = None):
    """Figure 3: Numerical vs analytical comparison. Professional styling."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Generating Figure 3...", total=None)
        
        fig, ax = plt.subplots(figsize=(10, 6.5))
        
        q0_values = []
        a_numerical = []
        a_analytical = []
        labels = []
        
        for case_name in ['A', 'B', 'C']:
            result = results[case_name]
            q0_values.append(result['q0'])
            a_numerical.append(result['a_final_numerical'] / result['a0'])
            a_analytical.append(result['a_final_analytical'] / result['a0'])
            labels.append(f"{result['q0']:.1f}")
        
        x = np.arange(len(q0_values))
        width = 0.38
        
        # Professional colors
        color_numerical = '#0173B2'
        color_analytical = '#DE8F05'
        
        bars1 = ax.bar(x - width/2, a_numerical, width, label='Numerical', color=color_numerical, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, a_analytical, width, label='Analytical', color=color_analytical, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel(r'Initial Mass Ratio $q_0 = M_1/M_2$', fontsize=13, labelpad=8)
        ax.set_ylabel(r'Final Semi-major Axis $a_f / a_0$', fontsize=13, labelpad=8)
        ax.set_title('Numerical vs Analytical Predictions', fontsize=15, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=False, edgecolor='black')
        ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.grid(True, alpha=0.25, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        # Add percentage difference annotations
        for i, (num, ana) in enumerate(zip(a_numerical, a_analytical)):
            diff = 100 * abs(num - ana) / ana
            y_pos = max(num, ana) + 0.025
            ax.text(i, y_pos, f'Δ={diff:.2f}%', ha='center', fontsize=9.5, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
        
        progress.update(task, completed=True)
    
    return fig


def generate_summary_table(results: Dict):
    """Print summary table."""
    print("\n" + "=" * 95)
    print("TABLE 1: Summary of Simulation Results (f_requested = 0.15)")
    print("=" * 95)
    header = f"{'Case':<6} {'q₀':<8} {'f_eff':<10} {'a_num/a₀':<12} {'a_ana/a₀':<12} {'Error':<10} {'ΔE/|E₀|':<12} {'Behavior'}"
    print(header)
    print("-" * 95)
    
    for case_name in ['A', 'B', 'C']:
        result = results[case_name]
        q0 = result['q0']
        f_eff = result['f_effective']
        a_num = result['a_final_numerical'] / result['a0']
        a_ana = result['a_final_analytical'] / result['a0']
        error = 100 * abs(a_num - a_ana) / a_ana
        dE = (result['E_final'] - result['E0']) / abs(result['E0'])
        
        if np.isclose(q0, 0.5):
            behavior = "Expansion"
        elif np.isclose(q0, 1.0):
            behavior = "Weak expansion"
        else:
            behavior = "Contraction"
        
        print(f"{case_name:<6} {q0:<8.2f} {f_eff:<10.4f} {a_num:<12.6f} {a_ana:<12.6f} {error:<10.3f}% {dE:<12.4f} {behavior}")
    
    print("=" * 95)
    print("Note: Angular momentum conservation is enforced by velocity rescaling (ΔL/L₀ ~ 10⁻¹⁵).")
    print("      Time normalized to initial orbital period T₀.")


# =============================================================================
# Main
# =============================================================================
def main(use_cache: bool = True):
    """Run complete simulation suite with optional caching.
    
    Args:
        use_cache: If True, check for cached results and skip simulation.
    """
    console.rule("[bold cyan]CONSERVATIVE MASS TRANSFER IN BINARY STAR SYSTEMS[/bold cyan]")
    console.print("[yellow]Version 2.0 - Osculating Semi-major Axis Method[/yellow]\n")
    
    results = None
    
    # Try to load cached results
    console.print("[bold]Step 1/5:[/bold] Checking for cached results...")
    results = load_results_from_json('simulation_results.json') if use_cache else None
    
    if results is not None:
        console.print("[green]Using cached data. Skipping simulation.[/green]\n")
    else:
        if use_cache:
            console.print("[yellow]No cache found. Running simulations.[/yellow]\n")
        
        # Validation
        console.print("[bold]Step 2/5:[/bold] Validation test")
        run_validation_test(verbose=True)
        console.print()
        
        # Simulations
        console.print("[bold]Step 3/5:[/bold] Running simulations")
        results = run_all_cases(verbose=True)
        console.print()
        
        # Save results to cache
        console.print("[bold]Step 4/5:[/bold] Caching results")
        save_results_to_json(results, 'simulation_results.json')
        console.print()
    
    # Figures
    console.print("[bold]Step 5/5:[/bold] Generating figures")
    generate_figure1_osculating(results, 'figure1_osculating_sma.png')
    generate_figure2_conservation(results, 'figure2_conservation.png')
    generate_figure3_comparison(results, 'figure3_comparison.png')
    
    console.print("[green]✓[/green] Saved: [cyan]figure1_osculating_sma.pdf[/cyan]")
    console.print("[green]✓[/green] Saved: [cyan]figure2_conservation.pdf[/cyan]")
    console.print("[green]✓[/green] Saved: [cyan]figure3_comparison.pdf[/cyan]")
    console.print()
    
    # Table
    console.print("[bold]Summary Table[/bold]")
    generate_summary_table(results)
    console.print()
    
    console.rule("[bold green]COMPLETE[/bold green]")
    
    return results


if __name__ == '__main__':
    results = main()