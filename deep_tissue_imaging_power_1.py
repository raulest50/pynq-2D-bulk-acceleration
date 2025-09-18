import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

# Estilo global más legible (no afecta tiempos de cómputo)
mpl.rcParams.update({
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

from deep_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser, campo_tem00
from deep_tissue_imaging.elementos.plotting import plot_field_intensity, plot_field_intensity_history
from deep_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido
import deep_tissue_imaging.propagators.propagation as prop
import deep_tissue_imaging.elementos.domain as Domain
from benchmark.phase_mask_manager import PhaseMaskManager
from benchmark.medir_psf_params import medir_psf_params
from benchmark.system_info import print_system_info

### 3. **Estimación de Consumo Energético**
# - Implementación basada en psutil para Windows con procesadores AMD
# - Estima el consumo de energía basado en el uso de CPU y el TDP del procesador
# - Proporciona estimaciones aproximadas del consumo durante la ejecución
# - Compatible con Windows y procesadores AMD Ryzen


def save_beam_profile(phi, X, Y, title, filename,
                      fs_title=20, fs_labels=16, fs_ticks=14):
    """Saves an intensity map |phi|^2 with improved style in PNG format."""
    X_um, Y_um = X * 1e6, Y * 1e6
    intensity = np.abs(phi) ** 2

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(X_um, Y_um, intensity, cmap='viridis', shading='auto')
    cbar = plt.colorbar(im, ax=ax)

    # Bold and larger labels
    ax.set_title(title, fontsize=fs_title, fontweight='bold')
    ax.set_xlabel('X (μm)', fontsize=fs_labels, fontweight='bold')
    ax.set_ylabel('Y (μm)', fontsize=fs_labels, fontweight='bold')
    cbar.set_label('Intensity (W/m²)', fontsize=fs_labels, fontweight='bold')

    # More readable and bold ticks
    ax.tick_params(axis='both', labelsize=fs_ticks)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')
    cbar.ax.tick_params(labelsize=fs_ticks)
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontweight('bold')

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Parametros de dominio

Lz = np.float32(361e-6)  # 200um
Nz = 361
dz = np.float32(Lz / Nz)  # 1um

Lx, Ly = np.float32(45e-6), np.float32(45e-6)  # 45um x 45um
Nx, Ny = 64, 64
dx = np.float32(Lx / Nx)  # 0.35um
dy = np.float32(Ly / Ny)  # 0.35um

x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
X, Y = np.meshgrid(x, y)

k0 = np.float32(2*np.pi / laser.wavelength)
k = np.float32(k0 * tejido.n_0)
sigma_phi = np.float32(k * tejido.Dn * tejido.l_s)
# Typical value for brain tissue (5 μm)
sigma_x = np.float32(5e-6)

domain = Domain.Domain(X, Y, Nx, Ny, Nz, dx, dy, dz, np.float32(1e-12), k0, k, sigma_phi, sigma_x)

phi0 = campo_tem00(X, Y, laser.w0, laser.I_peak)
# plot_field_intensity(phi0, X, Y)

# Create a phase mask manager
mask_manager = PhaseMaskManager(save_dir="./phase_masks")

# Print system information (does not affect timing)
print_system_info(save_json=True)

# Función para estimar el consumo de energía usando psutil
def estimate_power_consumption(func, *args, tdp=15, **kwargs):
    """
    Estima el consumo de energía durante la ejecución de una función.

    Args:
        func: Función a ejecutar
        *args, **kwargs: Argumentos para la función
        tdp: TDP del procesador en watts (15W para Ryzen 3 5300U)

    Returns:
        tuple: (resultado_función, estadísticas_energía)
    """
    try:
        import psutil
        import statistics
        import threading

        # Muestreo de CPU antes de ejecutar
        psutil.cpu_percent(interval=0.1)  # Primera llamada para inicializar

        # Medición de tiempo
        start_time = time.time()

        # Lista para almacenar muestras de uso de CPU
        cpu_samples = []

        # Variable para controlar el muestreo
        sampling_active = True

        # Función para muestrear CPU en segundo plano
        def sample_cpu():
            while sampling_active:
                cpu_samples.append(psutil.cpu_percent(interval=0.5))

        # Iniciar muestreo en segundo plano
        sampler = threading.Thread(target=sample_cpu)
        sampler.daemon = True
        sampler.start()

        # Ejecutar la función
        result = func(*args, **kwargs)

        # Detener muestreo
        sampling_active = False
        end_time = time.time()
        duration = end_time - start_time

        # Esperar a que termine el muestreo
        sampler.join(timeout=1.0)

        # Calcular estadísticas
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)

            # Estimación de potencia (TDP * % de uso)
            avg_power = tdp * (avg_cpu / 100.0)
            max_power = tdp * (max_cpu / 100.0)
            energy_joules = avg_power * duration

            stats = {
                "duration_seconds": duration,
                "avg_cpu_percent": avg_cpu,
                "max_cpu_percent": max_cpu,
                "avg_power_watts": avg_power,
                "max_power_watts": max_power,
                "energy_joules": energy_joules,
                "samples_count": len(cpu_samples)
            }
        else:
            stats = {
                "duration_seconds": duration,
                "error": "No CPU samples collected"
            }

        return result, stats
    except ImportError:
        print("psutil no disponible: No se medirá el consumo de energía.")
        print("Para habilitar, instale psutil: pip install psutil")
        return func(*args, **kwargs), {"error": "psutil no disponible"}
    except Exception as e:
        print(f"Error al estimar consumo de energía: {str(e)}")
        return func(*args, **kwargs), {"error": str(e)}

# Verificar disponibilidad de psutil para medición de energía
try:
    import psutil
    power_estimation_available = True
    print("Estimación de energía disponible: Usando psutil con TDP=15W (AMD Ryzen 3 5300U)")
except ImportError:
    power_estimation_available = False
    print("Estimación de energía no disponible: psutil no está instalado.")
    print("Para habilitar, instale psutil: pip install psutil")

# Ejecutar la propagación con medición de energía si está disponible
if power_estimation_available:
    # Usar la función de estimación de energía
    phi_history, energy_stats = estimate_power_consumption(
        prop.full_propagation_within_tissue,
        phi0, tejido, domain, mask_manager=mask_manager,
        tdp=15  # TDP para Ryzen 3 5300U
    )

    # Imprimir resultados de tiempo y energía
    print(f"Execution time: {energy_stats['duration_seconds']:.6f} seconds")

    if 'error' in energy_stats:
        print(f"Error en la estimación de energía: {energy_stats['error']}")
    else:
        print(f"Uso promedio de CPU: {energy_stats['avg_cpu_percent']:.1f}%")
        print(f"Uso máximo de CPU: {energy_stats['max_cpu_percent']:.1f}%")
        print(f"Potencia promedio estimada: {energy_stats['avg_power_watts']:.2f} Watts")
        print(f"Potencia máxima estimada: {energy_stats['max_power_watts']:.2f} Watts")
        print(f"Energía total estimada: {energy_stats['energy_joules']:.2f} Joules")
        print(f"Muestras de CPU tomadas: {energy_stats['samples_count']}")
else:
    # Ejecutar sin medición de energía
    start_time = time.time()
    phi_history = prop.full_propagation_within_tissue(phi0, tejido, domain, mask_manager=mask_manager)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
    print("No se realizó estimación de energía (psutil no disponible)")

# Save initial (step 0) and final (step Nz) beam profiles
save_beam_profile(phi_history[0], X, Y, 'Initial Beam Profile (Step 0)', 'beam_initial.png')
save_beam_profile(phi_history[-1], X, Y, f'Final Beam Profile (Step {Nz})', 'beam_final.png')

# Measure PSF parameters
z_positions = np.linspace(0, Lz, Nz+1)
focal_plane = phi_history[-1]  # Last slice is the focal plane
psf_params = medir_psf_params(focal_plane, X, Y, phi_history, z_positions, plot=True)

# Plot field intensity history
plot_field_intensity_history(phi_history, X, Y)
