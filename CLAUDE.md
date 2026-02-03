# CLAUDE.md - Guía de Proyecto para Claude Code

Este documento proporciona contexto y convenciones para trabajar con el proyecto **pynq-2D-bulk-acceleration**.

## Descripción del Proyecto

Simulación de imágenes en tejido profundo (*deep tissue imaging*) usando el **Método de Propagación de Haz (BPM - Beam Propagation Method)** con exploraciones de aceleración por hardware (GPU y FPGA vía PYNQ).

**Autor:** Esteban Raulest
**Estado:** Desarrollo activo - transicionando de CPU/GPU puro hacia aceleración FPGA via PYNQ.

---

## Comandos de Ejecución

```powershell
# Instalar dependencias
& $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry install

# Ejecutar simulación principal (CPU)
& $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry run python .\deep_tissue_imaging_1.py

# Ejecutar simulación Z-scan
& $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry run python .\z_scan_simulation_1.py

# Graficar resultados
& $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry run python .\results_plot.py

# Alias recomendado (si está configurado):
poe run python .\deep_tissue_imaging_1.py
```

---

## Estructura del Proyecto

```
pynq-2D-bulk-acceleration/
├── .junie/                     # Guidelines técnicas (HLS, PYNQ, teoría BPM)
├── deep_tissue_imaging/        # Módulo principal de simulación
│   ├── elementos/              # Objetos físicos: Domain, Lasers, Tejidos, Plotting
│   └── propagators/            # Solvers numéricos: Thomas, ADI, propagación completa
├── zscan/                      # Simulación Z-scan para caracterización óptica
├── benchmark/                  # Herramientas de análisis de rendimiento
│   ├── system_info.py          # Perfilado de hardware multiplataforma
│   ├── medir_psf_params.py     # Medición de PSF (Point Spread Function)
│   └── phase_mask_manager.py   # Generación reproducible de máscaras de fase
├── performance_data/           # Salida: métricas de ejecución (JSON)
├── phase_masks/                # Máscaras de fase cacheadas
├── outputs/                    # Perfiles de haz generados (PNG)
├── deep_tissue_imaging_1.py    # Punto de entrada principal (CPU)
├── z_scan_simulation_1.py      # Punto de entrada Z-scan
├── results_plot.py             # Visualización de resultados
└── pyproject.toml              # Dependencias (Poetry)
```

---

## Stack Tecnológico

| Categoría | Tecnología | Versión |
|-----------|------------|---------|
| Lenguaje | Python | 3.12 |
| Gestor de paquetes | Poetry + micromamba | 2.1.4 |
| Numérico | NumPy, SciPy | 2.2.1+, 1.16.0 |
| GPU (experimental) | CuPy | 13.4.1 (CUDA 12.x) |
| Visualización | Matplotlib | 3.10.0 |
| Hardware FPGA | PYNQ, Vitis HLS | 2025.1 |

---

## Convenciones de Código

### Idioma
- **Variables y funciones:** Español para términos de dominio (`tejido`, `dominio`, `mascara`, `desviacion_fase`)
- **Clases:** CamelCase en inglés (`Domain`, `PhaseMaskManager`, `Sample`)
- **Docstrings:** Inglés para APIs públicas, español aceptable para comentarios internos

### Tipos de Datos Numéricos
**CRÍTICO:** Siempre usar casting explícito para consistencia numérica:
```python
# Correcto
valor = np.float32(800e-9)
campo = np.complex64(Ex * fase)

# Incorrecto
valor = 800e-9  # float64 por defecto
```

### Patrón de Función de Dominio
```python
def mi_funcion(phi, tejido, domain, **kwargs):
    """
    Parameters
    ----------
    phi : np.ndarray (complex64)
        Campo eléctrico complejo
    tejido : object
        Propiedades del material (n0, alpha)
    domain : Domain
        Objeto de dominio con grids X, Y, Nz, dz
    """
    pass
```

### Instrumentación de Rendimiento
Todas las ejecuciones deben registrar tiempo y metadatos:
```python
import time
from benchmark.system_info import collect_system_info

start_time = time.time()
# ... simulación ...
execution_time = time.time() - start_time

# Guardar métricas
save_performance_data(execution_time, domain, psf_params, output_dir="./performance_data")
```

### Visualización
```python
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
})

fig.savefig(filename, dpi=300, bbox_inches='tight')
plt.close(fig)  # Siempre cerrar para liberar memoria
```

### Reproducibilidad
Máscaras de fase deben generarse con semillas fijas:
```python
np.random.seed(42 + mask_index)
theta = gaussian_filter(ruido, sigma=(sigma_y, sigma_x))
np.save(mask_file, theta)  # Cachear para reuso
```

---

## Algoritmos Clave

### Método BPM (Beam Propagation Method)
- **Base matemática:** Ecuación de Helmholtz escalar paraxial
- **Implementación:** Split-Step Fourier Method (SSFM) con operadores ADI
- **Solver:** Algoritmo de Thomas para sistemas tridiagonales

### Conclusiones sobre Aceleración GPU
**Importante:** El algoritmo ADI + Thomas NO es adecuado para GPU debido a:
- Baja intensidad aritmética
- Dependencias secuenciales
- Procesamiento fila/columna inherentemente serial

Ver `.junie/algorithm_gpu_suitability.md` para análisis detallado.

---

## Guidelines Técnicas (.junie/)

| Archivo | Propósito |
|---------|-----------|
| `hls_guidelines.md` | Pragmas Vitis HLS, interfaces AXI, tipos de precisión fija |
| `pynq_guidelines.md` | API PYNQ: Overlay, DMA, GPIO, Interrupt, drivers |
| `bpm-lecture.md` | Teoría matemática del BPM y Split-Step Fourier |
| `gaussian_beam_theory.md` | Parámetros de haz Gaussiano (w₀, z_R, Gouy) |
| `power_estimation_GPU.md` | Análisis energético: GPU 30× menos eficiente que CPU |

---

## Flujo de Trabajo para Nuevas Funcionalidades

1. **Leer código existente** antes de proponer cambios
2. **Usar `np.float32` / `np.complex64`** consistentemente
3. **Registrar métricas** de ejecución en `performance_data/`
4. **Cachear datos generados** (máscaras de fase) para reproducibilidad
5. **Cerrar figuras matplotlib** después de guardar
6. **Para HLS:** Consultar `.junie/hls_guidelines.md` para pragmas y interfaces
7. **Para PYNQ:** Consultar `.junie/pynq_guidelines.md` para drivers y DMA

---

## Archivos de Entrada Principal

| Script | Descripción |
|--------|-------------|
| `deep_tissue_imaging_1.py` | Simulación BPM en tejido (CPU) - genera perfiles de haz |
| `z_scan_simulation_1.py` | Simulación Z-scan para caracterización no lineal |
| `results_plot.py` | Post-procesamiento y visualización de resultados |

---

## Notas para el Asistente

- **No sobrecomplicar:** Mantener soluciones simples y enfocadas
- **Respetar el idioma mixto:** Español para dominio, inglés para APIs
- **Priorizar rendimiento:** Este es un proyecto de benchmarking; medir siempre
- **Validar numéricamente:** Comparar con golden model antes de cambios algorítmicos
- **Consultar .junie/:** Contiene conocimiento específico de HLS, PYNQ y física
