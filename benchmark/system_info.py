# benchmark/system_info.py
import sys
import platform
import os
import subprocess
import json


def _windows_cpu_info_via_powershell():
    """Devuelve info detallada del CPU usando PowerShell (Windows)."""
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Processor | Select-Object "
                "Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize | ConvertTo-Json",
            ],
            encoding="utf-8",
            errors="ignore",
        )
        data = json.loads(out)
        if isinstance(data, list):
            data = data[0]
        return {
            "cpu_model": data.get("Name"),
            "physical_cores": data.get("NumberOfCores"),
            "logical_cpus": data.get("NumberOfLogicalProcessors"),
            "cpu_max_mhz": data.get("MaxClockSpeed"),
            "l2_cache_kb": data.get("L2CacheSize"),
            "l3_cache_kb": data.get("L3CacheSize"),
        }
    except Exception:
        return {}


def _try_psutil_info(info: dict) -> dict:
    try:
        import psutil  # opcional
        info.setdefault("logical_cpus", psutil.cpu_count(logical=True))
        info.setdefault("physical_cores", psutil.cpu_count(logical=False))
        try:
            freq = psutil.cpu_freq()
            if freq:
                info.setdefault("cpu_max_mhz", round(freq.max or freq.current or 0.0, 1))
        except Exception:
            pass
        try:
            mem = psutil.virtual_memory()
            info.setdefault("ram_gb", round(mem.total / (1024 ** 3), 2))
        except Exception:
            pass
    except Exception:
        info.setdefault("logical_cpus", os.cpu_count())
    return info


def _try_cpuinfo(info: dict) -> dict:
    try:
        import cpuinfo  # opcional (py-cpuinfo)
        c = cpuinfo.get_cpu_info()
        if c:
            info.setdefault("cpu_model", c.get("brand_raw") or c.get("brand"))
            hz = c.get("hz_advertised_friendly") or c.get("hz_actual_friendly")
            if hz:
                info.setdefault("cpu_hz_advertised", hz)
            arch = c.get("arch")
            if arch:
                info.setdefault("arch", arch)
    except Exception:
        pass
    return info


def collect_system_info() -> dict:
    info = {
        "python": (sys.version.split()[0] if sys.version else None),
    }

    # Librerías científicas clave (no obligatorio)
    try:
        import numpy as _np
        info["numpy"] = _np.__version__
        # Opcional: backend BLAS
        try:
            import numpy as np
            from numpy import __config__ as _cfg
            blas = _cfg.get_info('blas_opt_info')
            if blas:
                info["numpy_blas"] = {k: (str(v) if not isinstance(v, (str, int, float)) else v) for k, v in blas.items()}
        except Exception:
            pass
    except Exception:
        pass

    # SO y arquitectura
    try:
        info["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"
        info["machine"] = platform.machine()
    except Exception:
        pass

    info = _try_psutil_info(info)
    info = _try_cpuinfo(info)

    # En Windows, enriquecer con WMI vía PowerShell
    if platform.system() == "Windows":
        win = _windows_cpu_info_via_powershell()
        for k, v in win.items():
            if v not in (None, "") and k not in info:
                info[k] = v

    return info


def format_summary_line(info: dict) -> str:
    parts = []
    if info.get("cpu_model"):
        parts.append(info["cpu_model"])
    if info.get("physical_cores") is not None and info.get("logical_cpus") is not None:
        parts.append(f"{info['physical_cores']}C/{info['logical_cpus']}T")
    elif info.get("logical_cpus") is not None:
        parts.append(f"{info['logical_cpus']} logical threads")
    if info.get("cpu_max_mhz"):
        parts.append(f"max {info['cpu_max_mhz']} MHz")
    if info.get("ram_gb"):
        parts.append(f"{info['ram_gb']} GB RAM")
    if info.get("os"):
        parts.append(info["os"])
    if info.get("python"):
        parts.append(f"Python {info['python']}")
    if info.get("numpy"):
        parts.append(f"NumPy {info['numpy']}")
    return " | ".join(str(p) for p in parts)


def print_system_info(save_json: bool = True, json_path: str = "system_info.json") -> dict:
    """
    Imprime un resumen (una línea) idóneo para el paper + detalles legibles.
    Si save_json=True guarda todos los campos en JSON. Devuelve el dict info.
    """
    info = collect_system_info()

    print("CPU/Plataforma (resumen para paper): ", format_summary_line(info))

    labels = {
        "cpu_model": "CPU model",
        "physical_cores": "Physical cores",
        "logical_cpus": "Logical CPUs",
        "cpu_max_mhz": "Max clock (MHz)",
        "l2_cache_kb": "L2 cache (KB)",
        "l3_cache_kb": "L3 cache (KB)",
        "ram_gb": "RAM (GB)",
        "arch": "CPU arch",
        "machine": "Machine",
        "os": "OS",
        "python": "Python",
        "numpy": "NumPy",
    }
    print("-- CPU/Platform details --")
    for k, lab in labels.items():
        v = info.get(k)
        if v is not None:
            print(f"{lab}: {v}")

    if save_json:
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return info


__all__ = ["collect_system_info", "format_summary_line", "print_system_info"]
