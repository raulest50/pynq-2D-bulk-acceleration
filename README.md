# pynq-2D-bulk-acceleration

Aceleración con PYNQ de simulación Z-scan y utilidades de propagación/visualización.

- Zscan_2.py: algoritmo solo PS, usando profiler para medir tiempos de ejecución.
- Zscan_normal.py: algoritmo normal, sin profiler, solo PS.
- deep_tissue_imaging_1.py: ejemplo de propagación en tejido y guardado de perfiles de haz.

---

## Método de ejecución con micromamba (nuevo)

Antes no era necesario usar micromamba para ejecutar este proyecto. Debido a restricciones institucionales (p. ej., bloqueos al instalador MSI de Python o falta de permisos de administrador), se adoptó un método totalmente en espacio de usuario usando micromamba + Poetry. Este método:
- No requiere permisos de administrador ni modifica el registro/PATH del sistema.
- Ofrece entornos aislados y reproducibles.
- Se integra bien con PyCharm y con flujos `poetry install` / `poetry run`.

Por favor, sigue siempre las políticas de tu institución. Todo este flujo ocurre en tu perfil de usuario.

---

## Requisitos

- Windows PowerShell (usuario normal, sin admin).
- Acceso a Internet para descargar el binario de micromamba y paquetes de conda-forge/PyPI.
- (Opcional) PyCharm si quieres configurar el intérprete del proyecto.

---

## Instalación rápida (sin admin) — micromamba + Python 3.12 + Poetry

Todos los comandos son para PowerShell. Pega cada bloque en una sola línea.

### 1) Descargar micromamba (binario portable)

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $dest="$env:USERPROFILE\micromamba.exe"; Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64" -OutFile $dest

Verificar:

& $env:USERPROFILE\micromamba.exe --version

### 2) Crear entorno en user-space con Python 3.12 + Poetry

$env:MAMBA_ROOT_PREFIX = "$env:USERPROFILE\mamba"; & $env:USERPROFILE\micromamba.exe create -y -p "$env:USERPROFILE\mamba\envs\py312" python=3.12 poetry -c conda-forge

Probar:

& $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" python --version

& $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry --version

---

## Uso del proyecto con Poetry (sin “activar” el entorno)

Desde la carpeta del repo (ajusta la ruta a la tuya):

cd "C:\Users\TU_USUARIO\Desktop\pynq-2D-bulk-acceleration"; & $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry install

Ejemplos de ejecución:

& $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry run python .\Zscan_2.py

& $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry run python .\Zscan_normal.py

& $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry run python .\deep_tissue_imaging_1.py

Sugerencia: si quieres evitar escribir la ruta larga, define un alias poe (ver sección “Atajo poe”).

---

## Flujo de trabajo con Poetry

- poetry init
- poetry add --dev pigar
- poetry install
- poetry env info --path  # para ver la ruta del intérprete
- poetry run pigar
- pigar generate
- poetry add -r requirements.txt

Nota: Si usas el alias poe, reemplaza poetry por poe (ej.: poe install).

---

## Configurar PyCharm

### Opción A: usar el Python del entorno micromamba

- File → Settings → Project → Python Interpreter → Add → Existing environment
- Selecciona: C:\Users\TU_USUARIO\mamba\envs\py312\python.exe
- Si PyCharm pide “Poetry executable”: C:\Users\TU_USUARIO\mamba\envs\py312\Scripts\poetry.exe

### Opción B: venv dentro del proyecto (opcional)

Si prefieres que Poetry cree .venv dentro del repo:

poe config virtualenvs.in-project true

poe env use python

poe install

Luego en PyCharm selecciona: C:\ruta\al\repo\.venv\Scripts\python.exe

---

## Atajo “poe” (alias de Poetry dentro del entorno)

Para la sesión actual:

function poe { & $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry @Args }

Para dejarlo permanente:

if (!(Test-Path $PROFILE)) { New-Item -Type File -Path $PROFILE -Force | Out-Null }; 'function poe { & $env:USERPROFILE\micromamba.exe run -p "$env:USERPROFILE\mamba\envs\py312" poetry @Args }' | Add-Content $PROFILE

Ahora puedes hacer:

- poe install
- poe run python .\Zscan_2.py
- poe add numpy

---

## Ejecución de ejemplos del repositorio

- Zscan_2.py: poe run python .\Zscan_2.py
- Zscan_normal.py: poe run python .\Zscan_normal.py
- deep_tissue_imaging_1.py: poe run python .\deep_tissue_imaging_1.py
  - Genera archivos PNG con perfiles de haz inicial y final (beam_inicial.png, beam_final.png) con títulos y etiquetas más legibles.

---

## Troubleshooting

### Descarga bloqueada / proxy corporativo

Usa BITS en lugar de Invoke-WebRequest:

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $dest="$env:USERPROFILE\micromamba.exe"; Start-BitsTransfer -Source "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64" -Destination $dest

### Poetry no resuelve alguna wheel con C/C++

Instala esa dependencia con conda primero (en el mismo prefijo):

& $env:USERPROFILE\micromamba.exe install -y -p "$env:USERPROFILE\mamba\envs\py312" <paquete> -c conda-forge

Luego ejecuta:

poe install  # o el comando completo con micromamba run + poetry install

### Usar comandos “estilo conda” (activar)

No es necesario, pero si lo quieres:

$hook = & $env:USERPROFILE\micromamba.exe shell hook -s powershell; if (!(Test-Path $PROFILE)) { New-Item -Type File -Path $PROFILE -Force | Out-Null }; $hook | Add-Content $PROFILE

Abre una nueva terminal y:

micromamba activate "C:\Users\TU_USUARIO\mamba\envs\py312"

---

## ¿Por qué micromamba?

- Sin admin / sin MSI: es un binario portable, no instala servicios ni modifica el registro.
- Aislado y reproducible: entornos en $USERPROFILE\mamba\envs\...
- Rápido y liviano: resuelve dependencias de conda-forge con buen rendimiento.
- Interoperable: convive con poetry y pip sin conflictos; PyCharm lo detecta fácil.

---

## Alternativas (por si GitHub está bloqueado)

- WinPython (portable ZIP): descomprimir y usar python.exe sin instalar.
- Python embeddable (ZIP oficial): muy minimalista; requiere habilitar pip manualmente.

---

## Estado del proyecto

- Python: 3.12.11 (entorno micromamba)
- Poetry: 2.1.4
- Entorno: C:\Users\TU_USUARIO\mamba\envs\py312\

---

## Notas finales

- Sustituye TU_USUARIO por tu usuario real en Windows.
- Si tu repo está en otra ruta, ajusta el cd y las rutas de PyCharm.
- Para mantener reproducibilidad, considera fijar versiones en pyproject.toml y/o generar un lockfile (poetry.lock).

