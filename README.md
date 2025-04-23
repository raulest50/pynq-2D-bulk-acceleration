# pynq-2D-bulk-acceleration

aceleracion con pynq de simulacion de z scan

* Zscan_2.py : algoritmo solo PS, usando profiler para medir tiempos de ejecucion
* Zscan_normal.py : algoritmo normal, sin profiler, solo PS

## poetry

```
poetry init
poetry add --dev pigar
poetry install
poetry env info --path
# configurar interpretador en 
poetry run pigar
pigar generate
poetry add -r requirements.txt
```

