# Registro de experimentos — ARC Prize 2026 ARC-AGI-2

Métrica oficial: exact-match por salida de test con 2 intentos, media sobre salidas. Validación local:
evaluación pública ARC-AGI-2 (120 tareas, 172 salidas). Decisiones con validación local, no con el LB.
Tabla CSV paralela: `experiments.csv`.

## E000 — control k01 (fork exacto LB 33,89), 28-10
- Hipótesis: el notebook público más fuerte declarado (Qwen3-4B grids SFT + TTT + DFS) corre igual en nuestra
  cuenta y da un control medido en el LB.
- v1 12:15 UTC: ERROR en commit. `FileNotFoundError` en `/kaggle/input/competitions/.../arc-agi_evaluation_challenges.json`:
  vía API Kaggle monta las entradas en otra raíz. Decisión: añadir celda que resuelve rutas (sin tocar la lógica).
- v2: pendiente de hueco GPU (2 sesiones por lotes de la cuenta ocupadas por otras sesiones).

## E001 — cobertura y precisión de ARC2_CHIMERA (CPU), 28-10
- Hipótesis H1a: los candidatos verificados de CHIMERA (regla local 3×3 inducida = principio CA de v10; programas
  exactos del DSL v8 con leave-one-out perfecto) tienen precisión alta y cubren alguna salida de evaluación.
- Config: `src/chimera_fill.py`, modos CA `vn_count, vn, moore_count, moore`; DSL `chimera_dsl.py`
  (copia sin cambios de `arc2_chimera/arc_solver.py`, sha256 9b4da221…).
- Resultado: pendiente.
- Resultado (28-10): 0/172 emitidos en evaluación ARC-AGI-2 (10,6 s CPU). Decisión: H1a rechazada en evaluación; medir en training.

## E000 (cont.) — control k01 v2, 29-10
- v2 COMPLETE. Commit en modo evaluación solo procesa 4 tareas (filtro original del notebook): 3/172 salidas = esperado.
- S01 enviado 29-10 05:08 UTC, ref 57759538, fork exacto. Decisión: control del LB.

## E002 — ARC2_CHIMERA v1 en training (CPU), 29-10
- 1000 tareas / 1076 salidas: emite 50, acierta 44 (precisión 0,88); dsl 38/43, ca 6/7; 49,5 s CPU.
- Decisión: el generador funciona en tareas tipo ARC-AGI-1; su ausencia en evaluación ARC-AGI-2 es de cobertura, no de fallo.

## E003 — CHIMERA CA v2 (rel3, ray4/8, rowcol, m5count, iterado), 29-10
- Hipótesis (JEV 02 «ampliar_ca_cpu», 0,43): familias de reglas locales más generales cubren salidas de evaluación.
- Código: trabajo/arc-agi-2/src/chimera_ca2.py (misma puerta: todos los pares + leave-one-out).
- Evaluación: 0/172 emitidos (16 s). Training: primer candidato 59 emitidos / 53 correctos; CA v2 11/11 correctos
  (ray4 5/5, rowcol 4/4, rel3 2/2); 9 salidas resueltas solo por CA v2.
- Decisión: H1 (CHIMERA añade salidas en ARC-AGI-2) sin apoyo en evaluación pública; se mantiene como componente de
  precisión ~1,0 en el envío (coste ~1 min CPU) y se mide en el LB (S02 vs S01). Negativo registrado.

## E004 — k02 = control + CHIMERA (commit Kaggle v1), 29-10
- COMPLETE sin error. Modo evaluación: CHIMERA 0 candidatos (coherente con E001/E003); puntuación 3/172 = control k01 (fusión neutra).
- Decisión: válido para S02 (30-10, tarea programada 00:13 UTC). Diferencia S02−S01 en LB = aporte CHIMERA en test oculto + ruido TTT.

## E005 — control por tarea en server-ia (1×3090), 29-10
- Entorno: trabajo/arc-agi-2/server (venv .venv2: torch 2.9.1, transformers 4.57.3, unsloth 2025.12.9; solver sin cambios salvo 3 rutas).
  Pila nueva (unsloth 2026.10, transformers 5.17) descartada: pérdida 0,0 y 0 candidatos (trabajo ffb53a45).
- Prueba 1 tarea (0934a4d8, trabajo 5cf43dfe): 0/1, 397 s, VRAM pico 14,5 GiB.
- Subconjunto fijo 24 tareas de evaluación (random.seed(20261029), server/subset24.txt), presupuesto 12 600 s: trabajo 32495d89b1c5419c.
