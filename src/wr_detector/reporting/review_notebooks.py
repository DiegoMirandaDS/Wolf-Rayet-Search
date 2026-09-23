"""Rebuild the compact, executed audit notebooks used for project handoff.

The notebooks intentionally consume the reproducible pipelines and experiment
history. They do not duplicate scientific transformations as notebook-only
logic.

Run from the repository root:

    python -m wr_detector.reporting.review_notebooks --execute
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import nbformat
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

LEGACY_AUDIT_NOTEBOOKS = (
    "00_Available_photometry_analysis.ipynb",
    "01_reference_catalog_audit.ipynb",
    "02_color_eda_linear_cuts.ipynb",
    "03_negative_reduction_audit.ipynb",
    "04_model_training_results.ipynb",
    "04_overfitting_and_tree_diagnostics.ipynb",
    "05_prediction_pool_audit.ipynb",
    "06_dimensionality_audit.ipynb",
    "07_prediction_pool_locus_superset_pilot.ipynb",
    "08_prediction_pool_exact_union_spatial_audit.ipynb",
)


def _markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def _code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip())


def _base_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }


def _setup_cell() -> nbformat.NotebookNode:
    return _code(
        """
from pathlib import Path
from IPython.display import Image, Markdown, display
import pandas as pd

PROJECT_ROOT = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "pyproject.toml").exists()
)

from wr_detector.reporting.project_summary import (
    AP_LEADER_RESULT_ID,
    LEADING_RESULT_ID,
    STABLE_RESULT_ID,
    collect_evidence,
    generate_figures,
)

PUBLIC_DIR = PROJECT_ROOT / "reports" / "public"
FIGURE_DIR = PUBLIC_DIR / "figures"
evidence = collect_evidence()
figure_paths = generate_figures(evidence, FIGURE_DIR)
display(Markdown(f"**Snapshot local:** `{evidence.snapshot_date}`"))
"""
    )


def _reference_notebook() -> nbformat.NotebookNode:
    cells = [
        _markdown(
            """
# Referencia, muestra negativa y locus de color

## TL;DR

- La referencia local contiene 705 WR del catálogo de Crowther v1.33; 442 tienen alias Gaia DR3 explícito.
- `relaxed_photometry` conserva 331/347 WR después del locus y reduce 66.787 negativos a 33.343.
- El locus usa seis planos intra-misión, HuberRegressor y `signed_log1p`; no usa colores cruzados entre misiones.
- La muestra SIMBAD es una colección controlada de contaminantes plausibles, no una estimación de toda la población no-WR.
"""
        ),
        _markdown(
            """
## Contexto y método

La referencia se descarga como snapshot con fecha y SHA-256. Solo se incorporan al
pipeline reproducible las entradas que declaran un alias Gaia DR3; no se infiere
identidad a partir de coordenadas. Gaia se enriquece con 2MASS y AllWISE mediante
crossmatch oficial y, cuando falta una contraparte, mediante fallback VizieR con
proveniencia y separación angular conservadas.

Las variantes `strict` y `relaxed` exigen Gaia G/BP/RP, 2MASS J/H/Ks y WISE W1/W2.
`strict` admite calidad A en las bandas IR requeridas y `relaxed` admite A o B.
"""
        ),
        _setup_cell(),
        _markdown("## Datos"),
        _code(
            """
catalogue = pd.DataFrame(
    {
        "etapa": ["Catálogo WR", "Alias Gaia DR3", "Fotometría relaxed", "Locus keep"],
        "WR": [
            evidence.catalogue["catalogue_rows"],
            evidence.catalogue["gaia_alias_rows"],
            int(
                evidence.locus.loc[
                    (evidence.locus["variant"] == "relaxed_photometry")
                    & (evidence.locus["class"] == "WR"),
                    "rows",
                ].iloc[0]
            ),
            int(
                evidence.locus.loc[
                    (evidence.locus["variant"] == "relaxed_photometry")
                    & (evidence.locus["class"] == "WR"),
                    "locus_keep",
                ].iloc[0]
            ),
        ],
    }
)
display(catalogue)
display(
    evidence.negative_types.rename(
        columns={"object_group": "clase", "rows": "fuentes"}
    )
)
"""
        ),
        _code(
            """
display(Image(filename=str(figure_paths["retention"]), width=1000))
display(Image(filename=str(figure_paths["negative_composition"]), width=900))
"""
        ),
        _markdown(
            r"""
## Resultado: definición del locus

Para cada color \(c\):

\[
T(c)=\operatorname{sign}(c)\ln(1+|c|)
\]

En cada plano \((x,y)\) se ajusta:

\[
T(y_i)=\beta_0+\beta_1T(x_i)+\epsilon_i,\qquad
r_i=T(y_i)-[\beta_0+\beta_1T(x_i)]
\]

El umbral del plano es \(q_{0.975}(|r|)\) medido sobre WR. Una fuente se marca
como outlier agregado si excede el umbral en al menos dos de los seis planos.

La formulación anterior basada en `log10` de colores positivos exigía que los
seis colores fueran positivos simultáneamente. Solo 2/347 WR de
`relaxed_photometry` y 0/306 de `strict_photometry` cumplían esa condición.
`signed_log1p` conserva el signo, está definida en cero y permite usar todas las
filas finitas.
"""
        ),
        _code(
            """
display(Image(filename=str(figure_paths["locus"]), width=1050))
display(Image(filename=str(figure_paths["sky"]), width=1000))
"""
        ),
        _markdown(
            """
## Takeaways

El filtro define una región de compatibilidad fotométrica, no una ventana
geográfica del plano galáctico. Los colores cruzados entre Gaia, 2MASS y WISE se
excluyen para evitar que offsets de misión, calibración o crossmatch definan el
locus. W1-W2 sí se conserva como feature, pero WISE no genera un plano porque no
se usan W3/W4.

Los conteos y gráficos se regeneran desde DuckDB/Parquet mediante
`python -m wr_detector.reporting.project_summary`.
"""
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata=_base_metadata())


def _model_notebook() -> nbformat.NotebookNode:
    cells = [
        _markdown(
            """
# Selección, evaluación y validación de modelos

## TL;DR

- `run_v3_main` compara 144 configuraciones: 8 variantes, 2 sets de features, 3 modelos y 3 samplers.
- La decisión se formula como ranking de objetos raros: AP, curvas PR, Recall@K, Precision@K y comportamiento a FPR bajo.
- No existe todavía un ganador operativo único: los perfiles líderes intercambian recuperación, pureza y estabilidad.
- La segunda capa WN/WC permanece como auditoría; aún no mejora el presupuesto top-K sin pérdida material de WR.
"""
        ),
        _markdown(
            """
## Contexto y método

El holdout se fija mediante hash estable de `source_id` y estratificación por
clase antes de reducir negativos. Aproximadamente 20% de WR y negativos se
reservan. Todos los WR restantes entrenan; los negativos de entrenamiento se
muestrean a razón 10:1 por estratos de cuantiles de color. Los negativos no
seleccionados forman `threshold_calibration`.

Los modelos son Random Forest, HistGradientBoosting y XGBoost. Se comparan
`none`, SMOTE y SMOTE-ENN. El remuestreo ocurre dentro de cada fold; `none` usa
ponderación nativa y las configuraciones muestreadas usan pesos neutros.
BayesSearchCV ejecuta 25 iteraciones y optimiza Average Precision.
"""
        ),
        _setup_cell(),
        _markdown("## Datos de evaluación"),
        _code(
            """
split = evidence.split.copy()
display(split)

selected_ids = [LEADING_RESULT_ID, AP_LEADER_RESULT_ID, STABLE_RESULT_ID]
columns = [
    "result_id",
    "dataset_variant",
    "model",
    "sampler",
    "holdout_average_precision",
    "holdout_recall_at_50",
    "holdout_recall_at_100",
    "holdout_precision_at_100",
    "holdout_recall_at_fpr_0p005",
    "overfit_warning_flag",
]
selected = evidence.model_results.loc[
    evidence.model_results["result_id"].isin(selected_ids), columns
].copy()
selected.insert(
    0,
    "perfil",
    selected["result_id"].map(
        {
            LEADING_RESULT_ID: "broad shortlist",
            AP_LEADER_RESULT_ID: "AP leader",
            STABLE_RESULT_ID: "accepted / stable",
        }
    ),
)
display(selected.drop(columns="result_id").sort_values("perfil"))
"""
        ),
        _markdown(
            r"""
## Métricas

\[
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},\qquad
\mathrm{FPR}=\frac{FP}{FP+TN}
\]

\[
\mathrm{AP}=\sum_n(R_n-R_{n-1})P_n
\]

Accuracy no decide el experimento: con decenas de millones de fuentes, predecir
casi todo como no-WR puede producir accuracy alta y una lista inútil. AP y PR
enfatizan la clase positiva; ROC se conserva como diagnóstico, ampliada en la
zona de FPR bajo.
"""
        ),
        _code(
            """
display(Image(filename=str(figure_paths["model"]), width=1100))
display(Image(filename=str(figure_paths["curves"]), width=1100))
"""
        ),
        _markdown(
            """
## Interpretación

El perfil `relaxed_photometry`/XGBoost/`none` recupera 48/68 WR dentro del
top-100 (Recall@100=70,6%; Precision@100=48,0%; AP=0,535), pero conserva una
advertencia de sobreajuste. El líder de AP con SMOTE alcanza AP=0,567, con
Recall@100=63,2%. El perfil `strict_poe_2`/XGBoost/SMOTE supera los controles de
estabilidad y alcanza Recall@100=73,8%, pero usa un holdout positivo menor y
produce una lista top-100 menos pura. Por eso los conteos absolutos no se
comparan entre variantes sin su denominador.
"""
        ),
        _code(
            """
display(Image(filename=str(figure_paths["importance"]), width=900))
"""
        ),
        _markdown(
            """
## Takeaways

La importancia interna del XGBoost destacado está dominada por W1-W2 y paralaje.
Es una descripción del estimador, no una inferencia causal. La selección final
debe fijar un presupuesto de seguimiento, revisar estabilidad/subtipos y
repetirse sobre la prediction pool exact-union terminada.
"""
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata=_base_metadata())


def _pool_notebook() -> nbformat.NotebookNode:
    cells = [
        _markdown(
            """
# Prediction pool: auditoría, reconstrucción y estado

## TL;DR

- El pool legacy de 58.037.788 filas es inmutable y no se usará para scoring definitivo.
- En 18 regiones, el filtro agregado perdió 1.960/26.369 fuentes compatibles con alguna variante exacta (7,43%); el máximo regional fue 18,90%.
- La corrección es una nueva adquisición inmutable con bitmask versionado y vistas lógicas por variante.
- El build exact-union terminó y auditó sus 675 tiles; el scoring de los cinco modelos completó y auditó 3.375/3.375 unidades.
"""
        ),
        _markdown(
            """
## Contexto y decisión

El primer pool usó una envolvente agregada construida a partir de loci de varias
variantes. La auditoría demostró que esa envolvente no era un superset
geométrico de su unión exacta. La nueva arquitectura separa:

1. adquisición Gaia por tiles;
2. almacenamiento Parquet inmutable;
3. bitmask de elegibilidad por variante;
4. vistas lógicas reproducibles;
5. scoring resumible por `result_id` sincronizado.

Esta separación permite reintentar consultas sin sobrescribir fuentes y cambiar
la selección de modelos sin volver a descargar toda Gaia.
"""
        ),
        _setup_cell(),
        _markdown("## Evidencia de la reconstrucción"),
        _code(
            """
audit = pd.DataFrame([evidence.pool_audit])
status = pd.DataFrame([evidence.pool_status])
display(audit.T.rename(columns={0: "valor"}))
display(status.T.rename(columns={0: "valor"}))
display(Image(filename=str(figure_paths["pool"]), width=1050))
"""
        ),
        _markdown(
            """
## Alcance y limitaciones

El proyecto consulta todo el cielo, pero restringe la inferencia a una envolvente
amplia de colores observados en WR y después aplica la elegibilidad exacta de
cada variante. No busca recuperar todas las estrellas del plano galáctico ni
usa una ventana espacial como etiqueta.

Las fuentes desconocidas del pool no son negativos confirmados. Por eso el
scoring masivo producirá una lista priorizada para revisión, no una estimación
directa de precisión científica. La confirmación sigue siendo espectroscópica.
"""
        ),
        _markdown(
            """
## Siguiente secuencia

1. Revisar los 20 candidatos prioritarios con imágenes y SED.
2. Priorizar espectroscopía para los cinco recomendados.
3. Actualizar SIMBAD solo con una ejecución live explícita.
4. Integrar las tablas ya calculadas en el Model Explorer.

## Takeaways

La adquisición masiva, la auditoría del pool y el scoring resumible ya están
cerrados. El ranking conserva el linaje completo hasta modelo, tile y snapshot
SIMBAD.
"""
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata=_base_metadata())


def _candidate_notebook() -> nbformat.NotebookNode:
    cells = [
        _markdown(
            """
# Auditoría de candidatos de la prediction pool

Vista de lectura del ranking reproducible. La selección, el consenso RRF,
el crossmatch SIMBAD y las disposiciones se calculan en
`wr_detector.pipelines.prediction_pool_candidates`.
"""
        ),
        _code(
            """
from pathlib import Path
from IPython.display import Markdown, display
import duckdb
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "pyproject.toml").exists()
)
CANDIDATE_DB = PROJECT_ROOT / "data/databases/prediction_pool_candidates.duckdb"
POOL_DB = PROJECT_ROOT / "data/databases/prediction_pool_exact_union_v1.duckdb"
if not CANDIDATE_DB.exists():
    raise FileNotFoundError(CANDIDATE_DB)

with duckdb.connect(str(CANDIDATE_DB), read_only=True) as con:
    run = con.execute("SELECT * FROM candidate_review_runs").fetchdf()
    models = con.execute("SELECT * FROM candidate_models").fetchdf()
    model_ranks = con.execute("SELECT * FROM candidate_model_ranks").fetchdf()
    consensus = con.execute(
        "SELECT * FROM candidate_consensus ORDER BY consensus_rank LIMIT 500"
    ).fetchdf()
    review = con.execute(
        "SELECT * FROM candidate_review ORDER BY consensus_rank"
    ).fetchdf()
    recommendations = con.execute(
        "SELECT * FROM candidate_recommendations ORDER BY followup_rank"
    ).fetchdf()
with duckdb.connect(str(POOL_DB), read_only=True) as con:
    pool_status = con.execute(
        '''
        SELECT b.status,
               COUNT(*) FILTER (WHERE t.status='completed') AS completed_tiles,
               SUM(t.acquired_pre_locus) AS acquired_rows,
               SUM(t.accepted_union) AS accepted_union,
               SUM(t.known_excluded) AS known_excluded,
               SUM(t.written) AS eligible_rows
        FROM exact_union_builds b
        JOIN exact_union_tiles t USING (pool_build_id)
        GROUP BY b.status
        '''
    ).fetchdf()
"""
        ),
        _markdown("## tl;dr"),
        _code(
            """
headline = recommendations.head(5)
display(Markdown(
    f"- Pool: **{int(pool_status.iloc[0]['completed_tiles'])}/675 tiles**, "
    f"estado **{pool_status.iloc[0]['status']}**.\\n"
    f"- Ranking: **{len(model_ranks):,}** filas top-K y "
    f"**{len(review):,}** candidatos enriquecidos.\\n"
    f"- SIMBAD: **{int(review['match_found'].sum())}** coincidencias entre "
    f"{len(review)} candidatos.\\n"
    f"- Seguimiento: **{len(recommendations)}** objetos revisados; "
    f"los primeros cinco se muestran al final."
))
"""
        ),
        _markdown(
            """
## Context & Methods

### Key Assumptions

- Los scores son valores de ordenamiento, no probabilidades calibradas.
- El consenso usa Reciprocal Rank Fusion (`k=60`) con pesos iguales.
- Una ausencia de SIMBAD no demuestra que el objeto sea desconocido.
- La confirmación de una estrella WR continúa siendo espectroscópica.
"""
        ),
        _markdown("## Data"),
        _code(
            """
display(pool_status)
display(run.T.rename(columns={0: "valor"}))
model_metric_columns = [
    "role",
    "result_id",
    "dataset_variant",
    "model",
    "sampler",
    "wr_holdout",
    "holdout_average_precision",
    "holdout_precision_at_50",
    "holdout_recall_at_50",
    "holdout_precision_at_100",
    "holdout_recall_at_100",
    "holdout_recall_at_fpr_0p001",
    "threshold_calibration_negative_pass_rate",
    "overfit_risk_score",
]
display(
    models[model_metric_columns]
    .sort_values("holdout_average_precision", ascending=False)
    .reset_index(drop=True)
)
display(
    model_ranks.groupby(
        ["role", "result_id", "dataset_variant", "model"], as_index=False
    ).agg(
        rows=("source_id", "size"),
        score_min=("score", "min"),
        score_max=("score", "max"),
    )
)
"""
        ),
        _markdown("## Results"),
        _code(
            """
top_sets = {
    result_id: set(group.nsmallest(500, "model_rank")["source_id"])
    for result_id, group in model_ranks.groupby("result_id")
}
labels = {
    result_id: group["role"].iloc[0]
    for result_id, group in model_ranks.groupby("result_id")
}
overlap = pd.DataFrame(index=top_sets, columns=top_sets, dtype=float)
for left, left_ids in top_sets.items():
    for right, right_ids in top_sets.items():
        union = left_ids | right_ids
        overlap.loc[left, right] = len(left_ids & right_ids) / len(union) if union else 0
overlap = overlap.rename(index=labels, columns=labels)
display(overlap.style.format("{:.1%}").background_gradient(cmap="magma"))
"""
        ),
        _code(
            """
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
review["model_support"].value_counts().sort_index().plot.bar(
    ax=axes[0], color="#6f42c1"
)
axes[0].set(
    title="Apoyo entre modelos",
    xlabel="Modelos en top-10.000",
    ylabel="Candidatos",
)
axes[1].scatter(
    review["galactic_l"],
    review["galactic_b"],
    c=review["consensus_rank"],
    s=14,
    cmap="viridis_r",
    alpha=0.75,
)
axes[1].set(
    title="Top 500 en coordenadas galácticas",
    xlabel="l [deg]",
    ylabel="b [deg]",
)
review["review_disposition"].value_counts().plot.bar(
    ax=axes[2], color="#e67e22"
)
axes[2].set(title="Contexto SIMBAD", xlabel="", ylabel="Candidatos")
axes[2].tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.show()
"""
        ),
        _code(
            """
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(review["G_BP"], review["G_RP"], s=12, alpha=0.55)
axes[0].set(title="Gaia intra-misión", xlabel="G - BP", ylabel="G - RP")
axes[1].scatter(review["J_H"], review["H_K"], s=12, alpha=0.55)
axes[1].set(title="2MASS intra-misión", xlabel="J - H", ylabel="H - Ks")
fig.tight_layout()
plt.show()
"""
        ),
        _code(
            """
review_columns = [
    "followup_rank",
    "consensus_rank",
    "source_id",
    "model_support",
    "variant_support",
    "best_model_rank",
    "G",
    "BP_RP",
    "J_K",
    "W1_W2",
    "parallax_over_error",
    "ruwe",
    "halpha_ew",
    "espels_class",
    "simbad_main_id",
    "simbad_main_type",
    "review_disposition",
    "review_comment",
    "simbad_url",
    "gaia_url",
    "gaia_adql",
]
display(recommendations[review_columns].head(20))
"""
        ),
        _markdown("## Takeaways"),
        _code(
            """
display(Markdown(
    "Los cinco objetos siguientes son prioridades de seguimiento, no "
    "confirmaciones WR. Se excluyeron del ranking de seguimiento los WR ya "
    "conocidos y las clasificaciones no-WR inequívocas, conservando siempre "
    "su rango consenso original para auditoría."
))
display(headline[review_columns].head(5))
"""
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata=_base_metadata())


def notebook_specs() -> dict[str, nbformat.NotebookNode]:
    return {
        "01_reference_and_color_locus.ipynb": _reference_notebook(),
        "02_model_selection_and_validation.ipynb": _model_notebook(),
        "03_prediction_pool_status.ipynb": _pool_notebook(),
        "04_prediction_pool_candidate_audit.ipynb": _candidate_notebook(),
    }


def _remove_superseded_notebooks() -> None:
    for name in LEGACY_AUDIT_NOTEBOOKS:
        path = NOTEBOOK_DIR / name
        if path.exists():
            path.unlink()


def _execute(
    notebook: nbformat.NotebookNode,
    *,
    timeout_seconds: int,
) -> nbformat.NotebookNode:
    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    return client.execute()


def rebuild_notebooks(*, execute: bool, timeout_seconds: int = 600) -> Iterable[Path]:
    """Replace superseded audit notebooks with compact executed narratives."""
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    _remove_superseded_notebooks()
    outputs: list[Path] = []
    for name, notebook in notebook_specs().items():
        if execute:
            notebook = _execute(notebook, timeout_seconds=timeout_seconds)
        path = NOTEBOOK_DIR / name
        nbformat.write(notebook, path)
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute every notebook before writing it.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    for output in rebuild_notebooks(execute=args.execute, timeout_seconds=args.timeout):
        print(output)


if __name__ == "__main__":
    main()
