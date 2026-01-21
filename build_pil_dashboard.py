from pathlib import Path
import base64
import mimetypes
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import eurostat
import requests

PAESI = ["IT", "DE", "FR", "ES", "NL", "PL", "EU27_2020"]
PAESI_LABEL = {
    "IT": "Italia",
    "DE": "Germania",
    "FR": "Francia",
    "ES": "Spagna",
    "NL": "Paesi Bassi",
    "PL": "Polonia",
    "EU27_2020": "Media UE",
}

YEAR_MIN = 1995

LOGO_URL = "https://www.umbertobertonelli.it/wp-content/uploads/2026/01/logo-grafici-sito.png"
LOGO_PATH = Path("logo.jpg")

DATASET_GDP = "nama_10_gdp"
NA_ITEM_GDP = "B1GQ"

UNIT_REAL_PREFERRED = "CLV10_MEUR"
BASE_YEAR_PREFERRED = 2000


def fmt_eur_mn(x):
    if x is None or pd.isna(x):
        return "n.d."
    return f"{x:,.0f}".replace(",", ".")


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "n.d."
    return f"{x:.2f}%"


def to_data_uri_from_file(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def to_data_uri_from_url(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "image/png").split(";")[0].strip()
    b64 = base64.b64encode(r.content).decode("ascii")
    return f"data:{ct};base64,{b64}"


def get_logo_source() -> str | None:
    if LOGO_PATH.exists():
        return to_data_uri_from_file(LOGO_PATH)
    if LOGO_URL:
        return to_data_uri_from_url(LOGO_URL)
    return None


def apply_logo_watermark(fig: go.Figure, logo_source: str | None) -> None:
    if not logo_source:
        return
    fig.add_layout_image(
        dict(
            source=logo_source,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            sizex=0.50,
            sizey=0.50,
            sizing="contain",
            opacity=0.12,
            layer="above",
        )
    )


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _is_year_col(c: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", str(c).strip()))


def _find_geo_time_col(columns) -> str:
    cols = list(columns)
    if "geo\\TIME_PERIOD" in cols:
        return "geo\\TIME_PERIOD"
    for c in cols:
        sc = str(c)
        if ("geo" in sc.lower()) and ("time_period" in sc.lower()):
            return c
    for c in cols:
        sc = str(c)
        if ("geo" in sc.lower()) and ("time" in sc.lower()):
            return c
    for c in cols:
        if str(c).strip().lower() == "geo":
            return c
    raise ValueError("Colonna geo non trovata nel dataset")


def load_gdp_long() -> tuple[pd.DataFrame, list[str], list[str]]:
    raw = eurostat.get_data(DATASET_GDP)
    if not raw or len(raw) < 2:
        raise RuntimeError("Download Eurostat fallito o dataset vuoto")

    header = raw[0]
    rows = raw[1:]
    df = pd.DataFrame(rows, columns=header)

    geo_col = _find_geo_time_col(df.columns)
    if geo_col != "geo\\TIME_PERIOD":
        df = df.rename(columns={geo_col: "geo\\TIME_PERIOD"})

    required = ["unit", "na_item", "geo\\TIME_PERIOD"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Colonna mancante nel dataset: {c}")

    df["na_item"] = df["na_item"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    df["geo\\TIME_PERIOD"] = df["geo\\TIME_PERIOD"].astype(str).str.strip()

    df = df[df["na_item"] == NA_ITEM_GDP].copy()
    df = df[df["geo\\TIME_PERIOD"].isin(PAESI)].copy()

    id_vars = [c for c in ["freq", "unit", "na_item", "geo\\TIME_PERIOD"] if c in df.columns]
    year_cols = [c for c in df.columns if c not in id_vars and _is_year_col(c)]
    if not year_cols:
        raise RuntimeError("Nessuna colonna anno trovata nel dataset dopo i filtri")

    long_df = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="Year", value_name="Value")
    long_df["Year"] = long_df["Year"].astype(str).str.strip()
    long_df["Value"] = _safe_num(long_df["Value"])

    long_df["Paese"] = long_df["geo\\TIME_PERIOD"].map(PAESI_LABEL).fillna(long_df["geo\\TIME_PERIOD"])
    long_df = long_df[["Year", "unit", "geo\\TIME_PERIOD", "Paese", "Value"]].copy()

    available_units = sorted(long_df["unit"].dropna().astype(str).unique().tolist())
    available_years = sorted(long_df["Year"].dropna().astype(str).unique().tolist())
    return long_df, available_years, available_units


def pick_unit(long_df: pd.DataFrame, preferred: str) -> str:
    units = sorted(long_df["unit"].dropna().astype(str).unique().tolist())
    if not units:
        raise RuntimeError("Nessuna unita disponibile nel dataset dopo i filtri")
    if preferred in units:
        return preferred
    for u in ["CLV10_MEUR", "CLV15_MEUR", "CLV20_MEUR", "CLV_I10", "CLV_I15", "CLV_I20"]:
        if u in units:
            return u
    return units[0]


def pivot_unit(long_df: pd.DataFrame, unit: str) -> pd.DataFrame:
    d = long_df[long_df["unit"].astype(str) == str(unit)].copy()
    p = d.pivot_table(index=["Year"], columns="geo\\TIME_PERIOD", values="Value", aggfunc="first").reset_index()
    p["Year"] = pd.to_numeric(p["Year"], errors="coerce")
    p = p.sort_values("Year")
    return p


def choose_base_year(p: pd.DataFrame, preferred: int) -> int:
    years = p["Year"].dropna().astype(int).tolist()
    if not years:
        return preferred
    if preferred in years:
        return preferred
    return min(years)


def last_common_year(p: pd.DataFrame, countries: list[str]) -> int | None:
    ok = p.copy()
    ok["ok"] = True
    for c in countries:
        if c not in ok.columns:
            ok["ok"] = False
        else:
            ok["ok"] = ok["ok"] & ok[c].notna()
    ok2 = ok[ok["ok"]].copy()
    if ok2.empty:
        yrs = p["Year"].dropna().astype(int).tolist()
        return max(yrs) if yrs else None
    return int(ok2["Year"].max())


def to_index_base100(p: pd.DataFrame, countries: list[str], base_year: int) -> pd.DataFrame:
    out = p.copy()
    base_row = out[out["Year"].astype(int) == int(base_year)]
    if base_row.empty:
        base_year = int(out["Year"].dropna().astype(int).min())
        base_row = out[out["Year"].astype(int) == int(base_year)]
        if base_row.empty:
            raise RuntimeError("Impossibile determinare base year per indice")

    base = base_row.iloc[0]
    for c in countries:
        if c in out.columns:
            b = base[c]
            out[c] = np.where(pd.notna(b) and b != 0, out[c] / b * 100.0, np.nan)
    return out


def yoy_growth(p: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    out = p.copy()
    out = out.sort_values("Year")
    for c in countries:
        if c in out.columns:
            out[c] = out[c].pct_change() * 100.0
    return out


def _apply_common_layout(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=56, r=22, t=90, b=78),
        title=dict(x=0, xanchor="left", y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )


def fig_lines_index(p_idx: pd.DataFrame, countries: list[str], base_year: int, last_year: int, logo_source: str | None) -> go.Figure:
    fig = go.Figure()
    for c in countries:
        if c not in p_idx.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=p_idx["Year"],
                y=p_idx[c],
                mode="lines",
                name=PAESI_LABEL.get(c, c),
                hovertemplate="Anno: %{x}<br>Indice: %{y:.1f}<extra></extra>",
            )
        )
    fig.update_layout(title=f"Pil reale, indice base 100 = {base_year} (ultimo anno {last_year})")
    fig.update_xaxes(title_text="Anno", automargin=True)
    fig.update_yaxes(title_text="Indice", automargin=True)
    _apply_common_layout(fig)
    apply_logo_watermark(fig, logo_source)
    return fig


def fig_gap_vs_it(p_idx: pd.DataFrame, partners: list[str], last_year: int, logo_source: str | None) -> go.Figure:
    fig = go.Figure()
    if "IT" not in p_idx.columns:
        _apply_common_layout(fig)
        return fig

    it = p_idx.set_index("Year")["IT"]
    for c in partners:
        if c not in p_idx.columns:
            continue
        s = p_idx.set_index("Year")[c] - it
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=f"{PAESI_LABEL.get(c, c)} meno Italia",
                hovertemplate="Anno: %{x}<br>Gap punti indice: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(title=f"Gap rispetto all Italia (punti indice), Pil reale (ultimo anno {last_year})")
    fig.update_xaxes(title_text="Anno", automargin=True)
    fig.update_yaxes(title_text="Punti indice", automargin=True)
    _apply_common_layout(fig)
    apply_logo_watermark(fig, logo_source)
    return fig


def fig_it_yoy(p_yoy: pd.DataFrame, last_year: int, logo_source: str | None) -> go.Figure:
    fig = go.Figure()
    if "IT" in p_yoy.columns:
        fig.add_trace(
            go.Bar(
                x=p_yoy["Year"],
                y=p_yoy["IT"],
                name="Italia",
                hovertemplate="Anno: %{x}<br>Variazione: %{y:.2f}%<extra></extra>",
            )
        )
    fig.update_layout(title=f"Pil reale Italia, crescita annua (ultimo anno {last_year})", showlegend=False)
    fig.update_xaxes(title_text="Anno", automargin=True)
    fig.update_yaxes(title_text="Percentuale", ticksuffix="%", automargin=True)
    _apply_common_layout(fig)
    apply_logo_watermark(fig, logo_source)
    return fig


def build_table_dataset(p_real: pd.DataFrame, p_idx: pd.DataFrame, p_yoy: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in p_real.iterrows():
        y = r["Year"]
        for c in PAESI:
            if c in p_real.columns:
                rows.append({"paese": PAESI_LABEL.get(c, c), "year": y, "metrica": "Pil reale", "value": r[c]})

    for _, r in p_idx.iterrows():
        y = r["Year"]
        for c in PAESI:
            if c in p_idx.columns:
                rows.append({"paese": PAESI_LABEL.get(c, c), "year": y, "metrica": "Indice base 100", "value": r[c]})

    for _, r in p_yoy.iterrows():
        y = r["Year"]
        for c in PAESI:
            if c in p_yoy.columns:
                rows.append({"paese": PAESI_LABEL.get(c, c), "year": y, "metrica": "Crescita YoY", "value": r[c]})

    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values(["metrica", "paese", "year"])
    return df


def make_filterable_table_html(df_table: pd.DataFrame, last_year: int, unit_real: str, base_year: int) -> str:
    d = df_table.copy()
    paesi_labels = [PAESI_LABEL.get(g, g) for g in PAESI]

    def option_list(items, all_label):
        out = [f'<option value="__ALL__">{all_label}</option>']
        out += [f'<option value="{x}">{x}</option>' for x in items]
        return "\n".join(out)

    rows_html = []
    for _, r in d.iterrows():
        v = r["value"]
        if r["metrica"] == "Crescita YoY":
            vtxt = "" if pd.isna(v) else f"{v:.2f}".replace(",", ".")
        else:
            vtxt = "" if pd.isna(v) else f"{v:,.0f}".replace(",", ".")
        ytxt = "" if pd.isna(r["year"]) else str(int(r["year"]))
        rows_html.append(
            "<tr>"
            f"<td>{r['paese']}</td>"
            f"<td style='text-align:right'>{ytxt}</td>"
            f"<td>{r['metrica']}</td>"
            f"<td style='text-align:right'>{vtxt}</td>"
            "</tr>"
        )

    return f"""
<div class="card">
  <h2>Tabella dati completa</h2>

  <div class="filters">
    <div>
      <div class="ft">Paese</div>
      <select id="f_country">
        {option_list(paesi_labels, "Tutti")}
      </select>
    </div>
    <div>
      <div class="ft">Metrica</div>
      <select id="f_metric">
        {option_list(["Pil reale", "Indice base 100", "Crescita YoY"], "Tutte")}
      </select>
    </div>
    <div>
      <div class="ft">Ricerca</div>
      <input id="f_search" type="text" placeholder="Testo libero">
    </div>
  </div>

  <div class="note">
    Fonte Eurostat {DATASET_GDP}. na_item {NA_ITEM_GDP}. unita {unit_real}. Indice base 100 = {base_year}. Ultimo anno {last_year}.
  </div>

  <div style="overflow:auto; max-height: 520px; border-radius: 12px; border: 1px solid rgba(15,23,42,0.10); background:#fff;">
    <table id="data_table">
      <thead>
        <tr>
          <th>Paese</th>
          <th style="text-align:right">Anno</th>
          <th>Metrica</th>
          <th style="text-align:right">Valore</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
  </div>
</div>

<script>
(function() {{
  const selC = document.getElementById("f_country");
  const selM = document.getElementById("f_metric");
  const inpS = document.getElementById("f_search");
  const tb = document.getElementById("data_table").getElementsByTagName("tbody")[0];

  function norm(x) {{
    return (x || "").toString().toLowerCase().trim();
  }}

  function apply() {{
    const c = selC.value;
    const m = selM.value;
    const s = norm(inpS.value);

    const rows = tb.getElementsByTagName("tr");
    for (let i = 0; i < rows.length; i++) {{
      const tds = rows[i].getElementsByTagName("td");
      const rc = tds[0].textContent;
      const ry = tds[1].textContent;
      const rm = tds[2].textContent;
      const rv = tds[3].textContent;

      let ok = true;
      if (c !== "__ALL__" && rc !== c) ok = false;
      if (m !== "__ALL__" && rm !== m) ok = false;

      if (ok && s) {{
        const blob = norm(rc + " " + ry + " " + rm + " " + rv);
        if (blob.indexOf(s) === -1) ok = false;
      }}

      rows[i].style.display = ok ? "" : "none";
    }}
  }}

  selC.addEventListener("change", apply);
  selM.addEventListener("change", apply);
  inpS.addEventListener("input", apply);
}})();
</script>
""".strip()


def build_html(
    out_path: Path,
    unit_real: str,
    base_year: int,
    last_year: int,
    fig_a: go.Figure,
    fig_b: go.Figure,
    fig_c: go.Figure,
    table_html: str,
    kpi: dict[str, str],
) -> None:
    fig_a_html = fig_a.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="fig_a",
        config={"responsive": True, "displaylogo": False},
    )
    fig_b_html = fig_b.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id="fig_b",
        config={"responsive": True, "displaylogo": False},
    )
    fig_c_html = fig_c.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id="fig_c",
        config={"responsive": True, "displaylogo": False},
    )

    html = f"""
<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pil Italia e confronto europeo</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #ffffff; color: #0f172a; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 22px 16px 34px 16px; }}
  h1 {{ font-size: 28px; margin: 0 0 10px 0; }}
  h2 {{ font-size: 18px; margin: 16px 0 10px 0; }}
  p {{ font-size: 15px; line-height: 1.6; margin: 10px 0; color: rgba(15,23,42,0.92); }}
  .card {{ background: rgba(15,23,42,0.03); border: 1px solid rgba(15,23,42,0.10); border-radius: 16px; padding: 14px 14px; margin: 12px 0; }}
  .kpi {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 10px; }}
  .kpi .box {{ background: #fff; border: 1px solid rgba(15,23,42,0.10); border-radius: 12px; padding: 10px 10px; }}
  .kpi .t {{ font-size: 12px; color: rgba(15,23,42,0.72); margin-bottom: 6px; }}
  .kpi .v {{ font-size: 18px; font-weight: 750; }}
  .note {{ font-size: 13px; color: rgba(15,23,42,0.78); margin-top: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ border: 1px solid rgba(15,23,42,0.10); padding: 8px 8px; text-align: left; }}
  th {{ background: rgba(15,23,42,0.04); position: sticky; top: 0; }}
  .filters {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 10px 0 10px 0; }}
  .filters select, .filters input {{
    width: 100%;
    padding: 10px 10px;
    border-radius: 12px;
    border: 1px solid rgba(15,23,42,0.14);
    background: #fff;
    font-size: 13px;
  }}
  .filters .ft {{ font-size: 12px; color: rgba(15,23,42,0.72); margin-bottom: 6px; }}
  @media (max-width: 1100px) {{
    .kpi {{ grid-template-columns: 1fr; }}
    .filters {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Pil Italia e confronto europeo</h1>

    <div class="card">
      <p>
        Pil reale Eurostat {DATASET_GDP}. Serie confrontabili tramite indice base 100.
        Il grafico dei gap mostra quanti punti indice i partner si sono mossi sopra o sotto l Italia.
      </p>

      <div class="kpi">
        <div class="box"><div class="t">Ultimo anno disponibile</div><div class="v">{kpi["last_year"]}</div></div>
        <div class="box"><div class="t">Pil reale Italia ultimo anno</div><div class="v">{kpi["it_last_real"]}</div></div>
        <div class="box"><div class="t">Crescita Italia YoY ultimo anno</div><div class="v">{kpi["it_last_yoy"]}</div></div>
        <div class="box"><div class="t">Indice Italia ultimo anno</div><div class="v">{kpi["it_last_idx"]}</div></div>
        <div class="box"><div class="t">Gap Germania meno Italia ultimo anno</div><div class="v">{kpi["gap_de_it"]}</div></div>
        <div class="box"><div class="t">Gap Spagna meno Italia ultimo anno</div><div class="v">{kpi["gap_es_it"]}</div></div>
      </div>

      <p class="note">unita {unit_real}. indice base 100 = {base_year}.</p>
    </div>

    <div class="card">{fig_a_html}</div>
    <div class="card">{fig_b_html}</div>
    <div class="card">{fig_c_html}</div>

    {table_html}
  </div>

<script>
(function() {{
  function isMobile() {{
    return window.matchMedia && window.matchMedia("(max-width: 768px)").matches;
  }}

  function applyMobileRelayout(divId, desktopTitle, mobileTitle) {{
    const gd = document.getElementById(divId);
    if (!gd || !window.Plotly) return;

    const mobile = isMobile();

    const titleText = mobile ? mobileTitle : desktopTitle;
    const margin = mobile ? {{ l: 58, r: 18, t: 120, b: 110 }} : {{ l: 56, r: 22, t: 90, b: 78 }};
    const xTitleStandoff = mobile ? 18 : 10;

    try {{
      Plotly.relayout(gd, {{
        "title.text": titleText,
        "title.x": 0,
        "title.xanchor": "left",
        "title.y": mobile ? 0.965 : 0.98,
        "title.yanchor": "top",
        "title.font.size": mobile ? 15 : 20,
        "margin": margin,
        "xaxis.title.standoff": xTitleStandoff,
        "xaxis.automargin": true,
        "yaxis.automargin": true
      }});
      Plotly.Plots.resize(gd);
    }} catch(e) {{}}
  }}

  function runAll() {{
    applyMobileRelayout(
      "fig_a",
      "Pil reale, indice base 100 = {base_year} (ultimo anno {last_year})",
      "Pil reale, indice base 100 = {base_year}<br>(ultimo anno {last_year})"
    );
    applyMobileRelayout(
      "fig_b",
      "Gap rispetto all Italia (punti indice), Pil reale (ultimo anno {last_year})",
      "Gap rispetto all Italia (punti indice)<br>Pil reale (ultimo anno {last_year})"
    );
    applyMobileRelayout(
      "fig_c",
      "Pil reale Italia, crescita annua (ultimo anno {last_year})",
      "Pil reale Italia, crescita annua<br>(ultimo anno {last_year})"
    );
  }}

  window.addEventListener("load", function() {{
    setTimeout(runAll, 200);
    setTimeout(runAll, 800);
  }});
  window.addEventListener("resize", function() {{
    clearTimeout(window.__gdpResizeT);
    window.__gdpResizeT = setTimeout(runAll, 120);
  }}, {{ passive: true }});
  window.addEventListener("orientationchange", function() {{
    setTimeout(runAll, 250);
  }}, {{ passive: true }});
}})();
</script>
</body>
</html>
""".strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main():
    out_path = Path("docs/pil_dashboard.html")

    logo_source = get_logo_source()

    long_df, _, _ = load_gdp_long()
    unit_real = pick_unit(long_df, UNIT_REAL_PREFERRED)

    p_real = pivot_unit(long_df, unit_real)
    p_real = p_real[p_real["Year"].notna() & (p_real["Year"].astype(int) >= YEAR_MIN)].copy()

    base_year = choose_base_year(p_real, BASE_YEAR_PREFERRED)

    last_year = last_common_year(p_real, PAESI)
    if last_year is None:
        raise RuntimeError("Impossibile determinare ultimo anno")

    p_idx = to_index_base100(p_real, PAESI, base_year)
    p_yoy = yoy_growth(p_real, PAESI)

    fig_a = fig_lines_index(p_idx, PAESI, base_year, last_year, logo_source)
    fig_b = fig_gap_vs_it(p_idx, ["DE", "FR", "ES", "NL", "PL", "EU27_2020"], last_year, logo_source)
    fig_c = fig_it_yoy(p_yoy, last_year, logo_source)

    df_table = build_table_dataset(p_real, p_idx, p_yoy)
    table_html = make_filterable_table_html(df_table, last_year, unit_real, base_year)

    def _val_at(df, col, year):
        r = df[df["Year"].astype(int) == int(year)]
        if r.empty or col not in r.columns:
            return np.nan
        v = r.iloc[0][col]
        return float(v) if pd.notna(v) else np.nan

    it_last_real = _val_at(p_real, "IT", last_year)
    it_last_idx = _val_at(p_idx, "IT", last_year)
    it_last_yoy = _val_at(p_yoy, "IT", last_year)

    de_last_idx = _val_at(p_idx, "DE", last_year)
    es_last_idx = _val_at(p_idx, "ES", last_year)

    kpi = {
        "last_year": str(last_year),
        "it_last_real": fmt_eur_mn(it_last_real),
        "it_last_idx": "n.d." if pd.isna(it_last_idx) else f"{it_last_idx:.1f}",
        "it_last_yoy": fmt_pct(it_last_yoy),
        "gap_de_it": "n.d." if pd.isna(de_last_idx) or pd.isna(it_last_idx) else f"{(de_last_idx - it_last_idx):.1f}",
        "gap_es_it": "n.d." if pd.isna(es_last_idx) or pd.isna(it_last_idx) else f"{(es_last_idx - it_last_idx):.1f}",
    }

    build_html(
        out_path=out_path,
        unit_real=unit_real,
        base_year=base_year,
        last_year=last_year,
        fig_a=fig_a,
        fig_b=fig_b,
        fig_c=fig_c,
        table_html=table_html,
        kpi=kpi,
    )

    print(str(out_path.resolve()))


if __name__ == "__main__":
    main()
