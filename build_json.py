from pathlib import Path
import json
import numpy as np
import pandas as pd
import eurostat

ECASES = ["50%", "67%", "80%", "100%", "125%", "167%"]

PAESI = ["IT", "FR", "DE", "ES", "PL", "EU27_2020"]
PAESI_LABEL = {
    "IT": "Italia",
    "FR": "Francia",
    "DE": "Germania",
    "ES": "Spagna",
    "PL": "Polonia",
    "EU27_2020": "Media UE",
}

ECASE_MAP = {
    "Single person without children earning 50% of the average earning": "50%",
    "Single person without children earning 67% of the average earning": "67%",
    "Single person without children earning 80% of the average earning": "80%",
    "Single person without children earning 100% of the average earning": "100%",
    "Single person without children earning 125% of the average earning": "125%",
    "Single person without children earning 167% of the average earning": "167%",
}

ESTRUCT_MAP = {
    "Gross earning": "RAL",
    "Net earning": "Netto",
    "Taxes": "Imposte",
    "Social Security": "Contributi",
    "Total": "Costo",
}

def load_data_long():
    raw = eurostat.get_data("earn_nt_net")
    df = pd.DataFrame(raw, columns=raw[0]).drop(index=0).reset_index(drop=True)

    for col in list(df.columns):
        try:
            dic = dict(eurostat.get_dic("earn_nt_net", col))
            df[col] = df[col].map(dic).fillna(df[col])
        except Exception:
            pass

    if "geo\\TIME_PERIOD" not in df.columns:
        for c in df.columns:
            if "TIME_PERIOD" in c and "geo" in c:
                df = df.rename(columns={c: "geo\\TIME_PERIOD"})
                break

    if "ecase" in df.columns:
        df["ecase"] = df["ecase"].map(ECASE_MAP).fillna(df["ecase"])
    if "estruct" in df.columns:
        df["estruct"] = df["estruct"].map(ESTRUCT_MAP).fillna(df["estruct"])

    if "currency" in df.columns:
        cur = df["currency"].astype(str)
        euro_mask = cur.str.contains("euro", case=False, na=False) | (cur == "EUR")
        if euro_mask.any():
            df = df.loc[euro_mask].copy()

    id_vars = [c for c in ["freq", "currency", "estruct", "ecase", "geo\\TIME_PERIOD"] if c in df.columns]
    year_cols = [c for c in df.columns if c not in id_vars]

    long_df = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="Year", value_name="Value")
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df["Year"] = long_df["Year"].astype(str)

    long_df = long_df[
        long_df["geo\\TIME_PERIOD"].isin(PAESI)
        & long_df["ecase"].isin(ECASES)
        & long_df["estruct"].isin(["Netto", "Costo", "Imposte", "Contributi", "RAL"])
    ].copy()

    long_df["Paese"] = long_df["geo\\TIME_PERIOD"].map(PAESI_LABEL).fillna(long_df["geo\\TIME_PERIOD"])
    long_df["Livello"] = long_df["ecase"].astype(str)
    long_df["Componente"] = long_df["estruct"].astype(str)

    long_df = long_df[["Year", "geo\\TIME_PERIOD", "Paese", "Livello", "Componente", "Value"]]
    available_years = sorted(long_df["Year"].dropna().unique().tolist())
    return long_df, available_years

def pivot_year(long_df, year):
    d = long_df[long_df["Year"] == str(year)].copy()
    p = d.pivot_table(
        index=["geo\\TIME_PERIOD", "Paese", "Livello"],
        columns="Componente",
        values="Value",
        aggfunc="first",
    ).reset_index()

    for c in ["Netto", "Costo", "Imposte", "Contributi", "RAL"]:
        if c not in p.columns:
            p[c] = np.nan

    p["Netto_su_Costo"] = (p["Netto"] / p["Costo"]) * 100.0
    p["Livello"] = pd.Categorical(p["Livello"], categories=ECASES, ordered=True)
    p = p.sort_values(["geo\\TIME_PERIOD", "Livello"])
    return p

def compute_kpi(pivot):
    p100 = pivot[pivot["Livello"].astype(str) == "100%"].copy()
    p100 = p100[p100["geo\\TIME_PERIOD"].isin(PAESI)].copy()

    def best_row(col):
        if not p100[col].notna().any():
            return None
        return p100.loc[p100[col].idxmax()]

    best_share = best_row("Netto_su_Costo")
    best_net = best_row("Netto")
    best_cost = best_row("Costo")

    def pack(row, col, pct=False):
        if row is None:
            return {"paese": None, "val": None}
        v = row[col]
        v = None if pd.isna(v) else float(v)
        return {"paese": str(row["Paese"]), "val": v}

    it100 = pivot[(pivot["geo\\TIME_PERIOD"] == "IT") & (pivot["Livello"].astype(str) == "100%")]
    eu100 = pivot[(pivot["geo\\TIME_PERIOD"] == "EU27_2020") & (pivot["Livello"].astype(str) == "100%")]

    def one(df, col):
        if df.empty:
            return None
        v = df.iloc[0][col]
        return None if pd.isna(v) else float(v)

    return {
        "italia_100_netto": one(it100, "Netto"),
        "italia_100_costo": one(it100, "Costo"),
        "italia_100_quota_netto": one(it100, "Netto_su_Costo"),
        "ue_100_quota_netto": one(eu100, "Netto_su_Costo"),
        "leader_quota_netto": pack(best_share, "Netto_su_Costo"),
        "leader_netto": pack(best_net, "Netto"),
        "leader_costo": pack(best_cost, "Costo"),
    }

def to_records(df, cols):
    out = df[cols].copy()
    out = out.replace({np.nan: None})
    return out.to_dict(orient="records")

def main():
    requested_year = "2024"
    out_path = Path("docs/earn_nt_net_dashboard.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    long_df, years = load_data_long()
    year = requested_year if requested_year in years else (years[-1] if years else requested_year)

    pivot = pivot_year(long_df, year)
    kpi = compute_kpi(pivot)

    payload = {
        "meta": {
            "source": "Eurostat earn_nt_net",
            "year": year,
            "paesi": PAESI,
            "paesi_label": PAESI_LABEL,
            "livelli": ECASES,
            "componenti": ["Netto", "Imposte", "Contributi", "Costo", "RAL"],
        },
        "kpi": kpi,
        "pivot": to_records(
            pivot,
            ["geo\\TIME_PERIOD", "Paese", "Livello", "Netto", "Imposte", "Contributi", "Costo", "RAL", "Netto_su_Costo"],
        ),
        "long": to_records(
            long_df[long_df["Year"] == str(year)],
            ["Paese", "Livello", "Componente", "Value"],
        ),
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(out_path.as_posix())

if __name__ == "__main__":
    main()
