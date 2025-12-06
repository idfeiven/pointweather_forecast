import os
import streamlit as st
import requests
import pandas as pd
import yaml
import importlib.util
import traceback


def nominatim_search(query: str, limit: int = 5) -> list:
    """Search Nominatim (OpenStreetMap) for a locality string.
    Returns a list of result dicts (may be empty).
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "jsonv2", "limit": limit}
    headers = {"User-Agent": "pointweather-streamlit-app"}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


# --- session state defaults
if "search_df" not in st.session_state:
    st.session_state["search_df"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = None
if "forecast_requested" not in st.session_state:
    st.session_state["forecast_requested"] = False


st.set_page_config(page_title="Buscador de localidades", layout="centered")

st.title("Buscador de localidades")
st.write("Busca una localidad usando la API de Nominatim (OpenStreetMap). Selecciona un resultado para ver sus coordenadas y ubicarlo en el mapa.")


query = st.text_input("Localidad", value=st.session_state.get("last_query", ""), placeholder="Ej. Barcelona, España")
cols = st.columns([3, 1])
with cols[0]:
    limit = st.slider("Resultados", min_value=1, max_value=20, value=5)
with cols[1]:
    search_btn = st.button("Buscar")


if search_btn:
    if not query or not query.strip():
        st.warning("Introduce una localidad para buscar.")
    else:
        with st.spinner("Buscando localidades..."):
            try:
                results = nominatim_search(query, limit=limit)
            except Exception as e:
                st.error(f"Error en la búsqueda: {e}")
                results = []

        rows = []
        for r in results:
            try:
                lat = float(r.get("lat"))
                lon = float(r.get("lon"))
            except Exception:
                lat = None
                lon = None
            rows.append({
                "display_name": r.get("display_name"),
                "lat": lat,
                "lon": lon,
                "type": r.get("type"),
                "class": r.get("class"),
            })

        df = pd.DataFrame(rows)
        st.session_state["search_df"] = df
        st.session_state["last_query"] = query
        # reset selection and forecast flags
        st.session_state["selected_index"] = None
        st.session_state["forecast_requested"] = False


# If we have previous search results, show them
if st.session_state["search_df"] is not None and not st.session_state["search_df"].empty:
    df = st.session_state["search_df"]
    st.dataframe(df[["display_name", "lat", "lon", "type"]])

    map_df = df.dropna(subset=["lat", "lon"]).rename(columns={"lat": "latitude", "lon": "longitude"})
    if not map_df.empty:
        st.map(map_df[["latitude", "longitude"]])

    # selection box (store index in session_state)
    default_index = st.session_state.get("selected_index")
    try:
        sel = st.selectbox("Selecciona un resultado", options=df.index.tolist(), index=int(default_index) if default_index is not None else 0, format_func=lambda i: df.at[i, "display_name"], key="sel_box")
    except Exception:
        sel = st.selectbox("Selecciona un resultado", options=df.index.tolist(), format_func=lambda i: df.at[i, "display_name"], key="sel_box2")

    st.session_state["selected_index"] = int(sel)
    sel_row = df.loc[st.session_state["selected_index"]]
    st.markdown("**Localidad seleccionada**")
    st.write(sel_row.to_dict())
    if pd.notna(sel_row["lat"]) and pd.notna(sel_row["lon"]):
        st.map(pd.DataFrame([{"latitude": sel_row["lat"], "longitude": sel_row["lon"]}]))

    # Request forecast button — set a flag in session_state so the request survives the rerun
    if st.button("Obtener pronóstico para esta localidad", key="get_forecast"):
        st.session_state["forecast_requested"] = True


# If forecast requested, perform download (this runs on rerun because flag is in session_state)
if st.session_state.get("forecast_requested"):
    with st.spinner("Descargando pronóstico..."):
        try:
            # Load config
            cfg_path = os.path.join(os.getcwd(), "download", "etc", "config.yaml")
            if not os.path.exists(cfg_path):
                st.error(f"Archivo de configuración no encontrado: {cfg_path}")
            else:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    config = yaml.safe_load(fh)

                # Load downloader module from file to avoid package import issues
                module_file = os.path.join(os.getcwd(), "download", "download_point_forecast.py")
                if not os.path.exists(module_file):
                    st.error(f"Módulo de descarga no encontrado: {module_file}")
                else:
                    spec = importlib.util.spec_from_file_location("download_point_forecast", module_file)
                    dp = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(dp)

                    det_models = config.get("det_models")
                    ens_models = config.get("ens_models")
                    variables = config.get("variables")
                    url_det = config.get("url_det")
                    url_ens = config.get("url_ens")

                    # Call downloader
                    locality_name = sel_row.get("display_name")
                    try:
                        fcst_df, ens_df = dp.download_point_forecast(locality_name,
                                                                   det_models=det_models,
                                                                   ens_models=ens_models,
                                                                   variables=variables,
                                                                   url_det=url_det,
                                                                   url_ens=url_ens)
                    except TypeError:
                        # fallback if function signature expects different args
                        fcst_df, ens_df = dp.download_point_forecast(locality_name, det_models, ens_models, variables, url_det, url_ens)

                    st.success("Pronóstico descargado")
                    st.subheader("Resumen de datos determinista")
                    st.write(fcst_df.head())

                    st.subheader("Resumen de datos ensemble")
                    st.write(ens_df.head())

                    # plotting options
                    det_numeric = [c for c in fcst_df.columns if c not in ("forecast_date", "model", "locality", "latitude", "longitude") and pd.api.types.is_numeric_dtype(fcst_df[c])]
                    if det_numeric:
                        col_to_plot = st.selectbox("Variable a plotear (determinista)", options=det_numeric, key="plot_det_col")
                        if st.button("Graficar determinista", key="plot_det_btn"):
                            plot_df = fcst_df[["forecast_date", col_to_plot]].dropna()
                            plot_df = plot_df.set_index("forecast_date")
                            st.line_chart(plot_df)
                    else:
                        st.info("No hay columnas numéricas disponibles para graficar en el dataset determinista.")

                    ens_numeric = [c for c in ens_df.columns if c != "forecast_date" and pd.api.types.is_numeric_dtype(ens_df[c])]
                    if ens_numeric:
                        ens_col = st.selectbox("Variable/miembro a plotear (ensemble)", options=ens_numeric, key="plot_ens_col")
                        if st.button("Graficar ensemble", key="plot_ens_btn"):
                            plot_df2 = ens_df[["forecast_date", ens_col]].dropna()
                            plot_df2 = plot_df2.set_index("forecast_date")
                            st.line_chart(plot_df2)

            # reset flag so it doesn't re-run repeatedly
            st.session_state["forecast_requested"] = False

        except Exception as e:
            st.error(f"Error descargando/mostrando pronóstico: {e}")
            st.text(traceback.format_exc())