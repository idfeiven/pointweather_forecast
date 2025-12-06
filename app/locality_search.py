import streamlit as st
import requests
import pandas as pd
import yaml
import importlib
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


st.set_page_config(page_title="Buscador de localidades", layout="centered")

st.title("Buscador de localidades")
st.write("Busca una localidad usando la API de Nominatim (OpenStreetMap). Selecciona un resultado para ver sus coordenadas y ubicarlo en el mapa.")

query = st.text_input("Localidad", placeholder="Ej. Barcelona, España")
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

        if not results:
            st.info("No se encontraron resultados.")
        else:
            # Normalize and show results
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
            st.dataframe(df[["display_name", "lat", "lon", "type"]])

            # Map: streamlit expects columns named latitude/longitude
            map_df = df.dropna(subset=["lat", "lon"]).rename(columns={"lat": "latitude", "lon": "longitude"})
            if not map_df.empty:
                st.map(map_df[["latitude", "longitude"]])

            sel = st.selectbox("Selecciona un resultado", options=df.index, format_func=lambda i: df.at[i, "display_name"]) 
            if sel is not None:
                sel_row = df.loc[sel]
                st.markdown("**Localidad seleccionada**")
                st.write(sel_row.to_dict())
                if pd.notna(sel_row["lat"]) and pd.notna(sel_row["lon"]):
                    st.map(pd.DataFrame([{"latitude": sel_row["lat"], "longitude": sel_row["lon"]}]))

                # Integración: botón para descargar pronóstico usando download/download_point_forecast.py
                if st.button("Obtener pronóstico para esta localidad"):
                    with st.spinner("Descargando pronóstico..."):
                        try:
                            # Cargar configuración
                            with open("download/etc/config.yaml", "r", encoding="utf-8") as fh:
                                config = yaml.safe_load(fh)

                            # Importar el módulo de descarga dinámicamente
                            dp = importlib.import_module("download.download_point_forecast")

                            det_models = config.get("det_models")
                            ens_models = config.get("ens_models")
                            variables = config.get("variables")
                            url_det = config.get("url_det")
                            url_ens = config.get("url_ens")

                            # Ejecutar la descarga (nota: puede tardar varios segundos)
                            fcst_df, ens_df = dp.download_point_forecast(sel_row["display_name"],
                                                                       det_models=det_models,
                                                                       ens_models=ens_models,
                                                                       variables=variables,
                                                                       url_det=url_det,
                                                                       url_ens=url_ens)

                            st.success("Pronóstico descargado")

                            # Mostrar un resumen y permitir seleccionar variable/columna para graficar
                            st.subheader("Resumen de datos determinista")
                            st.write(fcst_df.head())

                            st.subheader("Resumen de datos ensemble")
                            st.write(ens_df.head())

                            # Elegir columna para plotear en determinista
                            numeric_cols = [c for c in fcst_df.columns if c not in ("forecast_date", "model", "locality", "latitude", "longitude") and pd.api.types.is_numeric_dtype(fcst_df[c])]
                            if numeric_cols:
                                col_to_plot = st.selectbox("Variable a plotear (determinista)", options=numeric_cols)
                                if st.button("Graficar determinista"):
                                    plot_df = fcst_df[["forecast_date", col_to_plot]].dropna()
                                    plot_df = plot_df.set_index("forecast_date")
                                    st.line_chart(plot_df)
                            else:
                                st.info("No hay columnas numéricas disponibles para graficar en el dataset determinista.")

                            # Para ensemble: ofrecer lista de columnas
                            ens_numeric = [c for c in ens_df.columns if c != "forecast_date" and pd.api.types.is_numeric_dtype(ens_df[c])]
                            if ens_numeric:
                                ens_col = st.selectbox("Variable/miembro a plotear (ensemble)", options=ens_numeric)
                                if st.button("Graficar ensemble"):
                                    plot_df2 = ens_df[["forecast_date", ens_col]].dropna()
                                    plot_df2 = plot_df2.set_index("forecast_date")
                                    st.line_chart(plot_df2)

                        except Exception as e:
                            st.error(f"Error descargando/mostrando pronóstico: {e}")
                            st.text(traceback.format_exc())
