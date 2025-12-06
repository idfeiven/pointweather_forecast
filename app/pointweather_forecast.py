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


# Initialize session state
if "search_df" not in st.session_state:
    st.session_state["search_df"] = None
if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = 0
if "show_forecast" not in st.session_state:
    st.session_state["show_forecast"] = False


st.set_page_config(page_title="Buscador de localidades", layout="wide")

st.title("🌍 Buscador de localidades y pronósticos")
st.write("Busca una localidad usando la API de Nominatim (OpenStreetMap). Selecciona un resultado para obtener el pronóstico.")

# --- SEARCH SECTION ---
st.subheader("1. Buscar localidad")
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("Localidad (ej. Barcelona, España)", placeholder="Escribe el nombre de una ciudad...", key="search_input")

with col2:
    limit = st.slider("Resultados", min_value=1, max_value=20, value=5, key="result_limit")

if st.button("🔍 Buscar", key="search_button", use_container_width=True):
    if not query or not query.strip():
        st.warning("⚠️ Por favor, introduce una localidad para buscar.")
    else:
        with st.spinner("Buscando localidades..."):
            try:
                results = nominatim_search(query, limit=limit)
                if not results:
                    st.info("❌ No se encontraron resultados.")
                else:
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
                    st.session_state["selected_index"] = 0
                    st.session_state["show_forecast"] = False
                    st.success(f"✅ Se encontraron {len(df)} resultados")
            except Exception as e:
                st.error(f"❌ Error en la búsqueda: {e}")


# --- RESULTS SECTION ---
if st.session_state["search_df"] is not None and not st.session_state["search_df"].empty:
    st.subheader("2. Resultados")
    df = st.session_state["search_df"]

    # Show results table
    st.write("**Resultados encontrados:**")
    st.dataframe(
        df[["display_name", "lat", "lon", "type"]],
        use_container_width=True
    )

    # Show map
    map_df = df.dropna(subset=["lat", "lon"]).rename(
        columns={"lat": "latitude", "lon": "longitude"}
    )
    if not map_df.empty:
        st.map(map_df[["latitude", "longitude"]], zoom=4)

    # Selection dropdown
    st.subheader("3. Seleccionar localidad")
    selected_idx = st.radio(
        "Selecciona una localidad:",
        options=range(len(df)),
        format_func=lambda i: df.at[i, "display_name"],
        index=st.session_state["selected_index"],
        key="selection_radio"
    )
    st.session_state["selected_index"] = selected_idx

    # Show selected location details
    sel_row = df.loc[selected_idx]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latitud", f"{sel_row['lat']:.4f}")
    with col2:
        st.metric("Longitud", f"{sel_row['lon']:.4f}")
    with col3:
        st.metric("Tipo", sel_row["type"])

    # Show location on map
    if pd.notna(sel_row["lat"]) and pd.notna(sel_row["lon"]):
        st.map(
            pd.DataFrame([{"latitude": sel_row["lat"], "longitude": sel_row["lon"]}]),
            zoom=10
        )

    # Get forecast button
    st.subheader("4. Obtener pronóstico")
    if st.button("📊 Obtener pronóstico para esta localidad", key="get_forecast_btn", use_container_width=True):
        st.session_state["show_forecast"] = True
        st.rerun()


# --- FORECAST SECTION ---
if st.session_state.get("show_forecast") and st.session_state["search_df"] is not None:
    df = st.session_state["search_df"]
    sel_row = df.loc[st.session_state["selected_index"]]
    
    with st.spinner("⏳ Descargando pronóstico... (esto puede tardar unos minutos)"):
        try:
            # Load config
            cfg_path = os.path.join(os.getcwd(), "download", "etc", "config.yaml")
            if not os.path.exists(cfg_path):
                st.error(f"❌ Archivo de configuración no encontrado: {cfg_path}")
            else:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    config = yaml.safe_load(fh)

                # Load downloader module
                module_file = os.path.join(os.getcwd(), "download", "download_point_forecast.py")
                if not os.path.exists(module_file):
                    st.error(f"❌ Módulo de descarga no encontrado: {module_file}")
                else:
                    spec = importlib.util.spec_from_file_location("download_point_forecast", module_file)
                    dp = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(dp)

                    det_models = config.get("det_models", [])
                    ens_models = config.get("ens_models", [])
                    variables = config.get("variables", [])
                    url_det = config.get("url_det")
                    url_ens = config.get("url_ens")

                    locality_name = sel_row.get("display_name")
                    
                    try:
                        fcst_df, ens_df = dp.download_point_forecast(
                            locality_name,
                            det_models=det_models,
                            ens_models=ens_models,
                            variables=variables,
                            url_det=url_det,
                            url_ens=url_ens
                        )
                    except TypeError:
                        fcst_df, ens_df = dp.download_point_forecast(
                            locality_name, det_models, ens_models, variables, url_det, url_ens
                        )

                    st.success("✅ Pronóstico descargado correctamente")

                    # Display forecast data
                    st.subheader("5. Datos del pronóstico")
                    
                    tab1, tab2 = st.tabs(["Determinista", "Ensemble"])
                    
                    with tab1:
                        st.write("**Resumen de datos determinista:**")
                        st.dataframe(fcst_df.head(10), use_container_width=True)
                        
                        # Plotting options for deterministic
                        det_numeric = [
                            c for c in fcst_df.columns
                            if c not in ("forecast_date", "model", "locality", "latitude", "longitude")
                            and pd.api.types.is_numeric_dtype(fcst_df[c])
                        ]
                        if det_numeric:
                            col_to_plot = st.selectbox(
                                "Selecciona variable para graficar (determinista):",
                                options=det_numeric,
                                key="det_var_select"
                            )
                            plot_df = fcst_df[["forecast_date", col_to_plot, "model"]].dropna()
                            
                            # Group by model and plot
                            st.line_chart(
                                plot_df.set_index("forecast_date")[[col_to_plot]],
                                use_container_width=True
                            )
                        else:
                            st.info("No hay columnas numéricas disponibles en datos determinista.")

                    with tab2:
                        st.write("**Resumen de datos ensemble:**")
                        st.dataframe(ens_df.head(10), use_container_width=True)
                        
                        # Plotting options for ensemble
                        ens_numeric = [
                            c for c in ens_df.columns
                            if c != "forecast_date"
                            and pd.api.types.is_numeric_dtype(ens_df[c])
                        ]
                        if ens_numeric:
                            ens_col = st.selectbox(
                                "Selecciona variable/miembro para graficar (ensemble):",
                                options=ens_numeric,
                                key="ens_var_select"
                            )
                            plot_df2 = ens_df[["forecast_date", ens_col]].dropna()
                            
                            st.line_chart(
                                plot_df2.set_index("forecast_date")[[ens_col]],
                                use_container_width=True
                            )
                        else:
                            st.info("No hay columnas numéricas disponibles en datos ensemble.")

        except Exception as e:
            st.error(f"❌ Error descargando/mostrando pronóstico: {e}")
            st.text(traceback.format_exc())
            st.session_state["show_forecast"] = False
