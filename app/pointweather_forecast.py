import os
import yaml
import requests
import traceback
import pandas as pd
import importlib.util
import streamlit as st
import plotly.graph_objects as go


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

st.title("🌍Pointweather forecast⛅")
st.markdown("### Busca localidades y obtén gráficas con el pronóstico meteorológico determinista y ensemble.")
st.write("Busca una localidad usando la API de Nominatim (OpenStreetMap). Selecciona un resultado para obtener el pronóstico.")
st.write("Esta app ha sido desarrollada con ayuda de IA y utiliza datos de modelos de predicción del tiempo a través de open-meteo.com")

# --- SEARCH SECTION ---
st.subheader("Buscar localidad")
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("Localidad (ej. Barcelona, España)", placeholder="Escribe el nombre de una ciudad...", key="search_input")

with col2:
    limit = st.slider("Resultados", min_value=1, max_value=10, value=5, key="result_limit")

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
    # st.subheader("Resultados búsqueda localidad")
    df = st.session_state["search_df"]

    # Show results table
    # st.write("**Resultados encontrados:**")
    # st.dataframe(
    #     df[["display_name", "lat", "lon", "type"]],
    #     use_container_width=True
    # )

    # Show map
    # map_df = df.dropna(subset=["lat", "lon"]).rename(
    #     columns={"lat": "latitude", "lon": "longitude"}
    # )
    # if not map_df.empty:
    #     st.map(map_df[["latitude", "longitude"]], zoom=4)

    # Selection dropdown
    st.subheader("Seleccionar localidad")
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
    # if pd.notna(sel_row["lat"]) and pd.notna(sel_row["lon"]):
    #     st.map(
    #         pd.DataFrame([{"latitude": sel_row["lat"], "longitude": sel_row["lon"]}]),
    #         zoom=10
    #     )

    # Get forecast button
    st.subheader("Obtener pronóstico")
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
            cfg_path = os.path.join(os.getcwd(), "download", "etc", "config_download.yaml")
            if not os.path.exists(cfg_path):
                st.error(f"❌ Archivo de configuración no encontrado: {cfg_path}")
            else:
                # Read file as text first and normalize tabs -> spaces to avoid YAML tab errors
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    raw_cfg = fh.read()

                if "\t" in raw_cfg:
                    # Replace tabs with two spaces (YAML disallows tabs for indentation)
                    raw_cfg = raw_cfg.replace("\t", "  ")

                try:
                    config = yaml.safe_load(raw_cfg)
                except yaml.scanner.ScannerError as ye:
                    # Show a helpful error message in the UI with scanner details
                    st.error(f"❌ Error parseando YAML en {cfg_path}: {ye}")
                    st.text(str(ye))
                    # Re-raise so developers see the traceback in logs if needed
                    raise

                # Load downloader module
                module_file = os.path.join(os.getcwd(), "download", "download_point_forecast.py")
                if not os.path.exists(module_file):
                    st.error(f"❌ Módulo de descarga no encontrado: {module_file}")
                else:
                    spec = importlib.util.spec_from_file_location("download_point_forecast", module_file)
                    dp = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(dp)

                    # Extract configuration from nested structure
                    urls = config.get("urls", {})
                    models = config.get("models", {})
                    
                    url_det = urls.get("det")
                    url_ens = urls.get("ens")
                    det_models = models.get("det", [])
                    ens_models = models.get("ens", [])
                    variables = config.get("variables", [])
                    
                    # Validate that URLs were loaded
                    if not url_det or not url_ens:
                        st.error(f"❌ Error: URLs no encontradas en configuración. url_det={url_det}, url_ens={url_ens}")
                        st.stop()
                    
                    # Validate that models were loaded
                    if not det_models or not ens_models:
                        st.error(f"❌ Error: Modelos no encontrados en configuración. det_models={bool(det_models)}, ens_models={bool(ens_models)}")
                        st.stop()
                    
                    # Validate that variables were loaded
                    if not variables:
                        st.error(f"❌ Error: Variables no encontradas en configuración.")
                        st.stop()
                    
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
                    st.subheader("Datos del pronóstico")
                    
                    tab1, tab2 = st.tabs(["Determinista", "Ensemble"])
                    
                    with tab1:
                       
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
                            
                            # Create interactive plot with all models
                            plot_df = fcst_df[["forecast_date", col_to_plot, "model"]].dropna()

                            # Load plot config to get full variable names
                            plot_cfg = {}
                            try:
                                plot_cfg_path = os.path.join(os.getcwd(), 'plot', 'etc', 'config_plot.yaml')
                                if not os.path.exists(plot_cfg_path):
                                    plot_cfg_path = os.path.join(os.getcwd(), 'plot', 'etc', 'config_plot.yml')
                                with open(plot_cfg_path, 'r', encoding='utf-8') as pf:
                                    rawpf = pf.read()
                                if '\t' in rawpf:
                                    rawpf = rawpf.replace('\t', '  ')
                                plot_cfg = yaml.safe_load(rawpf) or {}
                            except Exception:
                                plot_cfg = {}

                            dict_name_vars_plot = plot_cfg.get('dict_name_vars', {})
                            dict_name_units_plot = plot_cfg.get('dict_name_units', {})
                            var_full = dict_name_vars_plot.get(col_to_plot, col_to_plot)
                            var_unit = dict_name_units_plot.get(col_to_plot, '')

                            fig = go.Figure()
                            for model in sorted(plot_df["model"].unique()):
                                model_data = plot_df[plot_df["model"] == model].sort_values("forecast_date")
                                fig.add_trace(go.Scatter(
                                    x=model_data["forecast_date"],
                                    y=model_data[col_to_plot],
                                    name=model,
                                    mode="lines",
                                    hovertemplate="<b>%{fullData.name}</b><br>Fecha: %{x|%Y-%m-%d %H:%M}<br>Valor: %{y:.2f}<extra></extra>"
                                ))

                            fig.update_layout(
                                title=f"Pronóstico determinista: {var_full}",
                                xaxis_title="Fecha",
                                yaxis_title=f"{var_full} {var_unit}",
                                hovermode="x unified",
                                height=500,
                                template="plotly_white"
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Also display static matplotlib plots using the plot module (if available)
                            try:
                                plot_module_file = os.path.join(os.getcwd(), 'plot', 'plot_point_forecast.py')
                                if os.path.exists(plot_module_file):
                                    specp = importlib.util.spec_from_file_location('plot_point_forecast', plot_module_file)
                                    plotmod = importlib.util.module_from_spec(specp)
                                    specp.loader.exec_module(plotmod)

                                    try:
                                        fig_static = plotmod.plot_forecast_data(fcst_df, col_to_plot)
                                        if fig_static is not None:
                                            st.subheader('Gráfica estática (determinista)')
                                            st.pyplot(fig_static)
                                    except Exception as e:
                                        st.warning(f"No se pudo generar la gráfica estática determinista: {e}")
                            except Exception:
                                pass
                        else:
                            st.info("No hay columnas numéricas disponibles en datos determinista.")
                        
                        st.write("**Datos de modelos deterministas en tabla:**")
                        st.dataframe(fcst_df, use_container_width=True)

                    with tab2:
                        
                        # Plotting options for ensemble
                        ens_numeric = [
                            c for c in ens_df.columns
                            if c != "forecast_date"
                            and pd.api.types.is_numeric_dtype(ens_df[c])
                        ]
                        if ens_numeric:
                            ens_col = st.selectbox(
                                "Selecciona variable para graficar (ensemble):",
                                options=dict_name_vars_plot.keys(),
                                key="ens_var_select"
                            )

                            # static ensemble plots
                            try:
                                plot_module_file = os.path.join(os.getcwd(), 'plot', 'plot_point_forecast.py')
                                if os.path.exists(plot_module_file):
                                    specp = importlib.util.spec_from_file_location('plot_point_forecast', plot_module_file)
                                    plotmod = importlib.util.module_from_spec(specp)
                                    specp.loader.exec_module(plotmod)

                                    try:
                                        fig_box = plotmod.plot_ens_boxplot(ens_df, ens_col)
                                        if fig_box is not None:
                                            st.subheader('Gráfica estática (ensemble - boxplot)')
                                            st.pyplot(fig_box)

                                        fig_quant = plotmod.plot_ens_forecast_data(ens_df, ens_col, quantiles=True)
                                        if fig_quant is not None:
                                            st.subheader('Gráfica estática (ensemble - quantiles)')
                                            st.pyplot(fig_quant)
                                    except Exception as e:
                                        st.warning(f"No se pudieron generar las gráficas estáticas ensemble: {e}")
                            except Exception:
                                pass
                        else:
                            st.info("No hay columnas numéricas disponibles en datos ensemble.")
                        
                        st.write("**Datos de todos los ensemble en tabla:**")
                        st.dataframe(ens_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error descargando/mostrando pronóstico: {e}")
            st.text(traceback.format_exc())
            st.session_state["show_forecast"] = False
