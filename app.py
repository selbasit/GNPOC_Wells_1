
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import plotly.express as px
from pyproj import Transformer

st.set_page_config(page_title="GNPOC Well Intelligence (Live ML)", layout="wide")
st.title("🚀 GNPOC Well Intelligence App (Live Training)")

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("GNPOC_Wells.csv")

    def parse_coord(coord_str):
        if pd.isna(coord_str): return None
        import re
        match = re.match(r'(\d+)([NSEW])', coord_str.upper())
        if match:
            val = int(match.group(1)) / 1e6
            return -val if match.group(2) in ['S', 'W'] else val
        return None

    df['LAT_DECIMAL'] = df['LATITUDE'].apply(parse_coord)
    df['LON_DECIMAL'] = df['LONGITUDE'].apply(parse_coord)

    cols = ['NORTHING', 'EASTING', 'MSL', 'BLOCK #', 'OPERATOR', 'WELL TYPE', 'LAT_DECIMAL', 'LON_DECIMAL']
    df = df[cols].replace("?", np.nan).dropna()
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    for col, enc in zip(categorical_cols, encoders):
        df[col] = enc.fit_transform(df[col])

    return df

num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")
numeric_cols = ["NORTHING", "EASTING", "MSL"]
categorical_cols = ["BLOCK #", "OPERATOR"]
encoders = [LabelEncoder(), LabelEncoder()]

df = load_and_prepare_data()

X = df[numeric_cols + categorical_cols]
y_class = LabelEncoder().fit_transform(df["WELL TYPE"])
y_lat = df["LAT_DECIMAL"]
y_lon = df["LON_DECIMAL"]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y_class)

lat_model = RandomForestRegressor(n_estimators=100, random_state=42)
lon_model = RandomForestRegressor(n_estimators=100, random_state=42)
lat_model.fit(X, y_lat)
lon_model.fit(X, y_lon)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

tab1, tab2, tab3 = st.tabs(["🎯 Predict WELL TYPE", "🧭 Cluster Map", "📍 Predict GPS"])

with tab1:
    st.subheader("Predict WELL TYPE")
    with st.form("class_form"):
        northing = st.number_input("NORTHING")
        easting = st.number_input("EASTING")
        msl = st.number_input("MSL")
        block = st.text_input("BLOCK #")
        operator = st.text_input("OPERATOR")
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([[northing, easting, msl, block, operator]], columns=numeric_cols + categorical_cols)
        input_df[numeric_cols] = num_imputer.transform(input_df[numeric_cols])
        input_df[categorical_cols] = cat_imputer.transform(input_df[categorical_cols])
        for col, enc in zip(categorical_cols, encoders):
            input_df[col] = enc.transform(input_df[col])
        pred = clf.predict(input_df)[0]
        st.success(f"Predicted WELL TYPE: {pred}")

with tab2:
    st.subheader("KMeans Clustering (3 groups)")
    df["Cluster"] = clusters
    fig = px.scatter(df, x="EASTING", y="NORTHING", color=df["Cluster"].astype(str), title="Well Clusters", height=600)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Predict Real-World GPS Coordinates")
    with st.form("gps_form"):
        northing = st.number_input("Northing", key="n")
        easting = st.number_input("Easting", key="e")
        msl = st.number_input("MSL", key="m")
        block = st.text_input("Block #", key="b")
        operator = st.text_input("Operator", key="o")
        submit = st.form_submit_button("Estimate GPS")

    if submit:
        input_df = pd.DataFrame([[northing, easting, msl, block, operator]], columns=numeric_cols + categorical_cols)
        input_df[numeric_cols] = num_imputer.transform(input_df[numeric_cols])
        input_df[categorical_cols] = cat_imputer.transform(input_df[categorical_cols])
        for col, enc in zip(categorical_cols, encoders):
            input_df[col] = enc.transform(input_df[col])
        pred_lat = lat_model.predict(input_df)[0]
        pred_lon = lon_model.predict(input_df)[0]

        transformer = Transformer.from_crs("epsg:32636", "epsg:4326", always_xy=True)
        lon_utm, lat_utm = transformer.transform(easting, northing)

        st.success(f"Predicted (ML): {pred_lat:.6f}, {pred_lon:.6f}")
        st.info(f"UTM Converted: {lat_utm:.6f}, {lon_utm:.6f}")
