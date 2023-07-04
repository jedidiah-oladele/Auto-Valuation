import math
import pandas as pd
import numpy as np
import streamlit as st
import yaml
import streamlit_authenticator as stauth

st.set_page_config(page_title="RS Score", layout="wide")

st.markdown(
    """
<h1 style="text-align:center;">RS Score</h1>""",
    unsafe_allow_html=True,
)


# Import the YAML file into script
with open("data/users.yaml") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

# Create the authenticator object
authenticator = stauth.authenticate.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)


# Render the login widget
name, authentication_status, username = authenticator.login("Login", "main")


if not st.session_state["authentication_status"]:
    if st.session_state["authentication_status"] == False:
        # Failed to login
        st.error("Username/password is incorrect")

    st.stop()


def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Radius of the Earth in kilometers
    radius = 6371.0

    # Calculate the differences between the latitudes and longitudes
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Apply the Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance


def calculate_proximity_score(distance):
    """
    Calculates a promixity score for a given distance, on a scale of minimum to maximum
    The maximum and minimum distance should be defined in kilometers
    """

    if distance > max_distance:  # Zero score for disances that are too far
        proximity_score = 0

    elif distance < min_distance:  # Zero score for distances that are too close
        proximity_score = 0

    else:
        # Assign a score based on where the distance lies within the maximum and minimum
        proximity_score = 1 - (
            (distance - min_distance) / (max_distance - min_distance)
        )

    return proximity_score


@st.cache_data()
def get_public_services_weights():
    # Read public services data
    PUBLIC_SERVICES_WEIGHTS = pd.read_csv("data/public_services_weights.csv")
    return PUBLIC_SERVICES_WEIGHTS


@st.cache_data()
def get_public_services_data():
    public_services_data = pd.read_csv("data/public_services.csv")

    # Clean data
    public_services_data["main_cat"] = public_services_data["main_cat"].apply(
        lambda x: x.strip().title()
    )
    public_services_data["sub_cat"] = public_services_data["sub_cat"].apply(
        lambda x: x.strip().title()
    )
    # Filter out irrelevant row
    public_services_data = public_services_data[
        (public_services_data["sub_cat"] != "Delete")
        & (public_services_data["sub_cat"] != "Delete")
    ]
    return public_services_data


@st.cache_data()  # To avoid duplicate calculations
def get_rs_score(listings_lat, listings_long):
    # Clean up function

    # Get distance of each public service from the listing
    public_services_data["distance"] = public_services_data.apply(
        lambda row: calculate_distance(
            lat1=listings_lat,
            lon1=listings_long,
            lat2=row["lat"],
            lon2=row["lng"],
        ),
        axis=1,
    )

    # Determine the proximity score for the each distances
    public_services_data["proximity_score"] = public_services_data["distance"].apply(
        lambda x: calculate_proximity_score(x)
    )

    # Keep the maximum proximity score in each sub category
    proximity_scores = (
        public_services_data.groupby(["main_cat", "sub_cat"])["proximity_score"]
        .max()
        .reset_index()
    )

    # Get the weighted score
    proximity_scores = pd.merge(
        proximity_scores,
        PUBLIC_SERVICES_WEIGHTS,
        how="left",
        on=["main_cat", "sub_cat"],
    )

    proximity_scores["weighted_score"] = (
        proximity_scores["proximity_score"] * proximity_scores["weight"]
    )

    # Get the scores for the various groups
    category_scores = (
        proximity_scores.groupby("main_cat")["weighted_score"].sum().reset_index()
    )

    # Find the average category score (final rs_score)
    rs_score = round(category_scores["weighted_score"].mean(), 4)

    return {
        "rs_score": rs_score,
        "category_scores": category_scores,
        "proximity_scores": proximity_scores,
    }


# Get data
PUBLIC_SERVICES_WEIGHTS = get_public_services_weights()
public_services_data = get_public_services_data()

# Get maximum and minimum distances
col1, col2 = st.columns(2)
max_distance = col1.number_input(
    "Maximum Distance (km)", value=10.0, min_value=0.0, step=0.01, format="%0.3f"
)
min_distance = col2.number_input(
    "Minimum Distance (km)", value=0.1, min_value=0.0, step=0.01, format="%0.3f"
)


tab2, tab1 = st.tabs(["Upload file", "Enter coordinates"])
with tab2:
    # CSV upload
    uploaded_file = st.file_uploader(
        "Upload a CSV file with lat and lng", accept_multiple_files=False
    )
    if uploaded_file is not None:
        # Read listing data
        real_estate_listings = pd.read_csv(uploaded_file)
        real_estate_listings = real_estate_listings.dropna(subset=["lat", "lng"])
        real_estate_listings["rs_score"] = np.nan

        # Display progress
        progress_bar = st.progress(0)
        progress_counter = 1
        total_length = len(real_estate_listings)

        for i, listing_row in real_estate_listings.iterrows():
            # Calculate rs_score for all rows

            results = get_rs_score(listing_row["lat"], listing_row["lng"])
            real_estate_listings.loc[i, "rs_score"] = results["rs_score"]

            progress_bar.progress(progress_counter / total_length)
            progress_counter += 1

        st.dataframe(real_estate_listings, use_container_width=True)

        # Export pulled data as a csv
        st.download_button(
            label="Download Results",
            data=real_estate_listings.to_csv(index=False).encode("utf-8"),
            file_name="real_estate_listings.csv",
        )


with tab1:
    # Single coordinates
    col1, col2 = st.columns(2)
    input_lat = col1.number_input(
        "Latitude", value=24.8033547, min_value=0.0, step=0.1, format="%0.7f"
    )
    input_lng = col2.number_input(
        "Longitude", value=46.6206349, min_value=0.0, step=0.1, format="%0.7f"
    )

    results = get_rs_score(input_lat, input_lng)

    rs_score = round(results["rs_score"], 4)
    category_scores = results["category_scores"]
    proximity_scores = results["proximity_scores"]

    # Display result
    st.success(f"RS Score: {rs_score}")

    if st.checkbox("Show Info"):
        st.write("Scores across main public services categories:")
        st.dataframe(category_scores, use_container_width=True)

        st.write("Scores across subcategories:")
        st.dataframe(proximity_scores, use_container_width=True)


authenticator.logout("Logout", "main")
