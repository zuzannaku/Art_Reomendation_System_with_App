from pathlib import Path
from urllib.parse import quote
import base64
import numpy as np
import pandas as pd
import requests
import streamlit as st
from textwrap import dedent
import json


PROCESSED_DATA_DIR = Path("/Users/zuzakutowska/Desktop/art_recomender/data/processed")

app_df = pd.read_csv(PROCESSED_DATA_DIR / "app_artworks_metadata.csv")
object_ids = np.load(PROCESSED_DATA_DIR / "object_ids.npy")
visual_similarity = np.load(PROCESSED_DATA_DIR / "visual_similarity.npy")
semantic_similarity = np.load(PROCESSED_DATA_DIR / "semantic_similarity.npy")

################## HELPERS #####################

# the main page will have 6 display categories to start exploring
CATEGORIES = {
    "Greek and Roman Art": {
        "description": "Artworks inspired by the cultures of ancient Greece and Rome.",
        "representative_object_id": 247009
    },
    "Medieval Art": {
        "description": "Sacred images and objects from the medieval world.",
        "representative_object_id": 470921
    },
    "Asian Art": {
        "description": "Ink paintings, scrolls, and works from across Asia.",
        "representative_object_id": 40081
    },
    "Islamic Art": {
        "description": "Manuscripts, paintings, and decorative works from Islamic cultures.",
        "representative_object_id": 446587
    },
    "European Paintings": {
        "description": "Paintings from European artistic traditions.",
        "representative_object_id": 459106
    },
    "Modern and Contemporary Art": {
        "description": "Modern works exploring new styles, subjects, and perspectives.",
        "representative_object_id": 459100
    }
}



@st.cache_data
def load_image_from_url(image_url):
    if pd.isna(image_url) or not str(image_url).strip():
        return None, None
    original_url = str(image_url).strip()
    urls_to_try = [
        original_url,
        original_url.replace("/original/", "/web-large/")
    ]
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.metmuseum.org/"
    }
    for url in urls_to_try:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=20
            )
            response.raise_for_status()
            content_type = response.headers.get(
                "Content-Type",
                "image/jpeg"
            )
            if response.content:
                return response.content, content_type
        except requests.RequestException:
            continue
    return None, None


def display_clickable_image( # i want the small images to be clickable and to make sure the user knows th eimage is clickable i add a hover effect
    image_data,
    content_type,
    target_url,
    alt_text,
    height=360
):
    encoded_image = base64.b64encode(image_data).decode("utf-8")

    image_html = f"""
    <style>
        .image-container {{
            width: 100%;
            height: {height}px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background-color: #f5f3ef;
            border-radius: 6px;
            cursor: pointer;
        }}

        .artwork-image {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            object-position: center;
            display: block;
            transition: transform 0.35s ease, filter 0.35s ease;
        }}

        .image-container:hover .artwork-image {{
            transform: scale(1.065);
            filter: brightness(0.96);
        }}
    </style>

    <a
        href="{target_url}"
        target="_self"
        style="display: block; text-decoration: none;"
    >
        <div class="image-container">
            <img
                src="data:{content_type};base64,{encoded_image}"
                alt="{alt_text}"
                class="artwork-image"
            >
        </div>
    </a>
    """

    st.html(image_html)
    
def display_detail_image( #
    image_data,
    content_type,
    alt_text,
    max_height=550
):
    encoded_image = base64.b64encode(image_data).decode("utf-8")

    image_html = f"""
    <div style="
        width: 100%;
        height: {max_height}px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        background-color: #f5f3ef;
        border-radius: 6px;
    ">
        <img
            src="data:{content_type};base64,{encoded_image}"
            alt="{alt_text}"
            style="
                max-width: 100%;
                max-height: 100%;
                width: auto;
                height: auto;
                object-fit: contain;
                display: block;
            "
        >
    </div>
    """

    st.html(image_html)

def show_women_star():
    st.html(
        """
        <style>
            .women-star-link {
                position: fixed;
                top: 4.5rem;
                right: 2rem;
                z-index: 9999;

                width: 42px;
                height: 42px;
                display: flex;
                align-items: center;
                justify-content: center;

                background-color: #ffe4ec;
                color: #c65f7c;
                border-radius: 50%;
                text-decoration: none;
                font-size: 1.35rem;

                box-shadow: 0 3px 10px rgba(122, 61, 82, 0.16);
                transition:
                    transform 0.2s ease,
                    background-color 0.2s ease,
                    box-shadow 0.2s ease;
            }

            .women-star-link:hover {
                transform: scale(1.1);
                background-color: #ffd4e1;
                box-shadow: 0 5px 14px rgba(122, 61, 82, 0.24);
            }
        </style>

        <a
            class="women-star-link"
            href="?gallery=women"
            target="_self"
            title="Explore artworks by women"
            aria-label="Explore artworks by women"
        >
            ★
        </a>
        """
    )

def get_recommendations(
    selected_object_id,
    semantic_weight,
    number_of_recommendations=12
):
    selected_object_id = int(selected_object_id)

    matching_indices = np.where(
        object_ids.astype(int) == selected_object_id
    )[0]

    if len(matching_indices) == 0:
        return pd.DataFrame()

    selected_index = matching_indices[0]
    visual_weight = (1.0 - semantic_weight) * 1.5

    combined_similarity = (
        semantic_weight * semantic_similarity[selected_index]
        + visual_weight * visual_similarity[selected_index]
    )

    ranked_indices = np.argsort(
        combined_similarity
    )[::-1]

    ranked_indices = ranked_indices[
        ranked_indices != selected_index
    ]

    recommended_indices = ranked_indices[
        :number_of_recommendations
    ]

    recommendation_ranking = pd.DataFrame({
        "object_id": object_ids[recommended_indices].astype(int),
        "similarity_score": combined_similarity[recommended_indices],
        "recommendation_rank": np.arange(
            len(recommended_indices)
        )
    })

    recommendations = recommendation_ranking.merge(
        app_df,
        on="object_id",
        how="left",
        validate="one_to_one"
    )

    recommendations = (
        recommendations
        .sort_values("recommendation_rank")
        .reset_index(drop=True)
    )

    return recommendations

CATEGORY_DEPARTMENTS = {
    "Asian Art": ["asian art"],
    "European Paintings": ["european paintings"],
    "Islamic Art": ["islamic art"],
    "Greek and Roman Art": ["greek and roman art"],
    "Medieval Art": ["medieval art", "the cloisters"],
    "Modern and Contemporary Art": ["modern and contemporary art"]
}



################### APP ###############################

st.set_page_config(
    page_title="Art Explorer",
    layout="wide"
)

#################### SESSION STATE ###################


if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

if "selected_object_id" not in st.session_state:
    st.session_state.selected_object_id = None

if "artwork_source" not in st.session_state:
    st.session_state.artwork_source = None


#################### URL NAVIGATION ###################

selected_category_from_url = st.query_params.get("category")
selected_gallery_from_url = st.query_params.get("gallery")
selected_artwork_from_url = st.query_params.get("artwork")
selected_source_from_url = st.query_params.get("source")


if selected_category_from_url in CATEGORIES:
    st.session_state.selected_category = selected_category_from_url
    st.session_state.page = "collection"


if selected_gallery_from_url == "women":
    st.session_state.selected_category = None
    st.session_state.page = "women"


if selected_artwork_from_url:
    try:
        st.session_state.selected_object_id = int(
            selected_artwork_from_url
        )

        st.session_state.artwork_source = (
            selected_source_from_url
            if selected_source_from_url
            else "collection"
        )

        st.session_state.page = "artwork"

    except (TypeError, ValueError):
        st.session_state.selected_object_id = None
        
###########################


def show_home_page():  
    show_women_star()
    
    st.markdown(
        """
        <h1 style="text-align: center; margin-bottom: 0.25rem;">
            Art Recommender
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    st.html(
        """
        <div style="
            max-width: 780px;
            margin: 0 auto 3rem auto;
            text-align: center;
            font-size: 1.05rem;
            line-height: 1.7;
            color: #555555;
        ">
            <p style="margin-bottom: 1rem;">
                This tool was created to support art discovery by providing
                personalized recommendations based on your interests.
            </p>

            <p style="margin-bottom: 0;">
                Start by picking a collection below. Once you select an individual
                artwork, you can explore more details about it and choose how
                similar artworks should be recommended.
            </p>
        </div>
        """
    )
    category_names = list(CATEGORIES.keys())

    for row_start in range(0, len(category_names), 3):
        columns = st.columns(3, gap="large")

        row_categories = category_names[
            row_start:row_start + 3
        ]

        for column_index, category_name in enumerate(
            row_categories
        ):
            category = CATEGORIES[category_name]

            representative_rows = app_df[
                app_df["object_id"]
                == category["representative_object_id"]
            ]

            with columns[column_index]:
                if representative_rows.empty:
                    st.warning(
                        f"Object ID "
                        f"{category['representative_object_id']} "
                        f"is not present in app_df."
                    )
                    continue

                representative_artwork = (
                    representative_rows.iloc[0]
                )

                image_data, content_type = load_image_from_url(
                    representative_artwork["primary_image"]
                )

                if image_data is not None:
                    category_url = quote(category_name)

                    display_clickable_image(
                        image_data=image_data,
                        content_type=content_type,
                        target_url=f"?category={category_url}",
                        alt_text=category_name,
                        height=360
                    )
                else:
                    st.warning(
                        f"Could not load the image for "
                        f"{category_name}."
                    )

                category_text = dedent(
                    f"""
                    <div style="
                        text-align: center;
                        min-height: 115px;
                        padding-top: 0.8rem;
                    ">
                        <h3 style="
                            margin-top: 0;
                            margin-bottom: 0.35rem;
                        ">{category_name}</h3>
                        <p style="
                            min-height: 48px;
                            margin-top: 0;
                            margin-bottom: 0.8rem;
                            color: #666666;
                            font-size: 0.9rem;
                            line-height: 1.4;
                        ">{category["description"]}</p>
                    </div>
                    """
                )

                st.markdown(
                    category_text,
                    unsafe_allow_html=True
                )

        st.write("")
        

def show_collection_page():
    selected_category = st.session_state.selected_category

    if st.button("← Back to home"):
        st.query_params.clear()
        st.session_state.selected_category = None
        st.session_state.selected_object_id = None
        st.session_state.page = "home"
        st.rerun()

    if selected_category is None:
        st.warning("No collection has been selected.")
        return

    st.markdown(
        dedent(
            f"""
            <h1 style="
                text-align: center;
                margin-bottom: 0.4rem;
            ">
                {selected_category}
            </h1>
            """
        ),
        unsafe_allow_html=True
    )

    st.markdown(
        dedent(
            f"""
            <p style="
                text-align: center;
                color: #666666;
                margin-top: 0;
                margin-bottom: 2.5rem;
            ">
                {CATEGORIES[selected_category]["description"]}
            </p>
            """
        ),
        unsafe_allow_html=True
    )

    selected_departments = CATEGORY_DEPARTMENTS[
        selected_category
    ]

    collection_df = app_df[
        app_df["department"]
        .fillna("")
        .str.lower()
        .isin(selected_departments)
    ].copy()

    collection_df = collection_df[
        collection_df["primary_image"].notna()
        & collection_df["primary_image"]
        .astype(str)
        .str.strip()
        .ne("")
    ]

    collection_df = collection_df.sample(
        n=min(60, len(collection_df)),
        random_state=42
    )

    for row_start in range(0, len(collection_df), 4):
        columns = st.columns(4, gap="medium")

        row_artworks = collection_df.iloc[
            row_start:row_start + 4
        ]

        for column_index, (_, artwork) in enumerate(
            row_artworks.iterrows()
        ):
            with columns[column_index]:
                image_data, content_type = load_image_from_url(
                    artwork["primary_image"]
                )

                title = (
                    str(artwork["title"]).strip()
                    if pd.notna(artwork["title"])
                    and str(artwork["title"]).strip()
                    else "Untitled"
                )

                artist = (
                    str(artwork["artist_display_name"]).strip()
                    if pd.notna(artwork["artist_display_name"])
                    and str(artwork["artist_display_name"]).strip()
                    else "Unknown artist"
                )

                century = (
                    str(artwork["century"]).strip()
                    if pd.notna(artwork["century"])
                    and str(artwork["century"]).strip()
                    else "Date unknown"
                )

                if image_data is not None:
                    artwork_id = int(artwork["object_id"])
                    collection_url = quote(selected_category)

                    artwork_url = (
                        f"?category={collection_url}"
                        f"&artwork={artwork_id}"
                    )

                    display_clickable_image(
                        image_data=image_data,
                        content_type=content_type,
                        target_url=artwork_url,
                        alt_text=title,
                        height=300
                    )
                else:
                    st.warning("Image could not be loaded.")

                metadata_text = dedent(
                    f"""
                    <div style="
                        text-align: center;
                        min-height: 120px;
                        padding-top: 0.6rem;
                    ">
                        <p style="
                            font-weight: 600;
                            margin-top: 0;
                            margin-bottom: 0.25rem;
                        ">{title}</p>
                        <p style="
                            color: #666666;
                            font-size: 0.85rem;
                            margin-top: 0;
                            margin-bottom: 0.15rem;
                        ">{artist}</p>
                        <p style="
                            color: #888888;
                            font-size: 0.8rem;
                            margin-top: 0;
                            margin-bottom: 0;
                        ">{century}</p>
                    </div>
                    """
                )

                st.markdown(
                    metadata_text,
                    unsafe_allow_html=True
                )

        st.write("")
     


def show_artwork_page():
    selected_object_id = st.session_state.selected_object_id
    selected_category = st.session_state.selected_category
    artwork_source = st.session_state.artwork_source

    back_label = (
        "← Back to Women Artists"
        if artwork_source == "women"
        else "← Back to collection"
    )

    if st.button(back_label):
        st.query_params.clear()
        st.session_state.selected_object_id = None

        if artwork_source == "women":
            st.session_state.page = "women"
            st.session_state.selected_category = None
            st.query_params["gallery"] = "women"

        else:
            st.session_state.page = "collection"

            if selected_category is not None:
                st.query_params["category"] = selected_category

        st.session_state.artwork_source = None
        st.rerun()

    selected_rows = app_df[
        app_df["object_id"] == selected_object_id
    ]

    if selected_rows.empty:
        st.warning(
            f"Artwork with object ID {selected_object_id} "
            "was not found."
        )
        return

    artwork = selected_rows.iloc[0]

    title = (
        str(artwork["title"]).strip()
        if pd.notna(artwork["title"])
        and str(artwork["title"]).strip()
        else "Untitled"
    )

    artist = (
        str(artwork["artist_display_name"]).strip()
        if pd.notna(artwork["artist_display_name"])
        and str(artwork["artist_display_name"]).strip()
        else "Unknown artist"
    )

    century = (
        str(artwork["century"]).strip()
        if pd.notna(artwork["century"])
        and str(artwork["century"]).strip()
        else "Date unknown"
    )

    medium = (
        str(artwork["medium"]).strip()
        if pd.notna(artwork["medium"])
        and str(artwork["medium"]).strip()
        else "Medium unknown"
    )

    country = (
        str(artwork["origin_country"]).strip()
        if pd.notna(artwork["origin_country"])
        and str(artwork["origin_country"]).strip()
        else "Origin unknown"
    )
    
    object_name = (
        str(artwork["object_name"]).strip()
        if pd.notna(artwork["object_name"])
        and str(artwork["object_name"]).strip()
        else "Object type unknown"
    )

    department = (
        str(artwork["department"]).strip()
        if pd.notna(artwork["department"])
        and str(artwork["department"]).strip()
        else "Department unknown"
    )

    tags = (
        str(artwork["tags"]).strip()
        if pd.notna(artwork["tags"])
        and str(artwork["tags"]).strip()
        else None
    )

    met_link = (
        str(artwork["link_resource"]).strip()
        if pd.notna(artwork["link_resource"])
        and str(artwork["link_resource"]).strip()
        else None
    )

    left_column, right_column = st.columns(
        [1.2, 1],
        gap="large"
    )

    with left_column:
        image_data, content_type = load_image_from_url(
            artwork["primary_image"]
        )

        if image_data is not None:
            display_detail_image(
                image_data=image_data,
                content_type=content_type,
                alt_text=title,
                max_height=650
            )
        else:
            st.warning("The artwork image could not be loaded.")

    with right_column:
        st.title(title)
        st.subheader(artist)
        
        artist_gender = (
            str(artwork["artist_gender"]).strip().lower()
            if pd.notna(artwork["artist_gender"])
            else ""
        )

        if artist_gender == "female":
            st.html(
                """
                <div style="
                    margin-top: 1.4rem;
                    padding: 1rem 1.2rem;
                    background-color: #ffe4ec;
                    border-radius: 10px;
                    color: #7a3d52;
                    line-height: 1.5;
                ">
                    <strong>Good catch!</strong>
                    You just discovered an artwork by a woman.
                    Works by women represent only 0.6% of this database.
                    <a
                        href="?gallery=women"
                        target="_self"
                        style="
                            color: #7a3d52;
                            font-weight: 600;
                            text-decoration: underline;
                        "
                    >
                        Explore the full Women Artists collection here.
                    </a>
                </div>
                """
            )       

        st.markdown(f"**Date:** {century}")
        st.markdown(f"**Medium:** {medium}")
        st.markdown(f"**Country:** {country}")
        st.markdown(f"**Object type:** {object_name}")
        st.markdown(f"**Department:** {department}")

        if tags:
            st.markdown(f"**Tags:** {tags}")

        if met_link:
            st.markdown(
                f'<a href="{met_link}" target="_blank" '
                f'style="display:inline-block; margin-top:1rem; color:inherit; '
                f'text-decoration:underline;">'
                f'Read more about this artwork at the official Met gallery.'
                f'</a>',
                unsafe_allow_html=True
            )
    st.markdown("---")

    st.markdown(
        """
        <h2 style="
            text-align: left;
            margin-bottom: 0.25rem;
        ">
            Explore similar artworks
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style="
            text-align: left;
            color: #666666;
            margin-top: 0;
            margin-bottom: 1rem;
        ">
            Adjust the balance between visual appearance and semantic context.
        </p>
        """,
        unsafe_allow_html=True
    )

    slider_area, empty_space = st.columns(
        [3, 7],
        gap="small"
    )

    with slider_area:
        st.html(
            """
            <style>
                /* Hide the automatic minimum and maximum numbers */
                div[data-testid="stSlider"] [data-testid="stTickBar"] {
                    display: none;
                }
            </style>
            """
        )

        semantic_weight = st.slider(
            "Recommendation balance",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            label_visibility="collapsed",
            key="recommendation_balance"
        )

        label_left, label_right = st.columns(2)

        with label_left:
            st.html(
                """
                <div style="
                    text-align: left;
                    color: #666666;
                    font-size: 0.85rem;
                    margin-top: -0.7rem;
                ">
                    Visual
                </div>
                """
            )

        with label_right:
            st.html(
                """
                <div style="
                    text-align: right;
                    color: #666666;
                    font-size: 0.85rem;
                    margin-top: -0.7rem;
                ">
                    Semantic
                </div>
                """
            )


    recommendations = get_recommendations(
        selected_object_id=selected_object_id,
        semantic_weight=semantic_weight,
        number_of_recommendations=12
    )

    st.write("")

    for row_start in range(0, len(recommendations), 4):
        columns = st.columns(4, gap="medium")

        row_recommendations = recommendations.iloc[
            row_start:row_start + 4
        ]

        for column_index, (_, recommendation) in enumerate(
            row_recommendations.iterrows()
        ):
            with columns[column_index]:
                recommendation_id = int(
                    recommendation["object_id"]
                )

                recommendation_title = (
                    str(recommendation["title"]).strip()
                    if pd.notna(recommendation["title"])
                    and str(recommendation["title"]).strip()
                    else "Untitled"
                )

                recommendation_artist = (
                    str(
                        recommendation["artist_display_name"]
                    ).strip()
                    if pd.notna(
                        recommendation["artist_display_name"]
                    )
                    and str(
                        recommendation["artist_display_name"]
                    ).strip()
                    else "Unknown artist"
                )

                recommendation_century = (
                    str(recommendation["century"]).strip()
                    if pd.notna(recommendation["century"])
                    and str(recommendation["century"]).strip()
                    else "Date unknown"
                )

                image_data, content_type = load_image_from_url(
                    recommendation["primary_image"]
                )

                recommendation_url = (
                    f"?artwork={recommendation_id}"
                    f"&source={artwork_source or 'collection'}"
                )

                if artwork_source == "women":
                    recommendation_url += "&gallery=women"

                elif selected_category is not None:
                    recommendation_url += (
                        f"&category={quote(selected_category)}"
                    )

                display_clickable_image(
                    image_data=image_data,
                    content_type=content_type,
                    target_url=recommendation_url,
                    alt_text=recommendation_title,
                    height=300
                )

                recommendation_text = f"""
                <div style="
                    text-align: center;
                    min-height: 120px;
                    padding-top: 0.6rem;
                ">
                    <p style="
                        font-weight: 600;
                        margin: 0 0 0.25rem 0;
                    ">{recommendation_title}</p>
                    <p style="
                        color: #666666;
                        font-size: 0.85rem;
                        margin: 0 0 0.15rem 0;
                    ">{recommendation_artist}</p>
                    <p style="
                        color: #888888;
                        font-size: 0.8rem;
                        margin: 0;
                    ">{recommendation_century}</p>
                </div>
                """

                st.html(recommendation_text)

        st.write("")


def show_women_page():
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #fff5f8;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("← Back to home"):
        st.query_params.clear()
        st.session_state.selected_object_id = None
        st.session_state.artwork_source = None
        st.session_state.page = "home"
        st.rerun()

    st.markdown(
        """
        <h1 style="
            text-align: center;
            margin-bottom: 0.5rem;
        ">
            Women Artists
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            max-width: 760px;
            margin: 0 auto 2.5rem auto;
            padding: 1rem 1.3rem;
            text-align: center;
            background-color: #ffe4ec;
            border-radius: 10px;
            color: #7a3d52;
            line-height: 1.5;
        ">
            Artworks by women represent only 0.6% of this database.
            This collection highlights the women whose work is present
            and makes their artworks easier to discover.
        </div>
        """,
        unsafe_allow_html=True
    )

    women_df = app_df[
        app_df["artist_gender"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("female")
    ].copy()

    women_df = women_df[
        women_df["primary_image"].notna()
        & women_df["primary_image"]
        .astype(str)
        .str.strip()
        .ne("")
    ]

    women_df = women_df.sample(
        n=min(24, len(women_df)),
        random_state=42
    )

    if women_df.empty:
        st.warning(
            "No artworks by women were found. "
            "Check the values in the artist_gender column."
        )
        return

    for row_start in range(0, len(women_df), 4):
        columns = st.columns(4, gap="medium")

        row_artworks = women_df.iloc[
            row_start:row_start + 4
        ]

        for column_index, (_, artwork) in enumerate(
            row_artworks.iterrows()
        ):
            with columns[column_index]:
                title = (
                    str(artwork["title"]).strip()
                    if pd.notna(artwork["title"])
                    and str(artwork["title"]).strip()
                    else "Untitled"
                )

                artist = (
                    str(artwork["artist_display_name"]).strip()
                    if pd.notna(artwork["artist_display_name"])
                    and str(artwork["artist_display_name"]).strip()
                    else "Unknown artist"
                )

                century = (
                    str(artwork["century"]).strip()
                    if pd.notna(artwork["century"])
                    and str(artwork["century"]).strip()
                    else "Date unknown"
                )

                image_data, content_type = load_image_from_url(
                    artwork["primary_image"]
                )

                if image_data is not None:
                    artwork_id = int(artwork["object_id"])

                    artwork_url = (
                        f"?gallery=women"
                        f"&source=women"
                        f"&artwork={artwork_id}"
                    )

                    display_clickable_image(
                        image_data=image_data,
                        content_type=content_type,
                        target_url=artwork_url,
                        alt_text=title,
                        height=300
                    )
                else:
                    st.warning("Image could not be loaded.")

                metadata_text = dedent(
                    f"""
                    <div style="
                        text-align: center;
                        min-height: 120px;
                        padding-top: 0.6rem;
                    ">
                        <p style="
                            font-weight: 600;
                            margin-top: 0;
                            margin-bottom: 0.25rem;
                        ">{title}</p>
                        <p style="
                            color: #7a5260;
                            font-size: 0.85rem;
                            margin-top: 0;
                            margin-bottom: 0.15rem;
                        ">{artist}</p>
                        <p style="
                            color: #98727f;
                            font-size: 0.8rem;
                            margin-top: 0;
                            margin-bottom: 0;
                        ">{century}</p>
                    </div>
                    """
                )

                st.markdown(
                    metadata_text,
                    unsafe_allow_html=True
                )

        st.write("")

if st.session_state.page == "home":
    show_home_page()

elif st.session_state.page == "collection":
    show_collection_page()

elif st.session_state.page == "artwork":
    show_artwork_page()



elif st.session_state.page == "women":
    show_women_page()
    
    
    