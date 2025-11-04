import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

language_popularity = {
    'English': 35.62, 'Simplified Chinese': 26.03, 'Russian': 9.09, 'Spanish - Spain': 4.11,
    'Portuguese - Brazil': 3.77, 'Korean': 3.59, 'German': 2.83, 'Japanese': 2.82,
    'French': 2.29, 'Polish': 1.6, 'Traditional Chinese': 1.37, 'Turkish': 1.19,
    'Thai': 0.94, 'Ukrainian': 0.69, 'Spanish - Latin America': 0.65, 'Italian': 0.6,
    'Czech': 0.54, 'Hungarian': 0.37, 'Portuguese - Portugal': 0.34, 'Dutch': 0.26,
    'Swedish': 0.26, 'Danish': 0.22, 'Vietnamese': 0.2, 'Finnish': 0.15, 'Indonesian': 0.13,
    'Norwegian': 0.12, 'Romanian': 0.11, 'Greek': 0.06, 'Bulgarian': 0.04, 'Arabic': 0.00
}
multiplayer_keywords = ['Multiplayer', 'MMO', 'Co-op', 'Online PvP', 'Online Co-Op', 'LAN Co-Op']

@st.cache_resource
def load_models_and_components():
    success_model = joblib.load('success_model.pkl')
    longevity_model = joblib.load('longevity_model.pkl')
    le_success = joblib.load('success_label_encoder.pkl')
    le_longevity = joblib.load('longevity_label_encoder.pkl')
    feature_list = joblib.load('final_feature_list.pkl')
    
    df = pd.read_csv('steam_data_final.csv')

    commercial_quantile_boundaries = [0, 0.20, 0.40, 0.60, 0.80, 0.95, 1.0]
    commercial_tier_labels = ['Niche or Flop', 'Modest Success', 'Solid Performer', 'Notable Hit', 'Major Hit', 'Blockbuster']
    df['SuccessTier'] = pd.qcut(df['CommercialSuccessScore'], q=commercial_quantile_boundaries, labels=commercial_tier_labels, duplicates='drop')

    longevity_quantile_boundaries = [0, 0.40, 0.75, 0.95, 1.0]
    longevity_tier_labels = ['Fades Quickly', 'Average Lifespan', 'High Retention', 'Evergreen']
    df['LongevityTier'] = pd.qcut(df['LongevityScore'], q=longevity_quantile_boundaries, labels=longevity_tier_labels, duplicates='drop')
    
    tags_exploded = df['Tags'].dropna().str.split(', ').explode()
    tag_counts = tags_exploded.value_counts()
    total_games = len(df)
    idf_weights = np.log(total_games / tag_counts)
    
    success_explainer = shap.TreeExplainer(success_model)
    longevity_explainer = shap.TreeExplainer(longevity_model)
    
    return success_model, longevity_model, le_success, le_longevity, feature_list, df, commercial_tier_labels, longevity_tier_labels, idf_weights.to_dict(), success_explainer, longevity_explainer

success_model, longevity_model, le_success, le_longevity, feature_list, df, commercial_tier_labels, longevity_tier_labels, idf_weights, success_explainer, longevity_explainer = load_models_and_components()

def calculate_language_score(selected_languages, popularity_data):
    total_score = 0
    matched_keys = set()
    for lang in selected_languages:
        for key, value in popularity_data.items():
            if lang.lower() in key.lower() and key not in matched_keys:
                total_score += value; matched_keys.add(key)
    return total_score

def find_similar_games_ranking(df, selected_tags, selected_genre_tags, predicted_success_tier, idf_weights, GENRE_TAGS):
    genre_match_mask = df['Tags'].apply(lambda x: len(set(str(x).split(', ')) & set(selected_genre_tags)) > 0 if pd.notna(x) else False)
    similar_games = df[genre_match_mask].copy()

    def calculate_hierarchical_similarity(game_tags_str, user_tags_set, user_genre_tags_set):
        if pd.isna(game_tags_str): return 0
        game_tags = set(str(game_tags_str).split(', '))
        
        shared_genres = game_tags.intersection(user_genre_tags_set)
        if not user_genre_tags_set:
            genre_score = 0.5
        else:
            genre_score = len(shared_genres) / len(user_genre_tags_set)
        
        user_other_tags = user_tags_set - user_genre_tags_set
        game_other_tags = game_tags - set(GENRE_TAGS)
        shared_other = game_other_tags.intersection(user_other_tags)
        
        if not shared_other:
            other_tags_score = 0
        else:
            intersection_weight = sum(idf_weights.get(tag, 0) for tag in shared_other)
            user_tags_weight = sum(idf_weights.get(tag, 0) for tag in user_other_tags)
            other_tags_score = intersection_weight / user_tags_weight if user_tags_weight > 0 else 0
            
        total_score = (0.7 * genre_score) + (0.3 * other_tags_score)
        return total_score

    similar_games['similarity_score'] = similar_games['Tags'].apply(
        calculate_hierarchical_similarity, args=(set(selected_tags), set(selected_genre_tags))
    )
    
    ranked_games = similar_games[similar_games['similarity_score'] > 0.7].copy()
    if ranked_games.empty or len(ranked_games) < 7: return None, None, 0, 0
    
    ranked_games.sort_values(by=['CommercialSuccessScore', 'similarity_score'], ascending=False, inplace=True)
    ranked_games.reset_index(drop=True, inplace=True)
    
    user_game_score = df.groupby('SuccessTier', observed=True)['CommercialSuccessScore'].min().get(predicted_success_tier, 0) + 1e-6
    insert_idx = ranked_games[ranked_games['CommercialSuccessScore'] < user_game_score].index
    user_rank = insert_idx[0] if len(insert_idx) > 0 else len(ranked_games)

    top_3 = ranked_games.head(3)[['Name', 'SuccessTier']]
    bottom_3 = ranked_games.tail(3)[['Name', 'SuccessTier']] if len(ranked_games) > 6 else pd.DataFrame()
    user_game_row = pd.DataFrame([{'Name': '🚀 YOUR GAME', 'SuccessTier': predicted_success_tier}], index=[f"#{user_rank + 1}"])
    top_3.index = [f"#{i+1}" for i in top_3.index]
    
    separator = pd.DataFrame([{'Name': '...', 'SuccessTier': '...'}], index=['...'])
    if not bottom_3.empty:
        n = len(ranked_games)
        total_ranked_count = n + 1
        bottom_3.index = [f"#{total_ranked_count-2}", f"#{total_ranked_count-1}", f"#{total_ranked_count}"]
        final_table = pd.concat([top_3, separator, user_game_row, separator, bottom_3])
    else:
        final_table = pd.concat([top_3, separator, user_game_row])

    return final_table, ranked_games, user_rank + 1, len(ranked_games) + 1

def get_tag_recommendations_with_impact(shap_df):
    potential_tags = shap_df[
        (shap_df['feature'].str.startswith('Tag_')) & 
        (shap_df['feature_value'] == 0) &
        (shap_df['feature'] != 'Tag_Free to Play')
    ].copy()
    
    potential_tags = potential_tags.sort_values('shap_value', ascending=False)
    top_3_tags = potential_tags.head(3)
    
    recommendations = []
    for _, row in top_3_tags.iterrows():
        tag_name = row['feature'].replace('Tag_', '')
        impact = row['shap_value']
        if impact > 0:
            recommendations.append((tag_name, impact))
    return recommendations

def generate_pricing_insights(similar_games_df):
    if similar_games_df is None or similar_games_df.empty: return

    st.subheader("💰 Pricing Insights")
    
    f2p_competitors = similar_games_df[similar_games_df['Price (USD)'] == 0]
    paid_competitors = similar_games_df[similar_games_df['Price (USD)'] > 0]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Price Distribution of Paid Competitors**")
        if not paid_competitors.empty:
            price_bins = [0.01, 5, 10, 15, 20, 30, 40, 60, 201]
            price_labels = [f"<$5", "$5-10", "$10-15", "$15-20", "$20-30", "$30-40", "$40-60", "$60+"]
            binned_prices = pd.cut(paid_competitors['Price (USD)'], bins=price_bins, labels=price_labels, right=False).value_counts().sort_index()
            st.bar_chart(binned_prices)
        else:
            st.info("No paid competitors found to analyze pricing.")

    with col2:
        st.metric("Average Price", f"${paid_competitors['Price (USD)'].mean():.2f}" if not paid_competitors.empty else "N/A")
        st.metric("Free to Play Competitors", f"{len(f2p_competitors)}")

tag_columns = [col for col in feature_list if col.startswith('Tag_')]
all_tag_names = sorted([tag.replace('Tag_', '') for tag in tag_columns])

GENRE_TAGS = ['Action', 'Adventure', 'Strategy', 'RPG', 'Indie', 'Simulation', 'Casual', 'Racing']
PLAYER_MODE_TAGS = ['Singleplayer', 'Multiplayer', 'Co-op', 'Online Co-Op', 'Early Access', 'PvP']
THEME_STYLE_TAGS = ['Pixel Graphics', 'Anime', 'Story Rich', 'Atmospheric', 'Great Soundtrack', 'Horror', 'Sci-fi', 'Fantasy']
EXCLUDED_TAGS = set(GENRE_TAGS + PLAYER_MODE_TAGS + THEME_STYLE_TAGS + ['Free to Play'])
OTHER_TAGS = [tag for tag in all_tag_names if tag not in EXCLUDED_TAGS]

st.set_page_config(layout="wide", page_title="GameSuccess Predictor")
st.title('GameSuccess Predictor 🎮')
st.markdown("An Intelligent Advisor for Indie Game Developers. Enter your game's pre-launch features to predict its potential.")

with st.sidebar:
    st.header("Enter Your Game's Features:")
    st.subheader("Business Model")
    is_f2p = st.checkbox("✅ Free to Play")
    price = st.number_input('Price (USD)', min_value=0.0, max_value=200.0, value=0.0 if is_f2p else 19.99, step=1.0, disabled=is_f2p)
    st.subheader("Technical Features")
    controller_support = st.checkbox('Controller Support', value=True)
    steam_deck_support = st.checkbox('Steam Deck Support', value=True)
    st.subheader("Game Tags")
    with st.expander("Genre Tags", expanded=True):
        selected_genre_tags = st.multiselect('Select Genre(s)', options=GENRE_TAGS, default=['Indie', 'Action'])
    with st.expander("Player Modes & Features", expanded=True):
        selected_player_mode_tags = st.multiselect('Select Player Modes', options=PLAYER_MODE_TAGS, default=['Singleplayer'])
    with st.expander("Theme & Style Tags"):
        selected_theme_tags = st.multiselect('Select Theme/Style', options=THEME_STYLE_TAGS)
    with st.expander("Other Tags"):
        selected_other_tags = st.multiselect('Select Other Tags', options=OTHER_TAGS)
    selected_tags = selected_genre_tags + selected_player_mode_tags + selected_theme_tags + selected_other_tags
    st.subheader("Localization")
    language_names = sorted(list(language_popularity.keys()))
    selected_languages = st.sidebar.multiselect('Select Supported Languages', options=language_names, default=['English', 'German', 'French'])

if st.button('🚀 Predict Success', use_container_width=True):
    
    if is_f2p and 'Free to Play' not in selected_tags:
        selected_tags.append('Free to Play')
        
    input_data = {'Price (USD)': 0.0 if is_f2p else price, 'Controller Support': 1 if controller_support else 0, 'Steam Deck Support': 1 if steam_deck_support else 0}
    input_data['IsMultiplayer'] = 1 if any(keyword in selected_tags for keyword in multiplayer_keywords) else 0
    input_data['LanguageMarketScore'] = calculate_language_score(selected_languages, language_popularity)
    for tag in all_tag_names:
        input_data[f'Tag_{tag}'] = 1 if tag in selected_tags else 0
    
    final_input_df = pd.DataFrame([input_data])[feature_list]

    success_probs = success_model.predict_proba(final_input_df)
    predicted_success_tier = le_success.inverse_transform(np.argmax(success_probs, axis=1))[0]
    
    st.header("📈 Prediction Results")
    st.balloons()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Commercial Success")
        success_probs_df = pd.DataFrame({'Tier': le_success.classes_, 'Probability': success_probs[0]})
        success_probs_df['Tier'] = pd.Categorical(success_probs_df['Tier'], categories=commercial_tier_labels, ordered=True)
        st.bar_chart(success_probs_df.sort_values('Tier').set_index('Tier'))
        st.metric("Most Likely Outcome", predicted_success_tier)
    with col2:
        st.subheader("Game Longevity")
        longevity_probs = longevity_model.predict_proba(final_input_df)
        longevity_probs_df = pd.DataFrame({'Tier': le_longevity.classes_, 'Probability': longevity_probs[0]})
        longevity_probs_df['Tier'] = pd.Categorical(longevity_probs_df['Tier'], categories=longevity_tier_labels, ordered=True)
        st.bar_chart(longevity_probs_df.sort_values('Tier').set_index('Tier'))
        st.metric("Most Likely Outcome", le_longevity.inverse_transform(np.argmax(longevity_probs, axis=1))[0])
    with col3:
        st.subheader("Market Reach")
        st.metric("Language Market Fit", f"{input_data['LanguageMarketScore']:.2f}%")
        st.info("The percentage of the Steam player base your game can reach.")

    st.markdown("---")
    st.header("💡 Recommendations & Insights")
    predicted_tier_index = np.argmax(success_probs, axis=1)[0]
    st.subheader(f"What's Driving the '{predicted_success_tier}' Prediction?")
    try:
        shap_values_for_class = shap.TreeExplainer(success_model)(final_input_df).values[0, :, predicted_tier_index]
        shap_df = pd.DataFrame({'feature': final_input_df.columns, 'shap_value': shap_values_for_class, 'feature_value': final_input_df.iloc[0].values})
        
        st.markdown("##### Impact of Your Selected Tags")
        impactful_tags = shap_df[(shap_df['feature'].str.startswith('Tag_')) & (shap_df['feature_value'] == 1)].copy()
        impactful_tags['feature'] = impactful_tags['feature'].str.replace('Tag_', '')
        impactful_tags.sort_values('shap_value', ascending=False, inplace=True)
        st.bar_chart(impactful_tags.set_index('feature'), y='shap_value')
        st.info("The chart above shows how each tag you selected influenced the final prediction.")
        st.subheader("Actionable Suggestions")
        tag_recs = get_tag_recommendations_with_impact(shap_df)
        if tag_recs:
            st.markdown("**Consider Adding These Tags for a Potential Boost:**")
            rec_list = "".join([f"- **{tag}**: Predicted to boost your score (Impact: {impact:+.2f})\n" for tag, impact in tag_recs])
            st.markdown(rec_list)
        else:
            st.markdown("No strong positive tags to recommend at this time.")

    except Exception as e:
        st.warning(f"Could not generate SHAP analysis. Error: {e}")

    st.markdown("---")
    st.header("📊 Market Analysis")
    display_table, raw_similar_df, user_rank, total_ranked = find_similar_games_ranking(df, selected_tags, selected_genre_tags, predicted_success_tier, idf_weights, GENRE_TAGS)
    
    st.subheader("Competitive Benchmark")
    if display_table is not None:
        st.write(f"Your game is predicted to rank approximately **#{user_rank}** out of **{total_ranked}** similar games on Steam.")
        st.dataframe(display_table, use_container_width=True)
    else:
        st.warning("Could not find enough similar games in the dataset to create a competitive benchmark.")
    
    st.markdown("<br>", unsafe_allow_html=True) 
    if raw_similar_df is not None:
        generate_pricing_insights(raw_similar_df)