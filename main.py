import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

# ==========================================
# 1. API CONFIGURATION
# ==========================================
SPOTIFY_CLIENT_ID = ''
SPOTIFY_CLIENT_SECRET = ''
SPOTIFY_REDIRECT_URI = ''

# ==========================================
# 2. DOWNLOADING DATA
# ==========================================
def get_user_spotify_data(kaggle_df, max_tracks_to_fetch=2000, target_playlists=None):
    print("Connecting to Spotify API...")
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope="playlist-read-private playlist-read-collaborative" 
    ))

    print("Downloading your playlists...")
    playlists = sp.current_user_playlists()
    playlist_tracks = []
    
    for playlist in playlists['items']:
        if not playlist or not playlist.get('id'):
            continue
            
        playlist_name = playlist.get('name', 'Nieznana')
        
        # FILTERING BY TARGET PLAYLISTS (if specified)
        if target_playlists and playlist_name not in target_playlists:
            continue


        print(f"\n -> Downloading from: {playlist_name}")
        
        try:
            results = sp.playlist_tracks(playlist['id'])
            tracks = results['items']
            
            while results['next'] and len(playlist_tracks) < max_tracks_to_fetch:
                results = sp.next(results)
                tracks.extend(results['items'])
                
            added_from_this_playlist = 0
            for item in tracks:
                if not isinstance(item, dict):
                    continue
                    
                track = item.get('track') or item.get('item')
                
                if track is None or not isinstance(track, dict):
                    continue
                    
                track_name = track.get('name')
                artists = track.get('artists', [])
                
                if track_name and len(artists) > 0:
                    artist_name = artists[0].get('name')
                    if artist_name:
                        playlist_tracks.append({
                            'match_name': str(track_name).lower().strip(),
                            'match_artist': str(artist_name).lower().strip(),
                            'Liked': 1
                        })
                        added_from_this_playlist += 1
                        
            print(f"    (Extracted: {added_from_this_playlist} valid tracks)")
            
        except spotipy.exceptions.SpotifyException:
            print(f"    [!] Omited '{playlist_name}' (no access / Spotify playlist).")
            continue
            
        if len(playlist_tracks) >= max_tracks_to_fetch:
            print("\nReached global limit of tracks to fetch.")
            break

    # REMOVING DUPLICATES
    liked_df = pd.DataFrame(playlist_tracks, columns=['match_name', 'match_artist', 'Liked'])
    
    if liked_df.empty:
        print("\n[!] ERROR: No tracks downloaded. Check the playlist name!")
        return liked_df
        
    liked_df = liked_df.drop_duplicates(subset=['match_name', 'match_artist'])
    print(f"\nDownloaded {len(liked_df)} unique tracks from all your playlists.")

    # PREPARING KAGGLE DATA FOR MATCHING
    kaggle_df['match_name'] = kaggle_df['track_name'].astype(str).str.lower().str.strip()
    kaggle_df['match_artist'] = kaggle_df['artists'].astype(str).str.split(';').str[0].str.lower().str.strip()

    # JOINING USER LIKES WITH KAGGLE DATABASE
    matched_likes = pd.merge(liked_df, kaggle_df, on=['match_name', 'match_artist'], how='inner')
    matched_likes = matched_likes.drop_duplicates(subset=['match_name', 'match_artist'])
    
    n_matched = len(matched_likes)
    print(f" Managed to match {n_matched} songs with the Kaggle database!")

    if n_matched < 10:
        print("Warning: You have too few matched songs to train the classification models.")
        return matched_likes

    #GENERATING BACKGROUND (NEGATIVE EXAMPLES)
    print("Generating background (negative examples)...")
    unliked_pool = kaggle_df[~kaggle_df['track_id'].isin(matched_likes['track_id'])]
    negatives_df = unliked_pool.sample(n=n_matched, random_state=42).copy()
    negatives_df['Liked'] = 0

    # JOINING LIKES AND NEGATIVES
    final_user_data = pd.concat([matched_likes, negatives_df], ignore_index=True)
    return final_user_data


# ==========================================
# 3. MAIN PROCESS
# ==========================================
print("Downloading local Kaggle dataset...")
kaggle_df = pd.read_csv('data/dataset.csv') 
df = get_user_spotify_data(kaggle_df, max_tracks_to_fetch=2000, target_playlists=[])

if len(df) > 20:
    num_features = ['duration_ms', 'danceability', 'energy', 'loudness', 
                    'speechiness', 'acousticness', 'instrumentalness', 
                    'liveness', 'valence', 'tempo']
    
    for col in num_features:
        df[col] = df[col].fillna(df[col].median())
        
    df['explicit'] = df['explicit'].apply(lambda x: 1 if x is True else 0)

    X = df[num_features + ['explicit']]
    y = df['Liked']
    X.columns = X.columns.astype(str)

    print("\nTraining models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #MODEL 1 (RANDOM FOREST)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_roc = roc_auc_score(y_test, rf_proba)

    #CROSS-VALIDATION
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    cv_roc_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')

    print(f"\n--- FINAL RESULTS ---")
    print(f"Dataset: {len(X)} tracks (50/50 zeros and ones). Used {X.shape[1]} audio features.")
    print(f"1. RANDOM FOREST (Test Set) -> Accuracy: {rf_acc:.2f} | ROC AUC: {rf_roc:.2f}")
    print(f"2. LOGISTIC REGRESSION (Cross-Validation) -> Average ROC AUC: {cv_roc_scores.mean():.2f}")

    # ==========================================
    # 4. RECOMMENDATION FUNCTION 
    # ==========================================
    def recommend_songs(model, all_songs_df, user_history_df, top_n=10):
        print("\nGenerating personalized recommendations...")
        
        known_tracks = user_history_df[user_history_df['Liked'] == 1]['match_name'].tolist()
        candidates = all_songs_df[~all_songs_df['match_name'].isin(known_tracks)].copy()

        num_features = ['duration_ms', 'danceability', 'energy', 'loudness', 
                        'speechiness', 'acousticness', 'instrumentalness', 
                        'liveness', 'valence', 'tempo']
        
        for col in num_features:
            candidates[col] = candidates[col].fillna(candidates[col].median())
            
        candidates['explicit'] = candidates['explicit'].apply(lambda x: 1 if x is True else 0)

        X_candidates = candidates[num_features + ['explicit']]
        X_candidates.columns = X_candidates.columns.astype(str)

        probabilities = model.predict_proba(X_candidates)[:, 1]
        candidates['Match_Probability'] = probabilities

        recommendations = candidates.sort_values(by='Match_Probability', ascending=False)
        recommendations = recommendations.drop_duplicates(subset=['track_name', 'artists'])

        return recommendations[['track_name', 'artists', 'Match_Probability']].head(top_n)

    #Running Recommendation Function
    top_10 = recommend_songs(rf_model, kaggle_df, df, top_n=10)

    print("\nYour Top 10 Recommendations:")
    print(top_10.to_string(index=False))

else:
    print("\nNot enough data from API to train the ML model.")