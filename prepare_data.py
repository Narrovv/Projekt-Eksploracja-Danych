import pandas as pd
import json

print("Reading MusicBrainz area data...")
# Dictionary to map area IDs to their names for quick lookup
area_dict = {}

with open(r'C:\Users\kaczm\Downloads\area\mbdump\area', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            # Saving area ID and name for later use
            area_dict[data['id']] = data.get('name', 'Unknown')
        except json.JSONDecodeError:
            continue

print("Reading MusicBrainz artist data...")
artist_country_data = []
seen_artists = set() # Set for quick duplicate checking

with open(r'C:\Users\kaczm\Downloads\artist\mbdump\artist', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            artist_name = data.get('name', '')
            if not artist_name:
                continue
            
            # Standardization of the name for matching with Kaggle
            match_name = str(artist_name).lower().strip()
            
            # Ignore duplicates
            if match_name in seen_artists:
                continue
            country_name = 'Unknown'
            area = data.get('area')
            
            if isinstance(area, dict):
                country_name = area.get('name', 'Unknown')
                # If the area name is 'Unknown', try to get it from the area_dict using the ID
                if country_name == 'Unknown' and 'id' in area:
                    country_name = area_dict.get(area['id'], 'Unknown')
            elif isinstance(area, str):
                country_name = area_dict.get(area, 'Unknown')
                
            artist_country_data.append({
                'match_artist': match_name,
                'country_name': country_name
            })
            seen_artists.add(match_name)
            
        except json.JSONDecodeError:
            continue

print("Converting dictionaries to DataFrame...")
artist_country = pd.DataFrame(artist_country_data)

print("Reading original Kaggle dataset...")
kaggle_df = pd.read_csv('data/dataset.csv')

print("Preparing Kaggle dataset for merging...")
kaggle_df['match_artist'] = kaggle_df['artists'].astype(str).str.split(';').str[0].str.lower().str.strip()

print("Enriching Kaggle dataset with country information...")
enriched_df = pd.merge(kaggle_df, artist_country, on='match_artist', how='left')

enriched_df['country_name'] = enriched_df['country_name'].fillna('Unknown')
enriched_df = enriched_df.drop(columns=['match_artist'])

output_filename = 'data/dataset_with_country.csv'
print(f"Saving enriched dataset to: {output_filename}...")
enriched_df.to_csv(output_filename, index=False)

print(f"Done! Enriched dataset saved to: {output_filename}")