import pandas as pd
import requests
import time

print("Wczytywanie bazy danych...")
df = pd.read_csv('goodreads_library_export.csv')

def clean_isbn(isbn):
    if pd.isna(isbn):
        return None
    cleaned = ''.join(c for c in str(isbn) if c.isdigit() or c.upper() == 'X')
    return cleaned if len(cleaned) in [10, 13] else None

df['Clean_ISBN'] = df['ISBN13'].apply(clean_isbn)
df['Clean_ISBN'] = df['Clean_ISBN'].fillna(df['ISBN'].apply(clean_isbn))

def get_data_from_openlibrary(isbn):
    if not isbn:
        return 'Unknown', None
    
    # Przedstawiamy się serwerowi, żeby uniknąć blokady (Rate Limit)
    headers = {'User-Agent': 'BookRecommendationProject/1.0 (Educational)'}
    
    genres = 'Unknown'
    rating = None
    
    # 1. Zapytanie o gatunki (Books API)
    books_url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    try:
        res_books = requests.get(books_url, headers=headers, timeout=5)
        if res_books.status_code == 200:
            data = res_books.json()
            book_key = f"ISBN:{isbn}"
            if book_key in data and 'subjects' in data[book_key]:
                subjects = [sub['name'] for sub in data[book_key]['subjects'][:3]]
                genres = ', '.join(subjects)
    except Exception:
        pass

    # 2. Zapytanie o OCENY (Search API - ograniczone tylko do pobrania oceny)
    search_url = f"https://openlibrary.org/search.json?q=isbn:{isbn}&fields=ratings_average"
    try:
        res_search = requests.get(search_url, headers=headers, timeout=5)
        if res_search.status_code == 200:
            data = res_search.json()
            if data.get('numFound', 0) > 0:
                rating = data['docs'][0].get('ratings_average', None)
    except Exception:
        pass
        
    return genres, rating

print(f"Rozpoczynam podwójne pobieranie metadanych (Gatunki + Oceny) dla {len(df)} książek...")
genres_list = []
ratings_list = []

for index, isbn in enumerate(df['Clean_ISBN']):
    if index % 50 == 0 and index > 0:
        print(f"Przetworzono {index} książek...")
        
    genres, rating = get_data_from_openlibrary(isbn)
    genres_list.append(genres)
    ratings_list.append(rating)
    time.sleep(0.5) 

df['OpenLibrary_Genre'] = genres_list
df['OpenLibrary_Rating'] = ratings_list

print("\nPobieranie zakończone! Próbka pierwszych wyników:")
print(df[['Title', 'Clean_ISBN', 'OpenLibrary_Genre', 'OpenLibrary_Rating']].head(10))

df.to_csv('goodreads_with_genres.csv', index=False)
print("\nZapisano nowy plik: goodreads_with_genres.csv")