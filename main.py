from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq, APIStatusError
import httpx
import json
import os
import re

load_dotenv(override=True)

app = FastAPI(title="Discovery Mode API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_MODEL = "llama-3.3-70b-versatile"
VIRAL_50_INDIA_PLAYLIST_ID = "37i9dQZEVXbMWDif5SCBJq"
TRENDING_FALLBACK_BLURB = "Trending right now"

DEEP_SYSTEM_PROMPT = (
    "You are a music discovery expert helping users break their listening loop. "
    "Return exactly 5 lesser-known song recommendations as a JSON array. "
    'Each item: {"artist_name": "...", "track_name": "...", "reason": "..."}. '
    "Reason: max 15 words, punchy like a friend recommending — not a critic. "
    "Do not recommend mainstream hits from the user's favorite artists. "
    "Consider music from all markets including Indian Bollywood, Hindi indie, regional Indian music, "
    "and other non-Western genres when relevant to the user's mood or input. "
    "Don't default to only Western artists."
)

QUICK_SYSTEM_PROMPT = (
    "You are a music discovery expert. Return exactly 10 song recommendations as a JSON array "
    "ordered from most familiar/safe (first) to most adventurous/unknown (last). "
    'Each item: {"artist_name": "...", "track_name": "...", "reason": "..."}. '
    "Reason: max 15 words, punchy and specific like a friend recommending. "
    "Consider music from all markets including Indian Bollywood, Hindi indie, regional Indian music, "
    "and other non-Western genres when relevant to the user's mood or input. "
    "Don't default to only Western artists."
)

TRENDING_SYSTEM_PROMPT = (
    "You write viral music blurbs. Return ONLY a JSON array of exactly 15 strings — "
    "one blurb per track in the same order given. Max 10 words each. No other text. "
    'Example: ["This heartbreak anthem hit different after that viral Reel."]'
)


def extract_json_text(content: str) -> str:
    text = content.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class DeepDiscoverRequest(BaseModel):
    prompt: str
    artists: str
    mood: str
    adventure_level: int = Field(ge=1, le=5)


class QuickDiscoverRequest(BaseModel):
    mood: str
    seed_artist: str | None = None


async def get_spotify_token() -> str:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
        )
        return response.json()["access_token"]


def calculate_novelty_scores(adventure_level: int) -> list[int]:
    min_score = 40 + (adventure_level - 1) * 10
    max_score = 55 + (adventure_level - 1) * 10
    return [round(min_score + i * (max_score - min_score) / 4) for i in range(5)]


def calculate_quick_novelty_scores() -> list[int]:
    return [round(30 + i * 60 / 9) for i in range(10)]


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def truncate_reason(reason: str) -> str:
    return truncate_words(reason, 15)


def truncate_blurb(blurb: str) -> str:
    return truncate_words(blurb, 10)


def parse_groq_json(content: str) -> list:
    text = extract_json_text(content)
    data = json.loads(text)
    if isinstance(data, dict) and "recommendations" in data:
        data = data["recommendations"]
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Expected non-empty JSON array")
    return data


def groq_chat(system_prompt: str, user_message: str, max_tokens: int = 600) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "paste_your_groq_key_here":
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured. Add your key to backend/.env",
        )

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.8,
            max_tokens=max_tokens,
        )
    except APIStatusError as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    return response.choices[0].message.content


def call_groq(
    user_message: str,
    system_prompt: str = DEEP_SYSTEM_PROMPT,
    max_tokens: int = 600,
) -> list:
    try:
        content = groq_chat(system_prompt, user_message, max_tokens)
    except APIStatusError as e:
        if e.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Discovery is taking a break — please try again in a moment",
            )
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    return parse_groq_json(content)


def parse_groq_blurbs(content: str, count: int) -> list[str]:
    text = extract_json_text(content)
    data = json.loads(text)
    if isinstance(data, dict) and "blurbs" in data:
        data = data["blurbs"]
    if not isinstance(data, list):
        raise ValueError("Expected JSON array of blurbs")

    blurbs: list[str] = []
    for item in data[:count]:
        if isinstance(item, str):
            blurbs.append(truncate_blurb(item))
        elif isinstance(item, dict):
            blurbs.append(truncate_blurb(item.get("blurb", item.get("reason", ""))))
    return blurbs


async def search_spotify_track(
    token: str, artist: str, track: str
) -> dict[str, str | None]:
    null_result = {
        "spotify_url": None,
        "preview_url": None,
        "album_image": None,
        "track_id": None,
    }

    queries = [f"track:{track} artist:{artist}", f"{track} {artist}"]
    async with httpx.AsyncClient() as client:
        for query in queries:
            response = await client.get(
                "https://api.spotify.com/v1/search",
                params={"q": query, "type": "track", "limit": 1},
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code != 200:
                continue
            items = response.json().get("tracks", {}).get("items", [])
            if not items:
                continue
            item = items[0]
            images = item.get("album", {}).get("images", [])
            return {
                "spotify_url": item.get("external_urls", {}).get("spotify"),
                "preview_url": item.get("preview_url"),
                "album_image": images[0]["url"] if images else None,
                "track_id": item.get("id"),
            }

    return null_result


async def fetch_trending_tracks_search_fallback(
    token: str, limit: int = 15
) -> list[dict]:
    """Fallback when playlist API is restricted — uses Search with market=IN."""
    seen_ids: set[str] = set()
    tracks: list[dict] = []
    queries = ["viral india", "bollywood hits", "hindi trending"]

    async with httpx.AsyncClient() as client:
        for query in queries:
            if len(tracks) >= limit:
                break
            response = await client.get(
                "https://api.spotify.com/v1/search",
                params={
                    "q": query,
                    "type": "track",
                    "market": "IN",
                    "limit": min(limit - len(tracks), 10),
                },
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code != 200:
                continue
            for item in response.json().get("tracks", {}).get("items", []):
                track_id = item.get("id")
                if not track_id or track_id in seen_ids:
                    continue
                seen_ids.add(track_id)
                artists = item.get("artists", [])
                images = item.get("album", {}).get("images", [])
                tracks.append(
                    {
                        "artist": artists[0]["name"] if artists else "",
                        "track": item.get("name", ""),
                        "spotify_url": item.get("external_urls", {}).get("spotify"),
                        "preview_url": item.get("preview_url"),
                        "album_image": images[0]["url"] if images else None,
                        "track_id": track_id,
                    }
                )
                if len(tracks) >= limit:
                    break

    if not tracks:
        raise HTTPException(
            status_code=502,
            detail="Failed to fetch trending tracks from Spotify. Please try again later.",
        )
    return tracks


async def fetch_playlist_tracks(token: str, playlist_id: str, limit: int = 15) -> list[dict]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
            params={
                "limit": limit,
                "market": "IN",
                "fields": "items(track(id,name,artists,external_urls,preview_url,album(images)))",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        if response.status_code != 200:
            return await fetch_trending_tracks_search_fallback(token, limit)

    tracks = []
    for item in response.json().get("items", []):
        track = item.get("track")
        if not track or not track.get("id"):
            continue
        artists = track.get("artists", [])
        images = track.get("album", {}).get("images", [])
        tracks.append(
            {
                "artist": artists[0]["name"] if artists else "",
                "track": track.get("name", ""),
                "spotify_url": track.get("external_urls", {}).get("spotify"),
                "preview_url": track.get("preview_url"),
                "album_image": images[0]["url"] if images else None,
                "track_id": track.get("id"),
            }
        )
        if len(tracks) >= limit:
            break

    if not tracks:
        return await fetch_trending_tracks_search_fallback(token, limit)

    return tracks


async def build_recommendations(
    groq_results: list, token: str, novelty_scores: list[int]
) -> list[dict]:
    recommendations = []
    for i, rec in enumerate(groq_results):
        artist = rec.get("artist_name", "")
        track = rec.get("track_name", "")
        reason = truncate_reason(rec.get("reason", ""))
        spotify_data = await search_spotify_track(token, artist, track)
        score = novelty_scores[i] if i < len(novelty_scores) else novelty_scores[-1]
        recommendations.append(
            {
                "artist": artist,
                "track": track,
                "reason": reason,
                **spotify_data,
                "novelty_score": score,
            }
        )
    return recommendations


async def fetch_groq_recommendations(
    user_message: str, system_prompt: str, max_tokens: int = 600
) -> list:
    for attempt in range(2):
        try:
            return call_groq(user_message, system_prompt, max_tokens)
        except HTTPException:
            raise
        except (json.JSONDecodeError, ValueError):
            if attempt == 1:
                raise HTTPException(
                    status_code=502,
                    detail="Failed to parse recommendations from AI. Please try again.",
                )
    return []


@app.get("/health")
async def health():
    token = await get_spotify_token()
    return {"status": "ok", "spotify": "connected", "token_preview": token[:20]}


@app.post("/recommend/deep")
async def recommend_deep(request: DeepDiscoverRequest):
    user_message = (
        f"Prompt: {request.prompt}\n"
        f"Favorite artists: {request.artists}\n"
        f"Mood: {request.mood}\n"
        f"Adventure level: {request.adventure_level}/5"
    )

    groq_results = await fetch_groq_recommendations(user_message, DEEP_SYSTEM_PROMPT)
    token = await get_spotify_token()
    novelty_scores = calculate_novelty_scores(request.adventure_level)
    recommendations = await build_recommendations(groq_results[:5], token, novelty_scores)
    return {"recommendations": recommendations}


@app.post("/recommend/quick")
async def recommend_quick(request: QuickDiscoverRequest):
    user_message = f"Mood: {request.mood}"
    if request.seed_artist:
        user_message += f"\nSeed artist: {request.seed_artist}"

    groq_results = await fetch_groq_recommendations(
        user_message, QUICK_SYSTEM_PROMPT, max_tokens=900
    )
    token = await get_spotify_token()
    novelty_scores = calculate_quick_novelty_scores()
    recommendations = await build_recommendations(groq_results[:10], token, novelty_scores)
    return {"recommendations": recommendations}


@app.get("/trending")
async def trending():
    token = await get_spotify_token()
    tracks = await fetch_playlist_tracks(token, VIRAL_50_INDIA_PLAYLIST_ID, limit=15)

    track_list = "\n".join(
        f"{i + 1}. {t['artist']} - {t['track']}" for i, t in enumerate(tracks)
    )
    user_message = f"Write a trending blurb for each track:\n{track_list}"

    blurbs = [TRENDING_FALLBACK_BLURB] * len(tracks)
    try:
        content = groq_chat(TRENDING_SYSTEM_PROMPT, user_message, max_tokens=400)
        parsed = parse_groq_blurbs(content, len(tracks))
        for i, blurb in enumerate(parsed):
            blurbs[i] = blurb
    except APIStatusError as e:
        if e.status_code != 429:
            raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    except (json.JSONDecodeError, ValueError):
        pass

    result = []
    for i, track in enumerate(tracks):
        result.append({**track, "blurb": blurbs[i] if i < len(blurbs) else TRENDING_FALLBACK_BLURB})

    return {"tracks": result}


@app.get("/search/artist")
async def search_artist_tracks(artist: str, limit: int = 3):
    token = await get_spotify_token()
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.spotify.com/v1/search",
            params={
                "q": f"artist:{artist}",
                "type": "track",
                "market": "IN",
                "limit": min(limit, 10),
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Spotify search failed: {response.text[:200]}",
            )

    tracks = []
    for item in response.json().get("tracks", {}).get("items", []):
        artists = item.get("artists", [])
        images = item.get("album", {}).get("images", [])
        tracks.append(
            {
                "artist": artists[0]["name"] if artists else artist,
                "track": item.get("name", ""),
                "spotify_url": item.get("external_urls", {}).get("spotify"),
                "preview_url": item.get("preview_url"),
                "album_image": images[0]["url"] if images else None,
                "track_id": item.get("id"),
            }
        )
    return {"tracks": tracks}
