"""Quick live facts without a back agent: a web search (or the weather service) answered by the front brain in the
same turn. A back agent takes a round of thinking and a tool call per question (half a minute on a busy server); a
search takes one to three seconds."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime

import httpx

WEATHER = re.compile(r"\b(weather|forecast|temperature|raining|snowing|how (hot|cold|warm))\b", re.I)
PLACE = re.compile(r"\b(?:in|at|for)\s+([A-Z][\w.'-]*(?:[\s,]+[A-Z][\w.'-]*){0,3})")


def place(text: str) -> str:
    m = PLACE.search(text)
    return m.group(1).strip(" ,.?!") if m else ""


async def weather(text: str, client: httpx.AsyncClient) -> str | None:
    """Current conditions and today's range from wttr.in (no key); where the question names no place, the server's."""

    r = await client.get(f"https://wttr.in/{place(text).replace(' ', '+')}", params={"format": "j1"})
    if r.status_code != 200:
        return None
    j = r.json()
    now, day = j["current_condition"][0], j["weather"][0]
    area = j.get("nearest_area", [{}])[0]
    where = ", ".join(x[0]["value"] for x in (area.get("areaName"), area.get("region")) if x)
    return (f"Weather now in {where}: {now['weatherDesc'][0]['value']}, {now['temp_F']}°F ({now['temp_C']}°C), feels like "
            f"{now['FeelsLikeF']}°F, wind {now['windspeedMiles']} mph, humidity {now['humidity']}%. Today: high "
            f"{day['maxtempF']}°F, low {day['mintempF']}°F, chance of rain {max(int(h['chanceofrain']) for h in day['hourly'])}%.")


def _search(query: str) -> list[dict]:
    from ddgs import DDGS

    return DDGS(timeout=4).text(query, max_results=6)


async def search(query: str) -> str | None:
    rows = await asyncio.to_thread(_search, query)
    return "\n".join(f"- {r['title']}: {r['body']}" for r in rows) or None


async def look_up(text: str, client: httpx.AsyncClient, timeout: float = 5) -> str | None:
    """What the web says now, as context for the front brain, or None (it then goes to the back agent)."""

    try:
        found = await asyncio.wait_for(weather(text, client) if WEATHER.search(text) else search(text), timeout)
    except Exception:
        return None
    if not found:
        return None
    return (f"[Live web lookup, {datetime.now().strftime('%A %d %B %Y, %H:%M')}]\n{found}\n\n[The user asked]\n{text}\n"
            f"[Answer from the lookup in one or two short spoken sentences; trust it over your own memory.]")
