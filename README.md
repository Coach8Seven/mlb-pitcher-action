# MLB Pitcher Strikeout GPT Action Helper

This helper gives your MLB custom GPT reliable last 5 and last 10 pitcher starts with strikeouts and pitch counts.

## What It Does

The endpoint:

`GET /pitcher-last-starts`

Accepts:

- `pitcher_name`: pitcher name, such as `Paul Skenes`
- `season`: season year, such as `2026`
- `limit`: number of starts to return, usually `10`
- `line`: strikeout prop line, such as `5.5`

Returns:

- date
- opponent
- innings pitched
- pitches thrown
- strikeouts
- last 5 average strikeouts
- last 10 average strikeouts
- last 5 average pitches
- last 10 average pitches
- hit rate versus the supplied strikeout line

## Local Test

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
python app.py
```

Test:

```bash
curl "http://localhost:8000/pitcher-last-starts?pitcher_name=Paul%20Skenes&season=2026&limit=10&line=5.5"
```

## Deploy

Deploy this folder to a public HTTPS service such as Render, Railway, Fly.io, or another simple web host.

Start command:

```bash
gunicorn app:app
```

After deployment, copy your public URL and replace this line in `openapi.yaml`:

```yaml
servers:
  - url: https://YOUR-DEPLOYED-DOMAIN.example.com
```

## Add To Your Custom GPT

1. Edit your MLB custom GPT.
2. Open the Actions section.
3. Create a new action.
4. Authentication: `None` for private testing.
5. Paste the updated `openapi.yaml`.
6. Test `getPitcherLastStarts`.

## Instruction To Add To MLB GPT

Add this sentence to your MLB GPT instructions if you have room:

When analyzing pitcher strikeout props, call `getPitcherLastStarts` to retrieve last 5 and last 10 starts with pitch counts before final rankings.

