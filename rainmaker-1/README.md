# Rainmaker v0.1

A first working version of the personal VN stock dashboard from the design doc.
It's a real app with real market data — but it implements a simplified subset
of the full knowledge base, not everything in the design doc yet. See "What's
implemented" below for the honest list.

## Getting it online (no Terminal needed)

Real-time stock data can't be fetched directly from a page on your phone or
laptop — the data providers don't allow that from a browser. It has to run
on a small server somewhere. The good news: getting one for free takes about
10 minutes of clicking, no command line at all.

**Step 1 — Put the code on GitHub (free account, just clicking)**
1. Go to [github.com](https://github.com) and sign up (free).
2. Click the **+** in the top right → **New repository**. Name it `rainmaker`, keep it Public, click **Create repository**.
3. On the new repo's page, click **uploading an existing file**.
4. Unzip the file I sent you, then drag the whole unzipped `rainmaker` folder's contents (not the outer folder itself — the files and folders *inside* it: `app.py`, `engine.py`, `static/`, etc.) into the upload box.
5. Scroll down, click **Commit changes**.

**Step 2 — Deploy it on Render (free, no credit card)**
1. Go to [render.com](https://render.com) and sign up (you can use your GitHub account to sign up — one click).
2. Click **New +** → **Web Service**.
3. Connect your GitHub account if asked, then select the `rainmaker` repo.
4. Render should auto-detect the settings from the `render.yaml` file included in the code. If it asks you to fill them in manually instead:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
5. Click **Create Web Service**. It'll take a few minutes to build the first time.
6. When it's done, Render gives you a URL like `https://rainmaker-xxxx.onrender.com` — that's your app, reachable from anywhere: iPhone, laptop, cellular data, doesn't matter.

**On your iPhone**: open that URL in Safari, tap Share → **Add to Home Screen**. You get a Rainmaker icon that opens full-screen like a real app.

**One free-tier trade-off worth knowing**: Render's free plan "sleeps" the app after 15 minutes with no visitors, then takes ~30-50 seconds to wake up on the next visit. For a personal dashboard you check a few times a day, this is barely noticeable — just expect a short pause if it's been idle a while. If you ever want it always-instant and reliably firing the 10:30/14:30 alerts even while asleep, see "Keeping it always-on" below.

**Data persistence note**: on Render's free tier, anything saved to disk (your portfolio positions) resets if you ever redeploy the code with an update. It survives normal sleep/wake, just not a redeploy. Use the **"Back up my portfolio data"** link at the bottom of the app occasionally — it downloads a small JSON file. If you ever need to restore after a redeploy, send me that file and I'll load it back in via the `/api/restore` endpoint.

## Keeping it always-on and alerts firing reliably (optional, free)

Because the free tier sleeps, the built-in 10:30/14:30 daily check might get
missed if nothing has visited the app recently. Fix it with a free "uptime
pinger" — also all clicking, no Terminal:

1. Go to [cron-job.org](https://cron-job.org) and make a free account.
2. Create two cron jobs that just visit your app's `/api/refresh` address —
   e.g. `https://rainmaker-xxxx.onrender.com/api/refresh` — scheduled for
   10:30 and 14:30 Vietnam time on weekdays.
3. That's it — those two visits both wake the app up *and* trigger the real
   check, so alerts fire reliably even if you haven't opened the app.

## Getting real push notifications on your iPhone (optional, free)

1. Install the free **ntfy** app from the App Store.
2. In the app, subscribe to a topic name you make up — something private and
   hard to guess, e.g. `richard-rainmaker-9f2k`.
3. On Render, open your service → **Environment** tab → add an environment
   variable... actually simpler: open `config.json` in your GitHub repo,
   click the pencil (edit) icon, set:
   ```json
   { "ntfy_topic": "richard-rainmaker-9f2k" }
   ```
   and commit — Render redeploys automatically with the change.

Now the 10:30/14:30 checks (and manual "Refresh" taps that find something
worth flagging) will push straight to your phone.

## What's implemented in v0.1

- **Real-time-ish price data** for any VN ticker, via the free `vnstock`
  library (a few minutes of caching, not tick-by-tick — this is a personal
  dashboard, not a trading terminal).
- **Portfolio tracking**: add a ticker, your buy price, and share count;
  Rainmaker tracks P/L and re-analyzes it automatically.
- **Plain-English recommendations** for each holding: buy more / hold / trim
  / sell / watch, each with a one-line reason and buy/stop/target numbers.
- **A simplified engine** covering: trend (moving averages), momentum
  (RSI/MACD), a breakout-with-volume check, and a basic "quiet buying vs.
  quiet selling" read from trading volume (On-Balance Volume). Foreign
  net-buy/sell is shown when the free data source actually provides it for
  that ticker — never guessed if it's not available.
- **A watchlist** of tickers you don't hold, screened from a fixed list of
  ~16 liquid VN30-style names, surfaced when something looks worth a look.
- **Portfolio vs. VN-Index**: the headline number at the top — are you
  beating the market or not.
- **Two scheduled checks a day** (10:30 and 14:30 Vietnam time, weekdays),
  logged in the Alerts panel, plus an optional push to your phone.
- **A "Refresh" button** for testing anytime, not just at the scheduled times.
- **A backup/restore endpoint** so your portfolio survives a redeploy.

## What's NOT implemented yet (from the fuller design doc)

- Support/resistance and chart-pattern recognition are simplified (20-day
  swing high/low only — not confirmed multi-touch levels or named patterns).
- No daily automated news research yet (macro/micro headlines) — the context
  strip is currently static, not pulled from real news.
- No block/put-through deal detection, no VN price-band awareness (ceiling/
  floor caps), no T+2.5 settlement tracking yet.
- The watchlist universe is a fixed list, not a full market scan.
- No fundamental analysis (valuation, financial statements).
- The self-check pass is basic (confidence downgrades on thin data, no
  fabricated numbers) — not yet the full contradiction-detection pass from
  the design doc.

All of the above are straightforward to add next — this version exists so
you can start actually using it and tell me what matters most to build out
first.

## Running it locally instead (if you ever get comfortable with Terminal)

```
cd rainmaker
./start.sh
```
Then open the address it prints on any device on the same WiFi. Free,
no account needed, but only works while your Mac is awake and running it.

## Files

- `app.py` — the web server and API
- `engine.py` — the analysis rules (indicators + recommendations)
- `data_source.py` — talks to vnstock, with caching and graceful failure
- `store.py` — saves your portfolio/watchlist/alerts to simple JSON files in `data/`
- `static/index.html` — the dashboard you see in the browser
- `config.json` — put your ntfy topic here for push notifications
- `render.yaml` — tells Render how to build and run the app automatically
