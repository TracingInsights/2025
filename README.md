<p align="center">
  <a href="https://tracinginsights.com/"><img src="Banner.png" alt="TracingInsights.com"></a>
</p>

<p align="center">
    <em>2025 Public F1 telemetry files - TracingInsights.com</em>
</p>

<p align="center">
<a href="https://github.com/sponsors/TracingInsights" target="_blank">
    <img src="https://img.shields.io/github/sponsors/TracingInsights?style=for-the-badge&logo=github" alt="GitHub Sponsors">
</a>
<a href="https://www.youtube.com/@TracingInsights" target="_blank">
    <img src="https://img.shields.io/youtube/channel/subscribers/UCehTBMnvLYkCbneNuC8d4_A?style=for-the-badge&logo=youtube" alt="YouTube Channel Subscribers">
</a>
<a href="https://bsky.app/profile/tracinginsights.com" target="_blank">
    <img src="https://img.shields.io/bluesky/followers/tracinginsights.com?style=for-the-badge&logo=bluesky" alt="Bluesky followers">
</a>
<a href="https://mas.to/@tracinginsights" target="_blank">
    <img src="https://img.shields.io/mastodon/follow/109801192184550885?domain=mas.to&style=for-the-badge&logo=mastodon" alt="Mastodon Follow">
</a>
<a href="https://www.reddit.com/r/formulaone/" target="_blank">
    <img src="https://img.shields.io/reddit/subreddit-subscribers/formulaone?style=for-the-badge&logo=reddit" alt="F1 Subreddit subscribers">
</a>
<a href="https://www.reddit.com/r/TracingInsights/" target="_blank">
    <img src="https://img.shields.io/reddit/subreddit-subscribers/TracingInsights?style=for-the-badge&logo=reddit" alt="TracingInsights Subreddit subscribers">
</a>
<a href="https://x.com/TracingInsights" target="_blank">
    <img src="https://img.shields.io/twitter/follow/TracingInsights?style=for-the-badge&logo=x" alt="X (formerly Twitter) Follow">
</a>
<a href="https://www.instagram.com/tracinginsights/" target="_blank">
    <img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram" alt="Instagram Follow">
</a>
<a href="https://www.threads.net/@tracinginsights" target="_blank">
    <img src="https://img.shields.io/badge/Threads-000000?style=for-the-badge&logo=threads" alt="Threads Follow">
</a>
<a href="https://www.facebook.com/tracinginsights" target="_blank">
    <img src="https://img.shields.io/badge/Facebook-3D82ED?style=for-the-badge&logo=facebook" alt="Facebook Follow">
</a>
<a href="https://whatsapp.com/channel/0029VaBAOXA1HspzlVJg2Z08" target="_blank">
    <img src="https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp" alt="WhatsApp Follow">
</a>
<a href="https://t.me/tracinginsights" target="_blank">
    <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram" alt="Telegram Follow">
</a>
<a href="https://www.tumblr.com/tracinginsights" target="_blank">
    <img src="https://img.shields.io/badge/Tumblr-36465D?style=for-the-badge&logo=tumblr" alt="Tumblr Follow">
</a>
<a href="https://tracinginsights.substack.com/" target="_blank">
    <img src="https://img.shields.io/badge/Substack-FF6719?style=for-the-badge&logo=substack" alt="Substack Follow">
</a>
<a href="https://www.linkedin.com/company/tracinginsights" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin" alt="LinkedIn Follow">
</a>
<a href="https://doi.org/10.5281/zenodo.17312802"><img src="https://zenodo.org/badge/927761260.svg" alt="DOI"></a>
</p>


## Analyze/Visualize the data at [f1tel.com](https://f1tel.com/) or [TracingInsights.com](https://tracinginsights.com/)

Ask AI about this repo/data

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TracingInsights-Archive/2025)

## Data For Other Seasons

<h2 align="center">

[2025](https://github.com/TracingInsights/2025) | [2024](https://github.com/TracingInsights/2024) | [2023](https://github.com/TracingInsights/2023) | [2022](https://github.com/TracingInsights/2022) | [2021](https://github.com/TracingInsights/2021) | [2020](https://github.com/TracingInsights/2020) | [2019](https://github.com/TracingInsights/2019) | [2018](https://github.com/TracingInsights/2018) | [Historical Data](https://github.com/TracingInsights-Archive/stats) | [Pitstops](https://github.com/TracingInsights-Archive/PitStops) | [Download Data in CSV](https://tracinginsights.com/analysis/download-raw-data/) | [Data Page](https://tracinginsights.com/data)

</h2>

## Circuit Corners & Sections (`corners.json`)

Every session folder has a `corners.json` describing the circuit layout as parallel column lists. Distances are resolved against the session's fastest lap.

- **Corners:** `CornerNumber`, `CornerLetter`, `CornerLabel` (e.g. `1`, `1A`), `X`, `Y`, `Angle`, `Distance` (meters from the start/finish line), `MVDistance` (MultiViewer's own fixed distance), `Rotation`.
- **Sections:** `Section` gives each corner's speed regime (`"Straight"`, `"Low"`, `"Medium"` or `"High"`). The bands themselves are in `SectionType`, `SectionStartPercent`, `SectionEndPercent`, `SectionStartDistance` and `SectionEndDistance`, one entry per band. They come from the curated `sections/<circuit>.json` files (percent of lap distance). The Chinese, Japanese and Las Vegas Grands Prix have no breakdown yet, so their band lists are empty and `Section` is `null`.
- **Marshal lights / sectors:** `MarshalLight*` and `MarshalSector*` (`Number`, `X`, `Y`, `Angle`, `Distance`, `MVDistance`).
- **Circuit:** `CircuitKey`, `CircuitName`, `Location`, `CountryName`, `CountryKey`, `CountryIocCode`, and pit-loss estimates `PitLossNormal`, `PitLossSC`, `PitLossVSC`.
- **Layout source:** `LayoutYear`, `LayoutRound`, `LayoutRaceDate`, `LayoutMeeting*`, `LayoutLap*`. MultiViewer doesn't publish every season, so the newest available layout is used, and these fields say which one.
- **Track outline:** `TrackX` and `TrackY` (plot one against the other to draw the track map), plus `TrackTime` and `MiniSectorIndex`.

Missing values are `null`. List fields that MultiViewer doesn't provide for a circuit are empty arrays.

## Driver & Car Images

Official Formula 1 images live under `images/`. Paths are built from the `team` and `driver` fields of `drivers.json`:

| Folder | Contents | Source |
|---|---|---|
| `images/drivers/<team>/<driver>.webp` | Full-body driver cutout (transparent WebP, 1229 × 3840) | [formula1.com/en/drivers](https://www.formula1.com/en/drivers) |
| `images/headshots/<team>/<driver>.png` | 840 × 840 headshot: the source asset behind the thumbnail that the `url` field in `drivers.json` links to | `media.formula1.com` |
| `images/cars/<team>.webp` | Car side view (transparent WebP) | [formula1.com/en/teams](https://www.formula1.com/en/teams) |

Every file is the original asset from `media.formula1.com`, not resized or re-encoded. Images are keyed by driver **and** team because pairings changed mid-season. Yuki Tsunoda and Liam Lawson swapped seats after two rounds, so the folder has both `Racing Bulls/TSU.webp` and `Red Bull Racing/TSU.webp`, and the same pair for `LAW`. `images/manifest.json` lists every file with the driver's name, pixel size and source URL. Reserve and test drivers exist on the host only as 128 × 128 images, so that is the size saved for them. Headshots exist only for drivers with a `url` in `drivers.json`.

```bash
curl -L -o VER.webp "https://raw.githubusercontent.com/TracingInsights/2025/main/images/drivers/Red%20Bull%20Racing/VER.webp"
```

To refresh the images, run `python3 download_images.py`. Add `--force` to re-download everything, or `--only cars` for a single set.

## Team Radio

Team radio clips published by [OpenF1](https://openf1.org) are saved as MP3 files in each driver's folder. Each session also has an index:

| Path | Contents |
|---|---|
| `<event>/<session>/<driver>/radio/<file>.mp3` | Original team radio recording from the F1 live timing archive |
| `<event>/<session>/team_radio.json` | Every clip of the session in time order: driver, driver number, UTC date, local file path and source URL |

Only clips that F1 broadcast are available, so most sessions have between a handful and a few dozen messages. `LapTimes.py` downloads the radios for each session it processes. To backfill the whole season, run `python3 team_radio.py`. Files that already exist are skipped.

## Fetch via CDN (jsDelivr)

Every push to this repository creates a GitHub release with a semver tag (e.g. `2026.288.1`, made from `<year>.<day-of-year>.<build>` of the release date). You can fetch any file through jsDelivr, pinned to that release:

```text
https://cdn.jsdelivr.net/gh/TracingInsights/2025@2026.288.1/Australian%20Grand%20Prix/Race/corners.json
```

For the latest data, look up the current release tag first. Both requests are fast and CORS-enabled:

```javascript
const release = await fetch(
  "https://api.github.com/repos/TracingInsights/2025/releases/latest"
).then(r => r.json());

const url = `https://cdn.jsdelivr.net/gh/TracingInsights/2025@${release.tag_name}/Australian%20Grand%20Prix/Race/corners.json`;
```

```bash
TAG=$(curl -s https://api.github.com/repos/TracingInsights/2025/releases/latest | jq -r .tag_name)
curl -L "https://cdn.jsdelivr.net/gh/TracingInsights/2025@${TAG}/Australian%20Grand%20Prix/Race/corners.json"
```

If you don't need a pinned release, you can use a branch directly (cached for 12 hours):

```text
https://cdn.jsdelivr.net/gh/TracingInsights/2025@main/Australian%20Grand%20Prix/Race/corners.json
```

> **Note:** jsDelivr serves individual files up to 20 MB. The latest release tag is also available from jsDelivr's own API:
> `https://data.jsdelivr.com/v1/packages/gh/TracingInsights/2025/resolved?specifier=latest`

## Frequently Asked Questions

Data availability timeline:

*   Practice sessions: Approximately 30 minutes after session completion
*   Qualifying sessions: Approximately 30 minutes after session completion
*   Sprint races: Approximately 30 minutes after race completion
*   Grand Prix races: Approximately 30 minutes after race completion

All telemetry data is processed and made available as quickly as possible after each session concludes.

Tip: For race weekends, check back shortly after each session to access the latest data. Historical data from previous seasons is available immediately.

You can analyze comprehensive telemetry data including:

*   Speed (km/h)
*   Throttle input percentage
*   Brake application
*   Gear selection
*   DRS activation zones
*   Lateral acceleration (G-forces)
*   Longitudinal acceleration
*   Track elevation changes
*   Engine RPM
*   Vertical acceleration

Fuel Correction is a feature that adjusts lap times to account for the changing weight of fuel in Formula 1 cars during a race. As cars burn fuel, they become progressively lighter, resulting in naturally faster lap times even without any improvement in driving or track conditions.

#### Why Fuel Correction matters:

*   **Weight impact:** Each kilogram of fuel adds approximately 0.03 seconds per lap to a car's time
*   **Race strategy analysis:** Helps distinguish between genuine pace improvements and those simply due to fuel load reduction
*   **Fair comparison:** Enables meaningful comparison between laps from different stages of a race
*   **Performance evaluation:** Provides a clearer picture of true car and driver performance throughout a race

#### How Fuel Correction is calculated:

Corrected Lap Time = Original Lap Time - (Fuel Weight Effect)

Where:

Fuel Weight Effect = Remaining Fuel (kg) × 0.03 seconds

Remaining Fuel = Initial Fuel × (1 - Current Lap / Total Race Laps)

The calculation uses these standard assumptions:

*   **Initial fuel load:**
    *   Full Race: 100kg 
    *   Sprint Race: 30kg

*   **Fuel consumption:** Linear usage throughout the race (equal amount burned each lap)
*   **Performance impact:** 0.03 seconds per kilogram of fuel per lap (industry standard approximation)

#### How to interpret Fuel-Corrected lap times:

*   **Early race laps:** Will show faster corrected times than their actual times (compensating for heavy fuel load)
*   **Mid-race laps:** Will show moderate correction
*   **Late race laps:** Will show minimal correction (as cars are already light on fuel)
*   **Comparing drivers:** Reveals true pace differences independent of fuel load advantages
*   **Stint analysis:** Helps identify genuine tire degradation separate from the fuel weight effect

#### When to use Fuel Correction:

*   **Race pace analysis:** When comparing laps from different stages of a race
*   **Strategy evaluation:** When analyzing the effectiveness of different pit stop strategies
*   **Driver performance:** When assessing a driver's consistency throughout a race
*   **Team comparisons:** When comparing the race pace of different teams


**Note:** Fuel correction is an approximation based on standard assumptions. Actual fuel loads and consumption rates may vary between teams, cars, and race conditions. The correction provides a useful analytical tool but should be considered alongside other performance metrics.

The Elevation Chart is a critical tool for verifying data integrity and alignment between different drivers' telemetry. Since track elevation is a fixed geographical feature, all drivers should show nearly identical elevation profiles regardless of their performance.

#### Why the Elevation Chart matters for data integrity:

*   **Reference alignment:** Track elevation is a constant physical property that all cars experience identically
*   **Data synchronization check:** Misaligned elevation profiles immediately indicate synchronization issues
*   **Error detection:** Helps identify corrupted or incorrectly processed telemetry data
*   **Baseline verification:** Establishes confidence in the reliability of other telemetry channels

#### How to use the Elevation Chart for data validation:

*   **Visual inspection:** Compare elevation profiles between multiple drivers - they should closely overlap
*   **Pattern matching:** Look for consistent peaks and valleys across all selected drivers
*   **Offset detection:** Check for horizontal shifts that indicate distance measurement errors
*   **Anomaly identification:** Identify unusual spikes or drops that may indicate sensor errors

#### Common data integrity issues revealed by the Elevation Chart:

*   **Horizontal misalignment:** Indicates distance measurement or synchronization errors
*   **Vertical scaling differences:** May indicate drivers going over kerbs/bumps
*   **Missing segments:** Reveals data gaps or transmission failures during recording
*   **Noise or jitter:** Can indicate sensor interference or data processing problems
*   **Complete mismatch:** Suggests incorrect lap selection or major data corruption

**Pro tip:** Always check the Elevation Chart first before analyzing other telemetry channels. If elevation profiles don't align properly, it's a strong indication that other telemetry comparisons (speed, throttle, braking) may be unreliable for that particular dataset.

**Note:** Minor variations in elevation data between drivers are normal due to sensor precision and sampling rates. However, significant differences that alter the overall profile shape indicate data integrity issues.

TracingInsights provides comprehensive Formula 1 data archives from 1950 to the present day, organized by season. Our data is collected from various sources including the Ergast/Jolpica API and F1 official feeds via FastF1.

#### Types of data available in our archives:

*   **Telemetry data:** Detailed car performance metrics including speed, throttle, brake, DRS usage, and acceleration
*   **Race results:** Complete race classifications, finishing positions, and points for all seasons
*   **Lap times:** Detailed timing information for every lap by each driver
*   **Pit stops:** Timing and duration of all pit stops during races
*   **Weather data:** Track and ambient conditions for different circuits and races
*   **Penalty points:** Current penalty point standings for each driver
*   **Historical statistics:** Complete race results database from 1950 onwards with driver and constructor standings

**Disclaimer:** This data is provided for free. TracingInsights is not affiliated with Formula 1, FIA, or any F1 team. All trademarks belong to their respective owners.

[Visit Data Archives Page](https://tracinginsights.com/data)

What kind of data is available in these repositories? 

Our repositories contain various types of Formula 1 data including:

*   Detailed telemetry data (speed, throttle, brake, DRS)
*   Acceleration data (lateral, longitudinal, vertical)
*   Technical data (RPM, gear, elevation)
*   Race results and qualifying times
*   Lap times and pit stop information
*   Driver and constructor standings

The data is organized by season and follows a consistent format for easy access and analysis.

 Can I use this data for my personal projects? 

Yes, you can use this data as you wish. We encourage creative use of the data for:

*   Data analysis and research
*   Visualization projects
*   Academic work
*   Personal applications

If you publish or share your work, attribution to TracingInsights is appreciated but not required.

 How often is the data updated? 

For current seasons, we typically update the data 30 minutes after each session of a grand prix. Historical seasons are complete and won't receive further updates unless corrections are needed.

 I found an error in the data. How can I report it? 

If you find any errors or inconsistencies in our data, you can:

*   Open an issue on the relevant GitHub repository
*   Contact us through our [contact page](https://tracinginsights.com/contact/)

We appreciate community contributions to improve data accuracy.

 Do I need an API key to access the data? 

No, you don't need an API key. All our data is freely available through the GitHub repositories. Simply:

1.   Clone or download the repository you're interested in
2.   Access the files directly from your local machine
3.   Or use GitHub's raw file URLs in your applications

This approach ensures the data remains accessible to everyone without authentication barriers.

Have a question that's not answered here? Check the README files in each repository for specific details about that season's data, or [contact us](https://tracinginsights.com/contact/) for additional help.

## Authors

- [@TracingInsights](https://github.com/TracingInsights)

## Support the project 

- [Report issues or Feature Requests](https://github.com/TracingInsights-Archive/2025/issues)
- [Sponsor](https://github.com/sponsors/TracingInsights)
- [Star the project on GitHub](https://github.com/TracingInsights-Archive/2025)

## Feedback

If you have any feedback, please reach out to us at https://tracinginsights.com/contact/

## Credits

 - [FastF1](https://github.com/theOehrly/Fast-F1)

 - [MultiViewer](https://github.com/f1multiviewer)

 - [Ergast/Jolpica](https://github.com/jolpica/jolpica-f1)

## License

[Apache-2.0 license](/LICENSE.md)


## Notice

TracingInsights and this website are unofficial and are not associated in any way with
the Formula 1 companies. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD
CHAMPIONSHIP, GRAND PRIX and related marks are trade marks of Formula One
Licensing B.V.