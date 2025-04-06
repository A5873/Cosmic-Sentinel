<div align="center">
  <img src="assets/images/logo.svg" alt="Cosmic Sentinel Logo" width="500">
  <h1>Cosmic Sentinel</h1>
  <p>
    <img src="https://img.shields.io/badge/Status-Beta-blue?style=flat-square" alt="Beta Status">
    <img src="https://img.shields.io/badge/Version-1.0.0--beta-orange?style=flat-square" alt="Version 1.0.0-beta">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License">
  </p>
</div>

A comprehensive space observation and monitoring platform for astronomers and space enthusiasts.

[Project Plan & Roadmap](PROJECT.md) | [Features](#features) | [Installation](#installation) | [Usage](#usage) | [Story](#project-story)

## Why This Software Matters

Cosmic Sentinel isn't just a space tracker - it's a powerful AI-enhanced observatory at your fingertips.
Whether you're a hobbyist looking at the night sky or a researcher analyzing space data, this app brings the universe to your desktop.

> Check out our detailed [Project Plan](PROJECT.md) for current status, roadmap, and future enhancements.

## Features

- **Planetary Tracking**: Track and visualize planetary positions and movements
- **Asteroid Monitoring**: Monitor near-Earth objects and potentially hazardous asteroids
- **Space Weather Monitor**: Track solar activity, geomagnetic conditions, and aurora forecasts
- **Reporting**: Generate detailed reports and visualizations

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/cosmic-sentinel.git
   cd cosmic-sentinel
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Create an account at [NASA API Portal](https://api.nasa.gov/) to get an API key
   - Either add your API key to `config.ini` or enter it in the application settings

## Configuration

The application can be configured through:

1. The `config.ini` file in the root directory
2. The application settings dialog
3. Command-line arguments

### Space Weather Monitor Configuration

Space weather monitoring can be configured with the following settings:

```ini
[space_weather]
monitoring_interval = 3600        # Update interval in seconds
auto_start_monitoring = false     # Start monitoring automatically
cache_expiration = 1              # Cache expiration in hours
solar_flare_alerts = true         # Enable solar flare alerts
cme_alerts = true                 # Enable CME alerts
geomagnetic_storm_alerts = true   # Enable geomagnetic storm alerts
aurora_alerts = true              # Enable aurora alerts
minimum_alert_severity = MODERATE # Minimum severity for alerts
```

## Usage

Run the application:
```
python main.py
```

## Project Story

### Why I Created This Project

I have been an amateur astronomer since childhood. I still remember the amazement I felt when first viewing the night sky.
As a software developer, I naturally became interested in combining my love of astronomy with my computer programming skills.

In 2020, I started to learn about formulas for calculating positions of the Moon and planets. I discovered many wonderful resources, including:
- Paul Schlyter's lucid and educational page [How to compute planetary positions](http://www.stjarnhimlen.se/comp/ppcomp.html)
- [Practical Astronomy with your Calculator](https://www.amazon.com/Practical-Astronomy-Calculator-Peter-Duffett-Smith/dp/0521356997), third edition, by Peter Duffett-Smith, Cambridge University Press
- [Astronomy on the Personal Computer](https://www.amazon.com/Astronomy-Personal-Computer-Oliver-Montenbruck/dp/3540672214/) by Oliver Montenbruck and Thomas Pfleger

## Technical Details

I implemented algorithms based on these resources. Over time, however, I noticed that they were not quite
as accurate as I would like. Their calculated positions differed from those reported by online tools
like [JPL Horizons](https://ssd.jpl.nasa.gov/horizons.cgi) and [Heavens Above](https://www.heavens-above.com/)
by large fractions of a degree in many cases.

In 2019 I renewed my interest in astronomy calculations, with the goal of creating something more accurate
that could be written in JavaScript to run inside a browser. I studied how professional
astronomers and space agencies did their calculations. First I looked at the United States Naval Observatory's
[NOVAS C 3.1](https://github.com/indigo-astronomy/novas) library. I quickly realized it could not be
ported to the browser environment, because it required very large (hundreds of megabytes)
precomputed ephemeris files.

This led in turn to studying the French *Bureau des Longitudes* model known as
[VSOP87](https://en.wikipedia.org/wiki/VSOP_(planets)). It requires more computation
but the data is much smaller, consisting of trigonometric power series coefficients.
However, it was still too large to fit in a practical web page.

Furthermore, these models were extremely complicated, and far more accurate than what I needed.
NOVAS, for example, performs relativistic calculations to correct for the bending
of light through the gravitational fields of planets, and time dilation due to different
non-inertial frames of reference! My humble needs did not require this herculean level
of complexity. So I decided to create Cosmic Sentinel with the following engineering goals:

- Support JavaScript, C, C#, and Python with the same algorithms, and verify them to produce identical results
- It would be well documented, relatively easy to use, and support a wide variety of common use cases

### Implementation Approach

The solution I settled on was to truncate the VSOP87 series to make it as small
as possible without exceeding the 1 arcminute error threshold.
I created a code generator that converts the truncated tables into C, C#, JavaScript,
and Python source code. Then I built unit tests that compare the calculations
against the NOVAS C 3.1 code operating on the DE405 ephemeris and other authoritative
sources, including the JPL Horizons tool. Basing on VSOP87 and verifying
against independent trusted sources provides extra confidence that everything is correct.

Pluto was a special case, because VSOP87 does not include a model for it. I ended up writing
a custom gravitation simulator for the major outer planets to model Pluto's orbit.
The results are verified against NOVAS and the model
[TOP2013](https://www.aanda.org/articles/aa/abs/2013/09/aa21843-13/aa21843-13.html).

## Support and Contribution

I am committed to maintaining this project for the long term, and I am happy to
answer questions about how to solve various astronomy calculation problems
using Cosmic Sentinel. Feel free to reach out on the
[discussions page](https://github.com/A5873/Cosmic-Sentinel/discussions) or
[submit a new issue](https://github.com/A5873/Cosmic-Sentinel/issues).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <img src="assets/images/icon.svg" alt="Cosmic Sentinel Icon" width="60">
  <p>Cosmic Sentinel &copy; 2025</p>
</div>
