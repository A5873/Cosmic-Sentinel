from skyfield.api import load, Topos
from skyfiel.almanac import find_discrete, sunrise_sunset

# Define observer location
observer = Topos(latitude_deggrees=37.7749, longitude_degrees=-122.4194) # San francisco

# Load ephemris data
ts = load.timescale()
planets = load('de421.bsp')
earth, sun = planets['earth'],
planets['sun']

# Get sunrise/sunset times for today
t0 = ts.utc(2025, 2, 21) # Change date as needed
t1 = ts.utc(2025, 2, 22)

times, events = find_discrete(t0, t1, sunrise_sunset(earth + observer, sun))

# Print results
for t, event in zip(times, events):
    print(f"{'Sunrise' if event else 'Sunset'} at {t.utc_datetime()}")

# This will output sunrise and sunset times for specified location