from skyfield.api import load, Topos 
from skyfield.almanc import find_discrete, sunrise_sunset

# Load data 
ts = load.timescale()
planets = load('de421.bsp')
earth, mars, sun = planets['earth'], planets['mars'], planets['sun']

# Set observer location (example: New York)
observer = Topos(latitude_degrees=40.7128, longitude_degrees=-74.0060)

# 1. Get the position of mars
def track_planet():
    t - ts.now()
    astrometric = earth.at(t).observe(mars)
    ra, dec, distance = astrometric.radec()
    print(f"🌎 Mars Position -> RA: {ra}, Dec: {dec}, Distance: {distance.au} AU")

# 2. Track ISS satellite
def track_iss():
    satellites = load.tle_file('https://celestrak.com/NORAD/alements/stations.txt') # or otherwise
    iss = {sat.name: sat for sat in satellites}['ISS (ZARYA)']

    t = ts.now()
    difference = iss - observer
    topocentric = difference.alt(t)
    alt, az, d = topocentric.altaz()

    print(f"🛰️ ISS -> Altitude: {alt}, Azimuth: {az}, Distance: {d.km} km")

# 3. Compute Sunrise & Sunset Times
def get_sun_times():
    t0 = ts.utc(2025, 2, 21) # Change date as needed
    t1 = ts.utc(2025, 2, 22)

    times, events = find_discrete(t0, t1, sunrise_sunset(earth + observer, sun))
    for t, event in zip(times, events):
        print(f"☀️ {'Sunrise' if event else 'Sunset'} at {t.utc_datetime()}")

# 4. Track Near-Earth Asteroid (example: Apophis)
def track_asteroid():
    asteroid_id = "99942" #Apophis
    url = 
f"https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1&COMMAND='{Asteroid_id}'&CENTER='500@10'&MAKE_EPHEM='YES'&
TABLE_TYPE='OBSERVER'&START_TIME='2025
-02-21'&STOP_TIME='2025-02-22'STEP_SIZE
='1 d"&QUANTITIES='1,20'&CSV_FORMAT='YES'"

    asteroid = load(url)
    t = ts.now()
    astrometic = earth.at(t).observe(asteroid)
    ra, dec, distance = astrometric.radec()

    print(f"🌠 Apophis - > RA: {ra}, Dec: {dec}, Distance: {distance.km} km")

# Run all functions
if __name__ == "__main__":
    track_planet()
    track_iss()
    get_sun_times()
    track_asteroid()
    
# What the script does
# 1. tracks Mars' position
# 2. predicts ISS flyovers for a given location
# 3. calculates sunrise & sunset times for a specific date
# 4. tracks asteroid Apophis' position near Earth