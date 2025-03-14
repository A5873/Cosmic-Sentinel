from skyfield.api import load, Topos

# Load satellite TLE data
satellites = load.tle_file('https://celestrak.com/NORAD/elements/stations.txt') # or otherwise
iss = {sat.name: sat for sat in satellites}['ISS (ZARYA)']

# Define observer location (e.g., New York)
observer = Topos(latitude_degrees=40.7128, longitude_degrees=-74.0060)

# Get current time
ts = load.timescale()
t = ts.now()

# Compute position relative to observer
difference = iss - observer
topocentric = difference.at(t)
alt, az, d = topocentric.altaz()

print(f"ISS Altitude: {alt}, Azimuth: {az}, Distance: {d.km} km")

# This tells you where to look for the ISS in the night sky