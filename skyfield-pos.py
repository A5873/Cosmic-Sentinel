from skyfield.api import load

# load planetary data 
planets = load('de421.bsp')
earth, mars = planets['earth'],
planets['mars']

# get the current time
ts = load.timescale()
t = ts.now()

# Compute position
astrometric = earth.at(t).observer(mars)
ra, dec, distance = astrometric.radec()

print(f"RA: {ra}, Dec: {dec}, Distance: {distance} AU")

# This should give the right ascension(RA), declination(Dec) and distance of Mars 
# from earth