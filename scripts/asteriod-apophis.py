from skyfield.api import load

# Load small body ephemeris data
eph = load('de421.bsp')
earth = eph['earth']

# Load timescale
ts = load.timescale()
t = ts.utc(2025, 2, 21)

# Load asteroid orbit (example: Apophis, NASA ID 99942)
from skyfield.api import Topos
apophis = load('https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1&COMMAND='
"'99942'&CENTER='500@10'&MAKE_EPHEM='YES'&TABLE_TYPE='OBSERVER'" 
"&START_TIME='2025-02-21'&STOP_TIME='2025-02-22'STEP_SIZE='1 d'"
"&QUANTITIES='1,20'&CSV_FORMAT='YES'")

# Get asteroid position
astrometic = earth.at(t).observe(apophis)
ra, dec, distance = astrometic.radec()

print(f"Apophis Position: RA {ra}, Dec {dec}, Distance: {distance.km} km")

# This gives the position of asteroid Apophis, a near-earth-object(NEO
