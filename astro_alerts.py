import time
from datetime import datetime
import matplotlib.pyplot as plt
from plyer import notification
from skyfield.api import load, Topos
from skyfield.almanac import find_discrete, sunrise_sunset

# Load planetary and satellite data
ts = load.timescale()
planets = load('de421.bsp')
earth, mars, sun = planets['earth'],
planets['mars'], planets['sun']

# Observer location (example: New York)
