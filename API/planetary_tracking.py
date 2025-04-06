"""
Planetary Tracking Module

This module provides functionality for tracking planets and calculating their positions,
alignments, and distances using the skyfield library.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
from skyfield.api import Loader, Topos, load, wgs84
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield.data import planet_radii
from skyfield.positionlib import ICRF, Apparent
from skyfield.units import Distance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlanetaryTracker:
    """
    Class for managing planetary calculations including positions, alignments,
    and distances from Earth.
    
    This class uses the skyfield library to perform astronomical calculations
    with high precision.
    """
    
    # Planetary bodies to track by default
    DEFAULT_PLANETS = [
        'sun', 'mercury', 'venus', 'earth', 'mars', 
        'jupiter barycenter', 'saturn barycenter', 
        'uranus barycenter', 'neptune barycenter'
    ]
    
    # Planet display names mapping
    PLANET_NAMES = {
        'sun': 'Sun',
        'mercury': 'Mercury',
        'venus': 'Venus',
        'earth': 'Earth',
        'mars': 'Mars',
        'jupiter barycenter': 'Jupiter',
        'saturn barycenter': 'Saturn',
        'uranus barycenter': 'Uranus',
        'neptune barycenter': 'Neptune'
    }
    
    def __init__(self, 
                 data_dir: str = 'data', 
                 planets: Optional[List[str]] = None,
                 observer_lat: float = 0.0,
                 observer_lon: float = 0.0,
                 observer_elevation: float = 0.0) -> None:
        """
        Initialize the PlanetaryTracker with specified parameters.
        
        Args:
            data_dir: Directory to store skyfield data files
            planets: List of planets to track (if None, uses DEFAULT_PLANETS)
            observer_lat: Observer's latitude in degrees
            observer_lon: Observer's longitude in degrees
            observer_elevation: Observer's elevation in meters
        """
        self.data_dir = data_dir
        self.planets_to_track = planets or self.DEFAULT_PLANETS
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Set up observer location
        self.observer_lat = observer_lat
        self.observer_lon = observer_lon
        self.observer_elevation = observer_elevation
        
        # Initialize ephemeris data
        self.ephemeris = None
        self.planets = {}
        self.earth = None
        self.timescale = None
        self.observer = None
        
        # Load the ephemeris data
        try:
            self._load_ephemeris()
            logger.info("Successfully initialized PlanetaryTracker")
        except Exception as e:
            logger.error(f"Failed to initialize PlanetaryTracker: {str(e)}")
            raise
    
    def _load_ephemeris(self) -> None:
        """
        Load planetary ephemeris data from skyfield.
        
        This internal method loads the necessary ephemeris data files
        and sets up the planet objects for calculations.
        
        Raises:
            Exception: If ephemeris data cannot be loaded
        """
        try:
            # Create a loader that saves data to our data directory
            loader = Loader(self.data_dir)
            
            # Load the timescale for time conversions
            self.timescale = loader.timescale()
            
            # Load the planetary ephemeris
            self.ephemeris = loader('de421.bsp')
            
            # Load all the requested planets
            for planet_name in self.planets_to_track:
                try:
                    self.planets[planet_name] = self.ephemeris[planet_name]
                except KeyError:
                    logger.warning(f"Planet '{planet_name}' not found in ephemeris")
            
            # Earth is used as the default observation point
            self.earth = self.ephemeris['earth']
            
            # Set up observer on Earth if coordinates are provided
            if all([self.observer_lat, self.observer_lon]):
                self.observer = wgs84.latlon(
                    self.observer_lat, 
                    self.observer_lon, 
                    elevation_m=self.observer_elevation
                )
            
            logger.info(f"Loaded ephemeris data for {len(self.planets)} planets")
        
        except Exception as e:
            logger.error(f"Error loading ephemeris data: {str(e)}")
            raise
    
    def set_observer_location(self, 
                              latitude: float, 
                              longitude: float, 
                              elevation: float = 0.0) -> None:
        """
        Set the geographical location of the observer.
        
        Args:
            latitude: Observer's latitude in degrees
            longitude: Observer's longitude in degrees
            elevation: Observer's elevation in meters
        """
        try:
            self.observer_lat = latitude
            self.observer_lon = longitude
            self.observer_elevation = elevation
            
            self.observer = wgs84.latlon(
                latitude, 
                longitude, 
                elevation_m=elevation
            )
            
            logger.info(f"Observer location set to: Lat {latitude}, Lon {longitude}, Elev {elevation}m")
        except Exception as e:
            logger.error(f"Failed to set observer location: {str(e)}")
            raise
    
    def get_planet_positions(self, 
                            timestamp: Optional[datetime] = None,
                            planets: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Calculate the current positions of planets.
        
        Args:
            timestamp: Time for calculations (default: current time)
            planets: List of planets to calculate positions for
                    (default: all planets in self.planets)
                    
        Returns:
            Dictionary mapping planet names to position information including:
                - ra: Right ascension in hours
                - dec: Declination in degrees
                - distance: Distance from Earth in AU
                - elongation: Angle from the Sun in degrees
                - magnitude: Visual magnitude (if available)
                - phase: Illuminated fraction
        """
        try:
            # Use current time if not specified
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            # Convert timestamp to skyfield time
            t = self.timescale.from_datetime(timestamp)
            
            # Use all loaded planets if not specified
            planet_list = planets or list(self.planets.keys())
            
            # Container for results
            positions = {}
            
            # Get Earth and Sun positions
            earth = self.ephemeris['earth']
            sun = self.ephemeris['sun']
            
            # Set up observer - either from the Earth center or a specific location
            if self.observer:
                observer = earth + self.observer
            else:
                observer = earth
            
            # Calculate for each planet
            for planet_name in planet_list:
                if planet_name not in self.planets:
                    logger.warning(f"Planet '{planet_name}' not found, skipping")
                    continue
                
                planet = self.planets[planet_name]
                
                # Get astrometric position
                astrometric = observer.at(t).observe(planet)
                
                # Get apparent position (including light-time and aberration)
                apparent = astrometric.apparent()
                
                # Get position in RA/Dec
                ra, dec, distance = apparent.radec()
                
                # Get altitude and azimuth if observer location is set
                alt, az, _ = apparent.altaz() if self.observer else (None, None, None)
                
                # Phase angle and illumination fraction (not applicable to the Sun)
                phase_angle = None
                phase_fraction = None
                magnitude = None
                
                if planet_name != 'sun':
                    # Get phase angle (sun-planet-observer)
                    sun_planet = observer.at(t).observe(sun).apparent()
                    planet_observer = apparent.position.km
                    sun_observer = sun_planet.position.km
                    
                    # Normalize vectors
                    planet_observer_norm = planet_observer / np.sqrt(np.sum(planet_observer**2))
                    sun_observer_norm = sun_observer / np.sqrt(np.sum(sun_observer**2))
                    
                    # Calculate phase angle
                    dot_product = np.dot(planet_observer_norm, sun_observer_norm)
                    phase_angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
                    
                    # Illuminated fraction
                    phase_fraction = (1 + np.cos(phase_angle * np.pi / 180)) / 2
                
                # Calculate elongation (angular separation from Sun)
                sun_pos = observer.at(t).observe(sun).apparent()
                _, _, _ = sun_pos.radec()
                
                # Calculate angular separation
                elongation = astrometric.separation_from(observer.at(t).observe(sun)).degrees
                
                # Store results
                positions[self.PLANET_NAMES.get(planet_name, planet_name)] = {
                    'ra': ra.hours,
                    'dec': dec.degrees,
                    'distance': distance.au,
                    'altitude': alt.degrees if alt else None,
                    'azimuth': az.degrees if az else None,
                    'elongation': elongation,
                    'phase_angle': phase_angle,
                    'phase_fraction': phase_fraction,
                    'magnitude': magnitude,
                    'timestamp': timestamp.isoformat(),
                }
            
            logger.info(f"Calculated positions for {len(positions)} planets at {timestamp}")
            return positions
            
        except Exception as e:
            logger.error(f"Error calculating planet positions: {str(e)}")
            raise
    
    def get_planetary_alignments(self, 
                               threshold_degrees: float = 5.0,
                               timestamp: Optional[datetime] = None,
                               planets: Optional[List[str]] = None) -> List[Dict]:
        """
        Find planetary alignments where planets appear close to each other.
        
        Args:
            threshold_degrees: Maximum angular separation to consider as an alignment
            timestamp: Time for calculations (default: current time)
            planets: List of planets to check for alignments (default: all planets)
            
        Returns:
            List of dictionaries describing alignments with:
                - planets: List of planets in the alignment
                - angular_separation: Separation in degrees
                - timestamp: Time of alignment
        """
        try:
            # Use current time if not specified
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            # Convert timestamp to skyfield time
            t = self.timescale.from_datetime(timestamp)
            
            # Use all loaded planets if not specified
            planet_list = planets or list(self.planets.keys())
            
            # Set up observer - either from the Earth center or a specific location
            if self.observer:
                observer = self.earth + self.observer
            else:
                observer = self.earth
            
            # Container for results
            alignments = []
            
            # Check all pairs of planets
            for i, planet1_name in enumerate(planet_list):
                if planet1_name not in self.planets:
                    continue
                    
                planet1 = self.planets[planet1_name]
                
                for planet2_name in planet_list[i+1:]:
                    if planet2_name not in self.planets:
                        continue
                        
                    planet2 = self.planets[planet2_name]
                    
                    # Skip if comparing a planet to itself
                    if planet1_name == planet2_name:
                        continue
                        
                    # Get astrometric positions
                    planet1_pos = observer.at(t).observe(planet1).apparent()
                    planet2_pos = observer.at(t).observe(planet2).apparent()
                    
                    # Calculate angular separation
                    separation = planet1_pos.separation_from(planet2_pos).degrees
                    
                    # Check if separation is below threshold
                    if separation <= threshold_degrees:
                        alignments.append({
                            'planets': [
                                self.PLANET_NAMES.get(planet1_name, planet1_name),
                                self.PLANET_NAMES.get(planet2_name, planet2_name)
                            ],
                            'angular_separation': separation,
                            'timestamp': timestamp.isoformat()
                        })
            
            logger.info(f"Found {len(alignments)} planetary alignments at {timestamp}")
            return alignments
            
        except Exception as e:
            logger.error(f"Error finding planetary alignments: {str(e)}")
            raise
    
    def get_planetary_distances(self,
                              from_body: str = 'earth',
                              timestamp: Optional[datetime] = None,
                              planets: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate distances between planets and a reference body (default: Earth).
        
        Args:
            from_body: Reference body to calculate distances from
            timestamp: Time for calculations (default: current time)
            planets: List of planets to calculate distances for
                    (default: all planets in self.planets)
                    
        Returns:
            Dictionary mapping planet names to distances in astronomical units (AU)
        """
        try:
            # Use current time if not specified
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            # Convert timestamp to skyfield time
            t = self.timescale.from_datetime(timestamp)
            
            # Use all loaded planets if not specified
            planet_list = planets or list(self.planets.keys())
            
            # Check if reference body exists
            if from_body not in self.planets:
                raise ValueError(f"Reference body '{from_body}' not found in loaded planets")
            
            reference_body = self.planets[from_body]
            
            # Container for results
            distances = {}
            
            # Calculate distance to each planet
            for planet_name in planet_list:
                if planet_name not in self.planets or planet_name == from_body:
                    continue
                    
                planet = self.planets[planet_name]
                
                # Calculate the position of the planet relative to the reference body
                position = reference_body.at(t).observe(planet)
                
                # Get the distance in AU
                distance = position.distance().au
                
                distances[self.PLANET_NAMES.get(planet_name, planet_name)] = distance
            
            logger.info(f"Calculated distances from {from_body} to {len(distances)} planets")
            return distances
            
        except Exception as e:
            logger.error(f"Error calculating planetary distances: {str(e)}")
            raise
    
    def predict_planet_position(self, 
                              planet_name: str, 
                              start_time: datetime,
                              end_time: datetime,
                              steps: int = 24) -> List[Dict]:
        """
        Predict the position of a planet over a time range.
        
        Args:
            planet_

