"""
Space Weather Monitor Module

This module provides functionality for monitoring and analyzing space weather conditions,
including solar flares, coronal mass ejections (CMEs), geomagnetic conditions, and
aurora forecasts using data from NASA DONKI API and NOAA Space Weather API.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import aiohttp
import asyncio
import requests
from requests.exceptions import RequestException, Timeout, HTTPError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpaceWeatherError(Exception):
    """Custom exception for space weather related errors."""
    pass


class SpaceWeatherEvent:
    """Class representing a space weather event like a solar flare or CME."""
    
    # Event types enumeration
    # Event types enumeration
    EVENT_TYPES = [
        "SOLAR_FLARE", 
        "CME", 
        "GEOMAGNETIC_STORM", 
        "RADIATION_BELT", 
        "AURORA", 
        "SOLAR_ENERGETIC_PARTICLE",
        "WSA_ENLIL",
        "RADIO_BLACKOUT",
        "HIGH_SPEED_STREAM",
        "MAGNETOPAUSE_CROSSING"
    ]
    # Severity levels
    SEVERITY_LEVELS = ["LOW", "MODERATE", "HIGH", "EXTREME", "UNKNOWN"]
    
    def __init__(
        self,
        event_id: str,
        event_type: str,
        start_time: datetime,
        source: str,
        severity: str = "UNKNOWN",
        end_time: Optional[datetime] = None,
        description: Optional[str] = None,
        link: Optional[str] = None,
        raw_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a space weather event.
        
        Args:
                self._fetch_cmes_async(),
                self._fetch_geomagnetic_storms_async(),
                self._fetch_aurora_forecast_async(),
                self._fetch_solar_conditions_async(),
                self._fetch_geomagnetic_conditions_async(),
                self._fetch_radio_blackouts_async(),
                self._fetch_solar_energetic_particles_async(),
                self._fetch_magnetopause_crossings_async(),
                self._fetch_high_speed_streams_async(),
                self._fetch_wsa_enlil_predictions_async()
            end_time: Time when the event ended (if applicable)
            description: Human-readable description of the event
            link: URL to more information about the event
            raw_data: Raw data from API for reference
        """
        self.event_id = event_id
        self.event_type = event_type
        self.start_time = start_time
        self.end_time = end_time
        self.source = source
        self.severity = severity if severity in self.SEVERITY_LEVELS else "UNKNOWN"
        self.description = description
        self.link = link
        self.raw_data = raw_data
    
    def get_geomagnetic_conditions(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get current geomagnetic conditions from NOAA's SWPC API.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing geomagnetic condition information
            
        Raises:
            SpaceWeatherError: If the API request fails
        """
        try:
            # The NOAA endpoint for geomagnetic conditions (planetary K index)
            endpoint = "planetary_k_index_1m.json"
            
            # Make the API request
            kp_data = self._make_noaa_request(endpoint, use_cache=use_cache)
            
            # Process the data
            if not kp_data:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "kp_index": None,
                    "activity_level": "UNKNOWN"
                }
            
            # Sort by time to get most recent first
            sorted_data = sorted(kp_data, key=lambda x: x.get("time_tag", ""), reverse=True)
            
            if not sorted_data:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "kp_index": None,
                    "activity_level": "UNKNOWN"
                }
            
            # Get most recent data point
            latest = sorted_data[0]
            kp_index = latest.get("kp_index", None)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "kp_index": kp_index,
                "data_time": latest.get("time_tag", "Unknown"),
                "observation_quality": "good"  # Default value
            }
            
            # Set activity level based on Kp index
            if kp_index is not None:
                if kp_index >= 8:
                    result["activity_level"] = "EXTREME"
                elif kp_index >= 6:
                    result["activity_level"] = "HIGH"
                elif kp_index >= 5:
                    result["activity_level"] = "MODERATE"
                else:
                    result["activity_level"] = "LOW"
                    
                # Update current conditions
                self.current_conditions["geomagnetic_activity"] = result["activity_level"]
                self.current_conditions["last_updated"] = datetime.now()
            else:
                result["activity_level"] = "UNKNOWN"
            
            return result
            
        except SpaceWeatherError as e:
            logger.error(f"Error fetching geomagnetic conditions: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error fetching geomagnetic conditions: {e}")
            raise SpaceWeatherError(f"Failed to retrieve geomagnetic conditions: {str(e)}")
    
    def get_solar_conditions(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get current solar conditions from NOAA's SWPC API.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing solar condition information
            
        Raises:
            SpaceWeatherError: If the API request fails
        """
        try:
            # The NOAA endpoint for solar conditions
            endpoint = "solar-wind/mag_1h.json"
            
            # Make the API request
            solar_data = self._make_noaa_request(endpoint, use_cache=use_cache)
            
            # Process the data
            result = {
                "timestamp": datetime.now().isoformat(),
                "observation_quality": "good"  # Default value
            }
            
            if not solar_data:
                result["activity_level"] = "UNKNOWN"
                return result
            
            # Sort by time to get most recent first
            sorted_data = sorted(solar_data, key=lambda x: x.get("time_tag", ""), reverse=True)
            
            if not sorted_data:
                result["activity_level"] = "UNKNOWN"
                return result
            
            # Get most recent data point
            latest = sorted_data[0]
            result["data_time"] = latest.get("time_tag", "Unknown")
            
            # Extract relevant fields
            bz = latest.get("bz", None)
            bt = latest.get("bt", None)
            
            if bz is not None:
                result["bz"] = bz
            if bt is not None:
                result["bt"] = bt
            
            # Set activity level based on magnetic field strength and orientation
            if bt is not None and bz is not None:
                if bt > 25 or bz < -15:
                    result["activity_level"] = "EXTREME"
                elif bt > 15 or bz < -10:
                    result["activity_level"] = "HIGH"
                elif bt > 10 or bz < -5:
                    result["activity_level"] = "MODERATE"
                else:
                    result["activity_level"] = "LOW"
                    
                # Update current conditions
                self.current_conditions["solar_activity"] = result["activity_level"]
                self.current_conditions["last_updated"] = datetime.now()
            else:
                result["activity_level"] = "UNKNOWN"
            
            return result
            
        except SpaceWeatherError as e:
            logger.error(f"Error fetching solar conditions: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error fetching solar conditions: {e}")
            raise SpaceWeatherError(f"Failed to retrieve solar conditions: {str(e)}")
    
    async def _monitoring_task(self):
        """
        Background task for periodically monitoring space weather conditions.
        This asynchronous task runs in the background and periodically 
        fetches updated space weather data.
        """
        logger.info("Space weather monitoring started")
        
        while self.monitoring_active:
            try:
                # Update all space weather data
                await self.update_current_conditions_async()
                
                # Check for significant events or changes
                self._check_for_significant_events()
                
                # Wait for the next monitoring interval
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
                # Continue monitoring despite errors
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    async def update_current_conditions_async(self):
        """
        Asynchronously update all current space weather conditions.
        This method fetches the latest data from all relevant sources.
        """
        try:
            # Create tasks for all API calls
            tasks = [
                self._fetch_solar_flares_async(),
                self._fetch_cmes_async(),
                self._fetch_geomagnetic_storms_async(),
                self._fetch_aurora_forecast_async(),
                self._fetch_solar_conditions_async(),
                self._fetch_geomagnetic_conditions_async()
            ]
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error updating conditions: {result}")
                    continue
                    
                # Result processing will be handled by each fetch method
                
            # Update timestamp
            self.current_conditions["last_updated"] = datetime.now()
            
            logger.info("Space weather conditions updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update space weather conditions: {e}")
    
    async def _fetch_solar_flares_async(self):
        """Asynchronously fetch solar flare data."""
        try:
            # Use aiohttp for async requests
            async with aiohttp.ClientSession() as session:
                # Form the request URL
                url = f"{self.NASA_DONKI_URL}/FLR"
                
                # Set up parameters
                start_date = datetime.now() - timedelta(days=2)
                end_date = datetime.now()
                params = {
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d"),
                    "api_key": self.nasa_api_key
                }
                
                # Make the request
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Process flares
                    # This is a simplified version - the full implementation would 
                    # create SpaceWeatherEvent objects
                    if data:
                        flare_count = len(data)
                        logger.info(f"Found {flare_count} recent solar flares")
                        
                        # Determine overall solar activity from flares
                        x_class = any(flare.get("classType", "").startswith("X") for flare in data)
                        m_class = any(flare.get("classType", "").startswith("M") for flare in data)
                        
                        if x_class:
                            self.current_conditions["solar_activity"] = "EXTREME"
                        elif m_class:
                            self.current_conditions["solar_activity"] = "HIGH"
                        
        except Exception as e:
            logger.error(f"Error fetching solar flares async: {e}")
            raise
    
    # Similar async methods for other data types
    # Similar async methods for other data types
    # Implementation of other async fetching methods
    async def _fetch_radio_blackouts_async(self):
        """Asynchronously fetch radio blackout data."""
        try:
            # Use aiohttp for async requests
            async with aiohttp.ClientSession() as session:
                # Form the request URL
                url = f"{self.NASA_DONKI_URL}/RBE"
                
                # Set up parameters
                start_date = datetime.now() - timedelta(days=2)
                end_date = datetime.now()
                params = {
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d"),
                    "api_key": self.nasa_api_key
                }
                
                # Make the request
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Process radio blackouts
                    if data:
                        blackout_count = len(data)
                        logger.info(f"Found {blackout_count} recent radio blackouts")
                        
                        # Update current events
                        for blackout in data:
                            try:
                                event_id = blackout.get("rbeID", "unknown")
                                
                                # Skip if we already have this event
                                if event_id in self.current_events:
                                    continue
                                    
                                # Extract needed data and create event
                                start_time = None
                                if "startTime" in blackout:
                                    try:
                                        start_time = datetime.fromisoformat(blackout["startTime"].replace("Z", "+00:00"))
                                    except (ValueError, TypeError):
                                        logger.warning(f"Invalid start time format for radio blackout {event_id}")
                                        
                                # Determine severity based on class
                                severity = "UNKNOWN"
                                blackout_class = blackout.get("classType", "")
                                
                                if blackout_class.startswith("X"):
                                    severity = "EXTREME"
                                elif blackout_class.startswith("M"):
                                    severity = "HIGH"
                                elif blackout_class.startswith("C"):
                                    severity = "MODERATE"
                                elif blackout_class.startswith("B") or blackout_class.startswith("A"):
                                    severity = "LOW"
                                
                                # Create description
                                description = f"Radio Blackout: Class {blackout_class}"
                                
                                # Create event
                                event = SpaceWeatherEvent(
                                    event_id=event_id,
                                    event_type="RADIO_BLACKOUT",
                                    start_time=start_time,
                                    end_time=None,  # Radio blackouts typically don't have end times in DONKI
                                    source="NASA DONKI",
                                    severity=severity,
                                    description=description,
                                    link=f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/RBE/{event_id}",
                                    raw_data=blackout
                                )
                                
                                # Add to current events
                                self.current_events[event_id] = event
                                
                            except Exception as e:
                                logger.warning(f"Error processing radio blackout data: {e}")
                                continue
                        
        except Exception as e:
            logger.error(f"Error fetching radio blackouts async: {e}")
            raise

    async def _fetch_solar_energetic_particles_async(self):
        """Asynchronously fetch solar energetic particle data."""
        try:
            # Use aiohttp for async requests
            async with aiohttp.ClientSession() as session:
                # Form the request URL
                url = f"{self.NASA_DONKI_URL}/SEP"
                
                # Set up parameters
                start_date = datetime.now() - timedelta(days=2)
                end_date = datetime.now()
                params = {
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d"),
                    "api_key": self.nasa_api_key
                }
                
                # Make the request
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Process SEP events
                    if data:
                        sep_count = len(data)
                        logger.info(f"Found {sep_count} recent solar energetic particle events")
                        
                        # Update current events
                        for sep in data:
                            try:
                                event_id = sep.get("sepID", "unknown")
                                
                                # Skip if we already have this event
                                if event_id in self.current_events:
                                    continue
                                    
                                # Extract needed data and create event
                                start_time = None
                                if "eventTime" in sep:
                                    try:
                                        start_time = datetime.fromisoformat(sep["eventTime"].replace("Z", "+00:00"))
                                    except (ValueError, TypeError):
                                        logger.warning(f"Invalid event time format for SEP {event_id}")
                                
                                # Determine severity based on different factors
                                severity = "HIGH"  # SEPs are usually high severity events
                                
                                # Create description
                                instruments = ", ".join([i.get("displayName", "") for i in sep.get("instruments", [])])
                                description = f"Solar Energetic Particle Event"
                                if instruments:
                                    description += f" detected by {instruments}"
                                
                                # Create event
                                event = SpaceWeatherEvent(
                                    event_id=event_id,
                                    event_type="SOLAR_ENERGETIC_PARTICLE",
                                    start_time=start_time,
                                    end_time=None,
                                    source="NASA DONKI",
                                    severity=severity,
                                    description=description,
                                    link=f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/SEP/{event_id}",
                                    raw_data=sep
                                )
                                
                                # Add to current events
                                self.current_events[event_id] = event
                                
                            except Exception as e:
                                logger.warning(f"Error processing SEP data: {e}")
                                continue
                        
        except Exception as e:
            logger.error(f"Error fetching solar energetic particles async: {e}")
            raise

    async def _fetch_magnetopause_crossings_async(self):
        """Asynchronously fetch magnetopause crossing data."""
        try:
            # Use aiohttp for async requests
            async with aiohttp.ClientSession() as session:
                # Form the request URL
                url = f"{self.NASA_DONKI_URL}/MPC"
                
                # Set up parameters
                start_date = datetime.now() - timedelta(days=2)
                end_date = datetime.now()
                params = {
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d"),
                    "api_key": self.nasa_api_key
                }
                
                # Make the request
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Process magnetopause crossings
                    if data:
                        mpc_count = len(data)
                        logger.info(f"Found {mpc_count} recent magnetopause crossings")
                        
                        # Update current events
                        for mpc in data:
                            try:
                                event_id = mpc.get("mpcID", "unknown")
                                
                                # Skip if we already have this event
                                if event_id in self.current_events:
                                    continue
                                    
                                # Extract needed data and create event
                                start_time = None
                                if "eventTime" in mpc:
                                    try:
                                        start_time = datetime.fromisoformat(mpc["eventTime"].replace("Z", "+00:00"))
                                    except (ValueError, TypeError):
                                        logger.warning(f"Invalid event time format for magnetopause crossing {event_id}")
                                
                                # Determine severity based on magnetopause crossing type
                                # Inbound (toward Earth) is typically more relevant for space weather
                                severity = "MODERATE"
                                if "crossing" in mpc and mpc["crossing"] == "inbound":
                                    severity = "HIGH"
                                
                                # Create description
                                crossing_type = mpc.get("crossing", "unknown")
                                spacecraft = mpc.get("spacecraft", "unknown spacecraft")
                                description = f"Magnetopause Crossing: {crossing_type.capitalize()} crossing by {spacecraft}"
                                
                                # Create event
                                event = SpaceWeatherEvent(
                                    event_id=event_id,
                                    event_type="MAGNETOPAUSE_CROSSING",
                                    start_time=start_time,
                                    end_time=None,
                                    source="NASA DONKI",
                                    severity=severity,
                                    description=description,
                                    link=f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/MPC/{event_id}",
                                    raw_data=mpc
                                )
                                
                                # Add to current events
                                self.current_events[event_id] = event
                                
                            except Exception as e:
                                logger.warning(f"Error processing magnetopause crossing data: {e}")
                                continue
                        
        except Exception as e:
            logger.error(f"Error fetching magnetopause crossings async: {e}")
            raise

    async def _fetch_high_speed_streams_async(self):
        """Asynchronously fetch high speed stream data."""
        try:
            # Use aiohttp for async requests
            async with aiohttp.ClientSession() as session:
                # Form the request URL
                url = f"{self.NASA_DONKI_URL}/HSS"
                
                # Set up parameters
                start_date = datetime.now() - timedelta(days=2)
                end_date = datetime.now()
                params = {
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d"),
                    "api_key": self.nasa_api_key
                }
                
                # Make the request
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Process high speed streams
                    if data:
                        hss_count = len(data)
                        logger.info(f"Found {hss_count} recent high speed streams")
                        
                        # Update current events
                        for hss in data:
                            try:
614
        try:
            # Use aiohttp for async requests
            async with aiohttp.ClientSession() as session:
                # Form the request URL
                url = f"{self.NASA_DONKI_URL}/CME"
                
                # Set up parameters
                start_date = datetime.now() - timedelta(days=2)
                end_date = datetime.now()
                params = {
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d"),
                    "api_key": self.nasa_api_key
                }
                
                # Make the request
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Process CMEs
                    if data:
                        cme_count = len(data)
                        logger.info(f"Found {cme_count} recent CMEs")
                        
                        # Update current events
                        for cme in data:
                            try:
                                cme_id = cme.get("activityID", "unknown")
                                
                                # Skip if we already have this event
                                if cme_id in self.current_events:
                                    continue
                                    
                                # Parse dates
                                start_time = None
                                if "startTime" in cme:
                                    try:
                                        start_time = datetime.fromisoformat(cme["startTime"].replace
    def _check_for_significant_events(self):
        """
        Check for significant space weather events that require notification.
        This method compares current conditions with previous state to detect
        significant changes or new events.
        """
        # Check for high-priority events
        high_priority_events = []
        
        # Solar flares (X and M class)
        for event_id, event in self.current_events.items():
            if event.event_type == "SOLAR_FLARE" and event.severity in ["HIGH", "EXTREME"]:
                if event_id not in self.event_history:
                    high_priority_events.append(event)
        
        # CMEs likely to impact Earth
        for event_id, event in self.current_events.items():
            if event.event_type == "CME" and event.severity in ["HIGH", "EXTREME"]:
                if event_id not in self.event_history:
                    high_priority_events.append(event)
        
        # Geomagnetic storms
        for event_id, event in self.current_events.items():
            if event.event_type == "GEOMAGNETIC_STORM" and event.severity in ["HIGH", "EXTREME"]:
                if event_id not in self.event_history:
                    high_priority_events.append(event)
        
        # Notify for high-priority events
        for event in high_priority_events:
            self._notify_event(event)
            
            # Add to history
            self.event_history.append(event.event_id)
            
        # Trim history if it gets too large
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-500:]
    
    def _notify_event(self, event: SpaceWeatherEvent):
        """
        Notify all registered callbacks about a space weather event.
        
        Args:
            event: The event to notify about
        """
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event notification callback: {e}")
    
    def register_callback(self, callback: callable) -> None:
        """
        Register a callback function to be notified of space weather events.
        
        Args:
            callback: Function that will be called when new space weather events occur.
                     The function must accept a SpaceWeatherEvent parameter.
        """
        if callback not in self.event_callbacks:
            self.event_callbacks.append(callback)
            logger.info("Registered new space weather event callback")
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate the duration of the event if end_time is available."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_ongoing(self) -> bool:
        """Determine if the event is still ongoing."""
        if self.end_time is None:
            return True
        return self.end_time > datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "source": self.source,
            "severity": self.severity,
            "description": self.description,
            "link": self.link,
            "is_ongoing": self.is_ongoing,
            "duration": str(self.duration) if self.duration else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpaceWeatherEvent':
        """Create an event from a dictionary."""
        start_time = datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None
        end_time = datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
        
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            start_time=start_time,
            end_time=end_time,
            source=data["source"],
            severity=data["severity"],
            description=data.get("description"),
            link=data.get("link"),
            raw_data=data.get("raw_data")
        )


class SpaceWeatherMonitor:
    """
    Class for monitoring space weather conditions from various data sources.
    
    This class provides methods to fetch and process space weather data 
    including solar flares, coronal mass ejections (CMEs), geomagnetic 
    conditions, and aurora forecasts.
    """
    
    # NASA DONKI API endpoint
    NASA_DONKI_URL = "https://api.nasa.gov/DONKI"
    
    # NOAA Space Weather Prediction Center endpoint
    NOAA_SWPC_URL = "https://services.swpc.noaa.gov/json"
    
    # Default cache directory
    DEFAULT_CACHE_DIR = Path("cache/space_weather")
    
    # Default cache expiration (in hours)
    DEFAULT_CACHE_EXPIRATION = 1  # Space weather data changes frequently
    
    # Monitoring intervals in seconds
    DEFAULT_MONITORING_INTERVAL = 3600  # 1 hour
    
    def __init__(
        self,
        nasa_api_key: str,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_expiration: int = DEFAULT_CACHE_EXPIRATION,
        auto_monitor: bool = False,
        monitoring_interval: int = DEFAULT_MONITORING_INTERVAL
    ):
        """
        Initialize the Space Weather Monitor.
        
        Args:
            nasa_api_key: NASA API key for DONKI API
            cache_dir: Directory to store cached data
            cache_expiration: Cache expiration time in hours
            auto_monitor: Whether to start automatic monitoring
            monitoring_interval: Interval between monitoring checks in seconds
        """
        self.nasa_api_key = nasa_api_key
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_expiration = cache_expiration
        self.monitoring_interval = monitoring_interval
        
        # Event storage
        self.current_events: Dict[str, SpaceWeatherEvent] = {}
        self.event_history: List[SpaceWeatherEvent] = []
        
        # Current conditions
        self.current_conditions = {
            "solar_activity": "UNKNOWN",
            "geomagnetic_activity": "UNKNOWN",
            "aurora_activity": "UNKNOWN",
            "last_updated": None
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Callback for event notifications
        self.event_callbacks: List[callable] = []
        
        # Start monitoring if requested
        if auto_monitor:
            self.start_monitoring()
    
    def _get_cache_path(self, endpoint: str, params: Dict[str, Any]) -> Path:
        """
        Generate a unique cache file path based on the request parameters.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Path to the cache file
        """
        # Create a unique filename from the endpoint and sorted parameters
        param_str = "_".join(f"{k}:{v}" for k, v in sorted(params.items()) 
                            if k != "api_key")
        filename = f"{endpoint.replace('/', '_')}_{param_str}.json"
        
        # Ensure the filename isn't too long
        if len(filename) > 255:
            import hashlib
            param_hash = hashlib.md5(param_str.encode()).hexdigest()
            filename = f"{endpoint.replace('/', '_')}_{param_hash}.json"
            
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if the cache file exists and is still valid.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False
            
        # Check if cache has expired
        file_time = cache_path.stat().st_mtime
        age_hours = (time.time() - file_time) / 3600
        
        return age_hours < self.cache_expiration
    
    def _read_cache(self, cache_path: Path) -> Dict[str, Any]:
        """
        Read data from cache file.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            Cached data as dictionary
            
        Raises:
            SpaceWeatherError: If the cache file cannot be read or parsed
        """
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error("Failed to parse NOAA API response")
            raise SpaceWeatherError("Invalid NOAA API response format")

    # Solar Flare methods
    def get_solar_flares(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> List[SpaceWeatherEvent]:
        """
        Get solar flare data from NASA's DONKI API.
        
        Args:
            start_date: Start date for data retrieval (default: 7 days ago)
            end_date: End date for data retrieval (default: current time)
            use_cache: Whether to use cached data if available
            
        Returns:
            List of SpaceWeatherEvent objects representing solar flares
            
        Raises:
            SpaceWeatherError: If the API request fails
        """
        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7)
            if end_date is None:
                end_date = datetime.now()
                
            # Format dates for API
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Make the API request
            params = {
                "startDate": start_str,
                "endDate": end_str
            }
            
            flare_data = self._make_nasa_request("FLR", params, use_cache)
            
            # If no data or empty list returned
            if not flare_data:
                logger.info("No solar flare data found for the specified period")
                return []
                
            # Process the flare data into SpaceWeatherEvent objects
            flares = []
            for flare in flare_data:
                try:
                    flare_id = flare.get("flrID", "unknown")
                    
                    # Parse dates
                    start_time = None
                    end_time = None
                    
                    if "beginTime" in flare:
                        try:
                            start_time = datetime.fromisoformat(flare["beginTime"].replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid begin time format for flare {flare_id}")
                    
                    if "endTime" in flare:
                        try:
                            end_time = datetime.fromisoformat(flare["endTime"].replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid end time format for flare {flare_id}")
                    
                    # Determine severity based on class (e.g., X1.5, M5.0)
                    classType = flare.get("classType", "")
                    severity = "UNKNOWN"
                    
                    if classType.startswith("X"):
                        severity = "EXTREME"
                    elif classType.startswith("M"):
                        severity = "HIGH"
                    elif classType.startswith("C"):
                        severity = "MODERATE"
                    elif classType.startswith("B") or classType.startswith("A"):
                        severity = "LOW"
                    
                    # Create link to event
                    link = f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/FLR/{flare_id}"
                    
                    # Create description
                    description = f"Solar Flare: Class {classType}"
                    if "sourceLocation" in flare:
                        description += f" from location {flare['sourceLocation']}"
                    
                    # Create event
                    event = SpaceWeatherEvent(
                        event_id=flare_id,
                        event_type="SOLAR_FLARE",
                        start_time=start_time,
                        end_time=end_time,
                        source="NASA DONKI",
                        severity=severity,
                        description=description,
                        link=link,
                        raw_data=flare
                    )
                    
                    flares.append(event)
                    
                except Exception as e:
                    logger.warning(f"Error processing flare data: {e}")
                    continue
                    
            logger.info(f"Retrieved {len(flares)} solar flares from {start_str} to {end_str}")
            return flares
            
        except SpaceWeatherError as e:
            logger.error(f"Error fetching solar flare data: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error fetching solar flare data: {e}")
            raise SpaceWeatherError(f"Failed to retrieve solar flare data: {str(e)}")
    
    def get_cmes(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> List[SpaceWeatherEvent]:
        """
        Get coronal mass ejection (CME) data from NASA's DONKI API.
        
        Args:
            start_date: Start date for data retrieval (default: 7 days ago)
            end_date: End date for data retrieval (default: current time)
            use_cache: Whether to use cached data if available
            
        Returns:
            List of SpaceWeatherEvent objects representing CMEs
            
        Raises:
            SpaceWeatherError: If the API request fails
        """
        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7)
            if end_date is None:
                end_date = datetime.now()
                
            # Format dates for API
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Make the API request
            params = {
                "startDate": start_str,
                "endDate": end_str
            }
            
            cme_data = self._make_nasa_request("CME", params, use_cache)
            
            # If no data or empty list returned
            if not cme_data:
                logger.info("No CME data found for the specified period")
                return []
                
            # Process the CME data into SpaceWeatherEvent objects
            cmes = []
            for cme in cme_data:
                try:
                    cme_id = cme.get("activityID", "unknown")
                    
                    # Parse dates
                    start_time = None
                    
                    if "startTime" in cme:
                        try:
                            start_time = datetime.fromisoformat(cme["startTime"].replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid start time format for CME {cme_id}")
                    
                    # Determine severity based on speed or other factors
                    severity = "UNKNOWN"
                    speed = None
                    
                    # Try to get speed from analysis
                    if "cmeAnalyses" in cme and cme["cmeAnalyses"]:
                        for analysis in cme["cmeAnalyses"]:
                            if "speed" in analysis:
                                speed = analysis.get("speed")
                                break
                    
                    # Set severity based on speed (km/s)
                    if speed:
                        if speed > 1500:
                            severity = "EXTREME"
                        elif speed > 1000:
                            severity = "HIGH"
                        elif speed > 500:
                            severity = "MODERATE"
                        else:
                            severity = "LOW"
                    
                    # Create link to event
                    link = f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/CME/{cme_id}"
                    
                    # Create description
                    description = "Coronal Mass Ejection"
                    if speed:
                        description += f" with speed {speed} km/s"
                    if "note" in cme:
                        description += f"\nNote: {cme['note']}"
                    
                    # Create event
                    event = SpaceWeatherEvent(
                        event_id=cme_id,
                        event_type="CME",
                        start_time=start_time,
                        end_time=None,  # CMEs don't typically have end times in DONKI
                        source="NASA DONKI",
                        severity=severity,
                        description=description,
                        link=link,
                        raw_data=cme
                    )
                    
                    cmes.append(event)
                    
                except Exception as e:
                    logger.warning(f"Error processing CME data: {e}")
                    continue
                    
            logger.info(f"Retrieved {len(cmes)} CMEs from {start_str} to {end_str}")
            return cmes
            
        except SpaceWeatherError as e:
            logger.error(f"Error fetching CME data: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error fetching CME data: {e}")
            raise SpaceWeatherError(f"Failed to retrieve CME data: {str(e)}")
    
    def get_geomagnetic_storms(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> List[SpaceWeatherEvent]:
        """
        Get geomagnetic storm data from NASA's DONKI API.
        
        Args:
            start_date: Start date for data retrieval (default: 7 days ago)
            end_date: End date for data retrieval (default: current time)
            use_cache: Whether to use cached data if available
            
        Returns:
            List of SpaceWeatherEvent objects representing geomagnetic storms
            
        Raises:
            SpaceWeatherError: If the API request fails
        """
        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7)
            if end_date is None:
                end_date = datetime.now()
                
            # Format dates for API
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Make the API request
            params = {
                "startDate": start_str,
                "endDate": end_str
            }
            
            storm_data = self._make_nasa_request("GST", params, use_cache)
            
            # If no data or empty list returned
            if not storm_data:
                logger.info("No geomagnetic storm data found for the specified period")
                return []
                
            # Process the storm data into SpaceWeatherEvent objects
            storms = []
            for storm in storm_data:
                try:
                    storm_id = storm.get("gstID", "unknown")
                    
                    # Parse dates
                    start_time = None
                    end_time = None
                    
                    if "startTime" in storm:
                        try:
                            start_time = datetime.fromisoformat(storm["startTime"].replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid start time format for storm {storm_id}")
                    
                    if "endTime" in storm:
                        try:
                            end_time = datetime.fromisoformat(storm["endTime"].replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid end time format for storm {storm_id}")
                    
                    # Determine severity based on Kp index
                    severity = "UNKNOWN"
                    kp_index = None
                    
                    if "allKpIndex" in storm and storm["allKpIndex"]:
                        # Use the maximum Kp index
                        kp_indices = [kp["kpIndex"] for kp in storm["allKpIndex"] if "kpIndex" in kp]
                        if kp_indices:
                            kp_index = max(kp_indices)
                    
                    # Set severity based on Kp index
                    if kp_index is not None:
                        if kp_index >= 8:
                            severity = "EXTREME"
                        elif kp_index >= 6:
                            severity = "HIGH"
                        elif kp_index >= 5:
                            severity = "MODERATE"
                        else:
                            severity = "LOW"
                    
                    # Create link to event
                    link = f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/GST/{storm_id}"
                    
                    # Create description
                    description = "Geomagnetic Storm"
                    if kp_index is not None:
                        description += f" with max Kp index of {kp_index}"
                    
                    # Create event
                    event = SpaceWeatherEvent(
                        event_id=storm_id,
                        event_type="GEOMAGNETIC_STORM",
                        start_time=start_time,
                        end_time=end_time,
                        source="NASA DONKI",
                        severity=severity,
                        description=description,
                        link=link,
                        raw_data=storm
                    )
                    
                    storms.append(event)
                    
                except Exception as e:
                    logger.warning(f"Error processing storm data: {e}")
                    continue
                    
            logger.info(f"Retrieved {len(storms)} geomagnetic storms from {start_str} to {end_str}")
            return storms
            
        except SpaceWeatherError as e:
            logger.error(f"Error fetching geomagnetic storm data: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error fetching geomagnetic storm data: {e}")
            raise SpaceWeatherError(f"Failed to retrieve geomagnetic storm data: {str(e)}")
    
    def get_aurora_forecast(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get aurora forecast data from NOAA's SWPC API.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing aurora forecast information
            
        Raises:
            SpaceWeatherError: If the API request fails
        """
        try:
            # The NOAA endpoint for aurora forecasts
            endpoint = "ovation_aurora_latest.json"
            
            # Make the API request
            aurora_data = self._make_noaa_request(endpoint, use_cache=use_cache)
            
            # Process the data
            forecast = {
                "timestamp": datetime.now().isoformat(),
                "forecast_time": aurora_data.get("Prediction_Time", "Unknown"),
                "coordinates": aurora_data.get("coordinates", []),
                "observation_quality": "good"  # Default value
            }
            
            # Calculate aurora activity level based on data
            if "coordinates" in aurora_data and aurora_data["coordinates"]:
                # Calculate average aurora probability across coordinates
                probabilities = []
                for coord in aurora_data["coordinates"]:
                    if isinstance(coord, dict) and "Aurora_Probability" in coord:
                        probabilities.append(float(coord["Aurora_Probability"]))
                
                if probabilities:
                    avg_probability = sum(probabilities) / len(probabilities)
                    
                    # Set activity level based on probability
                    if avg_probability > 0.7:
                        forecast["activity_level"] = "EXTREME"
                    elif avg_probability > 0.5:
                        forecast["activity_level"] = "HIGH" 
                    elif avg_probability > 0.3:
                        forecast["activity_level"] = "MODERATE"
                    else:
                        forecast["activity_level"] = "LOW"
                        
                    forecast["average_probability"] = avg_probability
                    
                    # Update current conditions
                    self.current_conditions["aurora_activity"] = forecast["activity_level"]
                    self.current_conditions["last_updated"] = datetime.now()
            
            return forecast
            
        except SpaceWeatherError as e:
            logger.error(f"Error fetching aurora forecast: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error fetching aurora forecast: {e}")
            raise SpaceWeatherError(f"Failed to retrieve aurora forecast: {str(e)}")
    
    def _write_cache(self, cache_path: Path, data: Dict[str, Any]) -> None:
        """
        Write data to cache file.
        
        Args:
            cache_path: Path to the cache file
            data: Data to cache
            
        Raises:
            SpaceWeatherError: If the cache file cannot be written
        """
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            logger.warning(f"Cache write error: {e}")
            raise SpaceWeatherError(f"Failed to write cache: {str(e)}")
    
    def _make_nasa_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make a request to NASA's DONKI API with caching.
        
        Args:
            endpoint: API endpoint to query
            params: Query parameters
            use_cache: Whether to use cached data if available
            
        Returns:
            API response data
            
        Raises:
            SpaceWeatherError: If the API request fails
        """
        # Prepare request parameters
        request_params = params or {}
        request_params["api_key"] = self.nasa_api_key
        
        url = f"{self.NASA_DONKI_URL}/{endpoint}"
        cache_path = self._get_cache_path(f"nasa_{endpoint}", request_params)
        
        # Try to use cached data if enabled and available
        if use_cache and self._is_cache_valid(cache_path):
            logger.debug(f"Using cached NASA data for {endpoint}")
            return self._read_cache(cache_path)
        
        try:
            logger.debug(f"Making NASA API request to {endpoint}")
            response = requests.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response if caching is enabled
            if use_cache:
                self._write_cache(cache_path, data)
                
            return data
            
        except Timeout:
            logger.error(f"Request to NASA {endpoint} timed out")
            raise SpaceWeatherError(f"NASA API request timed out: {url}")
            
        except HTTPError as e:
            logger.error(f"HTTP error from NASA API: {e}")
            # For 404, return empty list as this could mean no events
            if e.response.status_code == 404:
                return []
            raise SpaceWeatherError(f"NASA API HTTP error: {e}")
            
        except RequestException as e:
            logger.error(f"Request error to NASA API: {e}")
            raise SpaceWeatherError(f"NASA API request failed: {str(e)}")
            
        except json.JSONDecodeError:
            logger.error("Failed to parse NASA API response")
            raise SpaceWeatherError("Invalid NASA API response format")
    
    def _make_noaa_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make a request to NOAA's Space Weather Prediction Center API with caching.
        
        Args:
            endpoint: API endpoint to query
            params: Query parameters
            use_cache: Whether to use cached data if available
            
        Returns:
            API response data
            
        Raises:
            SpaceWeatherError: If the API request fails
        """
        # Prepare request parameters
        request_params = params or {}
        
        url = f"{self.NOAA_SWPC_URL}/{endpoint}"
        cache_path = self._get_cache_path(f"noaa_{endpoint}", request_params)
        
        # Try to use cached data if enabled and available
        if use_cache and self._is_cache_valid(cache_path):
            logger.debug(f"Using cached NOAA data for {endpoint}")
            return self._read_cache(cache_path)
        
        try:
            logger.debug(f"Making NOAA API request to {endpoint}")
            response = requests.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response if caching is enabled
            if use_cache:
                self._write_cache(cache_path, data)
                
            return data
            
        except Timeout:
            logger.error(f"Request to NOAA {endpoint} timed out")
            raise SpaceWeatherError(f"NOAA API request timed out: {url}")
            
        except HTTPError as e:
            logger.error(f"HTTP error from NOAA API: {e}")
            raise SpaceWeatherError(f"NOAA API HTTP error: {e}")
            
        except RequestException as e:
            logger.error(f"Request error to NOAA API: {e}")
            raise SpaceWeatherError(f"NOAA API request failed: {str(e)}")
            
        except json.JSONDecodeError:
            logger.error("Failed to parse NOAA API response")
            raise SpaceWeatherError("Invalid NOAA API response format")
            
    def start_monitoring(self) -> None:
        """
        Start the background monitoring task.
        
        This method starts an asynchronous task that periodically fetches
        and processes space weather data.
        """
        if not self.monitoring_active:
            self.monitoring_active = True
            
            # Create a new event loop if we're not already in one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Start the monitoring task
            self.monitoring_task = asyncio.ensure_future(self._monitoring_task())
            logger.info(f"Started space weather monitoring (interval: {self.monitoring_interval}s)")
            
    def stop_monitoring(self) -> None:
        """
        Stop the background monitoring task.
        """
        if self.monitoring_active:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                self.monitoring_task = None
                
            logger.info("Stopped space weather monitoring")
            
    def get_current_condition_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current space weather conditions.
        
        Returns:
            Dictionary containing current space weather conditions
        """
        return {
            "solar_activity": self.current_conditions.get("solar_activity", "UNKNOWN"),
            "geomagnetic_activity": self.current_conditions.get("geomagnetic_activity", "UNKNOWN"),
            "aurora_activity": self.current_conditions.get("aurora_activity", "UNKNOWN"),
            "last_updated": self.current_conditions.get("last_updated", None),
            "active_alerts": len([e for e in self.current_events.values() 
                               if e.severity in ["HIGH", "EXTREME"] and e.is_ongoing])
        }
