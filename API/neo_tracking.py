"""
NASA Near Earth Object (NEO) Tracking Module.

This module provides functionality to interact with NASA's NEO API,
fetch asteroid data, identify potentially hazardous asteroids, and
cache responses to minimize API calls.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import logging
import requests
from requests.exceptions import RequestException, Timeout, HTTPError

# Set up logging
logger = logging.getLogger(__name__)

class NEOAPIError(Exception):
    """Custom exception for NEO API related errors."""
    pass

class NEOTracker:
    """
    Class for handling NASA's Near Earth Object (NEO) API interactions.
    
    This class provides methods to fetch, process, and analyze asteroid data
    from NASA's NEO API with built-in caching and rate limiting.
    """
    
    # NASA's NEO API endpoint
    BASE_URL = "https://api.nasa.gov/neo/rest/v1"
    
    # Default cache directory
    DEFAULT_CACHE_DIR = Path("cache/neo_data")
    
    # Default cache expiration (in hours)
    DEFAULT_CACHE_EXPIRATION = 24
    
    # Rate limiting (requests per hour)
    RATE_LIMIT = 1000
    
    def __init__(
        self, 
        api_key: str, 
        cache_dir: Optional[Union[str, Path]] = None,
        cache_expiration: int = DEFAULT_CACHE_EXPIRATION,
        enable_rate_limiting: bool = True
    ):
        """
        Initialize the NEO Tracker.
        
        Args:
            api_key: NASA API key
            cache_dir: Directory to store cached data (default: cache/neo_data)
            cache_expiration: Cache expiration time in hours (default: 24)
            enable_rate_limiting: Whether to enforce rate limiting (default: True)
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_expiration = cache_expiration
        self.enable_rate_limiting = enable_rate_limiting
        self._last_request_time = 0.0
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"NEOTracker initialized with cache at {self.cache_dir}")
    
    def _rate_limit(self) -> None:
        """Apply rate limiting to avoid hitting NASA API limits."""
        if not self.enable_rate_limiting:
            return
            
        # Calculate minimum seconds between requests to stay under rate limit
        min_interval = 3600 / self.RATE_LIMIT  # seconds
        
        # Calculate elapsed time since last request
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        
        # If not enough time has passed, sleep for the remaining duration
        if elapsed < min_interval and self._last_request_time > 0:
            sleep_time = min_interval - elapsed
            logger.debug(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        # Update last request time
        self._last_request_time = time.time()
    
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
            NEOAPIError: If the cache file cannot be read or parsed
        """
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Cache read error: {e}")
            raise NEOAPIError(f"Failed to read cache: {str(e)}")
    
    def _write_cache(self, cache_path: Path, data: Dict[str, Any]) -> None:
        """
        Write data to cache file.
        
        Args:
            cache_path: Path to the cache file
            data: Data to cache
            
        Raises:
            NEOAPIError: If the cache file cannot be written
        """
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            logger.warning(f"Cache write error: {e}")
            raise NEOAPIError(f"Failed to write cache: {str(e)}")
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make an API request with caching and rate limiting.
        
        Args:
            endpoint: API endpoint to query
            params: Query parameters
            use_cache: Whether to use cached data if available
            
        Returns:
            API response data
            
        Raises:
            NEOAPIError: If the API request fails
        """
        # Prepare request parameters
        request_params = params or {}
        request_params["api_key"] = self.api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        cache_path = self._get_cache_path(endpoint, request_params)
        
        # Try to use cached data if enabled and available
        if use_cache and self._is_cache_valid(cache_path):
            logger.debug(f"Using cached data for {endpoint}")
            return self._read_cache(cache_path)
        
        # Apply rate limiting before making the request
        self._rate_limit()
        
        try:
            logger.debug(f"Making API request to {endpoint}")
            response = requests.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response if caching is enabled
            if use_cache:
                self._write_cache(cache_path, data)
                
            return data
            
        except Timeout:
            logger.error(f"Request to {endpoint} timed out")
            raise NEOAPIError(f"API request timed out: {url}")
            
        except HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise NEOAPIError(f"API HTTP error: {e}")
            
        except RequestException as e:
            logger.error(f"Request error: {e}")
            raise NEOAPIError(f"API request failed: {str(e)}")
            
        except json.JSONDecodeError:
            logger.error("Failed to parse API response")
            raise NEOAPIError("Invalid API response format")
    
    def get_feed(
        self, 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get asteroid feed data for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD) or datetime, defaults to today
            end_date: End date (YYYY-MM-DD) or datetime, defaults to 7 days after start
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing asteroid feed data
            
        Raises:
            NEOAPIError: If the date range is invalid or API request fails
            ValueError: If the date format is invalid
        """
        # Format and validate dates
        today = datetime.now()
        
        if start_date is None:
            start = today
        elif isinstance(start_date, str):
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("start_date must be in YYYY-MM-DD format")
        else:
            start = start_date
            
        if end_date is None:
            # Default to 7 days from start, but cap at today + 7 days (NASA API limit)
            end = min(start + timedelta(days=7), today + timedelta(days=7))
        elif isinstance(end_date, str):
            try:
                end = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("end_date must be in YYYY-MM-DD format")
        else:
            end = end_date
        
        # Validate date range (NASA API allows max 7 days)
        if (end - start).days > 7:
            raise NEOAPIError("Date range cannot exceed 7 days")
            
        if start > end:
            raise NEOAPIError("Start date must be before end date")
        
        # Format dates for API
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        
        params = {
            "start_date": start_str,
            "end_date": end_str
        }
        
        return self._make_request("feed", params, use_cache)
    
    def get_neo_by_id(self, asteroid_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get detailed information about a specific asteroid by ID.
        
        Args:
            asteroid_id: NASA JPL asteroid ID
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing detailed asteroid data
            
        Raises:
            NEOAPIError: If the API request fails
        """
        endpoint = f"neo/{asteroid_id}"
        return self._make_request(endpoint, use_cache=use_cache)
    
    def get_neo_browse(
        self, 
        page: int = 0, 
        size: int = 20,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Browse the overall asteroid dataset.
        
        Args:
            page: Page number for pagination
            size: Number of items per page (max 100)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing browsed asteroid data
            
        Raises:
            NEOAPIError: If the API request fails
            ValueError: If page or size is invalid
        """
        if page < 0:
            raise ValueError("Page number must be non-negative")
            
        if size < 1 or size > 100:
            raise ValueError("Size must be between 1 and 100")
            
        params = {
            "page": page,
            "size": size
        }
        
        return self._make_request("neo/browse", params, use_cache)
    
    def extract_hazardous_asteroids(
        self, 
        feed_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract potentially hazardous asteroids from feed data.
        
        Args:
            feed_data: Asteroid feed data returned by get_feed()
            
        Returns:
            List of potentially hazardous asteroids
        """
        hazardous_asteroids = []
        
        # Process each day's data
        for date, asteroids in feed_data.get("near_earth_objects", {}).items():
            for asteroid in asteroids:
                if asteroid.get("is_potentially_hazardous_asteroid", False):
                    # Add the date to the asteroid data for context
                    asteroid["close_approach_date"] = date
                    hazardous_asteroids.append(asteroid)
                    
        return hazardous_asteroids
    
    def get_hazardous_asteroids(
        self, 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all potentially hazardous asteroids for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD) or datetime, defaults to today
            end_date: End date (YYYY-MM-DD) or datetime, defaults to 7 days after start
            use_cache: Whether to use cached data if available
            
        Returns:
            List of potentially hazardous asteroids
            
        Raises:
            NEOAPIError: If the API request fails
        """
        feed_data = self.get_feed(start_date, end_date, use_cache)
        return self.extract_hazardous_asteroids(feed_data)
    
    def analyze_close_approaches(
        self, 
        asteroid_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze close approach data for an asteroid.
        
        Args:
            asteroid_data: Asteroid data returned by get_neo_by_id()
            
        Returns:
            List of analyzed close approach events
        """
        approaches = []
        
        for approach in asteroid_data.get("close_approach_data", []):
            # Extract useful data and convert units where needed
            velocity_kmh = approach.get("relative_velocity", {}).get("kilometers_per_hour")
            distance_km = approach.get("miss_distance", {}).get("kilometers")
            
            try:
                velocity = float(velocity_kmh) if velocity_kmh else None
                distance = float(distance_km) if distance_km else None
                
                approach_data = {
                    "date": approach.get("close_approach_date"),
                    "velocity_km_h": velocity,
                    "distance_km": distance,
                    "orbiting_body": approach.get("orbiting_body"),
                    # Add risk assessment based on distance
                    "risk_assessment": self._assess_risk(distance) if distance is not None else "Unknown",
                    "impact_energy": self._calculate_impact_energy(
                        asteroid_data.get("estimated_diameter", {})
                            .get("kilometers", {}).get("estimated_diameter_max"),
                        velocity
                    ) if velocity is not None else None
                }
                
                # Add additional timing information if available
                if "close_approach_date_full" in approach:
                    approach_data["date_full"] = approach["close_approach_date_full"]
                    
                # Add lunar distance for intuitive comparison
                if "miss_distance" in approach and "lunar" in approach["miss_distance"]:
                    approach_data["distance_lunar"] = float(approach["miss_distance"]["lunar"])
                
                approaches.append(approach_data)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing approach data: {e}")
                
        # Sort approaches by date
        approaches.sort(key=lambda x: x["date"])
        return approaches
    
    def _assess_risk(self, distance_km: float) -> str:
        """
        Assess the risk level of an asteroid based on its miss distance.
        
        Args:
            distance_km: Miss distance in kilometers
            
        Returns:
            Risk assessment as a string
        """
        # Lunar distance is approximately 384,400 km
        lunar_distance = 384400
        
        # Define risk thresholds (in kilometers)
        if distance_km is None:
            return "Unknown"
        elif distance_km < 7000:  # Approximately geostationary orbit
            return "Extreme"
        elif distance_km < lunar_distance * 0.5:  # Half lunar distance
            return "High"
        elif distance_km < lunar_distance:
            return "Moderate"
        elif distance_km < lunar_distance * 5:
            return "Low"
        else:
            return "Very Low"
    
    def _calculate_impact_energy(
        self, 
        diameter_km: Optional[float], 
        velocity_kmh: Optional[float]
    ) -> Optional[float]:
        """
        Calculate approximate impact energy in megatons of TNT.
        
        Uses a simplified formula based on diameter and velocity.
        
        Args:
            diameter_km: Asteroid diameter in kilometers
            velocity_kmh: Velocity in kilometers per hour
            
        Returns:
            Impact energy in megatons of TNT, or None if inputs are invalid
        """
        if diameter_km is None or velocity_kmh is None:
            return None
            
        try:
            # Convert velocity to km/s
            velocity_kms = velocity_kmh / 3600
            
            # Simplified impact energy calculation (in megatons of TNT)
            # Assumes average density of 3000 kg/m³
            volume = (4/3) * 3.14159 * (diameter_km / 2)**3
            mass = volume * 3000 * 10**9  # in kg
            energy = 0.5 * mass * (velocity_kms**2) / (4.184 * 10**15)  # Convert to megatons TNT
            
            return round(energy, 2)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error calculating impact energy: {e}")
            return None
    
    def validate_asteroid_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate asteroid data from the API.
        
        Args:
            data: Asteroid data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check required fields
        required_fields = ["id", "name", "neo_reference_id"]
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return False
                
        # Validate close approach data if present
        if "close_approach_data" in data:
            if not isinstance(data["close_approach_data"], list):
                logger.warning("close_approach_data is not a list")
                return False
                
            for approach in data["close_approach_data"]:
                required_approach_fields = ["close_approach_date", "miss_distance", "relative_velocity"]
                for field in required_approach_fields:
                    if field not in approach:
                        logger.warning(f"Missing required approach field: {field}")
                        return False
        
        return True
    
    def get_current_threats(
        self, 
        days_ahead: int = 7,
        minimum_risk: str = "Moderate",
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get asteroids that pose a significant threat in the coming days.
        
        Args:
            days_ahead: Number of days to look ahead
            minimum_risk: Minimum risk level to include
            use_cache: Whether to use cached data if available
            
        Returns:
            List of threatening asteroids with analysis
        """
        # Define risk levels in order from highest to lowest
        risk_levels = ["Extreme", "High", "Moderate", "Low", "Very Low"]
        
        # Validate the minimum risk level
        if minimum_risk not in risk_levels:
            raise ValueError(f"Invalid risk level. Must be one of: {', '.join(risk_levels)}")
            
        # Get minimum risk index (higher index = lower risk)
        min_risk_index = risk_levels.index(minimum_risk)
        
        # Get hazardous asteroids
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_ahead)
        hazardous = self.get_hazardous_asteroids(start_date, end_date, use_cache)
        
        threats = []
        
        # Analyze each hazardous asteroid
        for asteroid in hazardous:
            asteroid_id = asteroid.get("neo_reference_id")
            if not asteroid_id:
                continue
                
            # Get detailed data
            try:
                detailed_data = self.get_neo_by_id(asteroid_id, use_cache)
                approaches = self.analyze_close_approaches(detailed_data)
                
                # Filter approaches by date and risk level
                for approach in approaches:
                    approach_date = datetime.strptime(approach["date"], "%Y-%m-%d")
                    risk = approach.get("risk_assessment", "Unknown")
                    
                    # Check if the approach is within our time range
                    if (start_date.date() <= approach_date.date() <= end_date.date() and
                        risk in risk_levels[:min_risk_index+1]):
                        
                        # Add asteroid info to the threat
                        threat = {
                            "id": asteroid_id,
                            "name": detailed_data.get("name", "Unknown"),
                            "diameter_km": detailed_data.get("estimated_diameter", {})
                                .get("kilometers", {}).get("estimated_diameter_max"),
                            "hazardous": detailed_data.get("is_potentially_hazardous_asteroid"),
                            "approach": approach
                        }
                        
                        threats.append(threat)
                        
            except NEOAPIError as e:
                logger.warning(f"Error retrieving detailed data for asteroid {asteroid_id}: {e}")
                
        # Sort threats by risk level (highest first) and then by date
        threats.sort(
            key=lambda x: (
                risk_levels.index(x["approach"]["risk_assessment"]) 
                if x["approach"]["risk_assessment"] in risk_levels else len(risk_levels),
                x["approach"]["date"]
            )
        )
        
        return threats
    
    def generate_threat_report(self, days_ahead: int = 30) -> Dict[str, Any]:
        """
        Generate a comprehensive threat report for upcoming asteroid approaches.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            Report dictionary with threat statistics and details
        """
        # Get all levels of threats
        all_threats = self.get_current_threats(days_ahead, "Very Low")
        
        # Count threats by risk level
        risk_counts = {"Extreme": 0, "High": 0, "Moderate": 0, "Low": 0, "Very Low": 0}
        
        for threat in all_threats:
            risk = threat["approach"]["risk_assessment"]
            if risk in risk_counts:
                risk_counts[risk] += 1
                
        # Find closest approach
        closest_approach = None
        min_distance = float('inf')
        
        for threat in all_threats:
            distance = threat["approach"].get("distance_km")
            if distance is not None and distance < min_distance:
                min_distance = distance
                closest_approach = threat
                
        # Group threats by month
        monthly_distribution = {}
        
        for threat in all_threats:
            date_str = threat["approach"]["date"]
            month = date_str[:7]  # YYYY-MM format
            
            if month not in monthly_distribution:
                monthly_distribution[month] = 0
                
            monthly_distribution[month] += 1
            
        # Generate the report
        return {
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "days_analyzed": days_ahead,
            "total_threats": len(all_threats),
            "risk_distribution": risk_counts,
            "monthly_distribution": monthly_distribution,
            "closest_approach": closest_approach,
            "highest_risks": [t for t in all_threats if t["approach"]["risk_assessment"] in ["Extreme", "High"]][:5],
            "all_threats": all_threats
        }
