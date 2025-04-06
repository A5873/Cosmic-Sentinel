import os
import json
import logging
import base64
import hashlib
from typing import Any, Dict, Optional, Union
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from PyQt6.QtCore import QSettings, QSize, QPoint, QByteArray
from PyQt6.QtGui import QColor, QFont


class ConfigError(Exception):
    """Base exception for all configuration-related errors."""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""
    pass


class ConfigEncryptionError(ConfigError):
    """Exception raised when encryption/decryption operations fail."""
    pass


class ConfigFileError(ConfigError):
    """Exception raised when configuration file operations fail."""
    pass


class SettingsManager:
    """
    Manages application settings, configurations, and secure API key storage.
    
    This class provides functionality for:
    - Loading and saving application configurations
    - Securely storing and retrieving API keys
    - Managing UI preferences
    - Providing default configurations
    """
    
    # Default configurations
    DEFAULT_CONFIG = {
        "api": {
            "nasa_api_url": "https://api.nasa.gov",
            "skyfield_data_dir": "data/skyfield",
            "update_interval": 3600,  # seconds
            "cache_duration": 86400,  # seconds (24 hours)
        },
        "ui": {
            "theme": "dark",
            "font_size": 10,
            "accent_color": "#2980b9",
            "refresh_rate": 5,  # seconds
            "show_welcome_screen": True,
            "default_view": "planetary",
            "enable_animations": True,
            "data_update_notifications": True,
        },
        "planetary": {
            "default_bodies": ["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"],
            "show_dwarf_planets": False,
            "show_orbits": True,
            "orbit_segments": 100,
            "time_span": 365,  # days
        },
        "asteroid": {
            "default_min_diameter": 0.5,  # kilometers
            "default_max_diameter": 50,  # kilometers
            "hazard_threshold": 0.05,  # NEO hazard score
            "days_to_monitor": 30,
            "auto_refresh": True,
            "refresh_interval": 3600,  # seconds
        },
        "space_weather": {
            "monitoring_interval": 3600,  # seconds
            "auto_start_monitoring": False,
            "cache_expiration": 1,  # hours
            "solar_flare_alerts": True,
            "cme_alerts": True,
            "geomagnetic_storm_alerts": True,
            "aurora_alerts": True,
            "minimum_alert_severity": "MODERATE",  # LOW, MODERATE, HIGH, EXTREME
            "show_space_weather_status_in_statusbar": True,
            "update_interval": 900,  # seconds
            "aurora_view_hemisphere": "northern",  # northern, southern
            "fetch_solar_flares": True,
            "fetch_cmes": True,
            "fetch_geomagnetic_storms": True,
            "fetch_aurora_forecast": True,
            "fetch_solar_conditions": True,
            "fetch_geomagnetic_conditions": True,
        },
        "report": {
            "auto_generate": False,
            "report_dir": "reports/",
            "include_images": True,
            "image_format": "png",
            "image_dpi": 300,
        },
        "logging": {
            "level": "INFO",
            "log_to_file": True,
            "log_dir": "logs/",
            "max_log_size": 10485760,  # 10 MB
            "backup_count": 5,
        }
    }
    
    def __init__(self, app_name: str = "CosmicSentinel", organization: str = "AstroTech"):
        """
        Initialize the settings manager.
        
        Args:
            app_name: Name of the application
            organization: Organization name
        """
        self.app_name = app_name
        self.organization = organization
        self.qsettings = QSettings(organization, app_name)
        self.config_dir = os.path.join(os.path.expanduser("~"), f".{app_name.lower()}")
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.api_key_file = os.path.join(self.config_dir, "api_keys.enc")
        self.encryption_salt_file = os.path.join(self.config_dir, "salt")
        
        # Ensure config directory exists
        if not os.path.exists(self.config_dir):
            try:
                os.makedirs(self.config_dir)
            except OSError as e:
                logging.error(f"Failed to create config directory: {e}")
                raise ConfigFileError(f"Failed to create config directory: {e}")
        
        # Load or create configuration
        self.config = self._load_config()
        
        # Initialize encryption
        self._init_encryption()
        
    def _init_encryption(self) -> None:
        """Initialize encryption for secure storage of API keys."""
        try:
            # Check if salt exists, if not, create it
            if not os.path.exists(self.encryption_salt_file):
                salt = os.urandom(16)
                with open(self.encryption_salt_file, 'wb') as f:
                    f.write(salt)
            else:
                with open(self.encryption_salt_file, 'rb') as f:
                    salt = f.read()
            
            # Generate a machine-specific key based on hardware and OS information
            # This approach means the encryption is tied to this machine
            machine_id = self._get_machine_id()
            
            # Use PBKDF2 to derive a key from the machine ID and salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(machine_id.encode()))
            self.cipher = Fernet(key)
            
        except Exception as e:
            logging.error(f"Failed to initialize encryption: {e}")
            raise ConfigEncryptionError(f"Failed to initialize encryption: {e}")
    
    def _get_machine_id(self) -> str:
        """
        Get a unique machine identifier for encryption.
        
        This creates a reasonably unique ID based on available system information.
        """
        try:
            # Try to use more system-specific identifiers based on OS
            if os.name == 'nt':  # Windows
                import subprocess
                result = subprocess.check_output('wmic csproduct get uuid').decode()
                return result.split('\n')[1].strip()
            
            elif os.name == 'posix':  # Linux/Mac
                # Try to use machine-id from /etc/machine-id or /var/lib/dbus/machine-id
                if os.path.exists('/etc/machine-id'):
                    with open('/etc/machine-id', 'r') as f:
                        return f.read().strip()
                elif os.path.exists('/var/lib/dbus/machine-id'):
                    with open('/var/lib/dbus/machine-id', 'r') as f:
                        return f.read().strip()
                # Fall back to using the MAC address of the first network interface
                import uuid
                return str(uuid.getnode())
                
        except Exception as e:
            logging.warning(f"Failed to get system-specific machine ID: {e}")
        
        # Fallback: Use a combination of username and hostname
        import getpass
        import socket
        seed = f"{getpass.getuser()}@{socket.gethostname()}"
        
        # Use a consistent hash of the seed
        hash_obj = hashlib.sha256(seed.encode())
        return hash_obj.hexdigest()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default if it doesn't exist.
        
        Returns:
            Dict containing the configuration
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Update config with any new default keys that might not be in the loaded config
                config = self.DEFAULT_CONFIG.copy()
                self._update_dict_recursive(config, loaded_config)
                return config
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse config file: {e}")
                # Create a backup of the corrupted file
                if os.path.exists(self.config_file):
                    backup_file = f"{self.config_file}.backup"
                    try:
                        import shutil
                        shutil.copy2(self.config_file, backup_file)
                        logging.info(f"Backup of corrupted config created: {backup_file}")
                    except Exception as backup_err:
                        logging.error(f"Failed to create backup of corrupted config: {backup_err}")
                
                return self.DEFAULT_CONFIG.copy()
            except Exception as e:
                logging.error(f"Failed to load config file: {e}")
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config
            return self.DEFAULT_CONFIG.copy()
    
    def _update_dict_recursive(self, target: Dict, source: Dict) -> None:
        """
        Update a dictionary recursively, preserving nested structures.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with values to apply
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict_recursive(target[key], value)
            else:
                target[key] = value
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Failed to save config file: {e}")
            raise ConfigFileError(f"Failed to save configuration: {e}")
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        self.config = self.DEFAULT_CONFIG.copy()
        try:
            self.save_config()
            # Clear QSettings as well
            self.qsettings.clear()
        except Exception as e:
            logging.error(f"Failed to reset configuration: {e}")
            raise ConfigError(f"Failed to reset configuration: {e}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default if not found
        """
        try:
            return self.config[section][key]
        except KeyError:
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        # Ensure section exists
        if section not in self.config:
            self.config[section] = {}
        
        # Validate the value based on section and key
        self._validate_config_value(section, key, value)
        
        # Set the value
        self.config[section][key] = value
    
    def _validate_config_value(self, section: str, key: str, value: Any) -> None:
        """
        Validate a configuration value based on its section and key.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Define validation rules for specific sections and keys
        validation_rules = {
            "api": {
                "update_interval": lambda x: isinstance(x, int) and x > 0,
                "cache_duration": lambda x: isinstance(x, int) and x > 0,
            },
            "ui": {
                "font_size": lambda x: isinstance(x, int) and 6 <= x <= 24,
                "refresh_rate": lambda x: isinstance(x, (int, float)) and x > 0,
                "theme": lambda x: x in ["light", "dark", "system"],
            },
            "planetary": {
                "orbit_segments": lambda x: isinstance(x, int) and 10 <= x <= 1000,
                "time_span": lambda x: isinstance(x, int) and 1 <= x <= 36500,
            },
            "asteroid": {
                "default_min_diameter": lambda x: isinstance(x, (int, float)) and x >= 0,
                "default_max_diameter": lambda x: isinstance(x, (int, float)) and x > 0,
                "hazard_threshold": lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
                "days_to_monitor": lambda x: isinstance(x, int) and 1 <= x <= 365,
                "refresh_interval": lambda x: isinstance(x, int) and x >= 60,
            },
            "space_weather": {
                "monitoring_interval": lambda x: isinstance(x, int) and x >= 60,
                "update_interval": lambda x: isinstance(x, int) and x >= 60,
                "cache_expiration": lambda x: isinstance(x, (int, float)) and x > 0,
                "minimum_alert_severity": lambda x: x in ["LOW", "MODERATE", "HIGH", "EXTREME"],
                "aurora_view_hemisphere": lambda x: x in ["northern", "southern"],
            },
            "logging": {
                "level": lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "max_log_size": lambda x: isinstance(x, int) and x > 0,
                "backup_count": lambda x: isinstance(x, int) and x >= 0,
            }
        }
        
        # Check if we have a validation rule for this section and key
        if section in validation_rules and key in validation_rules[section]:
            rule = validation_rules[section][key]
            if not rule(value):
                raise ConfigValidationError(
                    f"Invalid value '{value}' for configuration [{section}].{key}"
                )
    
    def store_api_key(self, service_name: str, api_key: str) -> bool:
        """
        Securely store an API key for a service.
        
        Args:
            service_name: Name of the service (e.g., 'nasa', 'neows')
            api_key: API key to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing keys if any
            api_keys = self._load_api_keys()
            
            # Update/add the key
            api_keys[service_name] = api_key
            
            # Write back encrypted data
            self._save_api_keys(api_keys)
            return True
            
        except Exception as e:
            logging.error(f"Failed to store API key: {e}")
            raise ConfigEncryptionError(f"Failed to store API key: {e}")
    
    def get_api_key(self, service_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an API key for a service.
        
        Args:
            service_name: Name of the service
            default: Default value if key not found
            
        Returns:
            API key if found, default otherwise
        """
        try:
            api_keys = self._load_api_keys()
            return api_keys.get(service_name, default)
        except Exception as e:
            logging.error(f"Failed to retrieve API key: {e}")
            return default
    
    def _load_api_keys(self) -> Dict[str, str]:
        """
        Load encrypted API keys from file.
        
        Returns:
            Dictionary of service names to API keys
            
        Raises:
            ConfigEncryptionError: If decryption fails
        """
        if not os.path.exists(self.api_key_file):
            return {}
            
        try:
            with open(self.api_key_file, 'rb') as f:
                encrypted_data = f.read()
                
            if not encrypted_data:
                return {}
                
            # Decrypt the data
            decrypted_data = self.cipher.decrypt(encrypted_data)
            api_keys = json.loads(decrypted_data.decode('utf-8'))
            return api_keys
            
        except InvalidToken:
            logging.error("Failed to decrypt API keys: Invalid token")
            raise ConfigEncryptionError("Failed to decrypt API keys: Invalid token or corrupted data")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse API keys JSON: {e}")
            raise ConfigEncryptionError(f"Failed to parse API keys: {e}")
        except Exception as e:
            logging.error(f"Failed to load API keys: {e}")
            raise ConfigEncryptionError(f"Failed to load API keys: {e}")
    
    def _save_api_keys(self, api_keys: Dict[str, str]) -> None:
        """
        Save API keys to encrypted file.
        
        Args:
            api_keys: Dictionary of service names to API keys
            
        Raises:
            ConfigEncryptionError: If encryption fails
        """
        try:
            # Convert the dictionary to JSON
            json_data = json.dumps(api_keys).encode('utf-8')
            
            # Encrypt the data
            encrypted_data = self.cipher.encrypt(json_data)
            
            # Write to file
            with open(self.api_key_file, 'wb') as f:
                f.write(encrypted_data)
                
        except Exception as e:
            logging.error(f"Failed to save API keys: {e}")
            raise ConfigEncryptionError(f"Failed to save API keys: {e}")
    
    def delete_api_key(self, service_name: str) -> bool:
        """
        Delete an API key for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if successful, False if key not found
        """
        try:
            api_keys = self._load_api_keys()
            
            if service_name not in api_keys:
                return False
                
            del api_keys[service_name]
            self._save_api_keys(api_keys)
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete API key: {e}")
            raise ConfigEncryptionError(f"Failed to delete API key: {e}")
    
    # UI Theme Management
    def get_theme(self) -> str:
        """
        Get the current UI theme.
        
        Returns:
            Theme name: 'light', 'dark', or 'system'
        """
        return self.get("ui", "theme", "dark")
    
    def set_theme(self, theme: str) -> None:
        """
        Set the UI theme.
        
        Args:
            theme: Theme name ('light', 'dark', or 'system')
            
        Raises:
            ConfigValidationError: If theme is invalid
        """
        valid_themes = ["light", "dark", "system"]
        if theme not in valid_themes:
            raise ConfigValidationError(f"Invalid theme: {theme}. Must be one of {valid_themes}")
            
        self.set("ui", "theme", theme)
        self.save_config()
    
    def get_theme_colors(self) -> Dict[str, QColor]:
        """
        Get the color palette for the current theme.
        
        Returns:
            Dictionary of color names to QColor objects
        """
        theme = self.get_theme()
        
        # Define color palettes for different themes
        themes = {
            "light": {
                "background": QColor("#f5f5f5"),
                "foreground": QColor("#333333"),
                "primary": QColor(self.get("ui", "accent_color", "#2980b9")),
                "secondary": QColor("#27ae60"),
                "warning": QColor("#f39c12"),
                "danger": QColor("#e74c3c"),
                "info": QColor("#3498db"),
                "success": QColor("#2ecc71"),
                "panel": QColor("#ffffff"),
                "border": QColor("#dddddd"),
                "highlight": QColor("#f0f0f0"),
                "text_highlight": QColor("#2980b9"),
                "disabled": QColor("#aaaaaa"),
            },
            "dark": {
                "background": QColor("#121212"),
                "foreground": QColor("#e0e0e0"),
                "primary": QColor(self.get("ui", "accent_color", "#2980b9")),
                "secondary": QColor("#2ecc71"),
                "warning": QColor("#f39c12"),
                "danger": QColor("#e74c3c"),
                "info": QColor("#3498db"),
                "success": QColor("#2ecc71"),
                "panel": QColor("#1e1e1e"),
                "border": QColor("#333333"),
                "highlight": QColor("#2d2d2d"),
                "text_highlight": QColor("#3498db"),
                "disabled": QColor("#666666"),
            },
            "system": {}  # Will be determined at runtime based on system settings
        }
        
        # If system theme, determine based on system settings
        if theme == "system":
            import darkdetect
            system_theme = "dark" if darkdetect.isDark() else "light"
            return themes[system_theme]
            
        return themes[theme]
    
    def get_theme_fonts(self) -> Dict[str, QFont]:
        """
        Get the font settings for the current theme.
        
        Returns:
            Dictionary of font roles to QFont objects
        """
        font_size = self.get("ui", "font_size", 10)
        
        # Create base font
        base_font = QFont("Segoe UI", font_size)
        
        # Create fonts for different roles
        heading_font = QFont(base_font)
        heading_font.setBold(True)
        heading_font.setPointSize(int(font_size * 1.5))
        
        subtitle_font = QFont(base_font)
        subtitle_font.setBold(True)
        subtitle_font.setPointSize(int(font_size * 1.2))
        
        monospace_font = QFont("Consolas", font_size)
        
        small_font = QFont(base_font)
        small_font.setPointSize(int(font_size * 0.9))
        
        return {
            "base": base_font,
            "heading": heading_font,
            "subtitle": subtitle_font,
            "monospace": monospace_font,
            "small": small_font,
        }
    
    # Window state management
    def save_window_state(self, window_name: str, geometry: QByteArray, state: QByteArray) -> None:
        """
        Save window geometry and state.
        
        Args:
            window_name: Name of the window
            geometry: Window geometry as QByteArray
            state: Window state as QByteArray
        """
        self.qsettings.beginGroup(f"WindowState/{window_name}")
        self.qsettings.setValue("geometry", geometry)
        self.qsettings.setValue("state", state)
        self.qsettings.endGroup()
    
    def load_window_state(self, window_name: str) -> tuple[Optional[QByteArray], Optional[QByteArray]]:
        """
        Load window geometry and state.
        
        Args:
            window_name: Name of the window
            
        Returns:
            Tuple of (geometry, state) as QByteArrays, or (None, None) if not found
        """
        self.qsettings.beginGroup(f"WindowState/{window_name}")
        geometry = self.qsettings.value("geometry")
        state = self.qsettings.value("state")
        self.qsettings.endGroup()
        
        return geometry, state
    
    # Cache management
    def get_cache_path(self, service: str = "") -> str:
        """
        Get the path to the cache directory.
        
        Args:
            service: Optional service name to get service-specific cache
            
        Returns:
            Path to cache directory
        """
        cache_dir = os.path.join(self.config_dir, "cache")
        
        if service:
            cache_dir = os.path.join(cache_dir, service)
            
        # Ensure directory exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        return cache_dir
    
    def clear_recent_files(self) -> None:
        """Clear the recent files list."""
        self.qsettings.setValue("RecentFiles", [])
        
    # Space Weather settings methods
    
    def get_space_weather_settings(self) -> Dict[str, Any]:
        """
        Get all space weather settings.
        
        Returns:
            Dictionary of space weather settings
        """
        return self.config.get("space_weather", {})
    
    def get_space_weather_monitor_interval(self) -> int:
        """
        Get the space weather monitoring interval.
        
        Returns:
            Monitoring interval in seconds
        """
        return int(self.get("space_weather", "monitoring_interval", 3600))
    
    def get_space_weather_update_interval(self) -> int:
        """
        Get the space weather UI update interval.
        
        Returns:
            Update interval in seconds
        """
        return int(self.get("space_weather", "update_interval", 900))
    
    def is_space_weather_alert_enabled(self, alert_type: str) -> bool:
        """
        Check if a specific space weather alert type is enabled.
        
        Args:
            alert_type: Type of alert (solar_flare, cme, geomagnetic_storm, aurora)
            
        Returns:
            True if enabled, False otherwise
        """
        setting_key = f"{alert_type}_alerts"
        return bool(self.get("space_weather", setting_key, True))
    
    def get_minimum_alert_severity(self) -> str:
        """
        Get the minimum severity level for space weather alerts.
        
        Returns:
            Minimum severity level (LOW, MODERATE, HIGH, EXTREME)
        """
        return self.get("space_weather", "minimum_alert_severity", "MODERATE")
    
    def should_fetch_weather_data_type(self, data_type: str) -> bool:
        """
        Check if a specific space weather data type should be fetched.
        
        Args:
            data_type: Type of data (solar_flares, cmes, geomagnetic_storms, etc.)
            
        Returns:
            True if should fetch, False otherwise
        """
        setting_key = f"fetch_{data_type}"
        return bool(self.get("space_weather", setting_key, True))
    
    def get_aurora_view_hemisphere(self) -> str:
        """
        Get the preferred hemisphere for aurora views.
        
        Returns:
            Hemisphere name ("northern" or "southern")
        """
        return self.get("space_weather", "aurora_view_hemisphere", "northern")
    
    def should_auto_start_weather_monitoring(self) -> bool:
        """
        Check if space weather monitoring should start automatically.
        
        Returns:
            True if should auto-start, False otherwise
        """
        return bool(self.get("space_weather", "auto_start_monitoring", False))
    
    def should_show_space_weather_in_statusbar(self) -> bool:
        """
        Check if space weather status should be shown in the status bar.
        
        Returns:
            True if should show in status bar, False otherwise
        """
        return bool(self.get("space_weather", "show_space_weather_status_in_statusbar", True))
    
    def get_space_weather_cache_expiration(self) -> float:
        """
        Get the cache expiration time for space weather data.
        
        Returns:
            Cache expiration time in hours
        """
        return float(self.get("space_weather", "cache_expiration", 1.0))
        
        Args:
            service: Optional service name to clear only that service's cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_dir = self.get_cache_path()
            
            if service:
                # Clear only service-specific cache
                service_cache = os.path.join(cache_dir, service)
                if os.path.exists(service_cache):
                    import shutil
                    shutil.rmtree(service_cache)
                    os.makedirs(service_cache, exist_ok=True)
            else:
                # Clear all cache
                if os.path.exists(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    
            return True
            
        except Exception as e:
            logging.error(f"Failed to clear cache: {e}")
            return False
    
    # Recent file management
    def add_recent_file(self, file_path: str) -> None:
        """
        Add a file to the recent files list.
        
        Args:
            file_path: Path to the file
        """
        recent_files = self.get_recent_files()
        
        # Remove if already exists (to move it to the top)
        if file_path in recent_files:
            recent_files.remove(file_path)
            
        # Add to the beginning
        recent_files.insert(0, file_path)
        
        # Limit to 10 recent files
        recent_files = recent_files[:10]
        
        # Save the list
        self.qsettings.setValue("RecentFiles", recent_files)
    
    def get_recent_files(self) -> list[str]:
        """
        Get the list of recent files.
        
        Returns:
            List of file paths
        """
        return self.qsettings.value("RecentFiles", [])
    
    def clear_recent_files(self) -> None:
        """Clear the recent files list."""
        self.qsettings.setValue("RecentFiles", [])
