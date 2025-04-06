def _load_data_settings(self):
            return {
                "planet_update_interval": self.settings.value("data/planet_update_interval", 5, type=int),
                "asteroid_update_interval": self.settings.value("data/asteroid_update_interval", 60, type=int),
                "cache_directory": self.settings.value("data/cache_directory", ""),
                "cache_size_limit": self.settings.value("data/cache_size_limit", 500, type=int)
            }
        
def _load_notification_settings(self):
            """Load notification settings from QSettings with defaults."""
            default_settings = {
                "enabled": True,
                "types": {
                    "asteroid_alerts": True,
                    "planetary_events": True,
                    "space_weather": True,
                    "system": True
                },
                "methods": {
                    "desktop": True,
                    "email": False,
                    "email_address": ""
                },
                "thresholds": {
                    "neo_proximity": 5,
                    "solar_activity": "Medium (M-class)"
                }
            }
            
            settings_str = self.settings.value("notifications", "")
            if not settings_str:
                return default_settings
                
            try:
                return json.loads(settings_str)
            except json.JSONDecodeError:
                return default_settings
        
def showEvent(self, event):
            """Override showEvent to center the dialog on the parent window."""
            if self.parent():
                parent_geo = self.parent().geometry()
                self.move(
                    parent_geo.center().x() - self.width() // 2,
                    parent_geo.center().y() - self.height() // 2
                )
            super().showEvent(event)