    def on_time_changed(self, new_time):
        """Handle changes to the selected time."""
        self.current_time = new_time.toPyDateTime()
        self.update_data()
    
    def adjust_time(self, days=0, hours=0, minutes=0):
        """Adjust the current time by the specified amount."""
        delta = timedelta(days=days, hours=hours, minutes=minutes)
        self.current_time += delta
        self.date_time_edit.setDateTime(self.current_time)
        self.update_data()
    
    def toggle_real_time_updates(self, enabled):
        """Toggle real-time updates of planetary positions."""
        if enabled:
            self.update_timer.start(1000)  # Update every second
            self.date_time_edit.setEnabled(False)
            self.btn_prev_day.setEnabled(False)
            self.btn_next_day.setEnabled(False)
        else:
            self.update_timer.stop()
            self.date_time_edit.setEnabled(True)
            self.btn_prev_day.setEnabled(True)
            self.btn_next_day.setEnabled(True)
    
    def set_simulation_speed(self, speed):
        """Set the simulation speed multiplier."""
        # Speed is a value between 1 and 100
        self.simulation_speed = speed
        if self.update_timer.isActive():
            self.update_timer.setInterval(1000 // speed)
    
    def update_real_time(self):
        """Update the view in real-time mode."""
        time_delta = timedelta(seconds=self.simulation_speed)
        self.current_time += time_delta
        self.date_time_edit.setDateTime(self.current_time)
        self.update_data()
    
    def on_planet_selected(self, planet_name):
        """Handle planet selection from the dropdown."""
        if planet_name == "All Planets":
            # Reset view to show all planets
            self.plot_widget.setRange(xRange=(-10, 10), yRange=(-10, 10), padding=0.1)
            if hasattr(self, 'view_3d_widget'):
                self.view_3d_widget.setCameraPosition(distance=20)
        else:
            # Focus on selected planet
            self.planet_selected.emit(planet_name)
            self._focus_on_planet(planet_name)
    
    def _focus_on_planet(self, planet_name):
        """Focus the view on a specific planet."""
        planet_data = self.planet_data.get(planet_name.lower())
        if not planet_data:
            return
            
        # Update status panel with planet info
        self.selected_planet_label.setText(f"Selected Planet: {planet_name}")
        self.planet_distance_label.setText(
            f"Distance from Sun: {planet_data.distance_from_sun:.2f} AU"
        )
        self.planet_position_label.setText(
            f"Position: ({planet_data.position.x:.2f}, {planet_data.position.y:.2f}) AU"
        )
        self.planet_velocity_label.setText(
            f"Orbital Velocity: {planet_data.orbital_velocity:.2f} km/s"
        )
        
        # Focus 2D view
        x, y = planet_data.position.x, planet_data.position.y
        self.plot_widget.setRange(
            xRange=(x-2, x+2),
            yRange=(y-2, y+2),
            padding=0.1
        )
        
        # Focus 3D view if available
        if hasattr(self, 'view_3d_widget'):
            z = getattr(planet_data.position, 'z', 0)
            dist = (x**2 + y**2 + z**2)**0.5
            self.view_3d_widget.setCameraPosition(
                pos=QtGui.QVector3D(x, y, z),
                distance=dist * 2
            )
    
    def change_view_perspective(self, view_type):
        """Change the view perspective in 3D mode."""
        if not hasattr(self, 'view_3d_widget'):
            return
            
        if view_type == "Top View":
            self.view_3d_widget.setCameraPosition(azimuth=0, elevation=90)
        elif view_type == "Side View":
            self.view_3d_widget.setCameraPosition(azimuth=0, elevation=0)
        elif view_type == "Front View":
            self.view_3d_widget.setCameraPosition(azimuth=90, elevation=0)
        # Free View is handled by user interaction
    
    def toggle_orbit_display(self, show):
        """Toggle the display of orbital paths."""
        # Update 2D orbits
        for orbit in self.orbit_plots.values():
            orbit.setVisible(show)
        
        # Update 3D orbits
        for orbit in self.orbit_paths_3d.values():
            orbit.setVisible(show)
        
        self.update_data()
    
    def toggle_labels(self, show):
        """Toggle the display of planet labels."""
        for key, item in self.planet_plots.items():
            if "_label" in key:
                item.setVisible(show)
    
    def toggle_distances(self, show):
        """Toggle the display of distance indicators."""
        # Implementation depends on how you want to show distances
        # Could be lines, text, or both
        self.update_data()
    
    def closeEvent(self, event):
        """Handle cleanup when the widget is closed."""
        self.update_timer.stop()
        super().closeEvent(event)