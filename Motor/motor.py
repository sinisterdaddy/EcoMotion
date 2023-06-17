import math


class EnergyCalculator:
    def __init__(self):
        self.vehicle_mass = 2000  # kg
        self.mass_factor = 1.05
        self.acceleration = 0  # m^2 / s
        self.rolling_resistance_coefficient = 0.02  # coefficient of rolling resistance
        self.air_density = 1.225  # kg/m^3
        self.frontal_area = 2  # m^2
        self.aerodynamic_drag_coefficient = 0.5
        self.wind_speed = 0  # m/s
        self.road_angle = 0  # degrees

    def calculate_energy_consumption(self, road_angle, driving_speed):
        self.road_angle = road_angle
        rad_angle = math.radians(road_angle)
        rolling_resistance_power = (
            self.mass_factor * self.vehicle_mass * self.acceleration) +(
            self.vehicle_mass * 9.8 *
            self.rolling_resistance_coefficient * math.cos(rad_angle)
        )
        aerodynamic_drag_power = (
            0.5 * self.air_density * self.frontal_area * self.aerodynamic_drag_coefficient *
            (driving_speed - self.wind_speed) ** 2
        )
        gravitational_power = self.vehicle_mass * 9.8 * math.sin(rad_angle)
        total_power = (rolling_resistance_power +
                       aerodynamic_drag_power + gravitational_power) * driving_speed
        return total_power
