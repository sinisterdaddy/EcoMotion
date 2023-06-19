# import requests
# import urllib.parse
# class enviornment:
#     def __init__(self, start_position, end_position):
#         self.start_position = start_position
#         self.end_position = end_position
#         self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
#                            (1, 1), (-1, -1), (1, -1), (-1, 1)]
#         # Add any additional variables or initialization code as needed

#     # Add methods/functions to handle movement and exploration in different directions
#     def move(self, direction):
#         # Implement the logic for moving in a specific direction

#     def explore(self):
#         # Implement the logic for exploring in different directions

#         # Rest of the class code...

from datetime import datetime
import urllib.parse
import requests
from haversine import haversine
from Battery import LithiumIonBattery
from motor import EnergyCalculator
import math

import urllib.parse
import requests


class Environment:
    def __init__(self, origin_adr, destination_adr):
        self.origin = origin_adr
        self.destination = destination_adr
        self.latt = 0
        self.lngg = 0
        self.make_map()
        self.battery = LithiumIonBattery(50000)  # Wh
        self.EnergyCalculator = EnergyCalculator()
        self.charge_num = 0
        self.unreach_position_num = 0
        self.time = 0
        self.ii = 0
        self.status_dir_check = 0
        self.length = 1
        # self.s = requests.Session()
        self.env_height_km = 1

    def geocoding_api(self, address):
        # address: keyword of place
        geocode_api = 'https://maps.googleapis.com/maps/api/geocode/json?'
        geocode_url = geocode_api + \
            urllib.parse.urlencode({'address': address}) + \
            "&key=AZuesYuds12_dsakd23456sdeHf"
        # geocode_json = requests.get(geocode_url, timeout=10).json()
        session = requests.Session()
        geocode_json = session.get(geocode_url).json()
        self.geocode_json_status = geocode_json['status']
        self.geoposition_tuple = ('g', 'g')
        if self.geocode_json_status == 'OK':
            latt = 0
            lngg = 0
            geocode_data_results = geocode_json['results'][0]
            geocode_data_results_geometry = geocode_data_results['geometry']
            latt = geocode_data_results_geometry['location']['lat']
            lngg = geocode_data_results_geometry['location']['lng']
            self.latt = latt
            self.lngg = lngg
            geocode_data_results_formatted_address = geocode_data_results['formatted_address']
            geocode_data_results_types = geocode_data_results['types']
            geocode_data_results_place_id = geocode_data_results['place_id']
            self.geoposition = str(latt) + ',' + str(lngg)
            self.geoposition_tuple = (latt, lngg)
        else:
            self.geoposition = 'N/A'
        return self.geocode_json_status, self.geoposition, self.geoposition_tuple
    
    def elevation_api(self, location):
        # location = '51.4700223,-0.4542955'
        elevation_api = 'https://maps.googleapis.com/maps/api/elevation/json?'
        elevation_url = elevation_api + urllib.parse.urlencode({'locations': location}) + "&key=AZuesYuds12_dsakd23456sdeHf"
        # elevation_json = requests.get(elevation_url, timeout=10).json()
        session = requests.Session()
        elevation_json = session.get(elevation_url).json()
        self.elevation_json_status = elevation_json['status']
        if self.elevation_json_status == 'OK':
            elevation_data_results = elevation_json['results'][0]
            self.elevation_data_elevation = elevation_data_results['elevation']
            elevation_data_resolution = elevation_data_results['resolution']
        else:
            self.elevation_data_elevation = 'N/A'
            elevation_data_resolution = 'N/A'
        return self.elevation_json_status, self.elevation_data_elevation
    
    def directions_api(self, origin, destination):
        directions_api = 'https://maps.googleapis.com/maps/api/directions/json?'
        directions_url = directions_api + urllib.parse.urlencode({'origin': origin}) + '&' + urllib.parse.urlencode(
            {'destination': destination}) + '&' + urllib.parse.urlencode({'units': 'metric'}) + "&key=AZuesYuds12_dsakd23456sdeHf"
        # directions_json = requests.get(directions_url, timeout=10).json()
        session = requests.Session()
        directions_json = session.get(directions_url).json()
        self.directions_json_status = directions_json['status']
        if self.directions_json_status == 'OK':
            directions_data_routes = directions_json['routes'][0]
            directions_data_routes_bounds = directions_data_routes['bounds']  # all are 'southwest', 'northeast'
            directions_data_routes_legs = directions_data_routes['legs']
            self.directions_data_legs_steps = directions_data_routes_legs[0]['steps']
            #### process boundary ###
            north = directions_data_routes_bounds['northeast']['lat']
            east = directions_data_routes_bounds['northeast']['lng']
            south = directions_data_routes_bounds['southwest']['lat']
            west = directions_data_routes_bounds['southwest']['lng']
            self.bound = {'north': north, 'east': east, 'south': south, 'west': west}  # value of lat/lng
        else:
            self.directions_data_legs_steps = 'N/A'
            self.bound = 'N/A'
        return self.directions_json_status, self.directions_data_legs_steps, self.bound
    
    def make_map(self):
        origin_status, origin_position, origin_position_num = self.geocoding_api(self.origin)
        if origin_position_num == ('g', 'g'):
            origin_position_num = (self.latt, self.lngg)
        destination_status, destination_position, destination_position_num = self.geocoding_api(self.destination)
        if destination_position_num == ('g', 'g'):
            destination_position_num = (self.latt, self.lngg)
        direction_status, direction_steps, self.map_bound = self.directions_api(origin_position, destination_position)
        self.Google_steps = direction_steps
        while direction_status != 'OK':
            direction_status, direction_steps, self.map_bound = self.directions_api(origin_position, destination_position)
        self.current_position = origin_position_num  # position tuple (lat,lng)
        self.start_position = origin_position_num  # position tuple (lat,lng)
        self.end_position = destination_position_num  # position tuple (lat,lng)
        
    def stride_length(self, position):
        start_lat = self.start_position[0]
        start_lng = self.start_position[1]
        end_lat = self.end_position[0]
        end_lng = self.end_position[1]
        east = start_lng if start_lng > end_lng else end_lng
        west = start_lng if start_lng < end_lng else end_lng
        north = start_lat if start_lat > end_lat else end_lat
        south = start_lat if start_lat < end_lat else end_lat
        bound_a = (north, west)
        bound_b = (south, west)
        self.stride_bound_a = bound_a
        self.stride_bound_b = bound_b
        lat = position[0]
        right = (lat, east)
        left = (lat, west)
        height = haversine(bound_a, bound_b)  # km
        self.env_height_km = height
        wide = haversine(right, left)  # km
        # positive   # 500m per stride
        self.stride_height = (north - south) / (height * self.length)
        self.stride_wide = (east - west) / (wide * self.length)

    def step(self, action):
        self.step_reward = 0
        current_status = False
        step_history = []

        self.stride_length(self.current_position)
        stride_direction = -1 if action > 1 else 1

        if action == 0:  # north
            self.next_position = (
                self.current_position[0] + stride_direction * self.stride_height, self.current_position[1])
        elif action == 1:  # east
            self.next_position = (
                self.current_position[0], self.current_position[1] + stride_direction * self.stride_wide)
        elif action == 2:  # south
            self.next_position = (
                self.current_position[0] + stride_direction * self.stride_height, self.current_position[1])
        elif action == 3:  # west
            self.next_position = (
                self.current_position[0], self.current_position[1] + stride_direction * self.stride_wide)

        current = str(self.current_position[0]) + \
            ',' + str(self.current_position[1])
        next_position = str(
            self.next_position[0]) + ',' + str(self.next_position[1])

        self.status_dir_check = 0
        status, leg_step, bound = self.directions_api(current, next_position)
        self.status_dir_check = status

        if (
            status != 'OK' or
            self.next_position[0] > self.map_bound['north'] or
            self.next_position[0] < self.map_bound['south'] or
            self.next_position[1] > self.map_bound['east'] or
            self.next_position[1] < self.map_bound['west']
        ):
            # The step is not reachable
            if status != 'OVER_QUERY_LIMIT':
                self.step_reward = -1
                self.unreach_position_num += 1
            # The step is not available or out of map bound then go back to previous step
            self.next_position = self.current_position
        else:
            self.step_reward -= 0.1    # get -0.1 reward for every transition
            for i in range(len(leg_step)):
                start = (leg_step[i]['start_location']['lat'],
                        leg_step[i]['start_location']['lng'])
                end = (leg_step[i]['end_location']['lat'],
                    leg_step[i]['end_location']['lng'])
                duration = leg_step[i]['duration']['value']  # second
                distance = leg_step[i]['distance']['value']  # km
                start_position = str(start[0]) + ',' + str(start[1])
                end_position = str(end[0]) + ',' + str(end[1])
                status, height_start = self.elevation_api(start_position)
                status1, height_end = self.elevation_api(end_position)

                while status != 'OK' or status1 != 'OK':
                    status, height_start = self.elevation_api(start_position)
                    status1, height_end = self.elevation_api(end_position)

                elevation = height_end - height_start  # unit: m

                if duration <= 0:
                    duration = 1

                self.time += duration
                speed = math.sqrt(distance ** 2 + elevation ** 2) / duration  # m/s
                angle = math.atan2(distance * 1000, elevation)   # degree
                angle = angle if angle > 0 else 0
                power = self.need_energy.energy(angle=angle, V=speed)
                energy_consume = 0

                for t in range(duration):
                    charge = self.battery.use(duration=1, power=power)
                    energy_consume += self.battery.energy_consume
                    self.step_reward -= self.battery.energy_consume / 100000

                    if charge:   # this duration needs to charge the battery
                        self.charge_num += 1
                        self.battery.charge(50000)    # make it full capacity
                step_history.append(
                    [start, end, duration, distance, angle, speed, energy_consume])

            if (
                abs(self.next_position[0] - self.end_position[0]) < self.stride_height and
                abs(self.next_position[1] -
                    self.end_position[1]) < self.stride_wide
            ):
                self.step_reward = 1   # really close to end position within one step
                self.step_reward -= 0.1

                end_position = str(
                    self.end_position[0]) + ',' + str(self.end_position[1])
                statusE, leg_stepE, boundE = self.directions_api(
                    next_position, end_position)
                self.legE = leg_stepE

                if statusE == 'OK':
                    for i in range(len(self.legE)):
                        start = (self.legE[i]['start_location']['lat'],
                                self.legE[i]['start_location']['lng'])
                        end = (self.legE[i]['end_location']['lat'],
                            self.legE[i]['end_location']['lng'])
                        duration = self.legE[i]['duration']['value']  # second
                        distance = self.legE[i]['distance']['value']  # km
                        start_position = str(start[0]) + ',' + str(start[1])
                        end_position = str(end[0]) + ',' + str(end[1])
                        status, height_start = self.elevation_api(start_position)
                        status1, height_end = self.elevation_api(end_position)

                        while status != 'OK' or status1 != 'OK':                   # recheck again
                            status, height_start = self.elevation_api(
                                start_position)
                            status1, height_end = self.elevation_api(end_position)

                        elevation = height_end - height_start  # unit: m

                        if duration <= 0:
                            duration = 1

                        self.time += duration
                        speed = math.sqrt(
                            distance ** 2 + elevation ** 2) / duration  # m/s
                        angle = math.atan2(distance * 1000, elevation)   # degree
                        angle = angle if angle > 0 else 0   # we let downward as flat
                        power = self.need_energy.energy(angle=angle, V=speed)
                        energy_consume = 0

                        for t in range(duration):
                            charge = self.battery.use(duration=1, power=power)
                            energy_consume += self.battery.energy_consume
                            self.step_reward -= self.battery.energy_consume / 100000

                            if charge:   # this duration needs to charge the battery
                                self.charge_num += 1
                                # make it full capacity
                                self.battery.charge(50000)

                current_status = True
                self.current_position = self.start_position

        self.current_step_history = step_history
        self.current_position = self.next_position
        battery_SOC = self.battery.SOC
        return self.current_position, self.step_reward, current_status, self.charge_num, battery_SOC
    
    def origine_map_reward(self):
        leg = self.Google_step
        step_reward = 0
        time = 0
        for i in range(len(leg)):
            start = (leg[i]['start_location']['lat'], leg[i]['start_location']['lng'])
            end = (leg[i]['end_location']['lat'], leg[i]['end_location']['lng'])
            duration = leg[i]['duration']['value']  # second
            distance = leg[i]['distance']['value']  # km
            start_position = str(start[0]) + ',' + str(start[1])
            end_position = str(end[0]) + ',' + str(end[1])
            status, height_start = self.elevation_api(start_position)
            status1, height_end = self.elevation_api(end_position)
            
            while status != 'OK' or status1 != 'OK':
                status, height_start = self.elevation_api(start_position)
                status1, height_end = self.elevation_api(end_position)
            
            elevation = height_end - height_start  # unit: m
            
            if duration <= 0:
                duration = 1
            
            time += duration
            speed = math.sqrt(distance ** 2 + elevation ** 2) / duration  # m/s
            angle = math.atan2(distance * 1000, elevation)   # degree
            angle = angle if angle > 0 else 0
            power = self.need_energy.energy(angle=angle, V=speed)
            energy_consume = 0
            
            for t in range(duration):
                charge = self.battery.use(duration=1, power=power)
                energy_consume += self.battery.energy_consume
                step_reward -= self.battery.energy_consume / 100000
                
                if charge:   # this duration needs to charge the battery
                    self.charge_num += 1
                    self.battery.charge(50000)    # make it full capacity
                    # step_reward -= 0.1   # we deduct 0.1 point of reward when charge

        chargenum = self.charge_num
        SOC = self.battery.SOC
        self.battery.charge(50000)    # make it full capacity
        self.battery.use(0, 0)
        self.charge_num = 0

        return step_reward, chargenum, SOC, time
    
    def battery_charge(self):
        self.battery.charge(50000)
        self.battery.use(0, 0)
        # self.battery = lithium_ion_battery(50000) #Wh


    def battery_condition(self):
        soc = self.battery.SOC
        charge_num = self.charge_num
        return soc, charge_num
