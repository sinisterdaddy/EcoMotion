import numpy as np


class LithiumIonBattery:
    def __init__(self, total_capacity):
        # working voltage 3.8
        # cut-off voltage 2
        # we have 100 cells
        self.cell_voltage = 4.0
        self.cutoff_voltage = 2.0
        self.grade = (self.cell_voltage - self.cutoff_voltage) / 100.0
        self.total_capacity = total_capacity  # Wh
        self.capacity = total_capacity  # Wh
        self.ah = self.capacity / (self.cell_voltage * 100.0)
        self.need_charge = False
        self.energy_consumed = 0.0
        self.soc = 0.9
        # self.output = 1  # unit is C, c rate


    def use(self, duration, power):
        # duration in second
        # power in Watt  (j/s)
        # self.need_charge = False
        self.soc = self.capacity / self.total_capacity
        if self.soc > 0.9:
            self.soc = 0.9
        if self.soc <= 0.2:
            self.need_charge = True
        self.cell_voltage = self.soc * 100.0 * self.grade + self.cutoff_voltage  # we assume it is linear
        # self.capacity -= duration * outputrate * self.Ah * self.cell_voltage * 100 / 3600
        if not self.need_charge:
            self.energy_consumed = duration * power / 3600.0
            self.capacity -= self.energy_consumed
        return self.need_charge

    def charge(self, energy):
        if (energy + self.capacity) > self.total_capacity:
            self.capacity = self.total_capacity
            self.need_charge = False
        else:
            self.capacity += energy
            self.need_charge = False
