An EV_battery charging dataset for fault detection

Description: This dataset covering the period between October 2020 and October 2023 was collected from 30 real public EV charging stations which are operated by Shenzhen Auto Electric Power Plant Co., Ltd (Autosun), a leading company in EV Mega-Watt charging technology solution with a 250 MW public EV charging network in China.

We at first do the preprocessing here including: (1) down-sampling of normal operating data; (2) dividing charging stations into 8 data owners; (3) dropping incomplete or abnormal data. After this, we provide 2 files. The first one (processed_data.xlsx) is the pre-processed data. The second (processed_data_longer_than_30.xlsx) is the data which only includes charging sequences longer than 30 minutes. 

The second dataset, which will be used in our article, contains 21175 pieces of EV charging data with 1547432 sampling points, covering 10154 EVs that comprise various types of batteries. Each piece of charging data includes charging current, voltage and power at a 1 minute time interval. The battery fault data in this dataset can be categorized into two distinct types. The first type refers to faults that have been reported by engineering personnel, such as thermal runaway, unbalanced cell voltages, short-circuit and so on. The second type is characterized by excessively high temperatures that incur disconnection by the stations themselves.

The meaning of all columns is shown here:

id: unique id for every data point
transaction_id: unique id for every charging sequence
begin_time: begin time of a charging sequence
end_time: current sampling time
total_charging_kwh: charging energy for now
total_charging_min: charging time for now
current_soc: SOC for now
current_energy_meter_value: the energy of this battery at this time
chargingv: current charging voltage
charginga: current charging current
out_power: current charging power
charging_gun_temperature1: temperature of the charging gun
charging_gun_temperature2: temperature of the charging gun
types: Battery types, 03: LFP, 06: NMC, 04: LMO, 05: LCO, 07: LP
class_judge: data owner index
label: denote if it is a fault battery data, 1 means faulty, 0 means normal
