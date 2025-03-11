clc
clear all
close all

path = "/home/silvia/Documents/UNIPD_Dottorato/Oulu/mmWave_human/dataset/pna_data/";
sub = "sub8/";
filename = "sm_sub8_er.mat";
load(path + sub + filename);
timestamps = MeasData.timestamp;
d = datetime(timestamps,"ConvertFrom","datenum", 'TimeZone', 'Europe/Helsinki');
conv_ts = convertTo(d, 'posixtime', 'TimeZone', 'Europe/London');
MeasData.conv_timestamps = conv_ts;
save(path+sub+filename, "MeasData");