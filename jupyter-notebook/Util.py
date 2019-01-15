# utility functions for prcessing the Web QoE and QoS dataset 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime 

import Cdf
import sys 
import requests

# Global variables 
dataset = '../../dataset/2018-weblar/monroe_weblar_merged_201804.csv'

# colors 
C = {'google': 'C0', 'facebook': 'C1', 'youtube': 'C2', 'yahoo': 'C3', 'reddit': 'C4', \
     'wikipedia': 'C7', 'bbc': 'C6', 'microsoft': 'C9', 'amazon': 'C8', 'ebay': 'C5'}
M = {'google': 's', 'facebook': 'o', 'youtube': '3', 'yahoo': '8', 'reddit': 'x', \
     'wikipedia': '*', 'bbc': '>', 'microsoft': 'v', 'amazon': '.', 'ebay': 'p'}

C_O ={'NO_1': 'C0', 'SE_3': 'C1', 'NO_2': 'C3', 'SE_r': 'C4', 'SE_2': 'C9',\
       'SE_1': 'C8', 'NO_3': 'C7'}

M_O = {'NO_1': 's', 'SE_3': 'o', 'NO_2': '3', 'SE_r': 'd', 'SE_2': 'x',\
       'SE_1': '*', 'NO_3': '+'}

# Probe information
# Mobile node IDs
se_mobile = [372, 373, 374, 375, 378, 379, 380, 381, 382, 383, 384, 385, 406, 407, 
            408, 409, 410, 411, 412, 413, 414, 415, 418, 419, 420, 421, 422, 423, 
            426, 427,430, 431, 434, 435, 436, 437, 502, 504, 505, 506, 507, 508, 510, 511]
no_mobile = [206,228,229,254,255,261,289,290,291,292,296,297,304,305,366,367,368,369,
             448,449,450,451,456,457,460,461,462,463,552,553,554,555,556,557,558,559,
             560,561,562,563,564,565,566,567,568,569]
mobile_nodes = [206, 255, 261, 372, 415, 421, 422, 430, 451, 470, 472, 483, 486, 488, 
                489, 498, 500, 558, 559, 560, 566, 567, 568]
it_bus_mobile = [308, 309, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 
                 321, 326, 327, 330, 331, 336, 337, 341, 346, 347, 352, 353, 
                 354, 355, 388, 389, 398, 399, 404, 405]
it_truck_mobile = [144, 145, 146, 150, 151, 152, 153, 258, 264, 265, 266, 267, 274, 275]
no_stationary = [356, 357, 358, 359, 360, 361, 362, 363, 438, 439, 440, 441, 
                 444, 445, 452, 453, 454, 455, 470, 471, 472, 473]
se_stationary = [43 , 45 , 46 , 98 , 233, 234, 236, 238, 240, 249, 251, 400, 401, 402, 
                 403, 446, 447, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 
                 487, 488, 489, 490, 491, 492, 493, 494, 495, 498, 499, 500, 501, 608, 609]
all_mobile = se_mobile + no_mobile + it_bus_mobile + it_truck_mobile
all_stationary = no_stationary + se_stationary

# NSB-train to MONROE node ID mappring 
nsb_monroe = { 552:"75-55", 554:"75-52", 562:"75-50", 564:"75-49", 456:"75-46", 450:"75-45", 566:"75-27",
              558:"75-25", 212:"75-23", 460:"75-07", 568:"75-06", 560:"75-03", 556:"74-37", 462:"74-25", 
              304:"73-09", 368:"73-08", 296:"73-06", 366:"73-05", 291:"73-04", 448:"73-01", 292:"7-21772", 
              290:"5-21713", 254:"5-21726", 228:"5-21722", 553:"75-55", 555:"75-52", 563:"75-50", 565:"75-49", 
              457:"75-46", 451:"75-45", 567:"75-27", 559:"75-25", 461:"75-07", 569:"75-06", 561:"75-03", 
              463:"74-25", 305:"73-09", 369:"73-08", 297:"73-06", 367:"73-05", 261:"73-04", 449:"73-01", 
              206:"7-21772", 289:"5-21713", 255:"5-21726", 557:"74-37"}

# operator to country any
def change_op_cc(op):
    if op == '242 14':
        return('NO_1')
    elif op =='3':
        return('SE_3')
    elif op == 'N Telenor':
        return('NO_2')
    elif op == 'TELIA S':
        return('SE_r')
    elif op == 'Telenor SE':
        return('SE_2')
    elif op == 'Telia':
        return('SE_1') 
    elif op == 'Telia N' or op == 'NetCom':
        return('NO_3')
    elif op == 'vodafone IT':
        return('IT_1')
    elif op == 'I WIND':
        return('IT_2')
    elif op == 'I TIM':
        return('IT_3') 
    elif op == 'Orange':
        return('ES_1')
    elif op == 'YOIGO':
        return('ES_2')
    elif op == 'FIXED':
        return('FIXED')
    else: 
        return('Misc')
    
# box plots
def stop_plot_box(ax, title, shrink_right = False, tick_xlab = False, fs = 22, show_tick = True, plt_type = 'fetch_type'):
    ax.set_title(""); plt.suptitle("")
    if not tick_xlab: 
        ax.set_xticklabels('', fontsize = fs)
    
    #xticks = [100, 500, 1000, 5000, 10000]
    ax.grid(False)
    if plt_type == '3rt': 
        ax.set_xlim([1, 10])
    elif plt_type == '10rt':
        ax.set_xlim([1, 15])
    elif plt_type == 'atf_inst':
        ax.set_xlim([1, 10])
    elif plt_type == 'atf_intg':
        ax.set_xlim([1, 10])
        
    elif plt_type == 'u3rt': 
        ax.set_xlim([1, 10])
    elif plt_type == 'u10rt':
        ax.set_xlim([1, 20])
    elif plt_type == 'uatf_inst':
        ax.set_xlim([1, 20])
    elif plt_type == 'uatf_intg':
        ax.set_xlim([1, 20])
        
    elif plt_type == 'plt': 
        ax.set_xlim([3, 25])
    elif plt_type == 'ttfb':
        ax.set_xlim([0.2, 2])
    elif plt_type == 'dnslkup':
        ax.set_xlim([0, 0.2])
    elif plt_type == 'tcpcnct':
        ax.set_xlim([0, 0.5])
    elif plt_type == 'num_obj':
        ax.set_xlim([20, 200])
    elif plt_type == 'size_obj':
        ax.set_xlim([100, 9000])
    ax.set_xscale('linear')
    ax.set_xlabel('', fontsize=fs)
    ax.set_title(title,  fontsize=12)
    
    if plt_type == 'ttl':
        ax.set_xlim([5, 22])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('axes', -0.05))
        ax.spines['left'].set_position(('axes', -0.015))
        ax.set_xlabel('', fontsize=fs)
        ax.set_title(title,  fontsize=18, position=(0.79,0.19))

        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 4))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

        if not show_tick:
            ax.set_xticklabels("")
            
# sort box plot by column value 
def boxplot_sorted(df, by, column, ax, rot = 0):
    medianprops = dict(linestyle='--', color = 'blue', linewidth=2)
    props = dict(boxes="black", whiskers="black", medians="blue") #for colors -- change this for color scale
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    meds = df2.median().sort_values() # for sorting by median 
    return df2[meds.index].plot.box(ax = ax, rot = rot,  vert = False, color = props,
                                   medianprops = medianprops, showfliers = False, 
                                   return_type = "axes")

# create a list of dataset classified basedon URL
def tables_site(data):
    sites = {}
    for w in pd.unique(data['target']): 
        sites[w] = data.loc[data['target'] == w]
    return(sites)

# round time to date
def round_time_to_day(dtime):
    import datetime;    
    d = datetime.datetime.strptime(dtime, "%Y-%m-%d %H:%M:%S")
    d = d.replace(hour=0, minute=0, second=0)
    dtime = d.strftime('%Y-%m-%d')
    return dtime

# get the x value of the cdf
def get_icdf(caigo):
    pp = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95]
    ppf = []
    for p in pp:
        v = Cdf.Cdf.Value(caigo, p)
        ppf.append(v)
    return(ppf)

# create a list of dataset classified basedon probe ID
def tables_unit(data):
    sites = {}
    for w in pd.unique(data['unit_id']): 
        sites[w] = data.loc[data['unit_id'] == w]
    return(sites)

# create a list of dataset classified basedon date
def tables_date(data):
    sites = {}
    for w in pd.unique(data['dtime']): 
        sites[w] = data.loc[data['dtime'] == w]
    return(sites)

# create a list of dataset classified basedon latitude
def tables_lat(data):
    sites = {}
    for w in pd.unique(data['lat']): 
        sites[w] = data.loc[data['lat'] == w]
    return(sites)

# create a list of dataset classified basedon long
def tables_long(data):
    sites = {}
    for w in pd.unique(data['long']): 
        sites[w] = data.loc[data['long'] == w]
    return(sites)

# create a list of dataset classified basedon operator
def tables_operator(data):
    sites = {}
    for w in pd.unique(data['operator']): 
        sites[w] = data.loc[data['operator'] == w]
    return(sites)

def get_json_resource_from_absolute_uri(url, query_params):
    try: res = requests.get(url, params = query_params)
    except Exception as e: print(e, file=sys.stderr)
    else:
        try: res_json = res.json()
        except Exception as e: print(e, file=sys.stderr)
        else: 
            return res_json
        
def get_asn_from_endpoint(endpoint):
    asn = holder = None
    base_uri = 'https://stat.ripe.net'; url = '%s/data/prefix-overview/data.json'%base_uri
    params = {'resource' : endpoint}
    try: res = get_json_resource_from_absolute_uri(url, params)
    except Exception as e: print(e, file=sys.stderr); return None
    try:
        asns_holders = []
        for item in res['data']['asns']:
            asn = item['asn']; holder = item['holder']
            asns_holders.append((asn, holder))
    except Exception as e: print(e, file=sys.stderr)
    return asns_holders

# return base url from url
def get_base_url(url):
    return(url.rsplit('/')[2])

# return base url from url from set
def get_base_url_name(url):
    l = url.split('.')
    if len(l) > 2: 
        return(l[1])
    else:
        return(l[0])
            
# check if a measurement is from mobile node
def is_from_mobile(msmt_id, _mobile):
    if msmt_id in _mobile:
        return(1)
    else:
        return(0)
    
# Count ASes 
# input: a line that contains the list ASes separated with comma

def count_ases(line):
    ases = str(line).split(',')
    return(len(ases))


# Count cell ID change
# input: a line that contains the list ASes separated with comma
def count_cid_change(line):
    cid = str(line).split(',')
    return(len(cid) - 1) # 0 if doesn't change


# cell change cell ID change
def group_cid_change(cid):
    if cid == 0: return("0")
    elif cid > 0 and cid <= 10: return("[1 - 10]")
    elif cid > 10 and cid <= 20: return("[11 - 20]")
    elif cid > 20 and cid <= 30: return("[21 - 30]")
    elif cid > 30 and cid <= 40: return("[31 - 40]")
    else: return("41+")
    
# median value for RSSI, RSRQ, RSRP...
def _median(val):
    val_list = val.split(',')
    val_list = [int(x) for x in val_list]
    return(round(np.median(val_list)))
           
# median value for RSSI, RSRQ, RSRP...
def _mean(val):
    val_list = val.split(',')
    val_list = [int(x) for x in val_list]
    return(round(np.mean(val_list)))

# get measurement date time from the measurement ID (val)
def get_measurement_dtime(val):
    val_list = val.split('_')
    d = datetime.datetime.strptime(val_list[2], "%Y%m%d%H%M%S")
    d = d.replace(hour=0, minute=0, second=0)
    dtime = d.strftime('%Y-%m-%d')
    return(dtime) 
    

# Check weekday from measurement ID
def weekday(val):
    val_list = val.split('_')
    d = datetime.datetime.strptime(val_list[2], "%Y%m%d%H%M%S").weekday()
    
    return(d)
#     if d < 5:
#         return('Weekday')
#     else:
#         return('Weekend')

