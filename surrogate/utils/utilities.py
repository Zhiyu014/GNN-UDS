from swmm_api import read_inp_file
import datetime as dt
from os.path import exists,splitext,dirname,join
from swmm_api import read_inp_file
from swmm_api.input_file.sections.others import TimeseriesData,Timeseries
import pandas as pd
import numpy as np
from math import log10
import random

def get_inp_files(inp,arg,swmm_step=1,**kwargs):
    files = eval(arg['func'])(inp,arg=arg,swmm_step=swmm_step,**kwargs)
    return files

def split_file(file,arg,**kwargs):
    inp = read_inp_file(file)
    for k,v in inp.TIMESERIES.items():
        if arg['suffix'] is None or k.startswith(arg['suffix']):
            dura = v.data[-1][0] - v.data[0][0]
            st = (inp.OPTIONS['START_DATE'],inp.OPTIONS['START_TIME'])
            st = dt.datetime(st[0].year,st[0].month,st[0].day,st[1].hour,st[1].minute,st[1].second)
            dura = dura if type(dura) is dt.timedelta else dt.timedelta(hours=dura)
            et = st + dura + dt.timedelta(hours=arg['MIET'])
            inp.OPTIONS['END_DATE'],inp.OPTIONS['END_TIME'] = et.date(),et.time()
            inp.RAINGAGES[arg['gage']].Timeseries = k
            if not exists(arg['filedir']+k+'.inp'):
                inp.write_file(arg['filedir']+k+'.inp')
    events = [arg['filedir']+k+'.inp' for k in inp.TIMESERIES if arg['suffix'] is None or k.startswith(arg['suffix'])]
    return events


def generate_file(file, arg, swmm_step=1, pattern = 'Chicago_icm', filedir = None, rain_num = 1, replace = False):
    """
    Generate multiple inp files containing rainfall events
    designed by rainfall pattern.
    
    Parameters
    ----------
    base_inp_file : dir
        Path of the inp model.
    arg : dict
        rainfall arguments.
    pattern : str, optional
        'Chicago_icm'
    filedir : dir, optional
        The output dir. The default is None.
    rain_num : int, optional
        numbers of rainfall events. The default is 1.

    Returns
    -------
    files : list
        A list of inp files.

    """
    inp = read_inp_file(file)
    files = list()
    filedir = arg.get('filedir',dirname(file)) if filedir is None else filedir
    filedir = join(filedir,arg['suffix']+'_%s.inp')
    rain_num = arg.get('rain_num',rain_num)
    for i in range(rain_num):
        file = filedir%i
        files.append(file)
        if exists(file):
            step = read_inp_file(file)['OPTIONS']['ROUTING_STEP']
            if not replace and step.second == swmm_step:
                continue        

        if type(arg['P']) is tuple:
            p = random.randint(*arg['P'])
        elif type(arg['P']) is list:
            p = arg['P'][i]
        elif type(arg['P']) in [int,float]:
            p = arg['P']

        para = []
        for v in arg['params'].values():
            if type(v) is tuple:
                para.append(random.uniform(*v))
            elif type(v) in [int,float]:
                para.append(v)
        para += [p,arg['delta'],arg['dura']]

        # define simulation time on 01/01/2000
        start_time = dt.datetime(2000,1,1,0,0)
        end_time = start_time + dt.timedelta(minutes = arg['simu_dura'])
        inp.OPTIONS['START_DATE'] = start_time.date()
        inp.OPTIONS['END_DATE'] = end_time.date()
        inp.OPTIONS['START_TIME'] = start_time.time()
        inp.OPTIONS['END_TIME'] = end_time.time()
        inp.OPTIONS['REPORT_START_DATE'] = start_time.date()
        inp.OPTIONS['REPORT_START_TIME'] = start_time.time()
        inp.OPTIONS['ROUTING_STEP'] = dt.time(second=swmm_step)

        # calculate rainfall timeseries
        ts = eval(pattern)(para)
        ts = [[(start_time+dt.timedelta(hours=1)+dt.timedelta(minutes=t)).strftime('%m/%d/%Y %H:%M:%S'),va] for t,va in ts]
        inp['TIMESERIES'] = Timeseries.create_section()
        inp['TIMESERIES'].add_obj(TimeseriesData(Name = str(p)+'y',data = ts))
        inp.RAINGAGES['RG']['Timeseries'] = str(p)+'y'
        inp.RAINGAGES['RG']['Interval'] = str(int(arg['delta']//60)).zfill(2)+':'+str(int(arg['delta']%60)).zfill(2)

        inp.write_file(file)
    return files

# Generate a rainfall intensity file from a cumulative values in ICM
def Chicago_icm(para_tuple):
    A,C,n,b,r,P,delta,dura = para_tuple
    a = A*(1+C*log10(P))
    HT = a*dura/(dura+b)**n
    Hs = []
    for i in range(dura//delta+1):
        t = i*delta
        if t <= r*dura:
            H = HT*(r-(r-t/dura)*(1-t/(r*(dura+b)))**(-n))
        else:
            H = HT*(r+(t/dura-r)*(1+(t-dura)/((1-r)*(dura+b)))**(-n))
        Hs.append(H)
    tsd = np.diff(np.array(Hs))*60/delta
    ts = []
    for i in range(dura//delta):
        t = i*delta
        # key = str(1+t//60).zfill(2)+':'+str(t % 60).zfill(2)
        ts.append([t,tsd[i]])
    return ts

def generate_split_file(base_inp_file,
                        timeseries_file=None,
                        event_file=None,
                        filedir = None,
                        rain_num = None,
                        arg = None,
                        swmm_step = 1,
                        **kwargs):
    """
    Generate multiple inp files containing rainfall events
    separated from continous rainfall events.
    
    Parameters
    ----------
    base_inp_file : dir
        Path of the inp model.
    timeseries_file : dir
        Path of the rainfall timeseries data file.
    event_file : str
        Path of the rainfall event file (start or end time of each event).
    filedir : dir, optional
        The output dir. The default is None.
    rain_num : int, optional
        numbers of rainfall events. The default is 1.
    miet : int, optional
        minimum interevent time (min). The default is 120.
    Returns
    -------
    files : list
        A list of inp files.

    """
    if arg is not None:
        replace_rain = arg.get('replace_rain',False)
        MIET = arg.get('MIET',120)
        # if dura_range is None:
        dura_range = arg.get('duration_range',None)
        # if precip_range is None:
        precip_range = arg.get('precipitation_range',None)
        # if date_range is None:
        date_range = arg.get('date_range',None)
    # Read inp & data files & event file
    inp = read_inp_file(base_inp_file)

    if timeseries_file is None:
        timeseries_file = arg['rainfall_timeseries']
    tsf = pd.read_csv(timeseries_file,index_col=0)
    tsf['datetime'] = tsf['date']+' '+tsf['time']
    # tsf['datetime'] = tsf['datetime'].apply(lambda dti:datetime.strptime(dti, '%m/%d/%Y %H:%M:%S'))
    tsf.index = pd.to_datetime(tsf['datetime'])
    if arg.get('tide',False):
        tidets = pd.read_csv(arg['tide'],index_col=0)
        tidets['datetime'] = tidets['date']+' '+tidets['time']
        tidets.index = pd.to_datetime(tidets['datetime'])

    if event_file is None:
        event_file = arg.get('rainfall_events',splitext(timeseries_file)[0]+'_events.csv')
        if not exists(event_file):
            event_file = serapate_events(timeseries_file, MIET)
    
    events = pd.read_csv(event_file,index_col=0) if type(event_file) == str else event_file

    if dura_range is not None:
        events = events[events['Duration'].apply(lambda x:dura_range[0]<=x<=dura_range[1])]
    if precip_range is not None:
        events = events[events['Precipitation'].apply(lambda x:precip_range[0]<=x<=precip_range[1])]
    if date_range is not None:
        date_range = [dt.datetime.strptime(date,'%m/%d/%Y') for date in date_range]
        events['Date'] = pd.to_datetime(events['Date'])
        # events['Date'] = events['Date'].apply(lambda date:dt.datetime.strptime(date,'%m/%d/%Y'))
        events = events[events['Date'].apply(lambda x:date_range[0]<=x<=date_range[1])]

    filedir = arg.get('filedir') if filedir is None else filedir
    filedir = splitext(base_inp_file)[0] if filedir is None else filedir
    filedir += '_%s.inp'

    if type(rain_num) == int:
        # files = [filedir%idx for idx in range(rain_num)]
        events = events.sample(rain_num)
    elif type(rain_num) == list:
        events = events[events['Start'].apply(lambda x:x.split(':')[0].replace(' ','-') in rain_num)]
    # elif rain_num == 'all':
    #     files = [filedir%idx for idx in range(rain_num)]

    # # Skip generation if not replace
    # new_files = [file for file in files if not exists(file) or if_replace]
    # if len(new_files) == 0:
    #     return files

    files = list()
    events['Start'] = pd.to_datetime(events['Start'])
    events['End'] = pd.to_datetime(events['End'])
    for start_time,end_time in zip(events['Start'],events['End']):
        # Formulate the simulation periods
        # start_time = dt.datetime.strptime(start,'%m/%d/%Y %H:%M:%S')
        # end_time = dt.datetime.strptime(end,'%m/%d/%Y %H:%M:%S') + dt.timedelta(minutes = MIET)   
        end_time += dt.timedelta(minutes=MIET)

        file = filedir%start_time.strftime('%m_%d_%Y_%H')
        files.append(file)
        if exists(file):
            step = read_inp_file(file)['OPTIONS']['ROUTING_STEP']
            if not replace_rain and step.second == swmm_step:
                continue
        
        # rain = tsf[start_time < tsf['datetime']]
        # rain = rain[rain['datetime'] < end_time]
        rain = tsf[start_time:end_time]
        raindata = {col:[(dt,vol)
         for dt,vol in zip(rain.index,rain[col])]
          for col in rain.columns if col not in ['date','time','datetime']}

        for rg in inp.RAINGAGES.values():
            ts = rg.Timeseries
            inp.TIMESERIES[ts] = TimeseriesData(ts,raindata[ts])

        if arg.get('tide',False):
            tide = tidets[start_time-dt.timedelta(hours=1):end_time+dt.timedelta(hours=1)]
            tidedata = {col:[(dt,vol)
            for dt,vol in zip(tide.index,tide[col])]
            for col in tide.columns if col not in ['date','time','datetime']}
            for td in set(inp.OUTFALLS.frame['Data']):
                inp.TIMESERIES[td] = TimeseriesData(td,tidedata[td])

        inp.OPTIONS['START_DATE'] = start_time.date()
        inp.OPTIONS['END_DATE'] = end_time.date()
        inp.OPTIONS['START_TIME'] = start_time.time()
        inp.OPTIONS['END_TIME'] = end_time.time()
        inp.OPTIONS['REPORT_START_DATE'] = start_time.date()
        inp.OPTIONS['REPORT_START_TIME'] = start_time.time()
        inp.OPTIONS['ROUTING_STEP'] = dt.time(second=swmm_step)
        inp.write_file(file)
    return files




def serapate_events(timeseries_file,miet=120,event_file=None,replace=False):
    """
    Separate continous rainfall timeseries file into event-wise records.
    
    Parameters
    ----------
    timeseries_file : dir
        Path of the timeseries data.
    miet : int, optional
        minimum interevent time (min). The default is 120.
    event_file : dir
        Path of the event file to be saved.

    Returns
    -------
    event_file : dir
        Path of the event file to be saved.

    """
    if event_file is None:
        event_file = splitext(timeseries_file)[0]+'_events.csv'
        if exists(event_file) and not replace:
            return event_file

    tsf = pd.read_csv(timeseries_file,index_col=0)
    tsf = tsf.drop(tsf[tsf.sum(axis=1,numeric_only=True)==0].index)
    tsf['datetime'] = tsf['date']+' '+tsf['time']
    tsf['datetime'] = tsf['datetime'].apply(lambda dti:dt.datetime.strptime(dti, '%m/%d/%Y %H:%M:%S'))
    
    rain = tsf.reset_index(drop=True,level=None)
    start = [0] + rain[rain['datetime'].diff() > dt.timedelta(minutes = miet)].index.tolist()
    end = [ti-1 for ti in start[1:]] + [len(rain)-1]
    
    # Get start & end pairs of each rainfall event using month/day/year by SWMM
    pairs = [[rain.loc[ti,'datetime'],
    rain.loc[end[idx],'datetime']] 
    for idx,ti in enumerate(start)]
    events = pd.DataFrame(pairs,columns = ['Start','End'])
    events['Date'] = events['Start'].apply(lambda st:st.strftime('%m/%d/%Y'))
    events['Duration'] = events.apply(lambda row:(row['End'] - row['Start']).total_seconds()/60,axis=1)
    events['Precipitation'] = [tsf.loc[tsf.index[ti]:tsf.index[end[idx]]].sum(axis=0,numeric_only=True).mean() 
    for idx,ti in enumerate(start)]

    for col in ['Start','End']:
        events[col] = events[col].apply(lambda x: x.strftime('%m/%d/%Y %H:%M:%S'))

    events.to_csv(event_file)
    return event_file
    
