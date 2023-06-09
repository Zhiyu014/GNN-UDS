from swmm_api import read_inp_file
from datetime import timedelta,datetime
from os.path import exists,splitext
from datetime import datetime,timedelta
from swmm_api import read_inp_file
from swmm_api.input_file.sections.others import TimeseriesData
import pandas as pd


def get_inp_files(inp,arg,**kwargs):
    files = eval(arg['func'])(inp,arg=arg,**kwargs)
    return files

def generate_file(file,arg,**kwargs):
    inp = read_inp_file(file)
    for k,v in inp.TIMESERIES.items():
        if arg['suffix'] is None or k.startswith(arg['suffix']):
            dura = v.data[-1][0] - v.data[0][0]
            st = (inp.OPTIONS['START_DATE'],inp.OPTIONS['START_TIME'])
            st = datetime(st[0].year,st[0].month,st[0].day,st[1].hour,st[1].minute,st[1].second)
            dura = dura if type(dura) is timedelta else timedelta(hours=dura)
            et = st + dura + timedelta(hours=arg['MIET'])
            inp.OPTIONS['END_DATE'],inp.OPTIONS['END_TIME'] = et.date(),et.time()
            inp.RAINGAGES[arg['gage']].Timeseries = k
            if not exists(arg['filedir']+k+'.inp'):
                inp.write_file(arg['filedir']+k+'.inp')
    events = [arg['filedir']+k+'.inp' for k in inp.TIMESERIES if arg['suffix'] is None or k.startswith(arg['suffix'])]
    return events



def generate_split_file(base_inp_file,
                        timeseries_file=None,
                        event_file=None,
                        filedir = None,
                        rain_num = None,
                        arg = None,
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
    tsf['datetime'] = tsf['datetime'].apply(lambda dt:datetime.strptime(dt, '%m/%d/%Y %H:%M:%S'))

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
        date_range = [datetime.strptime(date,'%m/%d/%Y') for date in date_range]
        events['Date'] = events['Date'].apply(lambda date:datetime.strptime(date,'%m/%d/%Y'))
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
    for start,end in zip(events['Start'],events['End']):
        # Formulate the simulation periods
        start_time = datetime.strptime(start,'%m/%d/%Y %H:%M:%S')
        end_time = datetime.strptime(end,'%m/%d/%Y %H:%M:%S') + timedelta(minutes = MIET)   

        file = filedir%start_time.strftime('%m_%d_%Y_%H')
        files.append(file)
        if exists(file) == True and replace_rain == False:
            continue

        rain = tsf[start_time < tsf['datetime']]
        rain = rain[rain['datetime'] < end_time]
        raindata = [[[date+' '+time,vol]
         for date,time,vol in zip(rain['date'],rain['time'],rain[col])]
          for col in rain.columns if col not in ['date','time','datetime']]

        for idx,rg in enumerate(inp.RAINGAGES.values()):
            ts = rg.Timeseries
            inp.TIMESERIES[ts] = TimeseriesData(ts,raindata[idx])

        inp.OPTIONS['START_DATE'] = start_time.date()
        inp.OPTIONS['END_DATE'] = end_time.date()
        inp.OPTIONS['START_TIME'] = start_time.time()
        inp.OPTIONS['END_TIME'] = end_time.time()
        inp.OPTIONS['REPORT_START_DATE'] = start_time.date()
        inp.OPTIONS['REPORT_START_TIME'] = start_time.time()
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
    tsf['datetime'] = tsf['datetime'].apply(lambda dt:datetime.strptime(dt, '%m/%d/%Y %H:%M:%S'))
    
    rain = tsf.reset_index(drop=True,level=None)
    start = [0] + rain[rain['datetime'].diff() > timedelta(minutes = miet)].index.tolist()
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
    
