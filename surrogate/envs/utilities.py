from math import log10
from numpy import diff, array
from os.path import exists,splitext,isfile
import random
from datetime import datetime,timedelta
from swmm_api import read_inp_file,read_rpt_file,swmm5_run,read_out_file
from swmm_api.run import swmm5_run_parallel
from swmm_api.input_file.sections import Timeseries
from swmm_api.input_file.sections.others import TimeseriesData
import yaml
import pandas as pd
# This hyetograph is correct in a continous function, but incorrect with 5-min block.
def Chicago_Hyetographs(para_tuple):
    A,C,n,b,r,P,delta,dura = para_tuple
    a = A*(1+C*log10(P))
    ts = []
    for i in range(dura//delta):
        t = i*delta
        # key = str(1+t//60).zfill(2)+':'+str(t % 60).zfill(2)
        if t <= r*dura:
            ts.append([t, (a*((1-n)*(r*dura-t)/r+b)/((r*dura-t)/r+b)**(1+n))*60])
        else:
            ts.append([t, (a*((1-n)*(t-r*dura)/(1-r)+b)/((t-r*dura)/(1-r)+b)**(1+n))*60])
    # tsd = TimeseriesData(Name = name,data = ts)
    return ts

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
    tsd = diff(array(Hs))*60/delta
    ts = []
    for i in range(dura//delta):
        t = i*delta
        # key = str(1+t//60).zfill(2)+':'+str(t % 60).zfill(2)
        ts.append([t,tsd[i]])
    return ts


def generate_file(base_inp_file, para_file, pattern = 'Chicago_icm', filedir = None, rain_num = 1, replace = False):
    """
    Generate multiple inp files containing rainfall events
    designed by rainfall pattern.
    
    Parameters
    ----------
    base_inp_file : dir
        Path of the inp model.
    para_file : YAML
        Path of the rainfall pattern params.
    pattern : str, optional
        'Chicago_icm' or 'Chicago_Hyetographs'. The default is 'Chicago_icm'.
    filedir : dir, optional
        The output dir. The default is None.
    rain_num : int, optional
        numbers of rainfall events. The default is 1.

    Returns
    -------
    files : list
        A list of inp files.

    """
    inp = read_inp_file(base_inp_file)
    paras = yaml.load(open(para_file, "r"), yaml.FullLoader) if type(para_file) is str else para_file
    files = list()
    filedir = splitext(base_inp_file)[0] if filedir is None else filedir
    filedir = splitext(filedir)[0]+'_%s.inp'
    for i in range(rain_num):
        file = filedir%i
        files.append(file)
        if exists(file) == True and replace == False:
            continue        

        if type(paras['P']) is tuple:
            p = random.randint(*paras['P'])
        elif type(paras['P']) is list:
            p = paras['P'][i]
        elif type(paras['P']) in [int,float]:
            p = paras['P']

        delta = paras['delta']
        dura = paras['dura']
        simu_dura = paras['simu_dura']
        para = []
        for col in ['A','C','n','b','r']:
            v = paras['params'][col]
            if type(v) is tuple:
                para.append(random.uniform(*v))
            elif type(v) in [int,float]:
                para.append(v)
        para += [p,delta,dura]

        # define simulation time on 01/01/2000
        start_time = datetime(2000,1,1,0,0)
        end_time = start_time + timedelta(minutes = simu_dura)
        inp.OPTIONS['START_DATE'] = start_time.date()
        inp.OPTIONS['END_DATE'] = end_time.date()
        inp.OPTIONS['START_TIME'] = start_time.time()
        inp.OPTIONS['END_TIME'] = end_time.time()
        inp.OPTIONS['REPORT_START_DATE'] = start_time.date()
        inp.OPTIONS['REPORT_START_TIME'] = start_time.time()

        # calculate rainfall timeseries
        ts = eval(pattern)(para)
        ts = [[(start_time+timedelta(hours=1)+timedelta(minutes=t)).strftime('%m/%d/%Y %H:%M:%S'),va] for t,va in ts]
        inp['TIMESERIES'] = Timeseries.create_section()
        inp['TIMESERIES'].add_obj(TimeseriesData(Name = str(p)+'y',data = ts))
        inp.RAINGAGES['RG']['Timeseries'] = str(p)+'y'
        inp.RAINGAGES['RG']['Interval'] = str(int(delta//60)).zfill(2)+':'+str(int(delta%60)).zfill(2)

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
    


def generate_split_file(base_inp_file=None,
                        timeseries_file=None,
                        event_file=None,
                        filedir = None,
                        rain_num = 1,
                        rain_arg = None):
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
    if rain_arg is not None:
        replace_rain = rain_arg.get('replace_rain',False)
        MIET = rain_arg.get('MIET',120)
        # if dura_range is None:
        dura_range = rain_arg.get('duration_range',None)
        # if precip_range is None:
        precip_range = rain_arg.get('precipitation_range',None)
        # if date_range is None:
        date_range = rain_arg.get('date_range',None)
    # Read inp & data files & event file
    inp = read_inp_file(base_inp_file)

    if timeseries_file is None:
        timeseries_file = rain_arg['rainfall_timeseries']
    tsf = pd.read_csv(timeseries_file,index_col=0)
    tsf['datetime'] = tsf['date']+' '+tsf['time']
    tsf['datetime'] = tsf['datetime'].apply(lambda dt:datetime.strptime(dt, '%m/%d/%Y %H:%M:%S'))

    if event_file is None:
        event_file = rain_arg.get('rainfall_events',splitext(timeseries_file)[0]+'_events.csv')
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


    filedir = splitext(base_inp_file)[0] if filedir is None else filedir
    filedir = splitext(filedir)[0]+'_%s.inp'


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


def eval_control(event):
    if event.endswith('.inp'):
        event = event.replace('.inp','.rpt')
    rpt = read_rpt_file(event)
    flooding = rpt.flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
    outload = rpt.outfall_loading_summary['Total_Volume_10^6 ltr']
    # WSC in the chaohu model
    cso = outload.sum() - outload['WSC']
    return flooding,cso

def eval_pump(event,pumps):
    if event.endswith('.inp'):
        event = event.replace('.inp','.rpt')
    rpt = read_rpt_file(event)
    energy = rpt.pumping_summary.loc[pumps,'Power_Usage_Kw-hr'].sum()
    return energy

def get_flood_cso(event,outfalls=None,cumulative=False):
    rpt_step = read_inp_file(event).OPTIONS['REPORT_STEP']
    rpt_step = rpt_step.minute * 60 + rpt_step.second
    event = event.replace('.inp','.out')
    out = read_out_file(event)
    data = pd.DataFrame(out.get_part('system',variable='rainfall'))
    data.columns = ['Rainfall']
    flood = out.get_part('system')['flooding']
    if cumulative:
        # L/s --> 10^6 L
        flood = (flood*rpt_step/1e6).cumsum()
    data = pd.concat([data,flood],axis=1)
    data.columns = ['Rainfall','Flooding']
    if outfalls is not None:
        cso = sum([out.get_part('node',node)['total_inflow']
        for node in outfalls])
        if cumulative:
            cso = (cso*rpt_step/1e6).cumsum()
        data = pd.concat([data,cso],axis=1)
        data.columns = ['Rainfall','Flooding','CSO']
    return data



def get_depth_setting(event,tanks=None,pumps=None):
    if event.endswith('.inp'):
        event = event.replace('.inp','.out')
    out = read_out_file(event)
    data = pd.DataFrame(out.get_part('system',variable='rainfall'))
    data.columns = ['rainfall']
    
    if tanks is not None:
        depths = out.get_part('node',tanks,'depth')
        data = pd.concat([data,depths],axis=1)
        # data.columns = list(data.columns) + tanks
    if pumps is not None:
        settings = out.get_part('link',pumps,'capacity')
        data = pd.concat([data,settings],axis=1)
        # data.columns = list(data.columns) + pumps
    return data




def eval_closing_reward(event):
    '''
    Get the reward step-wise reward value
    Reward shaping:
    1. Operational reward: ±2
        1) If raining:  sum[(depth - ini_depth)/ini_depth] - sum[energy] * 0.2
        2) Not raining: sum[(ini_depth - depth)/ini_depth] - sum[energy] * 0.2
    2. Closing reward:  ±5 (gamma=0.95 n_steps=36 reward=2/(0.95**36)=4.16 --> 5)
        1) - 8 * (sum[flooding] + sum[overflow])/(sum[outflow] + sum[flooding] + final_storage) + 5
            cannot reach ±5 at all times so multiply -20 plus 15
        2) 10 * (HC[fl&CSO] - fl&CSO)/HC[fl&SCO]
            cannot reach ±5 at all times so multiply 10
    '''
    if event.endswith('.inp'):
        event = event.replace('.inp','.rpt')
    rpt = read_rpt_file(event)
    routing = rpt.flow_routing_continuity
    outload = rpt.outfall_loading_summary
    A,B = -20,10
    return A*(routing['Flooding Loss']['Volume_10^6 ltr'] +\
         routing['External Outflow']['Volume_10^6 ltr'] -\
             outload.loc['WSC','Total_Volume_10^6 ltr'] )/\
                sum([routing[col]['Volume_10^6 ltr']
                 for col in routing.keys() if 'Inflow' in col or 'Initial' in col]) + B



# For model predictive control
# TODO: options dict/list
def update_controls(eval_inp_file,options,j,ctrls):
    inp = read_inp_file(eval_inp_file)
    # Use control rules
    for idx,k in enumerate(inp['CONTROLS']):
        acts = inp['CONTROLS'][k].actions
        for i,act in enumerate(acts):
            act.value = str(options[act.label][ctrls[idx][i]])
            # act.value = setting[i]
        inp['CONTROLS'][k].actions = acts
    eval_inp_file = eval_inp_file.strip('.inp')+'_%s.inp'%j
    inp.write_file(eval_inp_file)
    return eval_inp_file
    
def eval_cost(rpt_file,target):
    rpt_helper = {'cumflooding':('node_flooding_summary','Total_Flood_Volume_10^6 ltr'),
    'totalinflow':('outfall_loading_summary','Total_Volume_10^6 ltr')}
    rpt = read_rpt_file(rpt_file)
    cost = []
    for ID,attr,weight in target:
        helper = rpt_helper[attr]
        table = getattr(rpt,helper[0])
        if table is None:
            cost.append(0)
            continue
        series = table[helper[1]]
        # if k[2] == 'ALL':
        #     target = series.sum()
        # elif k[2] == 'AVERAGE':
        #     target = series.mean()
        # elif k[2] == 'MAX':
        #     target = series.max()
        # elif k[2] == 'MIN':
        #     target = series.min()     
        # else:
        #     target = series[k[2]]
        if ID == 'system':
            target = series.sum()
        elif ID not in series.index:
            continue
        else:
            target = series[ID]
        cost.append(target*weight)
    return cost

def evaluate(eval_inp_file,target):
    rpt_file,_ = swmm5_run(eval_inp_file,create_out=False)
    cost = eval_cost(rpt_file,target)
    return cost


def evaluate_parallel(eval_inp_files,config):
    swmm5_run_parallel(eval_inp_files,processes = config['PROCESSES'])
    costs = []
    for file in eval_inp_files:
        rpt_file = file.replace('.inp','.rpt')
        cost = eval_cost(rpt_file,config['TARGET'])
        costs.append(cost)
    return costs