from emulator import Emulator
from dataloader import DataGenerator,generate_file
from datetime import datetime
from swmm_api import read_inp_file
from envs import shunqing

def parser():
    
    return

if __name__ == "__main__":
    env = shunqing()

    inp = read_inp_file(env.config['swmm_input'])
    events = generate_file(inp,env.config['rainfall'])
    dG = DataGenerator(env,seq_len=4)
    dG.generate(events,processes=1)

    args = parser()
    emul = Emulator(args.conv,args.edges,args.resnet,args.recurrent,args)

