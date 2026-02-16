import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from experiments.bandit_testbed.kernels import KernelType
from algorithms.utils.printing import ctext

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--D", dest="D", type=int, default=5)
parser.add_argument("--M", dest="M", type=int, default=1)
parser.add_argument("--K", dest="K", type=int, default=1)

parser.add_argument("--T", dest="T", type=int, default=20)

parser.add_argument("--seed", dest="seed", type=int, default=0)
parser.add_argument("--kernel", dest="kernel", type=int, default=None)
parser.add_argument("--delta", dest="delta", type=int, default=None)
parser.add_argument("--epsilon", dest="epsilon", type=int, default=None)

parser.add_argument("--style", type=str, default="stationary")

parser.add_argument("--optimistic-init", dest="optimistic_init", action='store_true')
parser.set_defaults(optimistic_init=False)

parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument("--verbose", action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)

parser.set_defaults(plot=False)

args = parser.parse_args()

kernel_type = KernelType(args.kernel)


# CHECK EXPERIMENT HAS BEEN RAN
experiment_str = "kernel={},T={},D={},M={},K={},eps={},delta={}"
experiment_str = experiment_str.format(kernel_type.name, args.T, args.D, args.M, args.K, args.epsilon, args.delta)

file_name = f"{experiment_str}.npz"
datapath = os.path.join("results", file_name)

if not os.path.exists(datapath):
    error_msg = "No Experiment Found for: kernel={}, T={}, D={}, M={}, K={}, eps={}, delta={}"
    error_msg = error_msg.format(kernel_type.name, args.T, args.D, args.M, args.K, args.epsilon, args.delta)
    print(ctext(error_msg, "red"))
    exit()

# GENERATE PLOTS
if not os.path.exists("plots"):
    os.mkdir("plots")

plotdir = os.path.join("plots", experiment_str)
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

data = np.load(f"{datapath}")


# PLOT THE MEAN OPTIMAL ACTION %
final_pct = data["final_pct"]
plt.figure(figsize=(15, 5))
plt.plot(final_pct[0])
plt.savefig(f"{plotdir}/final_pct.png")
plt.close()

# PLOT THE AVERAGE REWARD
avg_rs = data["avg_rs"]
plt.figure(figsize=(15, 5))
plt.plot(avg_rs[0])
plt.savefig(f"{plotdir}/avg_rs.png")
plt.close()
