#!/opt/software/anaconda/python-3.10.9/bin/python

from mpi4py import MPI

comm = MPI.COMM_WORLD #grabs communications means, finds a position in the data for something to be stored? (assuming)

nproc = comm.Get_size() #finds how many cores are being used, to determine the number of ranks needed under notation 1 integer for each core.

# The first processor is leader, so one fewer available to be a worker.
nworkers = nproc - 1

# samples
N = 100000000 # repeats for integrand calculation to get it to an accurate number of significant figures.
delta = 1.0 / N 

# integral
I = 0.0

def integrand(x):
  return(4.0 / (1.0 + x*x)) # x*x is faster than x**2 for massive repeats of calculations.

# By having a decision maker for each iteration, rank 0 chooses randomly which rank processes every iteration. The processing of this single iteration is then added to the
# total, this is repeated for each iteration. This is the key bottleneck with this code. To work around this, we can eliminate the leader core and just assign chunks of N to every
# available processor. This means it can all be calculated in parallel and then summated with comm.reduce.

# A third bottleneck lies in using python inherently, by using a for loop in a high-level langauage for the integrand calculations makes them quite slow. 
# To improve this we could attempt Numpy vectorisation? This would minimise the number of loops required to calculate the intgral.

if comm.Get_rank() == 0:

  # Leader: choose points to sample function, send to workers and
  # collect their contributions. Also calculate a sub-set of points.

  for i in range(0,N):

    # decide which rank evaluates this point, slow and unnesccary, tasks can be automatically distributed evenly amoung the cores.
    j = i % nproc

    # mid-point rule
    x = (i+0.5) * delta

	# this loop is pure shit and means we cant summate calculated values to find the final value of the integral. This slows computation further.

    if j == 0:
      # so do this locally using the leader machine
      y = integrand(x) * delta
    else:
      # communicate to a worker
      comm.send(x, dest=j)
      y = comm.recv(source=j)

    I += y

	# This section just shuts down all the cores, sequentially though then prints the integral. This is slow with a for loop.

  # Shut down the workers
  for i in range(1, nproc):
    comm.send(-1.0, dest=i)

  print("Bad Code Integral %.10f" % I)

# I believe the below section of the code is a failsafe to stop cores being used unnecessarily when integrand calculation fails? This won't have a large impact on computation speed.

else:

  # Worker: waiting for something to happen, then stop if sent message
  # outside the integral limits

  while True:

    x = comm.recv(source=0)

    if x < 0.0:
      # stop the worker
      break

    else:
      comm.send(integrand(x) * delta, dest=0)
