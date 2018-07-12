import numpy as np
from numba import jit,njit
from exponentiate import expm

### THIS SCRIPT NOW USES EFFECTIVE MISMATCHES! ###
    
def boltzmann_model_pos_dependent_faster(xdata,params):
    
    # Extract parameters
    DC = np.zeros(20)
    DI = np.zeros(20)
    DPAM = 0.0
    for i in range(41):
        if i<20:
            DC[i] = -1.0*params[i]
        elif i<40:
            DI[i-20] = params[i]
        elif i == 40:
            DPAM = params[i]
       
    
    # Precalculate pbound:
    bm_matrix = np.ones((400,21))
    bm_matrix[:,0] = DPAM
    bm_matrix[:,1:] = DC
    for i in range(1,21):
        for j in range(i,21):
            bm_matrix[(i-1)*20+(j-1),i] += DI[i-1]
            bm_matrix[(i-1)*20+(j-1),j] += DI[j-1]
    
    # Write out cumsum
    bm_matrix_sum = np.zeros((400,21))
    bm_matrix_sum[:,0] = DPAM
    for row in range(400):
        for column in range(1,21):
            bm_matrix_sum[row,column] = bm_matrix_sum[row,column-1] + bm_matrix[row,column]
    
    
    pbound = np.sum(np.exp(-bm_matrix_sum),axis=1)/(1.0+np.sum(np.exp(-bm_matrix_sum),axis=1))
    
    # Normalize by on-target
    ot1 = np.zeros(21)
    ot1[0] = DPAM
    for i in range(1,21):
        ot1[i] = ot1[i-1]+DC[i-1]
    ot = np.sum(np.exp(-ot1))/(1.0+np.sum(np.exp(-ot1)))
    pbound = pbound/ot
    
    ymodel = pbound[xdata]
            
    # Return the relative probability of being bound for this sequence
    return ymodel



def off_rate_all_in(xdata,params):

    # Extra:
    nr_datapoints = 3
    deltaT = 500.0

    # Extract parameters
    DC = np.zeros(20)
    DI = np.zeros(20)
    DPAM = 0.0
    attempt_rate = 0.0
    on_rate = 0.0
    for i in range(43):
        if i<20:
            DC[i] = -1.0*params[i]
        elif i<40:
            DI[i-20] = params[i]
        elif i == 40:
            DPAM = params[i]
        elif i == 41:
            attempt_rate = params[i]
        elif i == 42:
            on_rate = params[i]
    
    # ON TARGET:
    mismatch_pos = [0]
    slope_on_target = get_slope(DC,DI,DPAM,attempt_rate,on_rate,mismatch_pos,nr_datapoints,deltaT)
    
    # MISMATCHED:
    slope_array = np.zeros(400)
    for i in range(1,21):
        for j in range(i,21):
            slope_array[(i-1)*20+(j-1)] = get_slope(DC,DI,DPAM,attempt_rate,on_rate,[i,j],nr_datapoints,deltaT)
    
    
        
    # RETURN
    slope = np.zeros(len(xdata))
    for i in range(len(xdata)):
        slope[i] = slope_array[xdata[i]]
    return slope
    
@jit(cache=True)     
def get_energy_landscape(DC,DI,DPAM,mismatch_pos):
    
    A = 0
    A = len(mismatch_pos)
    mismatch_pos_unique = np.zeros(A)
    mismatch_pos_unique = np.unique(mismatch_pos)
    energy_landscape = np.zeros(22)
    energy_landscape[1] = DPAM
    energy_landscape[2:] = DC
    if mismatch_pos[0] == 0:
        return energy_landscape
    else:
        for mismatch in mismatch_pos_unique:
            energy_landscape[mismatch+1] += DI[mismatch-1]

        return energy_landscape
    
@njit(cache=True)      
def get_eq_dist(energy_landscape):
    energy_landscape_sum = np.zeros(len(energy_landscape))
    energy_landscape_sum[0] = energy_landscape[0]
    for i in range(1,len(energy_landscape)):
        energy_landscape_sum[i] = energy_landscape_sum[i-1] + energy_landscape[i]
    Z = np.sum(np.exp(-energy_landscape_sum))
    return np.exp(-energy_landscape_sum)/Z
    
    
@njit(cache=True)
def get_rates(energy_landscape_diff,attempt_rate,on_rate):
           
    # Construct array containing forward rates
    forward_rates = np.ones(len(energy_landscape_diff))*attempt_rate
    forward_rates[0] = on_rate
    forward_rates[-1] = 0.0
    
    # Construct array containing backward rates
    backward_rates = attempt_rate*np.exp(energy_landscape_diff)
    backward_rates[1] = on_rate*np.exp(energy_landscape_diff[1])
    backward_rates[0] = 0.0
    
    return forward_rates, backward_rates
    
@njit(cache=True)
def build_rate_matrix(forward_rates, backward_rates):

    diagonal1 = -(forward_rates+backward_rates)
    diagonal2 = backward_rates[1:]
    diagonal3 = forward_rates[:-1]
    rate_matrix = np.zeros((len(forward_rates),len(forward_rates)))                                 # Build the matrix
    for d in range(3):
        if d == 0:
            rate_matrix = rate_matrix + np.diag(diagonal1,k=0)
        elif d == 1:
            rate_matrix = rate_matrix + np.diag(diagonal2,k=1)
        elif d == 2:
            rate_matrix = rate_matrix + np.diag(diagonal3,k=-1)
    
    return rate_matrix
    
def get_slope(DC,DI,DPAM,attempt_rate,on_rate,mismatch_pos,nr_datapoints,deltaT):
    
    energy_landscape_diff = get_energy_landscape(DC,DI,DPAM,mismatch_pos)
    [forward_rates, backward_rates] = get_rates(energy_landscape_diff,attempt_rate,on_rate)
    rate_matrix = build_rate_matrix(forward_rates, backward_rates)
    rate_matrix[0][0] = 0.0
    rate_matrix[1][0] = 0.0
    
    simOccupancy = np.zeros((22,nr_datapoints))
    occupancy_array = get_eq_dist(energy_landscape_diff)
    occupancy_array[0] = 0.0
    occupancy_array = occupancy_array/np.sum(occupancy_array)
    
    for i in range(nr_datapoints):
        simOccupancy[:,i] = np.dot(expm(rate_matrix*deltaT*(i+1)),occupancy_array)
    unbound_fraction = simOccupancy[0,:]
    
    # Do linear fit to the bound fraction
    x_points = np.zeros(nr_datapoints)
    for x in range(1,nr_datapoints+1):
        x_points[x-1] = deltaT*x
    y_points = unbound_fraction
    pfit  = least_squares(x_points,y_points)
    slope = pfit[1]
    
    return slope
   
@njit(cache=True)    
def least_squares(x_points,y_points):

    size = len(x_points)
    X = np.ones((size,2))
    X[:,1] = x_points
    
    XT = np.ones((2,size))
    XT[1,:] = x_points
    
    Y = y_points
    
    a = np.dot(np.dot(np.linalg.inv(np.dot(XT,X)),XT),Y)
    
    return a
    

    
def on_rate_all_in(xdata,params):
    
    # Extra:
    nr_datapoints = 3
    deltaT = 500.0

    # Extract parameters
    DC = np.zeros(20)
    DI = np.zeros(20)
    DPAM = 0.0
    attempt_rate = 0.0
    on_rate = 0.0
    for i in range(43):
        if i<20:
            DC[i] = -1.0*params[i]
        elif i<40:
            DI[i-20] = params[i]
        elif i == 40:
            DPAM = params[i]
        elif i == 41:
            attempt_rate = params[i]
        elif i == 42:
            on_rate = params[i]    
    
    
    slope_array = np.zeros(400)
    for i in range(1,21):
        for j in range(i,21):
            
            energy_landscape_diff = get_energy_landscape(DC,DI,DPAM,[i,j])
            [forward_rates, backward_rates] = get_rates(energy_landscape_diff,attempt_rate,on_rate)
            rate_matrix = build_rate_matrix(forward_rates, backward_rates)
            
            occupancy_array = np.zeros(22)
            occupancy_array[0] = 1
            simOccupancy = np.zeros((22,nr_datapoints))
            for k in range(nr_datapoints):
                simOccupancy[:,k] = np.dot(expm(rate_matrix*deltaT*(k+1)),occupancy_array)
            bound_fraction = 1-simOccupancy[0,:]
            
            x_points = np.zeros(nr_datapoints)
            for x in range(1,nr_datapoints+1):
                x_points[x-1] = deltaT*x
            y_points = bound_fraction
            slope = np.sum( x_points*y_points )/np.sum( x_points*x_points )
            
            slope_array[(i-1)*20+(j-1)] = slope

    slope_result = np.zeros(len(xdata))
    for i in range(len(xdata)):
        slope_result[i] = slope_array[xdata[i]]
    return slope_result
    


### Final Model ###

def occ_final(xdata,params):
    
    # Extract parameters
    DC1   = -params[0]
    DC2   = -params[1]
    DI1   = params[2]
    DI2   = params[3]
    DPAM  = params[4]
    bp_dc = params[5]
    bp_di = params[6]
    
    DC = np.zeros(20)
    DI = np.zeros(20)
    
    DC[:int(np.floor(bp_dc))] = DC1
    DC[int(np.ceil(bp_dc)):]  = DC2
    DC[DC==0.0] = DC1*(bp_dc-np.floor(bp_dc)) + DC2*(np.ceil(bp_dc)-bp_dc)
    
    DI[:int(np.floor(bp_di))] = DI1
    DI[int(np.ceil(bp_di)):]  = DI2
    DI[DI==0.0] = DI1*(bp_di-np.floor(bp_di)) + DI2*(np.ceil(bp_di)-bp_di)
    
    # Precalculate pbound:
    bm_matrix = np.ones((400,21))
    bm_matrix[:,0] = DPAM
    bm_matrix[:,1:] = DC
    for i in range(1,21):
        for j in range(i,21):
            if i == j:
                bm_matrix[(i-1)*20+(j-1),i] += DI[i-1]
            else:
                bm_matrix[(i-1)*20+(j-1),i] += DI[i-1]
                bm_matrix[(i-1)*20+(j-1),j] += DI[j-1]
                
    # Write out cumsum
    bm_matrix_sum = np.zeros((400,21))
    bm_matrix_sum[:,0] = DPAM
    for row in range(400):
        for column in range(1,21):
            bm_matrix_sum[row,column] = bm_matrix_sum[row,column-1] + bm_matrix[row,column]
    
    
    pbound = np.sum(np.exp(-bm_matrix_sum),axis=1)/(1.0+np.sum(np.exp(-bm_matrix_sum),axis=1))
    
    # Normalize by on-target
    ot1 = np.zeros(21)
    ot1[0] = DPAM
    for i in range(1,21):
        ot1[i] = ot1[i-1]+DC[i-1]
    ot = np.sum(np.exp(-ot1))/(1.0+np.sum(np.exp(-ot1)))
    pbound = pbound/ot
    
    ymodel = pbound[xdata]
            
    # Return the relative probability of being bound for this sequence
    return ymodel
    
def on_final(xdata,params):
    
    # Extra:
    nr_datapoints = 3
    deltaT = 500.0

    # Extract parameters
    DC1   = -params[0]
    DC2   = -params[1]
    DI1   = params[2]
    DI2   = params[3]
    DPAM  = params[4]
    bp_dc = params[5]
    bp_di = params[6]
    attempt_rate = params[7]
    on_rate = params[8]
    
    DC = np.zeros(20)
    DI = np.zeros(20)
    
    DC[:int(np.floor(bp_dc))] = DC1
    DC[int(np.ceil(bp_dc)):]  = DC2
    DC[DC==0.0] = DC1*(bp_dc-np.floor(bp_dc)) + DC2*(np.ceil(bp_dc)-bp_dc)
    
    DI[:int(np.floor(bp_di))] = DI1
    DI[int(np.ceil(bp_di)):]  = DI2
    DI[DI==0.0] = DI1*(bp_di-np.floor(bp_di)) + DI2*(np.ceil(bp_di)-bp_di)
    
    
    slope_array = np.zeros(400)
    for i in range(1,21):
        for j in range(i,21):
            
            energy_landscape_diff = get_energy_landscape(DC,DI,DPAM,[i,j])
            [forward_rates, backward_rates] = get_rates(energy_landscape_diff,attempt_rate,on_rate)
            rate_matrix = build_rate_matrix(forward_rates, backward_rates)
            
            occupancy_array = np.zeros(22)
            occupancy_array[0] = 1
            simOccupancy = np.zeros((22,nr_datapoints))
            for k in range(nr_datapoints):
                simOccupancy[:,k] = np.dot(expm(rate_matrix*deltaT*(k+1)),occupancy_array)
            bound_fraction = 1-simOccupancy[0,:]
            
            x_points = np.zeros(nr_datapoints)
            for x in range(1,nr_datapoints+1):
                x_points[x-1] = deltaT*x
            y_points = bound_fraction
            slope = np.sum( x_points*y_points )/np.sum( x_points*x_points )
            
            slope_array[(i-1)*20+(j-1)] = slope

    slope_result = np.zeros(len(xdata))
    for i in range(len(xdata)):
        slope_result[i] = slope_array[xdata[i]]
    return slope_result

def off_final(xdata,params):

    # Extra:
    nr_datapoints = 3
    deltaT = 500.0

    # Extract parameters
    DC1   = -params[0]
    DC2   = -params[1]
    DI1   = params[2]
    DI2   = params[3]
    DPAM  = params[4]
    bp_dc = params[5]
    bp_di = params[6]
    attempt_rate = params[7]
    on_rate = params[8]
    
    DC = np.zeros(20)
    DI = np.zeros(20)
    
    DC[:int(np.floor(bp_dc))] = DC1
    DC[int(np.ceil(bp_dc)):]  = DC2
    DC[DC==0.0] = DC1*(bp_dc-np.floor(bp_dc)) + DC2*(np.ceil(bp_dc)-bp_dc)
    
    DI[:int(np.floor(bp_di))] = DI1
    DI[int(np.ceil(bp_di)):]  = DI2
    DI[DI==0.0] = DI1*(bp_di-np.floor(bp_di)) + DI2*(np.ceil(bp_di)-bp_di)
    
    # ON TARGET:
    mismatch_pos = [0]
    slope_on_target = get_slope(DC,DI,DPAM,attempt_rate,on_rate,mismatch_pos,nr_datapoints,deltaT)
    
    # MISMATCHED:
    slope_array = np.zeros(400)
    for i in range(1,21):
        for j in range(i,21):
            slope_array[(i-1)*20+(j-1)] = get_slope(DC,DI,DPAM,attempt_rate,on_rate,[i,j],nr_datapoints,deltaT)
    
    
        
    # RETURN
    slope = np.zeros(len(xdata))
    for i in range(len(xdata)):
        slope[i] = slope_array[xdata[i]]
    return slope



    
    
