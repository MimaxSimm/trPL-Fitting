# Import necessary libraries
import warnings, os, sys, shutil
# remove warnings from the output
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import torch, copy, uuid
import pySIMsalabim as sim
from pySIMsalabim.experiments.JV_steady_state import *
import ax, logging
from ax.utils.notebook.plotting import init_notebook_plotting, render

#Imported the github folder here.
sys.path.append('../') # add the path to the optimpv modul
from optimpv_github import *
from optimpv_github.RateEqfits.RateEqAgent import RateEqAgent
from optimpv_github.RateEqfits.RateEqModel import *
from optimpv_github.RateEqfits.Pumps import *

# Check that it is the right optimPV imported
import importlib
spec = importlib.util.find_spec('optimpv_github')

import httpimport
import sys
with httpimport.github_repo('MimaxSimm', 'trPL_Analysis', ref='master'):
    import trPL_importClass

from optimpv_github.axBOtorch.axBOtorchOptimizer import axBOtorchOptimizer
from botorch.acquisition.logei import qLogNoisyExpectedImprovement 
from ax.adapter.transforms.standardize_y import StandardizeY
from ax.adapter.transforms.unit_x import UnitX
from ax.adapter.transforms.remove_fixed import RemoveFixed
from ax.adapter.transforms.log import Log
from ax.generators.torch.botorch_modular.utils import ModelConfig
from ax.generators.torch.botorch_modular.surrogate import SurrogateSpec
from gpytorch.kernels import MaternKernel
from gpytorch.kernels import ScaleKernel
from botorch.models import SingleTaskGP

import copy
import pickle
from optimpv.general.general import *
from optimpv.scipyOpti.scipyOptimizer import ScipyOptimizer
import time



def importALL_data(dirs, L_layer = 800e-9, alpha = 36300*1e2, plot = False):
    trPL_list = []
    N0_list = []
    data2fit_list = []
    if (plot):
        fig, axes = plt.subplots(nrows=len(dirs), ncols=4, figsize=(12, 12))

    for idy, dir in enumerate(dirs):
        # Note that the BG and thicknesses dont affect the raw import, only processing for other funstionalities of the library.
        trPL = trPL_importClass.trPL_measurement_series(dir, BG = 1.55, thickness = L_layer, TRPL_denoise = 50, mode = "auto", retime = True, importPL=True, importSPV=False)
        trPL_list.append(trPL)
        # import Data, the :4 is to select only the last N.

        #Adapt N0 value
        Fluence = np.array(trPL.TRPL_powers)*trPL.BD_ratio*trPL.lambda_laser/(trPL.spot_area*np.array(trPL.TRPL_reprates_Hz)*trPL.hc)
        z_array = np.linspace(0, L_layer, 100)
        ns = alpha * Fluence[3]*np.exp(-alpha*z_array)
        plt.semilogy(z_array, ns)
        N0_ave = np.trapezoid(ns, x = z_array)/L_layer
        N0_list.append(N0_ave)
        
        power = []
        pmax = np.amax(trPL.TRPL_powers)
        cut = 1
        if(idy > 1):
            cut = 1e-5
        for idx, file in enumerate(trPL.TRPLs_files[:4]):
            t = trPL.TRPLs_ts[:,idx]
            PL = trPL.TRPLs_subsMean[:,idx]
            PL = PL[(t>=0) & (t<=cut)]
            t = t[(t>=0) & (t<=cut)]
            # interpolate the data to have a logarithmically spaced time axis
            t_log = np.logspace(np.log10(t[1]), np.log10(t[-1]), num=1000)
            t_log = np.insert(t_log, 0, 0)  # add 0 to the time array
            # interpolate the trPL values
            trPL_log = np.interp(t_log, t, PL)

            power.append(trPL.TRPL_powers[idx])
            if idx == 0:
                data2fit = {'t': t_log , 'trPL': trPL_log, 'G_frac': power[-1] / pmax * np.ones_like(t_log)}
            else:
                data2fit['t'] = np.concatenate((data2fit['t'], t_log ))
                data2fit['trPL'] = np.concatenate((data2fit['trPL'], trPL_log))
                data2fit['G_frac'] = np.concatenate((data2fit['G_frac'], power[-1] / pmax * np.ones_like(t_log)))
            
            if (plot):
                # plot the data
                ax = axes[idy, idx]
                ax.plot(t, PL, 'o', label='raw')
                ax.plot(t_log, trPL_log, '*', label='Interpolated trPL')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('trPL [a.u.]')
                ax.set_title(f'P = {power[-1]:.2e} $\\mu$W')
                ax.legend()

        data2fit= pd.DataFrame(data2fit)
        data2fit_list.append(data2fit)
    if (plot):
        plt.tight_layout()
        plt.show()

    return trPL_list, N0_list, data2fit_list

def define_baseParams(Lval = 800e-9, alphaval = 36300*1e2):
    # Define the parameters to be fitted
    params = []

    Eg = FitParam(name = 'Eg', value = 1.553, bounds = [0.5,2.0], log_scale = False, rescale = True, value_type = 'float', type='fixed', display_name=r'$E_g$', unit='eV', axis_type = 'linear')
    params.append(Eg)

    L = FitParam(name = 'L', value = Lval, bounds = [400e-9,1e-6], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$L$', unit='m', axis_type = 'linear',force_log=True)
    params.append(L)

    alpha = FitParam(name = 'alpha', value = alphaval, bounds = [1e6,1e8], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$\alpha$', unit='m$^{-1}$', axis_type = 'log',)
    params.append(alpha)

    N_cv = FitParam(name = 'N_cv', value = 2e24, bounds = [1e19,1e26], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$N_{cv}$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
    params.append(N_cv)

    k_direct = FitParam(name = 'k_direct', value = 1.96e-17, bounds = [5e-18,5e-17], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{direct}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(k_direct)

    mu_n = FitParam(name = 'mu_n', value = 1.2e-4, bounds = [1e-6,1e-3], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$\mu_n$', unit='m$^{2}$ V$^{-1}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(mu_n) # 4e-1*1e-4

    mu_p = FitParam(name = 'mu_p', value = 4e-5, bounds = [1e-6,1e-3], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$\mu_p$', unit='m$^{2}$ V$^{-1}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(mu_p) # 4e-1*1e-4

    N_t_bulk_1 = FitParam(name = 'N_t_bulk_1', value = 1.85e23, bounds = [1e19,1e26], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$N_{t,\text{bulk}}$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
    params.append(N_t_bulk_1)

    C_n_1 = FitParam(name = 'C_n_1', value = 4.24e-15, bounds = [1e-22,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{n,1}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(C_n_1)

    C_p_1 = FitParam(name = 'C_p_1', value = 8.85e-19, bounds = [1e-22,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{p,1}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(C_p_1)

    E_t_bulk_1 = FitParam(name = 'E_t_bulk_1', value = 0.2, bounds = [0.02,Eg.value-0.02], log_scale = False, rescale = True, value_type = 'float', type='range', display_name=r'$E_{t,\text{bulk}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=False)
    params.append(E_t_bulk_1)

    N_t_bulk_2 = FitParam(name = 'N_t_bulk_2', value = 1.36e21, bounds = [1e19,1e26], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$N_{t,\text{bulk}}$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
    params.append(N_t_bulk_2)

    C_n_2 = FitParam(name = 'C_n_2', value = 1e-14, bounds = [1e-22,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{n,2}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(C_n_2)

    C_p_2 = FitParam(name = 'C_p_2', value = 1.13e-16, bounds = [1e-22,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{p,2}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(C_p_2)

    E_t_bulk_2 = FitParam(name = 'E_t_bulk_2', value = 1.34, bounds = [0.02,Eg.value-0.02], log_scale = False, rescale = True, value_type = 'float', type='range', display_name=r'$E_{t,\text{bulk}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=False)
    params.append(E_t_bulk_2)

    N_t_bulk_3 = FitParam(name = 'N_t_bulk_3', value = 1.36e21, bounds = [1e19,1e26], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$N_{t,\text{bulk}}$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
    params.append(N_t_bulk_3)

    C_n_3 = FitParam(name = 'C_n_3', value = 1e-14, bounds = [1e-22,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{n,2}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(C_n_3)

    C_p_3 = FitParam(name = 'C_p_3', value = 1.13e-16, bounds = [1e-22,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{p,2}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(C_p_3)

    E_t_bulk_3 = FitParam(name = 'E_t_bulk_3', value = 1.34, bounds = [0.02,Eg.value-0.02], log_scale = False, rescale = True, value_type = 'float', type='range', display_name=r'$E_{t,\text{bulk}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=False)
    params.append(E_t_bulk_3)

    I_factor_PL = FitParam(name = 'I_factor_PL', value = 1.275e-22, bounds = [1e-27,1e-20], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$I_{\text{PL}}$', unit='-', axis_type = 'log', force_log=True)
    params.append(I_factor_PL) # in the following we weill fit the PL with the normalized log transformation so this factor is not useful and can be fixed to any value

    # original values
    params_orig = copy.deepcopy(params)
    num_free_params = 0
    dum_dic = {}
    for i in range(len(params)):
        if params[i].force_log:
            dum_dic[params[i].name] = np.log10(params[i].value)
        else:
            dum_dic[params[i].name] = params[i].value/params[i].fscale
    # we need this just to run the model to generate some fake data

        if params[i].type != 'fixed':
            num_free_params += 1

    return params, params_orig

def define_rateEq_andOpti(params, data_2fit, N0, parameter_constraints = [f'mu_n - mu_p >= 0']):
    # original values
    params_orig = copy.deepcopy(params)
    num_free_params = 0
    dum_dic = {}
    for i in range(len(params)):
        if params[i].force_log:
            dum_dic[params[i].name] = np.log10(params[i].value)
        else:
            dum_dic[params[i].name] = params[i].value/params[i].fscale
    # we need this just to run the model to generate some fake data
        if params[i].type != 'fixed'    :
            num_free_params += 1

    # Plot the data to be fitted and the initial guess
    time = data_2fit['t'].values # time in seconds
    X = np.asarray(data_2fit[['t', 'G_frac']])
    y = np.asarray(data_2fit['trPL'])
    fpu = 10e3 # Frequency of the pump laser in Hz
    N0 = N0
    background = 0e28 # Background illumination 

    # Define the Agent and the target metric/loss function
    metric = 'nrmse'
    loss = 'linear' # 'nrmse' or 'mse' or 'soft_l1' or 'linear'
    pump_args = { 'fpu': fpu , 'background' : background, 'N0': N0,}
    exp_format = 'trPL' # experiment format
    model = partial(DBTD_multi_trap, method='LSODA', dimensionless=False, timeout=120, timeout_solve=120)

    RateEq = RateEqAgent(params, [X], [y], model = model, pump_model = initial_carrier_density, pump_args = pump_args, fixed_model_args = {}, metric = metric, 
                                    loss = loss,minimize=True,exp_format=exp_format,detection_limit=0e-5,  compare_type ='normalized_log',do_G_frac_transform=True,parallel=False)

    model_gen_kwargs_list = None
    # Here we add some constraints to the parameters to help the optimizer
    
    model_kwargs_list = [{},{"torch_device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),'botorch_acqf_class':qLogNoisyExpectedImprovement,'transforms':[RemoveFixed, Log,UnitX, StandardizeY],
    'surrogate_spec':SurrogateSpec(model_configs=[ModelConfig(botorch_model_class=SingleTaskGP,covar_module_class=ScaleKernel, covar_module_options={'base_kernel':MaternKernel(nu=2.5, ard_num_dims=num_free_params)})])}]

    if not(parameter_constraints == None):
        optimizer_turbo = axBOtorchOptimizer(params = params, agents = RateEq, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,600], batch_size = [10,4], 
        ax_client = None,  max_parallelism = 100, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = model_gen_kwargs_list, name = 'ax_opti',parallel_agents= True, parameter_constraints = parameter_constraints)
    else:
        # For optimization down the line where mun and mup are fixed.
        optimizer_turbo = axBOtorchOptimizer(params = params, agents = RateEq, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,600], batch_size = [10,4], 
        ax_client = None,  max_parallelism = 100, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = model_gen_kwargs_list, name = 'ax_opti',parallel_agents= True)

    return RateEq, optimizer_turbo

def import_previous_run_params(file_path, mu_fixed=False):

    params, params_orig = define_baseParams()

    with open(file_path, 'rb') as file:
        RateEq = pickle.load(file)

    

    for p in (params):
        for pNew in (RateEq.params):
            if(p.name == pNew.name):
                p.value = pNew.value

        if (mu_fixed):
            if ("mu" in p.name or "k_direct" in p.name):
                p.type = "fixed"
        else:
            if ("k_direct" in p.name):
                p.type = "fixed"

    for p, pB in zip(RateEq.params, params):
        print(p.name)
        assert p.value == pB.value
        print(pB.value, pB.type)

    return params

if __name__ == "__main__":
    print("Imported optimpv", spec)

    # Import the Data
    dir2Q3_before = os.path.abspath('./PowerDep-Before/Fitted')
    dir2Q3_after21H = os.path.abspath('./PowerDep-After20Hours1SUN')

    dir3Q2_before = os.path.abspath('./3Q2/Before')
    dir3Q2_after21H = os.path.abspath('./3Q2/After21H')

    savedir = os.path.abspath('./Results-allData_RUN4_LSODA_10tries_withBestFitfromRUN3')

    dirs = [dir2Q3_before, dir2Q3_after21H, dir3Q2_before, dir3Q2_after21H]
    
    trPL_list, N0_list, data2fit_list = importALL_data(dirs, L_layer = 800e-9, alpha = 36300*1e2) #L and alpha are measured from the films.

    params = import_previous_run_params(file_path=r"./Results-allData_RUN3_LSODA_10tries_NewConstraints/02-2Q3Before-nrmse00173-RateEqAgent.pickle", mu_fixed = True)

    names = ["-2Q3Before", "-2Q3After", "-3Q2Before", "-3Q2After"]
    exp_format = 'trPL'
    #First loops, do 10 optimisatiosn for each condition, save the data appropriately.
    for id_samp, (N0, data) in enumerate(zip(N0_list, data2fit_list)):

        if(id_samp == 0):
            continue
        
        #params, params_orig = define_baseParams()
        nrmse_best = 1e5
        RateEq_best = None

        y_experimental = np.asarray(data['trPL'])
        X_experimental = np.asarray(data[['t', 'G_frac']])

        time_init = time.time()
        # Do the first 10 iterations
        for i in range(10):
            
            print("----------------------------"+names[id_samp]+"; Iteration:", i, "-------------------------------")
            # time_hours = (time.time() - time_init)/3600
            # if (time_hours > 24):
            #     break

            params_loop = copy.deepcopy(params)
            constraints =  [f' -N_t_bulk_1 - 0.5 * C_n_1 - 0.5 * C_p_1  - N_t_bulk_2 - 0.5 * C_n_2 - 0.5 * C_p_2 - N_t_bulk_3 - 0.5 * C_n_3 - 0.5 * C_p_3<= -5',f'E_t_bulk_1-E_t_bulk_2 <= -0.1',f'E_t_bulk_2 - E_t_bulk_3 <= -0.1']
            RateEq, optimizer_turbo = define_rateEq_andOpti(params_loop, data, N0, parameter_constraints = constraints)

            turbo = copy.deepcopy(optimizer_turbo)

            
            if (i == 0):
                RateEq_best = copy.deepcopy(RateEq)

            try:
                turbo.optimize_turbo(force_continue=False,kwargs_turbo_state={'failure_tolerance':8}) # run the optimization with turbo

                ax_client = turbo.ax_client # get the ax client
                turbo.update_params_with_best_balance() # update the params list in the turbo with the best parameters
                RateEq.params = turbo.params # update the params list in the agent with the best parameters
                
                y_test = RateEq.run({},exp_format=exp_format)
                y_transformed, y_pred_transformed = transform_data(y_experimental,y_test, transform_type='normalized_log', do_G_frac_transform=True, X=X_experimental)
                nrmse = calc_metric(y_transformed, y_pred_transformed, metric_name='nrmse')

                if (nrmse < nrmse_best):
                    nrmse_best = nrmse
                    RateEq_best = copy.deepcopy(RateEq)
                    print("---------------------------- New Best nrmse:", nrmse_best)

                # Save result
                file_path = os.path.join(savedir,'0'+str(i)+names[id_samp]+f'-nrmse{nrmse*10000:05.0f}-optimizer.pickle')
                with open(file_path, 'wb') as file:
                    pickle.dump(turbo, file)

                file_path = os.path.join(savedir,'0'+str(i)+names[id_samp]+f'-nrmse{nrmse*10000:05.0f}-RateEqAgent.pickle')
                with open(file_path, 'wb') as file:
                    pickle.dump(RateEq, file)

            except: 
                print("---------------------------- Failed Opti Round", i, "for", names[id_samp])
                nrmse = 0.1
                # Save result
                file_path = os.path.join(savedir,'0'+str(i)+names[id_samp]+f'-nrmse{nrmse*10000:05.0f}-optimizer.pickle')
                with open(file_path, 'wb') as file:
                    pickle.dump(turbo, file)

                file_path = os.path.join(savedir,'0'+str(i)+names[id_samp]+f'-nrmse{nrmse*10000:05.0f}-RateEqAgent.pickle')
                with open(file_path, 'wb') as file:
                    pickle.dump(RateEq, file)


        for id_samp2, (N02, data2) in enumerate(zip(N0_list, data2fit_list)):
            i = i + 1
            params_loop = copy.deepcopy(params)

            for p in params_loop:
                for pB in RateEq_best.params:
                    if (p.name == pB.name):
                        p.value = pB.value
                    if (p.name == 'k_direct' or 'mu' in p.name):
                        p.type = 'fixed'

            #No parameter contraints, as we are fixing mu and k_direct.
            constraints =  [f' -N_t_bulk_1 - 0.5 * C_n_1 - 0.5 * C_p_1  - N_t_bulk_2 - 0.5 * C_n_2 - 0.5 * C_p_2 - N_t_bulk_3 - 0.5 * C_n_3 - 0.5 * C_p_3<= -5',f'E_t_bulk_1-E_t_bulk_2 <= -0.1',f'E_t_bulk_2 - E_t_bulk_3 <= -0.1']
            RateEq2, optimizer_turbo2 = define_rateEq_andOpti(params_loop, data2, N02, parameter_constraints=constraints)

            params_scipy = copy.deepcopy(RateEq2.params) # make copies of the parameters
            RateEq_scipy = copy.deepcopy(RateEq2) # make a copy of the agent
            
            scipyOpti = ScipyOptimizer(params=params_scipy, agents=RateEq_scipy, method='SLSQP', options={'max_nfev':2000,'xtol':1e-13}, name='scipy_opti', 
                                        parallel_agents=True, max_parallelism=os.cpu_count()-1, verbose_logging=True)
            try:
                #scipyOpti.optimize() # run the optimization with scipy
                RateEq_scipy.params = scipyOpti.params

                y_experimental = np.asarray(data2['trPL'])
                X_experimental = np.asarray(data2[['t', 'G_frac']])

                y_test = RateEq_scipy.run({},exp_format=exp_format)
                y_transformed, y_pred_transformed = transform_data(y_experimental,y_test, transform_type='normalized_log', do_G_frac_transform=True, X=X_experimental)
                nrmse = calc_metric(y_transformed, y_pred_transformed, metric_name='nrmse')

                # Save result
                file_path = os.path.join(savedir,str(i)+names[id_samp2]+f'-nrmse{nrmse*10000:05.0f}-scipyoptimizer-10s-from'+names[id_samp]+'.pickle')
                with open(file_path, 'wb') as file:
                     pickle.dump(scipyOpti, file)
    
                file_path = os.path.join(savedir,str(i)+names[id_samp2]+f'-nrmse{nrmse*10000:05.0f}-scipyRateEqAgent-10s-from'+names[id_samp]+'.pickle')
                with open(file_path, 'wb') as file:
                     pickle.dump(RateEq_scipy, file)
            except:
                print("---------------------------- Failed Scipy for", names[id_samp2], "with", names[id_samp])