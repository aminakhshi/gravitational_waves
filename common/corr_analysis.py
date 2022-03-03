import numpy as np
import h5py as hf
import pycbc.waveform as wf
import matplotlib.pyplot as plt
from .utils import p_corr
import pickle
import tarfile
import wget
import copy
import os
from pycbc import pnutils

def match_pair_v1(event_name, data, fs_ref=4096., save_out = True, local_path = None):
    """

    :param event_name:
    :param data:
    :param fs_ref:
    :return:
    """

    first_run = ['GW150914', 'GW151012', 'GW151226']
    second_run = ['GW170104', 'GW170608', 'GW170729',
                  'GW170809', 'GW170814', 'GW170818', 'GW170823']
    third_run = ['GW190412', 'GW190814', 'GW190521']
    # lines 787, 901, 902, 1023, 1039

    if save_out:
        if local_path:
            event_folder = local_path
        else:
            event_folder = 'results/{}'.format(event_name)
            if_exist = os.path.isdir(event_folder)
            if not if_exist:
                os.makedirs(event_folder)
                
    with open("linksDict", "rb") as file:
        get_link = pickle.load(file)
    link_ = get_link[event_name]
    if event_name in third_run:
        print("downloading from: \n" + link_)
        file_ = wget.download(link_)
        print("download finished. \n")
        tar = tarfile.open(file_, "r:")
        tar.extractall()
        tar.close()
        h_file = hf.File(event_name + '.h5')
    else:
        print("downloading from: \n" + link_)
        file_ = wget.download(link_)
        print("download finished. \n")
        h_file = hf.File(file_)

    if event_name in first_run:
        h_samp = h_file['IMRPhenomPv2_posterior']
        aprx_method = 'IMRPhenomPv2'
        variables_ = list(h_samp.dtype.names)
        ''' variables_ :
        ['costheta_jn', 'luminosity_distance_Mpc', 'right_ascension', 'declination', 
        'm1_detector_frame_Msun', 'm2_detector_frame_Msun', 'spin1', 'spin2', 'costilt1', 'costilt2'] 
        '''
        test_ = h_samp[variables_[0]]
        post_realizations = np.zeros((len(test_), len(variables_)))

    elif event_name in second_run:
        # LIGO
        h_samp = h_file['IMRPhenomPv2_posterior']
        aprx_method = 'IMRPhenomPv2'
        variables_ = list(h_samp.dtype.names)
        ''' variables_ :
        ['costheta_jn', 'luminosity_distance_Mpc', 'right_ascension', 'declination', 
        'm1_detector_frame_Msun', 'm2_detector_frame_Msun', 'spin1', 'spin2', 'costilt1', 'costilt2'] 
        '''
        # # Soumi et al:
        # h_file = hf.File('./posteriors_Soumi/' + event_name + '/posteriors_thinned.hdf')
        # h_samp = h_file['samples']
        # aprx_method = 'IMRPhenomPv2'
        # variables_ = list(h_samp.keys())
        # ''' variables_ :
        # ['dec', 'distance', 'inclination', 'mass1', 'mass2', 'polarization', 'ra', 'spin1_a',
        # 'spin1_azimuthal', 'spin1_polar', 'spin2_a', 'spin2_azimuthal', 'spin2_polar', 'tc']
        # '''
        test_ = h_samp[variables_[0]][()]
        post_realizations = np.zeros((len(test_), len(variables_)))

    elif event_name in third_run:
        if event_name == 'GW190412':
            h_samp = h_file['C01:IMRPhenomPv2']['posterior_samples']  # for GW190412
            aprx_method = 'IMRPhenomPv2'  # for GW190412
        if event_name == 'GW190425':
            h_samp = h_file['C01:IMRPhenomPv2_NRTidal-HS']['posterior_samples']  # for GW190425
            aprx_method = 'IMRPhenomPv2'  # for GW190425
        if event_name == 'GW190521':
            h_samp = h_file['C01:IMRPhenomPv3HM']['posterior_samples']  # for GW190521
            aprx_method = 'IMRPhenomPv3HM'  # for GW190521
        if event_name == 'GW190814':
            h_samp = h_file['C01:IMRPhenomD']['posterior_samples']  # for GW190814
            aprx_method = 'IMRPhenomD'  # for GW190814
        # h_samp.dtype
        variables_ = ['mass_1', 'mass_2', 'spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z',
                      'ra', 'dec', 'iota']
        variables_to_pass_td = ['mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z',
                                'ra', 'dec', 'iota']
        test_ = h_samp[variables_[0]][()]
        post_realizations = np.zeros((len(test_), len(variables_)))

    # upLim = 20
    # to_save = np.empty([upLim, 4])
    to_save = np.empty((len(test_), 4))
    to_save[:] = np.nan

    for i in range(len(variables_)):
        post_realizations[:, i] = h_samp[variables_[i]]

    time_array = data[:, 0]
    H_extract = data[:, 1]
    L_extract = data[:, 2]
    time_array_H = time_array
    time_array_L = time_array

    realizations_to_eval = range(len(test_))
    # realizations_to_eval = range(upLim)

    count_ind = -1
    for real_ind in realizations_to_eval:

        count_ind += 1
        print(event_name + ', processing realization: #' + str(count_ind))
        values_ = post_realizations[real_ind, :]

        if event_name in first_run:
            zip_obj_ = zip(variables_, values_)
            dict_ = dict(zip_obj_)
            polar_to_cart = pnutils.jframe_to_l0frame(mass1=dict_['m1_detector_frame_Msun'],
                                                                mass2=dict_['m2_detector_frame_Msun'],
                                                                f_ref=10,
                                                                thetajn=np.arccos(dict_['costheta_jn']),
                                                                spin1_a=dict_['spin1'],
                                                                spin2_a=dict_['spin2'],
                                                                spin1_polar=np.arccos(dict_['costilt1']),
                                                                spin2_polar=np.arccos(dict_['costilt2']))
            params_to_pass_td = {**dict_, **polar_to_cart}
            hp, hc = wf.get_td_waveform(params_to_pass_td,
                                        mass1=dict_['m1_detector_frame_Msun'],
                                        mass2=dict_['m2_detector_frame_Msun'],
                                        inclination=np.arccos(dict_['costheta_jn']),
                                        distance=dict_['luminosity_distance_Mpc'],
                                        approximant=aprx_method,
                                        delta_t=1.0 / fs_ref,
                                        f_lower=10)

        elif event_name in second_run:
            zip_obj_ = zip(variables_, values_)
            dict_ = dict(zip_obj_)
            #LIGO:
            polar_to_cart = pnutils.jframe_to_l0frame(mass1=dict_['m1_detector_frame_Msun'],
                                                                mass2=dict_['m2_detector_frame_Msun'],
                                                                f_ref=10,
                                                                thetajn=np.arccos(dict_['costheta_jn']),
                                                                spin1_a=dict_['spin1'],
                                                                spin2_a=dict_['spin2'],
                                                                spin1_polar=np.arccos(dict_['costilt1']),
                                                                spin2_polar=np.arccos(dict_['costilt2']))
            params_to_pass_td = {**dict_, **polar_to_cart}
            hp, hc = wf.get_td_waveform(params_to_pass_td,
                                        mass1=dict_['m1_detector_frame_Msun'],
                                        mass2=dict_['m2_detector_frame_Msun'],
                                        inclination=np.arccos(dict_['costheta_jn']),
                                        distance=dict_['luminosity_distance_Mpc'],
                                        approximant=aprx_method,
                                        delta_t=1.0 / fs_ref,
                                        f_lower=10)

            # # Soumi et al:
            # polar_to_cart = wf.waveform_modes.jframe_to_l0frame(mass1=dict_['mass1'],
            #                                                     mass2=dict_['mass2'],
            #                                                     f_ref=10,
            #                                                     thetajn=dict_['inclination'],
            #                                                     spin1_a=dict_['spin1_a'],
            #                                                     spin2_a=dict_['spin2_a'],
            #                                                     spin1_polar=dict_['spin1_polar'],
            #                                                     spin2_polar=dict_['spin2_polar'])
            # params_to_pass_td = {**dict_, **polar_to_cart}
            # hp, hc = wf.get_td_waveform(params_to_pass_td,
            #                             approximant=aprx_method,
            #                             delta_t=1.0 / fs_ref,
            #                             f_lower=10)

        elif event_name in third_run:
            zip_obj_ = zip(variables_to_pass_td, values_)
            dict_ = dict(zip_obj_)
            hp, hc = wf.get_td_waveform(dict_,
                                        approximant=aprx_method,
                                        delta_t=1.0 / fs_ref,
                                        f_lower=20)

        waveform_H = hp.data
        waveform_L = hc.data

        # make them the same size
        if len(waveform_H) > len(H_extract):
            ringdown_ = np.where(waveform_H == np.max(waveform_H))  # to make the ringdown central
            waveform_H = waveform_H[
                         int(ringdown_ - np.fix(len(H_extract) / 2)):int(ringdown_ + np.fix(len(H_extract) / 2))]
            H_extract_ = H_extract[0:2 * int(np.fix(len(H_extract) / 2))]  # to compensate for above cut
            # time_array_H_ = time_array_H[0:2 * int(np.fix(len(time_array_H) / 2))]  # to compensate for above cut

        if len(waveform_H) < len(H_extract):
            H_extract_ = H_extract[:len(waveform_H)]
            # time_array_H_ = time_array_H[:len(waveform_H)]

        # make them the same size
        if len(waveform_L) > len(L_extract):
            ringdown_ = np.where(waveform_L == np.max(waveform_L))  # to make the ringdown central
            waveform_L = waveform_L[
                         int(ringdown_ - np.fix(len(L_extract) / 2)):int(ringdown_ + np.fix(len(L_extract) / 2))]
            L_extract_ = L_extract[0:2 * int(np.fix(len(L_extract) / 2))]  # to compensate for above cut
            # time_array_L_ = time_array_L[0:2 * int(np.fix(len(time_array_L) / 2))]  # to compensate for above cut

        if len(waveform_L) < len(L_extract):
            L_extract_ = L_extract[:len(waveform_L)]
            # time_array_L_ = time_array_L[:len(waveform_L)]

        # NOTICE: the input extraction should be centered around the ringdown
        corr_check_range = np.arange(-500, 500)
        corr_arr_H = np.empty(len(corr_check_range))
        corr_arr_L = np.empty(len(corr_check_range))
        H_extract_ = H_extract_ / np.std(H_extract_)
        waveform_H = waveform_H / np.std(waveform_H)
        L_extract_ = L_extract_ / np.std(L_extract_)
        waveform_L = waveform_L / np.std(waveform_L)
        cntr = -1
        for tau in corr_check_range:
            cntr += 1
            cr, _ = p_corr(waveform_H, H_extract_, tau)
            corr_arr_H[cntr] = cr
            cr, _ = p_corr(waveform_L, L_extract_, tau)
            corr_arr_L[cntr] = cr

        # check for nans
        corr_arr_H[np.isnan(corr_arr_H)] = 0
        corr_arr_L[np.isnan(corr_arr_L)] = 0

        # check for non-relevant quantities for p_corr coeff
        corr_arr_H[(corr_arr_H > 1) | (corr_arr_H < -1)] = 0
        corr_arr_L[(corr_arr_L > 1) | (corr_arr_L < -1)] = 0

        max_corr_pos_H = np.where(np.abs(corr_arr_H) == np.max(np.abs(corr_arr_H)))
        # print('the H1 associated coefficient is: ' + str(corr_arr_H[max_corr_pos_H]))
        max_corr_pos_L = np.where(np.abs(corr_arr_L) == np.max(np.abs(corr_arr_L)))
        # print('the L1 associated coefficient is: ' + str(corr_arr_L[max_corr_pos_L]))

        tau_H = corr_check_range[max_corr_pos_H][0]  # 0 index to ensure uniqueness
        tau_L = corr_check_range[max_corr_pos_L][0]  # 0 index to ensure uniqueness
        delay_ = (tau_H - tau_L) / fs_ref

        to_save[real_ind, 0] = corr_arr_H[max_corr_pos_H]
        to_save[real_ind, 1] = corr_arr_L[max_corr_pos_L]
        to_save[real_ind, 2] = delay_
        # to_save[real_ind, 3] = polarization

        print('H1_corr: ' + str(corr_arr_H[max_corr_pos_H]) +
              '\nL1_corr: ' + str(corr_arr_L[max_corr_pos_L]) +
              '\ndelay: ' + str(delay_))

    np.savetxt(os.path.join(event_folder,event_name + '_postMatch.txt'), to_save)

    # # #
    # illustrating the best match:
    # # #

    corr_file = copy.deepcopy(to_save)
    abs_summed_corr = np.sum(np.abs(corr_file[:, :2]), axis=1)
    a_ = np.where(abs_summed_corr == np.max(abs_summed_corr))
    ind_to_eval_ = a_[0].item()  # converts the array to int
    delay_dist = corr_file[np.where(corr_file[:, 2] != 0), 2]
    print('H1_corr: ' + str(corr_file[ind_to_eval_, 0]) +
          '\nL1_corr: ' + str(corr_file[ind_to_eval_, 1]) +
          '\ndelay: ' + str(corr_file[ind_to_eval_, 2]))
    print('mean time-delay between best matches is: \n' +
          str(1000 * np.mean(delay_dist)) + ' -+ ' + str(1000 * np.std(delay_dist)) + ' ms')
    # to illustrate arbitrary realization this section shall be commented

    print(event_name + ', processing realization (max corr.): #' + str(ind_to_eval_))
    values_ = post_realizations[ind_to_eval_, :]

    if event_name in first_run:
        zip_obj_ = zip(variables_, values_)
        dict_ = dict(zip_obj_)
        polar_to_cart = pnutils.jframe_to_l0frame(mass1=dict_['m1_detector_frame_Msun'],
                                                            mass2=dict_['m2_detector_frame_Msun'],
                                                            f_ref=10,
                                                            thetajn=np.arccos(dict_['costheta_jn']),
                                                            spin1_a=dict_['spin1'],
                                                            spin2_a=dict_['spin2'],
                                                            spin1_polar=np.arccos(dict_['costilt1']),
                                                            spin2_polar=np.arccos(dict_['costilt2']))
        params_to_pass_td = {**dict_, **polar_to_cart}
        hp, hc = wf.get_td_waveform(params_to_pass_td,
                                    mass1=dict_['m1_detector_frame_Msun'],
                                    mass2=dict_['m2_detector_frame_Msun'],
                                    inclination=np.arccos(dict_['costheta_jn']),
                                    distance=dict_['luminosity_distance_Mpc'],
                                    approximant=aprx_method,
                                    delta_t=1.0 / fs_ref,
                                    f_lower=10)

    elif event_name in second_run:
        # LIGO:
        polar_to_cart = pnutils.jframe_to_l0frame(mass1=dict_['m1_detector_frame_Msun'],
                                                            mass2=dict_['m2_detector_frame_Msun'],
                                                            f_ref=10,
                                                            thetajn=np.arccos(dict_['costheta_jn']),
                                                            spin1_a=dict_['spin1'],
                                                            spin2_a=dict_['spin2'],
                                                            spin1_polar=np.arccos(dict_['costilt1']),
                                                            spin2_polar=np.arccos(dict_['costilt2']))
        params_to_pass_td = {**dict_, **polar_to_cart}
        hp, hc = wf.get_td_waveform(params_to_pass_td,
                                    mass1=dict_['m1_detector_frame_Msun'],
                                    mass2=dict_['m2_detector_frame_Msun'],
                                    inclination=np.arccos(dict_['costheta_jn']),
                                    distance=dict_['luminosity_distance_Mpc'],
                                    approximant=aprx_method,
                                    delta_t=1.0 / fs_ref,
                                    f_lower=10)

        # # Soumi et al:
        # polar_to_cart = wf.waveform_modes.jframe_to_l0frame(mass1=dict_['mass1'],
        #                                                     mass2=dict_['mass2'],
        #                                                     f_ref=10,
        #                                                     thetajn=dict_['inclination'],
        #                                                     spin1_a=dict_['spin1_a'],
        #                                                     spin2_a=dict_['spin2_a'],
        #                                                     spin1_polar=dict_['spin1_polar'],
        #                                                     spin2_polar=dict_['spin2_polar'])
        # params_to_pass_td = {**dict_, **polar_to_cart}
        # hp, hc = wf.get_td_waveform(params_to_pass_td,
        #                             approximant=aprx_method,
        #                             delta_t=1.0 / fs_ref,
        #                             f_lower=10)

    elif event_name in third_run:
        zip_obj_ = zip(variables_to_pass_td, values_)
        dict_ = dict(zip_obj_)
        hp, hc = wf.get_td_waveform(dict_,
                                    approximant=aprx_method,
                                    delta_t=1.0 / fs_ref,
                                    f_lower=20)

    waveform_H = hp.data
    waveform_L = hc.data

    np.savetxt(os.path.join(event_folder,'Best_fit_waveform_H_' + event_name + '.txt'), waveform_H)
    np.savetxt(os.path.join(event_folder,'Best_fit_waveform_L_' + event_name + '.txt'), waveform_L)

    # zeropadding waveforms to ensure no size-related issues!
    waveform_H = np.concatenate((waveform_H, np.zeros([5000, ])))
    waveform_L = np.concatenate((waveform_L, np.zeros([5000, ])))
    # waveform_V = np.concatenate((waveform_V, np.zeros([5000, ])))

    # make them the same size
    if len(waveform_H) > len(H_extract):
        ringdown_ = np.where(waveform_H == np.max(waveform_H))  # to make the ringdown central
        ringdown_ = ringdown_[0].item()
        waveform_H_ = waveform_H[
                      int(ringdown_ - np.fix(len(H_extract) / 2)):
                      int(ringdown_ + np.fix(len(H_extract) / 2))]
        H_extract_ = H_extract[0:2 * int(np.fix(len(H_extract) / 2))]  # to compensate for above cut
        time_array_H_ = time_array_H[0:2 * int(np.fix(len(time_array_H) / 2))]  # to compensate for above cut

    if len(waveform_H) < len(H_extract):
        H_extract_ = H_extract[:len(waveform_H)]
        time_array_H_ = time_array_H[:len(waveform_H)]

    # make them the same size
    if len(waveform_L) > len(L_extract):
        ringdown_ = np.where(waveform_L == np.max(waveform_L))  # to make the ringdown central
        ringdown_ = ringdown_[0].item()
        waveform_L_ = waveform_L[
                      int(ringdown_ - np.fix(len(L_extract) / 2)):
                      int(ringdown_ + np.fix(len(L_extract) / 2))]
        L_extract_ = L_extract[0:2 * int(np.fix(len(L_extract) / 2))]  # to compensate for above cut
        time_array_L_ = time_array_L[0:2 * int(np.fix(len(time_array_L) / 2))]  # to compensate for above cut

    if len(waveform_L) < len(L_extract):
        L_extract_ = L_extract[:len(waveform_L)]
        time_array_L_ = time_array_L[:len(waveform_L)]

    corr_check_range = np.arange(-500, 500)
    corr_arr_H = np.empty(len(corr_check_range))
    corr_arr_L = np.empty(len(corr_check_range))
    H_extract_ = H_extract_ / np.std(H_extract_)
    waveform_H_ = waveform_H_ / np.std(waveform_H_)
    L_extract_ = L_extract_ / np.std(L_extract_)
    waveform_L_ = waveform_L_ / np.std(waveform_L_)
    cntr = -1
    for tau in corr_check_range:
        cntr += 1
        cr, cr_err = p_corr(waveform_H_, H_extract_, tau)
        corr_arr_H[cntr] = cr
        cr, cr_err = p_corr(waveform_L_, L_extract_, tau)
        corr_arr_L[cntr] = cr

    # check for nans
    corr_arr_H[np.isnan(corr_arr_H)] = 0
    corr_arr_L[np.isnan(corr_arr_L)] = 0

    # check for non-relevant quantities for p_corr coeff
    corr_arr_H[(corr_arr_H > 1) | (corr_arr_H < -1)] = 0
    corr_arr_L[(corr_arr_L > 1) | (corr_arr_L < -1)] = 0

    max_corr_pos_H = np.where(np.abs(corr_arr_H) == np.max(np.abs(corr_arr_H)))
    max_corr_pos_L = np.where(np.abs(corr_arr_L) == np.max(np.abs(corr_arr_L)))

    tau_H = corr_check_range[max_corr_pos_H][0]  # 0 index to ensure uniqueness
    tau_H = tau_H.item()
    tau_L = corr_check_range[max_corr_pos_L][0]  # 0 index to ensure uniqueness
    tau_L = tau_L.item()

    print('delay: ' + str((tau_H - tau_L) / fs_ref))
    #
    #
    # HERE ON THE WAVEFORM AND THE EXTRACTED DATA ARE EXACTLY MATCHED TO FIX THE CORR FRAME
    #
    #
    waveform_H_ = waveform_H[
                  int((ringdown_ - tau_H) - np.fix(len(H_extract) / 2)):
                  int((ringdown_ - tau_H) + np.fix(len(H_extract) / 2))]
    waveform_L_ = waveform_L[
                  int((ringdown_ - tau_L) - np.fix(len(L_extract) / 2)):
                  int((ringdown_ - tau_L) + np.fix(len(L_extract) / 2))]

    waveform_H_ = waveform_H_ / np.std(waveform_H_)
    waveform_L_ = waveform_L_ / np.std(waveform_L_)

    cntr = -1
    for tau in corr_check_range:
        cntr += 1
        cr, cr_err = p_corr(waveform_H_, H_extract_, tau)
        corr_arr_H[cntr] = cr
        cr, cr_err = p_corr(waveform_L_, L_extract_, tau)
        corr_arr_L[cntr] = cr

    # check for nans
    corr_arr_H[np.isnan(corr_arr_H)] = 0
    corr_arr_L[np.isnan(corr_arr_L)] = 0

    # check for non-relevant quantities for p_corr coeff
    corr_arr_H[(corr_arr_H > 1) | (corr_arr_H < -1)] = 0
    corr_arr_L[(corr_arr_L > 1) | (corr_arr_L < -1)] = 0

    max_corr_pos_H = np.where(np.abs(corr_arr_H) == np.max(np.abs(corr_arr_H)))
    max_corr_pos_L = np.where(np.abs(corr_arr_L) == np.max(np.abs(corr_arr_L)))

    tau_H = corr_check_range[max_corr_pos_H][0]  # 0 index to ensure uniqueness
    tau_L = corr_check_range[max_corr_pos_L][0]  # 0 index to ensure uniqueness

    print('tau_H_matched: ' + str(tau_H.item()) + '\ntau_L_matched: ' + str(
        tau_L.item()))  # double check to ensure both=0

    corr_frame_size = np.fix(0.18 * fs_ref)
    center = np.fix(len(H_extract_) / 2) - np.fix(0.00 * fs_ref)  # center is movable ... watch the extract size limits!

    center_H = center - tau_H.item()  # =center for the previous runs
    x = waveform_H_[int(center_H - np.fix(corr_frame_size / 2)):
                    int(center_H + np.fix(corr_frame_size / 2))]
    y = H_extract_[int(center - np.fix(corr_frame_size / 2)):
                   int(center + np.fix(corr_frame_size / 2))]
    t = time_array_H_[int(center - np.fix(corr_frame_size / 2)):
                      int(center + np.fix(corr_frame_size / 2))]
    cr_H, _ = p_corr(x, y, 0)

    plt.figure(dpi=250)
    plt.plot(t, np.sign(cr_H) * x, label="waveform")
    plt.plot(t, y, label="extraction")
    plt.title(event_name + '_H1, corr: ' + str(cr_H))
    plt.legend()
    plt.savefig(os.path.join(event_folder,event_name + '_H1_bestMatchIllustration.png'))

    center_L = center - tau_L.item()  # =center for the previous runs
    x = waveform_L_[int(center_L - np.fix(corr_frame_size / 2)):
                    int(center_L + np.fix(corr_frame_size / 2))]
    y = L_extract_[int(center - np.fix(corr_frame_size / 2)):
                   int(center + np.fix(corr_frame_size / 2))]
    t = time_array_L_[int(center - np.fix(corr_frame_size / 2)):
                      int(center + np.fix(corr_frame_size / 2))]
    cr_L, _ = p_corr(x, y, 0)

    plt.figure(dpi=250)
    plt.plot(t, np.sign(cr_L) * x, label="waveform")
    plt.plot(t, y, label="extraction")
    plt.title(event_name + '_L1, corr: ' + str(cr_L))
    plt.legend()
    plt.savefig(os.path.join(event_folder,event_name + '_L1_bestMatchIllustration.png'))

    return to_save

def match_pair(event_name, data, fs_ref=4096., save_out = True, local_path = None):
    """

    :param event_name:
    :param data:
    :param fs_ref:
    :return:
    """

    # local_path = ?
    # lines 787, 901, 902, 1023, 1039

    O2_source = "SoumiEtAl"
    # O2_source = "LIGO"

    first_run = ['GW150914', 'GW151012', 'GW151226']
    second_run = ['GW170104', 'GW170608', 'GW170729',
                  'GW170809', 'GW170814', 'GW170818', 'GW170823']
    third_run = ['GW190412', 'GW190814', 'GW190521']
    
    if save_out:
        if local_path:
            event_folder = local_path
        else:
            event_folder = 'results/{}'.format(event_name)
            if_exist = os.path.isdir(event_folder)
            if not if_exist:
                os.makedirs(event_folder)
                
    # download source files
    with open("linksDict_soumi", "rb") as file:
        get_link = pickle.load(file)
    link_ligo = get_link[event_name]

    if event_name in first_run:
        #
        print("downloading from: \n" + link_ligo)
        file_ = wget.download(link_ligo)
        print("download finished. \n")
        h_file = hf.File(file_)
        #
        h_samp = h_file['IMRPhenomPv2_posterior']
        aprx_method = 'IMRPhenomPv2'
        variables_ = list(h_samp.dtype.names)
        ''' variables_ :
        ['costheta_jn', 'luminosity_distance_Mpc', 'right_ascension', 'declination', 
        'm1_detector_frame_Msun', 'm2_detector_frame_Msun', 'spin1', 'spin2', 'costilt1', 'costilt2'] 
        '''
        pol_id = "gen"
        test_ = h_samp[variables_[0]]
        post_realizations = np.zeros((len(test_), len(variables_)))

    elif event_name in second_run:
        if O2_source == "LIGO":
            # LIGO:
            print("downloading from: \n" + link_ligo)
            file_ = wget.download(link_ligo)
            print("download finished. \n")
            h_file = hf.File(file_)
            #
            h_samp = h_file['IMRPhenomPv2_posterior']
            aprx_method = 'IMRPhenomPv2'
            variables_ = list(h_samp.dtype.names)
            ''' variables_ :
            ['costheta_jn', 'luminosity_distance_Mpc', 'right_ascension', 'declination', 
            'm1_detector_frame_Msun', 'm2_detector_frame_Msun', 'spin1', 'spin2', 'costilt1', 'costilt2'] 
            '''
            pol_id = "gen"
        elif O2_source == "SoumiEtAl":
            # Soumi et al:
            # download source files
            with open("linksDict_SoumiEtAl", "rb") as file:
                get_link_soumi = pickle.load(file)
            link_soumi = get_link_soumi[event_name]
            print("downloading from: \n" + link_soumi)
            file_ = wget.download(link_soumi)
            print("download finished. \n")
            h_file = hf.File(file_)
            #
            h_samp = h_file['samples']
            aprx_method = 'IMRPhenomPv2'
            variables_ = list(h_samp.keys())
            ''' variables_ :
            ['dec', 'distance', 'inclination', 'mass1', 'mass2', 'polarization', 'ra', 'spin1_a',
            'spin1_azimuthal', 'spin1_polar', 'spin2_a', 'spin2_azimuthal', 'spin2_polar', 'tc']
            '''
            pol_id = "fix"
        test_ = h_samp[variables_[0]][()]
        post_realizations = np.zeros((len(test_), len(variables_)))

    elif event_name in third_run:
        #
        print("downloading from: \n" + link_ligo)
        file_ = wget.download(link_ligo)
        print("download finished. \n")
        tar = tarfile.open(file_, "r:")
        tar.extractall()
        tar.close()
        h_file = hf.File(event_name + '.h5')
        #
        if event_name == 'GW190412':
            h_samp = h_file['C01:IMRPhenomPv2']['posterior_samples']  # for GW190412
            aprx_method = 'IMRPhenomPv2'  # for GW190412
        if event_name == 'GW190425':
            h_samp = h_file['C01:IMRPhenomPv2_NRTidal-HS']['posterior_samples']  # for GW190425
            aprx_method = 'IMRPhenomPv2'  # for GW190425
        if event_name == 'GW190521':
            h_samp = h_file['C01:IMRPhenomPv3HM']['posterior_samples']  # for GW190521
            aprx_method = 'IMRPhenomPv3HM'  # for GW190521
        if event_name == 'GW190814':
            h_samp = h_file['C01:IMRPhenomD']['posterior_samples']  # for GW190814
            aprx_method = 'IMRPhenomD'  # for GW190814
        # h_samp.dtype
        variables_ = ['mass_1', 'mass_2', 'spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z',
                      'ra', 'dec', 'iota', 'psi']
        variables_to_pass_td = ['mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z',
                                'ra', 'dec', 'inclination', 'polarization']
        pol_id = "fix"
        test_ = h_samp[variables_[0]][()]
        post_realizations = np.zeros((len(test_), len(variables_)))

    for i in range(len(variables_)):
        post_realizations[:, i] = h_samp[variables_[i]]

    time_array = data[:, 0]
    H_extract = data[:, 1]
    L_extract = data[:, 2]
    time_array_H = time_array
    time_array_L = time_array

    if pol_id == "gen":
        pol_arr = np.linspace(0, 2 * np.pi, 15)
    else:
        pol_arr = [0]

    upLim = 1500
    to_save = np.empty([len(pol_arr) * upLim, 4])
    realizations_to_eval = range(upLim)
    to_save[:] = np.nan

    # upLim = len(test_)
    # to_save = np.empty([len(pol_arr) * upLim, 4])
    # realizations_to_eval = range(upLim)
    # to_save[:] = np.nan

    count_ind = -1
    for real_ind in realizations_to_eval:

        for pol_ in pol_arr:

            count_ind += 1
            print(event_name + ', processing realization: #' + str(count_ind) + "/" +
                  str((len(pol_arr) * upLim) - 1))
            values_ = post_realizations[real_ind, :]

            if event_name in first_run:
                zip_obj_ = zip(variables_, values_)
                dict_ = dict(zip_obj_)
                polar_to_cart = pnutils.jframe_to_l0frame(mass1=dict_['m1_detector_frame_Msun'],
                                                                    mass2=dict_['m2_detector_frame_Msun'],
                                                                    f_ref=10,
                                                                    thetajn=np.arccos(dict_['costheta_jn']),
                                                                    spin1_a=dict_['spin1'],
                                                                    spin2_a=dict_['spin2'],
                                                                    spin1_polar=np.arccos(dict_['costilt1']),
                                                                    spin2_polar=np.arccos(dict_['costilt2']))
                params_to_pass_td = {**dict_, **polar_to_cart}
                hp, hc = wf.get_td_waveform(params_to_pass_td,
                                            mass1=dict_['m1_detector_frame_Msun'],
                                            mass2=dict_['m2_detector_frame_Msun'],
                                            inclination=np.arccos(dict_['costheta_jn']),
                                            distance=dict_['luminosity_distance_Mpc'],
                                            approximant=aprx_method,
                                            delta_t=1.0 / fs_ref,
                                            f_lower=10)
                declination = dict_['declination']
                right_ascension = dict_['right_ascension']
                polarization = pol_

            elif event_name in second_run:
                zip_obj_ = zip(variables_, values_)
                dict_ = dict(zip_obj_)
                if O2_source == "LIGO":
                    # LIGO:
                    polar_to_cart =pnutils.jframe_to_l0frame(mass1=dict_['m1_detector_frame_Msun'],
                                                                        mass2=dict_['m2_detector_frame_Msun'],
                                                                        f_ref=10,
                                                                        thetajn=np.arccos(dict_['costheta_jn']),
                                                                        spin1_a=dict_['spin1'],
                                                                        spin2_a=dict_['spin2'],
                                                                        spin1_polar=np.arccos(dict_['costilt1']),
                                                                        spin2_polar=np.arccos(dict_['costilt2']))
                    params_to_pass_td = {**dict_, **polar_to_cart}
                    hp, hc = wf.get_td_waveform(params_to_pass_td,
                                                mass1=dict_['m1_detector_frame_Msun'],
                                                mass2=dict_['m2_detector_frame_Msun'],
                                                inclination=np.arccos(dict_['costheta_jn']),
                                                distance=dict_['luminosity_distance_Mpc'],
                                                approximant=aprx_method,
                                                delta_t=1.0 / fs_ref,
                                                f_lower=10)
                    declination = dict_['declination']
                    right_ascension = dict_['right_ascension']
                    polarization = pol_

                elif O2_source == "SoumiEtAl":
                    # Soumi et al:
                    polar_to_cart = pnutils.jframe_to_l0frame(mass1=dict_['mass1'],
                                                                        mass2=dict_['mass2'],
                                                                        f_ref=10,
                                                                        thetajn=dict_['inclination'],
                                                                        spin1_a=dict_['spin1_a'],
                                                                        spin2_a=dict_['spin2_a'],
                                                                        spin1_polar=dict_['spin1_polar'],
                                                                        spin2_polar=dict_['spin2_polar'])
                    params_to_pass_td = {**dict_, **polar_to_cart}
                    hp, hc = wf.get_td_waveform(params_to_pass_td,
                                                approximant=aprx_method,
                                                delta_t=1.0 / fs_ref,
                                                f_lower=10)
                    declination = dict_['dec']
                    right_ascension = dict_['ra']
                    polarization = dict_['polarization']

            elif event_name in third_run:
                zip_obj_ = zip(variables_to_pass_td, values_)
                dict_ = dict(zip_obj_)
                hp, hc = wf.get_td_waveform(dict_,
                                            approximant=aprx_method,
                                            delta_t=1.0 / fs_ref,
                                            f_lower=20)
                declination = dict_['dec']
                right_ascension = dict_['ra']
                polarization = dict_['psi']

            # applying antenna pattern
            det_h1 = det.Detector('H1')
            det_l1 = det.Detector('L1')
            # det_v1 = det.Detector('V1')

            signal_h1 = det_h1.project_wave(hp, hc, right_ascension, declination, polarization)
            signal_l1 = det_l1.project_wave(hp, hc, right_ascension, declination, polarization)
            # signal_v1 = det_v1.project_wave(hp, hc, right_ascension, declination, polarization)

            waveform_H = signal_h1.data
            waveform_L = signal_l1.data
            # waveform_V = signal_v1.data

            # waveform_H = hp.data
            # waveform_L = hp.data

            # make them the same size
            if len(waveform_H) > len(H_extract):
                ringdown_ = np.where(waveform_H == np.max(waveform_H))  # to make the ringdown central
                waveform_H = waveform_H[
                             int(ringdown_ - np.fix(len(H_extract) / 2)):int(ringdown_ + np.fix(len(H_extract) / 2))]
                H_extract_ = H_extract[0:2 * int(np.fix(len(H_extract) / 2))]  # to compensate for above cut
                # time_array_H_ = time_array_H[0:2 * int(np.fix(len(time_array_H) / 2))]  # to compensate for above cut

            if len(waveform_H) < len(H_extract):
                H_extract_ = H_extract[:len(waveform_H)]
                # time_array_H_ = time_array_H[:len(waveform_H)]

            # make them the same size
            if len(waveform_L) > len(L_extract):
                ringdown_ = np.where(waveform_L == np.max(waveform_L))  # to make the ringdown central
                waveform_L = waveform_L[
                             int(ringdown_ - np.fix(len(L_extract) / 2)):int(ringdown_ + np.fix(len(L_extract) / 2))]
                L_extract_ = L_extract[0:2 * int(np.fix(len(L_extract) / 2))]  # to compensate for above cut
                # time_array_L_ = time_array_L[0:2 * int(np.fix(len(time_array_L) / 2))]  # to compensate for above cut

            if len(waveform_L) < len(L_extract):
                L_extract_ = L_extract[:len(waveform_L)]
                # time_array_L_ = time_array_L[:len(waveform_L)]

            # NOTICE: the input extraction should be centered around the ringdown
            corr_check_range = np.arange(-500, 500)
            corr_arr_H = np.empty(len(corr_check_range))
            corr_arr_L = np.empty(len(corr_check_range))
            H_extract_ = H_extract_ / np.std(H_extract_)
            waveform_H = waveform_H / np.std(waveform_H)
            L_extract_ = L_extract_ / np.std(L_extract_)
            waveform_L = waveform_L / np.std(waveform_L)
            cntr = -1
            for tau in corr_check_range:
                cntr += 1
                cr, _ = p_corr(waveform_H, H_extract_, tau)
                corr_arr_H[cntr] = cr
                cr, _ = p_corr(waveform_L, L_extract_, tau)
                corr_arr_L[cntr] = cr

            # check for nans
            corr_arr_H[np.isnan(corr_arr_H)] = 0
            corr_arr_L[np.isnan(corr_arr_L)] = 0

            # check for non-relevant quantities for p_corr coeff
            corr_arr_H[(corr_arr_H > 1) | (corr_arr_H < -1)] = 0
            corr_arr_L[(corr_arr_L > 1) | (corr_arr_L < -1)] = 0

            max_corr_pos_H = np.where(np.abs(corr_arr_H) == np.max(np.abs(corr_arr_H)))
            # print('the H1 associated coefficient is: ' + str(corr_arr_H[max_corr_pos_H]))
            max_corr_pos_L = np.where(np.abs(corr_arr_L) == np.max(np.abs(corr_arr_L)))
            # print('the L1 associated coefficient is: ' + str(corr_arr_L[max_corr_pos_L]))

            if len(max_corr_pos_H[0]) == 1 and len(max_corr_pos_L[0]) == 1:
                to_save[count_ind, 0] = corr_arr_H[max_corr_pos_H]
                to_save[count_ind, 1] = corr_arr_L[max_corr_pos_L]
                to_save[count_ind, 2] = polarization
                to_save[count_ind, 3] = real_ind
            else:
                to_save[count_ind, 0] = 0
                to_save[count_ind, 1] = 0
                to_save[count_ind, 2] = polarization
                to_save[count_ind, 3] = real_ind
                continue

            print('H1_corr: ' + str(corr_arr_H[max_corr_pos_H]) +
                  '\nL1_corr: ' + str(corr_arr_L[max_corr_pos_L]))

    np.savetxt(os.path.join(event_folder,event_name + '_postMatch.txt'), to_save)

    # # #
    # illustrating the best match:
    # # #

    corr_file = copy.deepcopy(to_save)
    abs_summed_corr = np.sum(np.abs(corr_file[:, :2]), axis=1)
    a_ = np.where(abs_summed_corr == np.max(abs_summed_corr))
    ind_to_eval_ = a_[0][0].item()  # converts the array to int
    print('H1_corr: ' + str(corr_file[ind_to_eval_, 0]) +
          '\nL1_corr: ' + str(corr_file[ind_to_eval_, 1]))
    # to illustrate arbitrary realization this section shall be commented

    print(event_name + ', processing realization (max corr.): #' + str(ind_to_eval_))
    values_ = post_realizations[int(to_save[ind_to_eval_, 3]), :]
    polarization = to_save[ind_to_eval_, 2]

    if event_name in first_run:
        zip_obj_ = zip(variables_, values_)
        dict_ = dict(zip_obj_)
        polar_to_cart = pnutils.jframe_to_l0frame(mass1=dict_['m1_detector_frame_Msun'],
                                                            mass2=dict_['m2_detector_frame_Msun'],
                                                            f_ref=10,
                                                            thetajn=np.arccos(dict_['costheta_jn']),
                                                            spin1_a=dict_['spin1'],
                                                            spin2_a=dict_['spin2'],
                                                            spin1_polar=np.arccos(dict_['costilt1']),
                                                            spin2_polar=np.arccos(dict_['costilt2']))
        params_to_pass_td = {**dict_, **polar_to_cart}
        hp, hc = wf.get_td_waveform(params_to_pass_td,
                                    mass1=dict_['m1_detector_frame_Msun'],
                                    mass2=dict_['m2_detector_frame_Msun'],
                                    inclination=np.arccos(dict_['costheta_jn']),
                                    distance=dict_['luminosity_distance_Mpc'],
                                    approximant=aprx_method,
                                    delta_t=1.0 / fs_ref,
                                    f_lower=10)
        declination = dict_['declination']
        right_ascension = dict_['right_ascension']
        polarization = polarization

    elif event_name in second_run:
        zip_obj_ = zip(variables_, values_)
        dict_ = dict(zip_obj_)
        if O2_source == "LIGO":
            # LIGO:
            polar_to_cart = pnutils.jframe_to_l0frame(mass1=dict_['m1_detector_frame_Msun'],
                                                                mass2=dict_['m2_detector_frame_Msun'],
                                                                f_ref=10,
                                                                thetajn=np.arccos(dict_['costheta_jn']),
                                                                spin1_a=dict_['spin1'],
                                                                spin2_a=dict_['spin2'],
                                                                spin1_polar=np.arccos(dict_['costilt1']),
                                                                spin2_polar=np.arccos(dict_['costilt2']))
            params_to_pass_td = {**dict_, **polar_to_cart}
            hp, hc = wf.get_td_waveform(params_to_pass_td,
                                        mass1=dict_['m1_detector_frame_Msun'],
                                        mass2=dict_['m2_detector_frame_Msun'],
                                        inclination=np.arccos(dict_['costheta_jn']),
                                        distance=dict_['luminosity_distance_Mpc'],
                                        approximant=aprx_method,
                                        delta_t=1.0 / fs_ref,
                                        f_lower=10)
            declination = dict_['declination']
            right_ascension = dict_['right_ascension']
            polarization = polarization

        elif O2_source == "SoumiEtAl":
            # Soumi et al:
            polar_to_cart = pnutils.jframe_to_l0frame(mass1=dict_['mass1'],
                                                                mass2=dict_['mass2'],
                                                                f_ref=10,
                                                                thetajn=dict_['inclination'],
                                                                spin1_a=dict_['spin1_a'],
                                                                spin2_a=dict_['spin2_a'],
                                                                spin1_polar=dict_['spin1_polar'],
                                                                spin2_polar=dict_['spin2_polar'])
            params_to_pass_td = {**dict_, **polar_to_cart}
            hp, hc = wf.get_td_waveform(params_to_pass_td,
                                        approximant=aprx_method,
                                        delta_t=1.0 / fs_ref,
                                        f_lower=10)
            declination = dict_['dec']
            right_ascension = dict_['ra']
            polarization = polarization

    elif event_name in third_run:
        zip_obj_ = zip(variables_to_pass_td, values_)
        dict_ = dict(zip_obj_)
        hp, hc = wf.get_td_waveform(dict_,
                                    approximant=aprx_method,
                                    delta_t=1.0 / fs_ref,
                                    f_lower=20)
        declination = dict_['dec']
        right_ascension = dict_['ra']
        polarization = polarization

    # applying antenna pattern
    det_h1 = det.Detector('H1')
    det_l1 = det.Detector('L1')
    # det_v1 = det.Detector('V1')

    signal_h1 = det_h1.project_wave(hp, hc, right_ascension, declination, polarization)
    signal_l1 = det_l1.project_wave(hp, hc, right_ascension, declination, polarization)
    # signal_v1 = det_v1.project_wave(hp, hc, right_ascension, declination, polarization)

    waveform_H = signal_h1.data
    waveform_L = signal_l1.data
    # waveform_V = signal_v1.data

    # waveform_H = hp.data
    # waveform_L = hp.data

    np.savetxt(os.path.join(event_folder,'Best_fit_waveform_H' + event_name + '.txt'), waveform_H)
    np.savetxt(os.path.join(event_folder,'Best_fit_waveform_L' + event_name + '.txt'), waveform_L)

    # zeropadding waveforms to ensure no size-related issues!
    waveform_H = np.concatenate((waveform_H, np.zeros([5000, ])))
    waveform_L = np.concatenate((waveform_L, np.zeros([5000, ])))
    # waveform_V = np.concatenate((waveform_V, np.zeros([5000, ])))

    # make them the same size
    if len(waveform_H) > len(H_extract):
        ringdown_ = np.where(waveform_H == np.max(waveform_H))  # to make the ringdown central
        ringdown_ = ringdown_[0].item()
        waveform_H_ = waveform_H[
                      int(ringdown_ - np.fix(len(H_extract) / 2)):
                      int(ringdown_ + np.fix(len(H_extract) / 2))]
        H_extract_ = H_extract[0:2 * int(np.fix(len(H_extract) / 2))]  # to compensate for above cut
        time_array_H_ = time_array_H[0:2 * int(np.fix(len(time_array_H) / 2))]  # to compensate for above cut

    if len(waveform_H) < len(H_extract):
        H_extract_ = H_extract[:len(waveform_H)]
        time_array_H_ = time_array_H[:len(waveform_H)]

    # make them the same size
    if len(waveform_L) > len(L_extract):
        ringdown_ = np.where(waveform_L == np.max(waveform_L))  # to make the ringdown central
        ringdown_ = ringdown_[0].item()
        waveform_L_ = waveform_L[
                      int(ringdown_ - np.fix(len(L_extract) / 2)):
                      int(ringdown_ + np.fix(len(L_extract) / 2))]
        L_extract_ = L_extract[0:2 * int(np.fix(len(L_extract) / 2))]  # to compensate for above cut
        time_array_L_ = time_array_L[0:2 * int(np.fix(len(time_array_L) / 2))]  # to compensate for above cut

    if len(waveform_L) < len(L_extract):
        L_extract_ = L_extract[:len(waveform_L)]
        time_array_L_ = time_array_L[:len(waveform_L)]

    corr_check_range = np.arange(-500, 500)
    corr_arr_H = np.empty(len(corr_check_range))
    corr_arr_L = np.empty(len(corr_check_range))
    H_extract_ = H_extract_ / np.std(H_extract_)
    waveform_H_ = waveform_H_ / np.std(waveform_H_)
    L_extract_ = L_extract_ / np.std(L_extract_)
    waveform_L_ = waveform_L_ / np.std(waveform_L_)
    cntr = -1
    for tau in corr_check_range:
        cntr += 1
        cr, cr_err = p_corr(waveform_H_, H_extract_, tau)
        corr_arr_H[cntr] = cr
        cr, cr_err = p_corr(waveform_L_, L_extract_, tau)
        corr_arr_L[cntr] = cr

    # check for nans
    corr_arr_H[np.isnan(corr_arr_H)] = 0
    corr_arr_L[np.isnan(corr_arr_L)] = 0

    # check for non-relevant quantities for p_corr coeff
    corr_arr_H[(corr_arr_H > 1) | (corr_arr_H < -1)] = 0
    corr_arr_L[(corr_arr_L > 1) | (corr_arr_L < -1)] = 0

    max_corr_pos_H = np.where(np.abs(corr_arr_H) == np.max(np.abs(corr_arr_H)))
    max_corr_pos_L = np.where(np.abs(corr_arr_L) == np.max(np.abs(corr_arr_L)))

    tau_H = corr_check_range[max_corr_pos_H][0]  # 0 index to ensure uniqueness
    tau_H = tau_H.item()
    tau_L = corr_check_range[max_corr_pos_L][0]  # 0 index to ensure uniqueness
    tau_L = tau_L.item()

    print('delay: ' + str((tau_H - tau_L) / fs_ref))
    #
    #
    # HERE ON THE WAVEFORM AND THE EXTRACTED DATA ARE EXACTLY MATCHED TO FIX THE CORR FRAME
    #
    #
    waveform_H_ = waveform_H[
                  int((ringdown_ - tau_H) - np.fix(len(H_extract) / 2)):
                  int((ringdown_ - tau_H) + np.fix(len(H_extract) / 2))]
    waveform_L_ = waveform_L[
                  int((ringdown_ - tau_L) - np.fix(len(L_extract) / 2)):
                  int((ringdown_ - tau_L) + np.fix(len(L_extract) / 2))]

    waveform_H_ = waveform_H_ / np.std(waveform_H_)
    waveform_L_ = waveform_L_ / np.std(waveform_L_)

    cntr = -1
    for tau in corr_check_range:
        cntr += 1
        cr, cr_err = p_corr(waveform_H_, H_extract_, tau)
        corr_arr_H[cntr] = cr
        cr, cr_err = p_corr(waveform_L_, L_extract_, tau)
        corr_arr_L[cntr] = cr

    # check for nans
    corr_arr_H[np.isnan(corr_arr_H)] = 0
    corr_arr_L[np.isnan(corr_arr_L)] = 0

    # check for non-relevant quantities for p_corr coeff
    corr_arr_H[(corr_arr_H > 1) | (corr_arr_H < -1)] = 0
    corr_arr_L[(corr_arr_L > 1) | (corr_arr_L < -1)] = 0

    max_corr_pos_H = np.where(np.abs(corr_arr_H) == np.max(np.abs(corr_arr_H)))
    max_corr_pos_L = np.where(np.abs(corr_arr_L) == np.max(np.abs(corr_arr_L)))

    tau_H = corr_check_range[max_corr_pos_H][0]  # 0 index to ensure uniqueness
    tau_L = corr_check_range[max_corr_pos_L][0]  # 0 index to ensure uniqueness

    corr_frame_size = np.fix(0.18 * fs_ref)
    center = np.fix(len(H_extract_) / 2) - np.fix(0.00 * fs_ref)  # center is movable ... watch the extract size limits!

    center_H = center - tau_H.item()  # =center for the previous runs
    x = waveform_H_[int(center_H - np.fix(corr_frame_size / 2)):
                    int(center_H + np.fix(corr_frame_size / 2))]
    y = H_extract_[int(center - np.fix(corr_frame_size / 2)):
                   int(center + np.fix(corr_frame_size / 2))]
    t = time_array_H_[int(center - np.fix(corr_frame_size / 2)):
                      int(center + np.fix(corr_frame_size / 2))]
    cr_H, _ = p_corr(x, y, 0)

    plt.figure(dpi=250)
    plt.plot(t, np.sign(cr_H) * x, label="waveform")
    plt.plot(t, y, label="extraction")
    plt.title(event_name + '_H1, corr: ' + str(cr_H))
    plt.legend()
    plt.savefig(os.path.join(event_folder,event_name + '_H1_bestMatchIllustration.png'))

    center_L = center - tau_L.item()  # =center for the previous runs
    x = waveform_L_[int(center_L - np.fix(corr_frame_size / 2)):
                    int(center_L + np.fix(corr_frame_size / 2))]
    y = L_extract_[int(center - np.fix(corr_frame_size / 2)):
                   int(center + np.fix(corr_frame_size / 2))]
    t = time_array_L_[int(center - np.fix(corr_frame_size / 2)):
                      int(center + np.fix(corr_frame_size / 2))]
    cr_L, _ = p_corr(x, y, 0)

    plt.figure(dpi=250)
    plt.plot(t, np.sign(cr_L) * x, label="waveform")
    plt.plot(t, y, label="extraction")
    plt.title(event_name + '_L1, corr: ' + str(cr_L))
    plt.legend()
    plt.savefig(os.path.join(event_folder,event_name + '_L1_bestMatchIllustration.png'))

    return to_save
