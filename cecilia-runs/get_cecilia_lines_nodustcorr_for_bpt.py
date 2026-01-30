import os
import numpy as np 
from astropy.io import fits
import pickle as dill
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

cat_to_cue_mosfire = {'[O II]-3726.04':'O  2 3726.03A',
                      '[O II]-3728.8':'O  2 3728.81A',
                       'HBeta-4861.35':'H  1 4861.32A',
                       '[O III]-4958.911':'O  3 4958.91A',
                      '[O III]-5006.843':'O  3 5006.84A',
                       'HAlpha-6562.79':'H  1 6562.80A'
                      }
cat_to_cue_g395m = {'He I-7065.19':'He 1 7065.25A', 
                    '[Ar III]-7135.8':'Ar 3 7135.79A', 
                '[O II]-7319.92':'O  2 7323.00A', 
                '[O II]-7330.19':'O  2 7332.00A',
                '[Ar III]-7751.06':'Ar 3 7751.11A', 
                'P10-9014.91':'H  1 9014.91A',
                '[S III]-9068.6':'S  3 9068.62A', 
                'P9-9229.014':'H  1 9229.02A', 
                '[S III]-9530.6':'S  3 9530.62A',
                'P8-9545.969':'H  1 9545.97A', 
            #     'P7-10049.369':'H  1 1.00494m', 
                'He I-10830.339':'He 1 1.08303m', 
                'P6-10938.17':'H  1 1.09381m',  
            #     '[Fe II]-12567.46':'Fe 2 1.25668m',
                'P5-12818.072':'H  1 1.28181m'}
cat_to_cue_g235m = {
            'Hgamma-4340.437':'H  1 4340.46A', 
            '[O III]-4363.209':'O  3 4363.21A',
            'He I-4471.48':'He 1 4471.50A', 
            'He II-4685.703':'He 2 4685.68A', 
            '[Ar IV]-4711.35':'Ar 4 4711.26A', 
            '[Ar IV]-4740.2':'Ar 4 4740.12A', 
            'Hbeta-4861.35':'H  1 4861.32A', 
            '[O III]-4958.911':'O  3 4958.91A', 
            '[O III]-5006.843':'O  3 5006.84A',
            '[N II]-5754.59':'N  2 5754.61A', 
            'He I-5875.621':'He 1 5875.66A',
            '[O I]-6300.304':'O  1 6300.30A',
            '[S III]-6312.06':'S  3 6312.06A', 
            '[O I]-6363.776':'O  1 6363.78A', 
            '[N II]-6548.05':'N  2 6548.05A',
            'Halpha-6562.79':'H  1 6562.80A',
            '[N II]-6583.45':'N  2 6583.45A', 
            'He I-6678.151':'He 1 6678.15A',
            '[S II]-6716.44':'S  2 6716.44A', 
            '[S II]-6730.815':'S  2 6730.82A', 
            'He I-7065.19':'He 1 7065.25A',
            '[Ar III]-7135.8':'Ar 3 7135.79A', 
            '[O II]-7319.92':'O  2 7323.00A', 
            '[O II]-7330.19':'O  2 7332.00A',
            '[Ar III]-7751.06':'Ar 3 7751.11A',
            'P10-9014.91':'H  1 9014.91A',
            '[S III]-9068.6':'S  3 9068.62A', 
            'P9-9229.014':'H  1 9229.02A', 
            '[S III]-9530.6':'S  3 9530.62A', 
            'P8-9545.969':'H  1 9545.97A',
            # 'P7-10049.369':'H  1 1.00494m'
            }

line_wav = np.array([9.23150e+02, 9.26226e+02, 9.30748e+02, 9.37804e+02, 9.49743e+02,
        9.72537e+02, 1.02572e+03, 1.21567e+03, 1.40602e+03, 1.54819e+03,
        1.55077e+03, 1.64041e+03, 1.66081e+03, 1.66615e+03, 1.75000e+03,
        1.81456e+03, 1.85472e+03, 1.86279e+03, 1.88271e+03, 1.89203e+03,
        1.90668e+03, 1.90873e+03, 2.14277e+03, 2.32095e+03, 2.32350e+03,
        2.32469e+03, 2.32540e+03, 2.32693e+03, 2.32812e+03, 2.42428e+03,
        2.47100e+03, 2.66035e+03, 2.66915e+03, 2.79553e+03, 2.80271e+03,
        3.10918e+03, 3.34218e+03, 3.72163e+03, 3.72603e+03, 3.72881e+03,
        3.79790e+03, 3.83538e+03, 3.86876e+03, 3.88864e+03, 3.88905e+03,
        3.96747e+03, 3.97007e+03, 4.06860e+03, 4.07635e+03, 4.10173e+03,
        4.34046e+03, 4.36321e+03, 4.47150e+03, 4.62157e+03, 4.68568e+03,
        4.71126e+03, 4.72000e+03, 4.74012e+03, 4.86132e+03, 4.95891e+03,
        5.00684e+03, 5.19182e+03, 5.20026e+03, 5.51771e+03, 5.53787e+03,
        5.57734e+03, 5.75461e+03, 5.87566e+03, 6.30030e+03, 6.31206e+03,
        6.36378e+03, 6.54805e+03, 6.56280e+03, 6.58345e+03, 6.67815e+03,
        6.71644e+03, 6.73082e+03, 7.06525e+03, 7.13579e+03, 7.32300e+03,
        7.33200e+03, 7.33215e+03, 7.75111e+03, 8.57870e+03, 8.72713e+03,
        9.01491e+03, 9.06862e+03, 9.12360e+03, 9.22902e+03, 9.53062e+03,
        9.54597e+03, 9.85000e+03, 1.00494e+04, 1.03364e+04, 1.08291e+04,
        1.08303e+04, 1.09381e+04, 1.25668e+04, 1.28181e+04, 1.73621e+04,
        1.81741e+04, 1.87510e+04, 1.94456e+04, 2.16553e+04, 2.62515e+04,
        3.03837e+04, 3.29609e+04, 3.73954e+04, 4.05113e+04, 4.65251e+04,
        5.12726e+04, 5.90660e+04, 6.98337e+04, 7.45782e+04, 7.50045e+04,
        8.98898e+04, 1.05076e+05, 1.23685e+05, 1.28101e+05, 1.43639e+05,
        1.55509e+05, 1.87078e+05, 2.18253e+05, 3.28629e+05, 3.34704e+05,
        3.48046e+05, 3.60036e+05, 5.18004e+05, 5.73238e+05, 6.06263e+05,
        6.31679e+05, 8.83323e+05, 1.21769e+06, 1.45495e+06, 1.57636e+06,
        2.05283e+06, 3.70269e+06, 6.09590e+06])

zdf = pd.read_csv('/home/carolinevr/cecilia_redshifts.txt', sep=r'\s+')  # dataframe of cecilia redshifts
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
zdf['dist'] = cosmo.luminosity_distance(zdf['z']).to(u.cm).value
zdf['lum_fac'] = 4*np.pi*zdf['dist']**2

abund_dir = '/home/carolinevr/ABUND_20251204'


dir = os.listdir(abund_dir) 
i =0
in_cue_235 = list(cat_to_cue_g235m.keys())
in_cue_395 = list(cat_to_cue_g395m.keys())
in_cue_mos = list(cat_to_cue_mosfire.keys())

for file in dir:
    line_list = np.array(['H  1 923.150A', 'H  1 926.226A', 'H  1 930.748A', 'H  1 937.804A',
        'H  1 949.743A', 'H  1 972.537A', 'H  1 1025.72A', 'H  1 1215.67A',
        'S  4 1406.02A', 'C  4 1548.19A', 'C  4 1550.77A', 'He 2 1640.41A',
        'O  3 1660.81A', 'O  3 1666.15A', 'N  3 1750.00A', 'Ne 3 1814.56A',
        'Al 3 1854.72A', 'Al 3 1862.79A', 'Si 3 1882.71A', 'Si 3 1892.03A',
        'C  3 1906.68A', 'C  3 1908.73A', 'N  2 2142.77A', 'O  3 2320.95A',
        'C  2 2323.50A', 'C  2 2324.69A', 'C  2 2325.40A', 'C  2 2326.93A',
        'C  2 2328.12A', 'Ne 4 2424.28A', 'O  2 2471.00A', 'Al 2 2660.35A',
        'Al 2 2669.15A', 'Mg 2 2795.53A', 'Mg 2 2802.71A', 'Ar 3 3109.18A',
        'Ne 3 3342.18A', 'S  3 3721.63A', 'O  2 3726.03A', 'O  2 3728.81A',
        'H  1 3797.90A', 'H  1 3835.38A', 'Ne 3 3868.76A', 'He 1 3888.64A',
        'H  1 3889.05A', 'Ne 3 3967.47A', 'H  1 3970.07A', 'S  2 4068.60A',
        'S  2 4076.35A', 'H  1 4101.73A', 'H  1 4340.46A', 'O  3 4363.21A',
        'He 1 4471.50A', 'C  1 4621.57A', 'He 2 4685.68A', 'Ar 4 4711.26A',
        'Ne 4 4720.00A', 'Ar 4 4740.12A', 'H  1 4861.32A', 'O  3 4958.91A',
        'O  3 5006.84A', 'Ar 3 5191.82A', 'N  1 5200.26A', 'Cl 3 5517.71A',
        'Cl 3 5537.87A', 'O  1 5577.34A', 'N  2 5754.61A', 'He 1 5875.66A',
        'O  1 6300.30A', 'S  3 6312.06A', 'O  1 6363.78A', 'N  2 6548.05A',
        'H  1 6562.80A', 'N  2 6583.45A', 'He 1 6678.15A', 'S  2 6716.44A',
        'S  2 6730.82A', 'He 1 7065.25A', 'Ar 3 7135.79A', 'O  2 7323.00A',
        'O  2 7332.00A', 'Ar 4 7332.15A', 'Ar 3 7751.11A', 'Cl 2 8578.70A',
        'C  1 8727.13A', 'H  1 9014.91A', 'S  3 9068.62A', 'Cl 2 9123.60A',
        'H  1 9229.02A', 'S  3 9530.62A', 'H  1 9545.97A', 'C  1 9850.00A',
        'H  1 1.00494m', 'S  2 1.03364m', 'He 1 1.08291m', 'He 1 1.08303m',
        'H  1 1.09381m', 'Fe 2 1.25668m', 'H  1 1.28181m', 'H  1 1.73621m',
        'H  1 1.81741m', 'H  1 1.87510m', 'H  1 1.94456m', 'H  1 2.16553m',
        'H  1 2.62515m', 'H  1 3.03837m', 'H  1 3.29609m', 'H  1 3.73954m',
        'H  1 4.05113m', 'H  1 4.65251m', 'H  1 5.12726m', 'H  1 5.90660m',
        'Ar 2 6.98337m', 'H  1 7.45782m', 'H  1 7.50045m', 'Ar 3 8.98898m',
        'S  4 10.5076m', 'H  1 12.3685m', 'Ne 2 12.8101m', 'Cl 2 14.3639m',
        'Ne 3 15.5509m', 'S  3 18.7078m', 'Ar 3 21.8253m', 'P  2 32.8629m',
        'S  3 33.4704m', 'Si 2 34.8046m', 'Ne 3 36.0036m', 'O  3 51.8004m',
        'N  3 57.3238m', 'P  2 60.6263m', 'O  1 63.1679m', 'O  3 88.3323m',
        'N  2 121.769m', 'O  1 145.495m', 'C  2 157.636m', 'N  2 205.283m',
        'C  1 370.269m', 'C  1 609.590m'])
    if (file != "cecilia_2593_3.abund.fits") & (file != "cecilia_2593_35.abund.fits"):  # BX537 doesn't exist in mosfire, MD41 has no nirspec data

        # Initialise dictionaries
        line_lum = np.zeros(len(line_list))
        line_lum_up = np.zeros(len(line_list))
        line_lum_unc = np.zeros(len(line_list))
        if file.endswith('.fits'):
            pass
        else:
            continue
        # print(file)
        path = os.path.join(abund_dir, file)
        # Open fits file
        hdu = fits.open(path)
        # Get object ID
        id = file.split('_')[2]
        id = int(id.split('.')[0])
        print("\n",id)
        # Get precomputed factor to convert intensity to luminosity
        lum_fac = zdf['lum_fac'][zdf['IDNo'] == id].item()
        if i == 0:
            # Initialise line lists
            lines235 = (hdu[1].data)
            llist_235 = np.char.add(np.char.add(lines235['NAME'], '-'), lines235['WAVELENGTH'].astype(str))
            to_use_235 = np.isin(llist_235, in_cue_235)
            lines395 = hdu[3].data 
            llist_395 = np.char.add(np.char.add(lines395['NAME'], '-'), lines395['WAVELENGTH'].astype(str))
            to_use_395 = np.isin(llist_395, in_cue_395)
            mos = hdu[7].data 
            llist_mos = np.char.add(np.char.add(mos['NAMES'], '-'), mos['WAVELENGTH'].astype(str))
            to_use_mos = np.isin(llist_mos, in_cue_mos)
            i = 1
        else: 
            mos = hdu[7].data

        # Get G235M data
        fluxes235 = hdu[1].data 

        ha = fluxes235['FLUX'][lines235['NAME'] == 'Halpha'].item()
        ha_err = fluxes235['F_UNC'][lines235['NAME'] == 'Halpha'].item()
        ha_ind_mos = (mos['NAMES']=='HAlpha')
        ha_mos = mos['FLUX'][ha_ind_mos]

        hb = fluxes235['FLUX'][lines235['NAME'] == 'Hbeta'].item()
        hb_err = fluxes235['F_UNC'][lines235['NAME'] == 'Hbeta'].item()
        hb_ind_mos = (mos['NAMES']=="HBeta")
        hb_mos = mos['FLUX'][hb_ind_mos]

        # If Ha is not detected, use Hb to normalise MOSFIRE and NIRSpec fluxes. 
        if (ha_mos != 0) & (ha != 0):
            # Normalise relative to Ha
            print("using ha as a normalizing line")
            print("ha nirs",ha, ha_err, "ha mos",ha_mos, mos["I_UNC_CORR"][ha_ind_mos])
            print("lum fac", lum_fac)
            
            mos_norm = mos['FLUX'][to_use_mos]/mos['FLUX'][ha_ind_mos]
            
            # Convert to luminosity and propagate uncertainty
            mos_lum = mos_norm * ha * lum_fac
            mos_lum_err = mos_lum* np.sqrt(np.divide(mos['F_UNC'][to_use_mos], mos['FLUX'][to_use_mos], where=mos['FLUX'][to_use_mos] > 0)**2
                                            + (mos['F_UNC'][ha_ind_mos]/mos['FLUX'][ha_ind_mos])**2 + (ha_err/ha)**2) 
            # mos_lum_err = mos['I_UNC_CORR'][to_use_mos]/mos['FLUX'][4] * lum_fac * ha

            # Record
            for j, line in enumerate(mos_lum):
                if line > mos_lum_err[j]:
                    line_lum[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(line)
                    line_lum_unc[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(mos_lum_err[j])
                else:
                    #print(in_cue_mos[j], "is an upper lim")
                    line_lum_up[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(mos_lum_err[j]) * 3
                    line_lum_unc[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(mos_lum_err[j])
                    #print("new line lum upper limit", line_lum_up[line_list == cat_to_cue_mosfire[in_cue_mos[j]]], "\n")
        

        elif (hb_mos != 0) & (hb != 0):
            print("using hb as a normalizing line")
            print("ha nirs", ha, "ha mos", ha_mos)
            print("hb nirs", hb, "hb mos", hb_mos)
            
            # Normalise MOSFIRE lines to MOSFIRE Hb
            mos_norm = mos['FLUX'][to_use_mos]/mos['FLUX'][hb_ind_mos]
            # Convert to luminosity
            mos_lum = mos_norm * hb * lum_fac

            if hb == 0:
                print(f'{id} No NIRSpec Hb')
            if mos['FLUX'][hb_ind_mos] == 0:
                print(f'{id} no MOSFIRE Hb')
                        
            # Propagate uncertainty on MOSFIRE lines
            mos_lum_err =  mos_lum * np.sqrt(np.divide(mos['F_UNC'][to_use_mos], mos['FLUX'][to_use_mos], where=mos['FLUX'][to_use_mos] > 0)**2 +
                                            (mos['F_UNC'][hb_ind_mos]/mos['FLUX'][hb_ind_mos])**2 + (hb_err/hb)**2) 
            for j, line in enumerate(mos_lum):
                if line > mos_lum_err[j]:
                    line_lum[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(line)
                    line_lum_unc[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(mos_lum_err[j])
                else:
                    line_lum_up[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(mos_lum_err[j]) * 3
                    line_lum_unc[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(mos_lum_err[j])

        else: # if Ha nor Hb overlap, use O3
            print("using o3 as a normalizing line")
            print("ha nirs", ha, "ha mos", ha_mos)
            print("hb nirs", hb, "hb mos", hb_mos)


            o3_ind_mos = (mos['WAVELENGTH']==5006.843)
            o3_mos = mos['FLUX'][o3_ind_mos]
            mos_norm = mos["FLUX"][to_use_mos]/mos["FLUX"][o3_ind_mos]

            # get g235m o3
            o3 = fluxes235['FLUX'][lines235['WAVELENGTH'] == 5006.843].item()
            o3_err = fluxes235['F_UNC'][lines235['WAVELENGTH'] == 5006.843].item()

            mos_lum = mos_norm * o3 * lum_fac
            mos_lum_err = mos_lum * np.sqrt(np.divide(mos['F_UNC'][to_use_mos], mos['FLUX'][to_use_mos], where=mos['FLUX'][to_use_mos] > 0)**2
                                            + (mos["F_UNC"][o3_ind_mos]/mos["FLUX"][o3_ind_mos])**2 +(o3_err/o3)**2)
            
            for j, line in enumerate(mos_lum):
                if line > mos_lum_err[j]:
                    line_lum[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(line)
                    line_lum_unc[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(mos_lum_err[j])
                else:
                    line_lum_up[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(mos_lum_err[j]) * 3
                    line_lum_unc[line_list == cat_to_cue_mosfire[in_cue_mos[j]]] = float(mos_lum_err[j])
            
        # Get usable cdline intensities in G235M and convert to luminosity
        lum_235 = fluxes235['FLUX'][to_use_235] * lum_fac
        lum_err_235 = fluxes235['F_UNC'][to_use_235] * lum_fac


        # Record line luminosities
        # line_lum_up[line_list == cat_to_cue_g235m[in_cue_235[j]]] = float(lum_err_235[j]) * 3
        # line_lum_unc[line_list == cat_to_cue_g235m[in_cue_235[j]]] = float(lum_err_235[j])
        # line_lum[line_list == cat_to_cue_g235m[in_cue_235[j]]] = float(line)

        for j, line in enumerate(lum_235):
            #print(cat_to_cue_g235m[in_cue_235[j]], "lum", line_lum[line_list == cat_to_cue_g235m[in_cue_235[j]]],"uncertainty", line_lum_unc[line_list == cat_to_cue_g235m[in_cue_235[j]]], "upper lim", line_lum_up[line_list == cat_to_cue_g235m[in_cue_235[j]]])
            if line > lum_err_235[j]:
                # print("line detected in nirs")
                #print("luminosity branch")
                line_lum[line_list == cat_to_cue_g235m[in_cue_235[j]]] = float(line)
                line_lum_unc[line_list == cat_to_cue_g235m[in_cue_235[j]]] = float(lum_err_235[j])
                # print(cat_to_cue_g235m[in_cue_235[j]], line_lum[line_list == cat_to_cue_g235m[in_cue_235[j]]], line_lum_unc[line_list == cat_to_cue_g235m[in_cue_235[j]]], line_lum_up[line_list == cat_to_cue_g235m[in_cue_235[j]]], "\n")
                
            elif (line_lum[line_list == cat_to_cue_g235m[in_cue_235[j]]] != 0):
                # print("line not detected in nirs, but is already detected in mos")
                # print(cat_to_cue_g235m[in_cue_235[j]], line_lum[line_list == cat_to_cue_g235m[in_cue_235[j]]], line_lum_unc[line_list == cat_to_cue_g235m[in_cue_235[j]]], line_lum_up[line_list == cat_to_cue_g235m[in_cue_235[j]]], "\n")
                
                continue
            elif lum_err_235[j] != 0:  # if the line exists, but is not over the SN threshold

                line_lum_up[line_list == cat_to_cue_g235m[in_cue_235[j]]] = float(lum_err_235[j]) * 3
                line_lum_unc[line_list == cat_to_cue_g235m[in_cue_235[j]]] = float(lum_err_235[j])

            else:
                continue

            # print("after", cat_to_cue_g235m[in_cue_235[j]], "lum", line_lum[line_list == cat_to_cue_g235m[in_cue_235[j]]], "uncertainty", line_lum_unc[line_list == cat_to_cue_g235m[in_cue_235[j]]], "upper lim", line_lum_up[line_list == cat_to_cue_g235m[in_cue_235[j]]], "\n")

        # Get usable line intensities in G395M and convert to luminosity
        fluxes395 = hdu[3].data
        lum_395 = fluxes395['FLUX'][to_use_395] * lum_fac
        lum_err_395 = fluxes395['F_UNC'][to_use_395] * lum_fac
        for j, line in enumerate(lum_395):
            # If the the uncertainty for that line is already non-zero (i.e., was also measured in G235M)
            if line_lum_unc[line_list == cat_to_cue_g395m[in_cue_395[j]]] != 0:
                # If this is a line luminosity measurement and not an upper limit
                if line_lum[line_list == cat_to_cue_g395m[in_cue_395[j]]] != 0:
                    # If the SNR of the G235M measurement is higher than the G395M measurement, continue to the next emission line

                    if line_lum[line_list == cat_to_cue_g395m[in_cue_395[j]]]/line_lum_unc[line_list == cat_to_cue_g395m[in_cue_395[j]]] > np.nan_to_num(line/lum_err_395[j]):  # otherwise it was comparing to a nan
                        continue
                    # Else, replace G235M measurement with G395M
                    else:
                        line_lum[line_list == cat_to_cue_g395m[in_cue_395[j]]] = float(line)
                        line_lum_unc[line_list == cat_to_cue_g395m[in_cue_395[j]]] = float(lum_err_395[j])
                # If this line is an upper limit in G235M
                if line_lum_up[line_list == cat_to_cue_g395m[in_cue_395[j]]] != 0:
                    # If this is not an upper limit in G395M, adopt the line in G395M. Set upper limit to 0
                    if line > lum_err_395[j]:
                        line_lum[line_list == cat_to_cue_g395m[in_cue_395[j]]] = float(line)
                        line_lum_unc[line_list == cat_to_cue_g395m[in_cue_395[j]]] = float(lum_err_395[j])
                        line_lum_up[line_list == cat_to_cue_g395m[in_cue_395[j]]] = 0
                    # If it's still an upper limit but a more stringent upper limit, adopt G395M. 
                    elif lum_err_395[j] < line_lum_unc[line_list == cat_to_cue_g395m[in_cue_395[j]]]:
                        line_lum_up[line_list == cat_to_cue_g395m[in_cue_395[j]]] = float(lum_err_395[j]) * 3
                        if line_lum[line_list == cat_to_cue_g395m[in_cue_395[j]]] == 0:
                            line_lum_unc[line_list == cat_to_cue_g395m[in_cue_395[j]]] = float(lum_err_395[j])
                    # Else, continue to the next emission line
                    else:
                        continue
            # If not already recorded in G235M, proceed normally.
            elif line > lum_err_395[j]:
                line_lum[line_list == cat_to_cue_g395m[in_cue_395[j]]] = float(line)
                line_lum_unc[line_list == cat_to_cue_g395m[in_cue_395[j]]] = float(lum_err_395[j])
            else:
                line_lum_up[line_list == cat_to_cue_g395m[in_cue_395[j]]] = float(lum_err_395[j]) * 3
                if line_lum[line_list == cat_to_cue_g395m[in_cue_395[j]]] == 0:
                    line_lum_unc[line_list == cat_to_cue_g395m[in_cue_395[j]]] = float(lum_err_395[j])
        hdu.close()

        dic = {'z':zdf['z'][zdf['IDNo'] == id].item(), 
            'line_name': line_list, 'line_wav':line_wav,
            'line_lum':line_lum, 'line_lum_up':line_lum_up,
            'line_lum_unc':line_lum_unc}
        name = zdf['GalaxyName'][zdf['IDNo'] == id].item()

        # # break
        with open('/home/carolinevr/PAPER/line-luminosities-nodustcorr/{}.pkl'.format(name), 'wb') as f:
            dill.dump(dic, f)

        print('/home/carolinevr/PAPER/line-luminosities-nodustcorr/{}.pkl written'.format(name))