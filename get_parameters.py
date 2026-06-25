from pathlib import Path
import warnings
import numpy as np
import os


# Parameters used in the data generation process.
Database_dir = Path('database')

def get_params(arg=0):
    task_id = int(arg[1])
    params = dict(
        # path containing background noise recordings
        database_dir = Database_dir,
        mixturepath = Database_dir , # root path to the synthesized mixture files
        db_path = 'source_datasets/single_source_samples',
        materials_path = 'source_datasets/material_absorption',
        ontology_path =  'source_datasets/ontology.json',
        # noisepath = Database_dir / 'TAU-SRIR_DB/TAU-SNoise_DB', 
        min_samples_per_class = 30,
        metric_threshold = 10., # filter out the sound event class with lower F-scores
        mixture_duration = 60., # seconds
        start_delay = 10, # seconds
        audio_format = 'both', # 'foa' , 'mic' or 'both'
        db_name = 'seld', 
        seed = 2024, # fix the seed for reproducibility
        chunksize = 128,
        max_workers = 128,
        nb_mixtures = 10000, # number of mixtures 
        ################ Sound Event Parameters ################
        nb_events_per_classes = -1, # -1 means all events
        target_classes = 'all', # all classes are considered as target
        interf_classes = [], # no interference
        max_polyphony_target = 3,
        max_polyphony_interf = 1,
        ################ SRIR Parameters ################
        #### mic array parameters #####
        SH_order = 1, # spherical harmonic order
        array_type = 'rigid', # 'rigid' or 'open'
        SH_type = 'real', # shperical harmony type, 'real' or 'complex'
        radius = 0.042, # radius of the spherical array
        mic_pos = [[45,35],[-45,-35],[135,-35],[-135,35]],
        #### Room Parameters #####
        # [[lx_min, lx_max],[ly_min, ly_max],[lz_min, lz_max]]
        room_size_range = [[4., 20.], [4., 20.], [3., 10.]],
        # [value_min, value_max]
        temperature_range = [15, 35], # degree Celsius
        humidity_range = [0, 100],
        RT60_range = [0.2, 2], # in seconds. None for absorption of materials from database.
        mic_pos_range_percentage = [0.4, 0.6], # percentage of the room size
        src_pos_from_walls = 0.5,
        src_pos_from_listener = 1,
        method = 'hybrid', # 'hybrid' or 'ism'
        tools = 'pyroomacoustics', # 'gpuRIR' or 'pyroomacoustics' or 'smir'
        add_noise = False,
        snr_set = [6, 31], # dB
        add_interf = False,
        dataset_type = 'test', # NOTE: synthesizing training sets or test sets

        # Additional args for optiimzation
        # Windows/performance controls
        measure_rt60 = False,
        write_sum = False,
        print_each_mixture = False,
        fast_audio_loader = True,
    )

    if task_id == 1:
        ################################################################################
        #### Default for sound events from AudioSet or FSD50K (training set)
        ####   and SRIRs are computationally generated
        ####   (./data_generator/data_synthesis.py)
        ################################################################################
        params['db_name'] = 'FSD50K'
        params['dataset_type'] = 'train' # NOTE: synthesizing training sets or test sets
        params['nb_mixtures'] = 10
        params['max_polyphony_target'] = 1
        params['output_dir'] = 'seld_{}_{}_ov{}_{}'.format(
            params['db_name'], params['nb_mixtures'], 
            params['max_polyphony_target'], params['dataset_type'])
        params['mixturepath'] /= params['output_dir']
        params['audio_format'] = 'both'
        params['RT60_range'] = None # or [0.2, 2.]
        params['chunksize'] = 2
        params['max_workers'] = 16
    elif task_id == 2:
        ################################################################################
        #### Default for sound events from AudioSet and FSD50K (test set), 
        ####   and SRIRs from TAU-SRIR DB 
        ####   (./data_generator/data_synthesis_test.py)
        ################################################################################
        params['db_name'] = 'FSD50K'
        params['audio_format'] = 'both'
        params['dataset_type'] = 'test' # NOTE: synthesizing training sets or test sets
        params['nb_mixtures'] = 18 # TODO
        params['max_polyphony_target'] = 1
        params['output_dir'] = 'seld_{}_tau{}_ov{}_{}'.format(
            params['db_name'], params['nb_mixtures'], 
            params['max_polyphony_target'], params['dataset_type'])
        params['mixturepath'] /= params['output_dir']
        params['rooms'] = [[1, 2, 3, 4, 5, 6, 8, 9, 10]]
        params['mixture_duration'] = 60
        params['event_time_per_layer'] = 40
        params['chunksize'] = 22
        params['max_workers'] = 5
        TAU_SRIR_DB = Database_dir / 'TAU-SRIR_DB'
        params['rirpath'] = TAU_SRIR_DB / 'TAU-SRIR_DB'
        params['noisepath'] = TAU_SRIR_DB / 'TAU-SNoise_DB'
        params['add_noise'] = False
    elif task_id == 3:
        ################################################################################
        #### ET-CLR Phase 1 pretraining dataset
        #### Clean, static, single-source FOA mixtures.
        #### Use this for the current ET-CLR setup.
        ################################################################################
        params['db_name'] = 'FSD50K'
        params['dataset_type'] = 'train'

        # Corpus size
        params['nb_mixtures'] = 3600          # 60 s each
        params['mixture_duration'] = 60.0
        params['audio_format'] = 'foa'         # ET-CLR uses FOA WXYZ-derived features only

        # Output
        params['output_dir'] = 'etclr_phase1_{}_{}mix_{}s_foa_ov{}_{}'.format(
            params['db_name'],
            params['nb_mixtures'],
            int(params['mixture_duration']),
            1,
            params['dataset_type']
        )
        params['mixturepath'] /= params['output_dir']

        # Sound-event setup
        params['target_classes'] = 'all'
        params['interf_classes'] = []
        params['nb_events_per_classes'] = -1
        params['max_polyphony_target'] = 1
        params['max_polyphony_interf'] = 0
        params['add_interf'] = False
        params['start_delay'] = 1.0

        # Room/SRIR setup
        params['room_size_range'] = [[4., 12.], [4., 12.], [2.5, 5.]]
        params['RT60_range'] = [0.2, 1.2]
        params['src_pos_from_walls'] = 0.7
        params['src_pos_from_listener'] = 1.0
        params['mic_pos_range_percentage'] = [0.35, 0.65]

        # Keep synthesis clean; your ET-CLR already applies feature augmentations.
        params['add_noise'] = False
        params['snr_set'] = [20, 40]

        # Computation
        params['chunksize'] = 32
        params['max_workers'] = 4

        params['measure_rt60'] = False
        params['write_sum'] = False
        params['print_each_mixture'] = False
        params['fast_audio_loader'] = True

        # ET-CLR crop-manifest settings
        params['make_etclr_manifest'] = True
        params['etclr_phase'] = 1
        params['etclr_crop_duration'] = 3.0
        params['etclr_crop_hop'] = 1.5
        params['etclr_frame_rate'] = 10          # DCASE-style metadata: 100 ms frames
        params['etclr_min_active_fraction'] = 0.60
        params['etclr_max_polyphony'] = 1
        params['etclr_require_single_source'] = True
        params['etclr_split_name'] = 'pretrain'

    elif task_id == 4:
        ################################################################################
        #### ET-CLR Phase 2 pretraining dataset
        #### Harder multi-source FOA mixtures for robustness after Phase 1.
        ################################################################################
        params['db_name'] = 'FSD50K'
        params['dataset_type'] = 'train'

        # Corpus size
        params['nb_mixtures'] = 3600
        params['mixture_duration'] = 60.0
        params['audio_format'] = 'foa'

        # Output
        params['output_dir'] = 'etclr_phase2_{}_{}mix_{}s_foa_ov{}_{}'.format(
            params['db_name'],
            params['nb_mixtures'],
            int(params['mixture_duration']),
            3,
            params['dataset_type']
        )
        params['mixturepath'] /= params['output_dir']

        # Sound-event setup
        params['target_classes'] = 'all'
        params['interf_classes'] = []
        params['nb_events_per_classes'] = -1
        params['max_polyphony_target'] = 3
        params['max_polyphony_interf'] = 0
        params['add_interf'] = False
        params['start_delay'] = 1.0

        # Harder room/SRIR setup
        params['room_size_range'] = [[4., 20.], [4., 20.], [3., 10.]]
        params['RT60_range'] = [0.2, 2.0]
        params['src_pos_from_walls'] = 0.5
        params['src_pos_from_listener'] = 1.0
        params['mic_pos_range_percentage'] = [0.3, 0.7]

        # Add moderate synthetic noise only in Phase 2.
        params['add_noise'] = True
        params['snr_set'] = [10, 30]

        # Computation
        params['chunksize'] = 32
        params['max_workers'] = 4

        params['measure_rt60'] = False
        params['write_sum'] = False
        params['print_each_mixture'] = False
        params['fast_audio_loader'] = True

        # ET-CLR crop-manifest settings
        params['make_etclr_manifest'] = True
        params['etclr_phase'] = 2
        params['etclr_crop_duration'] = 3.0
        params['etclr_crop_hop'] = 1.5
        params['etclr_frame_rate'] = 10
        params['etclr_min_active_fraction'] = 0.50
        params['etclr_max_polyphony'] = 3
        params['etclr_require_single_source'] = False
        params['etclr_split_name'] = 'pretrain_hard'

    elif task_id == 5:
        ################################################################################
        #### ET-CLR Phase 2 pretraining dataset
        #### Harder multi-source FOA mixtures for robustness after Phase 1.
        ################################################################################
        params['db_name'] = 'FSD50K'
        params['dataset_type'] = 'train'

        # Corpus size
        params['nb_mixtures'] = 3600
        params['mixture_duration'] = 60.0
        params['audio_format'] = 'foa'

        # Output
        params['output_dir'] = 'leseld_phase2_{}_{}mix_{}s_foa_ov{}_{}'.format(
            params['db_name'],
            params['nb_mixtures'],
            int(params['mixture_duration']),
            3,
            params['dataset_type']
        )
        params['mixturepath'] /= params['output_dir']

        # Sound-event setup
        params['target_classes'] = 'all'
        params['interf_classes'] = []
        params['nb_events_per_classes'] = -1
        params['max_polyphony_target'] = 3
        params['max_polyphony_interf'] = 0
        params['add_interf'] = False
        params['start_delay'] = 1.0

        # Harder room/SRIR setup
        params['room_size_range'] = [[4., 20.], [4., 20.], [3., 10.]]
        params['RT60_range'] = [0.2, 2.0]
        params['src_pos_from_walls'] = 0.5
        params['src_pos_from_listener'] = 1.0
        params['mic_pos_range_percentage'] = [0.3, 0.7]

        # Add moderate synthetic noise only in Phase 2.
        params['add_noise'] = True
        params['snr_set'] = [10, 30]

        # Computation
        params['chunksize'] = 32
        params['max_workers'] = 4

        params['measure_rt60'] = False
        params['write_sum'] = False
        params['print_each_mixture'] = False
        params['fast_audio_loader'] = True

        # ET-CLR crop-manifest settings
        params['make_etclr_manifest'] = True
        params['etclr_phase'] = 2
        params['etclr_crop_duration'] = 3.0
        params['etclr_crop_hop'] = 1.5
        params['etclr_frame_rate'] = 10
        params['etclr_min_active_fraction'] = 0.50
        params['etclr_max_polyphony'] = 3
        params['etclr_require_single_source'] = False
        params['etclr_split_name'] = 'pretrain_hard'
    
    params['mic_pos'] = np.array(params['mic_pos'])

    if params['tools'] == 'smir' and params['array_type'] == 'open':
        warnings.warn('SMIR does not support open array, change to rigid array!')
        params['array_type'] = 'rigid'
    if params['tools'] != 'pyroomacoustics' and params['method'] == 'hybrid':
        warnings.warn('Hybrid method only support pyroomacoustics, change to ISM!')
        params['method'] = 'ism'

    if params['mixturepath'].exists():
        if input('The mixture path {} exists! Do you want to overwrite it? (y/n)'.format(params['mixturepath'])) != 'y':
            exit(0)

    return params

if __name__ == '__main__':
    get_params()
