
import os
import numpy as np
import json
import h5py
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm


all_storys = ['odetostepfather','legacy','life','avatar','undertheinfluence','myfirstdaywiththeyankees',
              'howtodraw','souls','naked','alternateithicatom','wheretheressmoke']
story_tr_length = {"alternateithicatom": 363, "souls": 375, "avatar": 387, "legacy": 420, "odetostepfather": 424, "undertheinfluence": 324, "howtodraw": 374,
                    "myfirstdaywiththeyankees": 378, "naked": 442, "life": 450, "wheretheressmoke": 311}


def identity(x):
    return x

class StoryFmriDataset(Dataset):
    def __init__(self, all_fmri=None, all_stim=[], fmri_transform=identity, sentence_transform=identity):
        self.all_fmri = all_fmri
        self.all_stim = all_stim
        self.fmri_transform = fmri_transform
        self.sentence_transform = sentence_transform

    def __len__(self):
        return len(self.all_stim)
    
    def pad_tokens(self, item):
        pass 
    
    def __getitem__(self, index):
        sentence = self.sentence_transform(self.all_stim[index])
        if self.all_fmri is not None:
            fmri = self.fmri_transform(np.expand_dims(self.all_fmri[index], axis=0))
            return {'fmri': fmri, 'stimulus': sentence}
        else:
            return {'stimulus': sentence}
        
class VideoFmriDataset(Dataset):
    def __init__(self, all_fmri=None, all_stim=None, fmri_index_list=[], stim_index_list=[],
                 fmri_transform=identity, sentence_transform=identity, is_test=False, hdf_reload=False):
        self.all_fmri = all_fmri
        self.all_stim = all_stim
        self.fmri_index_list = fmri_index_list
        self.stim_index_list = stim_index_list
        self.fmri_transform = fmri_transform
        self.sentence_transform = sentence_transform
        self.is_test = is_test
        self.hdf_reload = hdf_reload

    def __len__(self):
        return len(self.stim_index_list)
    
    def pad_tokens(self, item):
        pass 
    
    def __getitem__(self, index):
        
        if self.all_fmri is not None:
            if not self.is_test:
                sub, stid, intvideo_begin, video_end = self.stim_index_list[index]
                _, _, fmri_tr_begin, fmri_tr_end = self.fmri_index_list[index]
                if self.hdf_reload:
                    all_stim = h5py.File(self.all_stim[stid], 'r')
                    target = all_stim['stimuli'][intvideo_begin:video_end]
                    all_stim.close()
                    return {'fmri': self.all_fmri[sub][fmri_tr_begin:fmri_tr_end], 
                            'stimulus': target}
                else:
                    return {'fmri': self.all_fmri[sub][fmri_tr_begin:fmri_tr_end], 
                            'stimulus': self.all_stim[stid][intvideo_begin:video_end]}
            else:
                sub, intvideo_begin, video_end = self.stim_index_list[index]
                _, fmri_tr_begin, fmri_tr_end = self.fmri_index_list[index]
                if self.hdf_reload:
                    all_stim = h5py.File(self.all_stim[0], 'r')
                    target = all_stim['stimuli'][intvideo_begin:video_end]
                    all_stim.close()
                    return {'fmri': self.all_fmri[sub][fmri_tr_begin:fmri_tr_end], 
                            'stimulus': target}
                else:
                    return {'fmri': self.all_fmri[sub][fmri_tr_begin:fmri_tr_end], 
                            'stimulus': self.all_stim[0][intvideo_begin:video_end]}
        else:
            if not self.is_test:
                sub, stid, intvideo_begin, video_end = self.stim_index_list[index]
                if self.hdf_reload:
                    all_stim = h5py.File(self.all_stim[stid], 'r')
                    target = all_stim['stimuli'][intvideo_begin:video_end]
                    all_stim.close()
                    return {'stimulus': target}
                else:
                    return {'stimulus': self.all_stim[stid][intvideo_begin:video_end]}
            else:
                sub, intvideo_begin, video_end = self.stim_index_list[index]
                if self.hdf_reload:
                    all_stim = h5py.File(self.all_stim[0], 'r')
                    target = all_stim['stimuli'][intvideo_begin:video_end]
                    all_stim.close()
                    return {'stimulus': target}
                else:
                    return {'stimulus': self.all_stim[0][intvideo_begin:video_end]}



def load_video_fmri_time_index(tr=2.0, wanted_video_duration=4.0, freq=15, stimulus_index="full", subject_list=['S1']):
    #wanted_video_duration=4.0的单位是秒， freq=15的单位是帧/秒， stimulus_index对应刺激hdf文件名中数字
    total_frames_per_video = 9000
    tr_video_index_list = []
    tr_fmri_index_list = []
    if stimulus_index == "full":
        stimulus_index_list = range(12)
    else:
        stimulus_index_list = [stimulus_index]

    for stid in stimulus_index_list:
        for ii in range(0, total_frames_per_video, int(wanted_video_duration*freq)):
            video_begin = ii
            video_end = ii + wanted_video_duration*freq
            fmri_tr_begin = stid*300 + np.ceil(ii/freq/tr)
            fmri_tr_end = stid*300 + np.ceil((ii + wanted_video_duration*freq)/freq/tr)
            for sub in subject_list:
                tr_video_index_list.append([sub, stid, int(video_begin), int(video_end)])
                tr_fmri_index_list.append([sub, stid, int(fmri_tr_begin), int(fmri_tr_end)])

    te_video_index_list = []
    te_fmri_index_list = []
    for ii in range(0, total_frames_per_video, int(wanted_video_duration*freq)):
        video_begin = ii
        video_end = ii + wanted_video_duration*freq
        fmri_tr_begin = np.ceil(ii/freq/tr)
        fmri_tr_end = np.ceil((ii + wanted_video_duration*freq)/freq/tr)
        for sub in subject_list:
            te_video_index_list.append([sub, int(video_begin), int(video_end)])
            te_fmri_index_list.append([sub, int(fmri_tr_begin), int(fmri_tr_end)])
    
    return tr_video_index_list, tr_fmri_index_list, te_video_index_list, te_fmri_index_list

def load_video_stimuli(video_hdf_dir='/cw/liir_data/NoCsBack/cross_modal_brain/video_stimuli/', number_of_video_loaded=12):
    tr_video_names = ['train_%s.hdf' % f'{i:02}' for i in range(number_of_video_loaded)]
    tr_all_vnh5 = []
    te_all_vnh5 = []
    for vn in tqdm(tr_video_names):
        vnh5 = h5py.File(os.path.join(video_hdf_dir, vn), 'r')
        tr_all_vnh5.append(np.array(vnh5.get('stimuli')))
        vnh5.close()

    vnh5 = h5py.File(os.path.join(video_hdf_dir, 'test.hdf'), 'r')
    te_all_vnh5.append(np.array(vnh5.get('stimuli')))
    vnh5.close()

    return tr_all_vnh5, te_all_vnh5

def load_video_stimuli_filename(video_hdf_dir='/cw/liir_data/NoCsBack/cross_modal_brain/video_stimuli/', number_of_video_loaded=12):
    tr_video_names = ['train_%s.hdf' % f'{i:02}' for i in range(number_of_video_loaded)]
    tr_all_vnh5 = []
    te_all_vnh5 = []
    for vn in tqdm(tr_video_names):
        tr_all_vnh5.append(os.path.join(video_hdf_dir, vn))

    te_all_vnh5.append(os.path.join(video_hdf_dir, 'test.hdf'))

    return tr_all_vnh5, te_all_vnh5

def load_video_fmri(fmri_dir='/cw/liir_data/NoCsBack/cross_modal_brain/video_fmri', subject_list=['S1']):
    tr_fmri_dict = {}
    te_fmri_dict = {}
    
    for subject in subject_list:
        print('Loading subject %s' % subject)
        tr_fmri_fn = os.path.join(fmri_dir, subject, 'train.npy')
        te_fmri_fn = os.path.join(fmri_dir, subject, 'test.npy')
        tr_fmri_dict[subject] = np.load(tr_fmri_fn)
        te_fmri_dict[subject] = np.load(te_fmri_fn)

    return tr_fmri_dict, te_fmri_dict

def create_video_fmri_dataset(video_hdf_dir='/cw/liir_data/NoCsBack/cross_modal_brain/video_stimuli/',
                              fmri_dir='/cw/liir_data/NoCsBack/cross_modal_brain/video_fmri',
                              load_first_n_sub=None, load_specific_sub=None, 
                              fmri_transform=identity, sentence_transform=identity, load_only_video_filename=False):
    
    if load_specific_sub is not None:
        subject_list = load_specific_sub
    elif load_first_n_sub is not None:
        subject_list = ['S%s' % f'{i+1}' for i in range(load_first_n_sub)]
    else:
        subject_list = ['S%s' % f'{i+1}' for i in range(12)]
    print('Loading video stimuli from hdf')
    if load_only_video_filename:
        tr_all_video_stimuli, te_all_video_stimuli = load_video_stimuli_filename(video_hdf_dir)
    else:
        tr_all_video_stimuli, te_all_video_stimuli = load_video_stimuli(video_hdf_dir)
    print('Loading video fmri')
    tr_video_index_list, tr_fmri_index_list, te_video_index_list, te_fmri_index_list = load_video_fmri_time_index(subject_list=subject_list)
    tr_fmri_dict, te_fmri_dict = load_video_fmri(fmri_dir, subject_list)

    return VideoFmriDataset(all_fmri=tr_fmri_dict, all_stim=tr_all_video_stimuli, fmri_index_list=tr_fmri_index_list,
                             stim_index_list=tr_video_index_list, hdf_reload=load_only_video_filename), \
            VideoFmriDataset(all_fmri=te_fmri_dict, all_stim=te_all_video_stimuli, fmri_index_list=te_fmri_index_list,
                                stim_index_list=te_video_index_list, hdf_reload=load_only_video_filename)


    

def load_story_tr_time(story, tsv_dir="/cw/liir_code/NoCsBack/jingyuan/cross_modal/transcripts", 
                       tr=2.0, pad=5):
    tsv_file = os.path.join(tsv_dir, story+'.tsv')
    length_dict = story_tr_length
    length = length_dict[story]
    stimuli = []
    tr_index = []
    with open(tsv_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            fields = line.strip('\n').split('\t')
            tr_begin = min(np.ceil(int(fields[0])/1000/tr) + pad, length)
            tr_end = min(np.ceil(int(fields[1])/1000/tr) + pad, length)
            stimuli.append([fields[-1]])
            tr_index.append([int(tr_begin), int(tr_end)])
            
    return stimuli, tr_index

def create_story_fmri_dataset(fmri_dir='/cw/liir_data/NoCsBack/cross_modal_brain/story_fmri', load_first_n_sub=None, load_specific_sub=None, 
                    load_first_n_story=None, load_specific_story=None, test_story=None, test_ratio=0.2, load_fmri=True):
    if load_first_n_story is not None:
        tr_story_list = all_storys[:load_first_n_story]
    elif load_specific_story is not None:
        tr_story_list = load_specific_story
    else:
        tr_story_list = all_storys
    
    # if test_story is not None and test_story in tr_story_list:
    #     tr_story_list.remove(test_story)

    if load_specific_sub is not None:
        subject_list = load_specific_sub
    elif load_first_n_sub is not None:
        subject_list = ['S%s' % f'{i+1}' for i in range(load_first_n_sub)]
    else:
        subject_list = ['S%s' % f'{i+1}' for i in range(12)]

    all_tr_stimuli = []
    all_te_stimuli = []
    if load_fmri:
        all_tr_fmri = []
        all_te_fmri = []
        for ii in subject_list:
            for story in tr_story_list:
                tr_fmri_fn = os.path.join(fmri_dir, ii, f'{story}.npy')
                tr_fmri_npy = np.load(tr_fmri_fn)
                stimuli, tr_index = load_story_tr_time(story)
                if test_story is None:
                    for sid, stim in enumerate(stimuli[:int(len(stimuli)*(1-test_ratio))]):
                        all_tr_stimuli.append(stim)
                        all_tr_fmri.append(tr_fmri_npy[tr_index[sid][0]:tr_index[sid][1]])
                    for sid, stim in enumerate(stimuli[int(len(stimuli)*(1-test_ratio)):]):
                        all_te_stimuli.append(stim)
                        all_te_fmri.append(tr_fmri_npy[tr_index[sid][0]:tr_index[sid][1]])
                else:
                    for sid, stim in enumerate(stimuli):
                        all_tr_stimuli.append(stim)
                        all_tr_fmri.append(tr_fmri_npy[tr_index[sid][0]:tr_index[sid][1]])


            if test_story is not None:
                stimuli, tr_index = load_story_tr_time(test_story)
                te_fmri_fn = os.path.join(fmri_dir, ii, f'{test_story}.npy')
                te_fmri_npy = np.load(te_fmri_fn)
                for sid, stim in enumerate(stimuli):
                    all_te_stimuli.append(stim)
                    all_te_fmri.append(te_fmri_npy[tr_index[sid][0]:tr_index[sid][1]])
           

        return StoryFmriDataset(all_fmri=all_tr_fmri, all_stim=all_tr_stimuli), StoryFmriDataset(all_fmri=all_te_fmri, all_stim=all_te_stimuli)
    
    else:
        for ii in subject_list:
            for story in tr_story_list:
                stimuli, _ = load_story_tr_time(story)
                if test_story is None:
                    for sid, stim in enumerate(stimuli[:int(len(stimuli)*(1-test_ratio))]):
                        all_tr_stimuli.append(stim)
                    for sid, stim in enumerate(stimuli[int(len(stimuli)*(1-test_ratio)):]):
                        all_te_stimuli.append(stim)
                else:
                    for sid, stim in enumerate(stimuli):
                        all_tr_stimuli.append(stim)

        if test_story is not None:
            stimuli, _ = load_story_tr_time(test_story)
            for sid, stim in enumerate(stimuli):
                all_te_stimuli.append(stim)

    
    return StoryFmriDataset(all_stim=all_tr_stimuli), StoryFmriDataset(all_stim=all_te_stimuli)




# if __name__ == '__main__':
    # tsv_file = '/cw/liir_code/NoCsBack/jingyuan/cross_modal/transcripts/adollshouse.tsv'
    # length_json = '/cw/liir_code/NoCsBack/jingyuan/cross_modal/respdict.json'
    # stim_tr_dict = load_story_tr_time(tsv_file, length_json)
    # print(stim_tr_dict)
    # video_index_list, fmri_index_list = load_video_tr_time(stimulus_index=1)
    # print(video_index_list[-10:], fmri_index_list[-10:])
    # tr_video_fmri, te_video_fmri = load_video_fmri(load_first_n_sub=2)
    # print(tr_video_fmri['S1'].shape, tr_video_fmri['S2'].shape)
    # print(te_video_fmri['S1'].shape, te_video_fmri['S2'].shape)














