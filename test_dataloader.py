from dataloader import create_story_fmri_dataset, create_video_fmri_dataset

if __name__ == '__main__':

    tr_story_ds, te_story_ds = create_story_fmri_dataset(fmri_dir='/cw/liir_data/NoCsBack/cross_modal_brain/story_fmri',
                                                        load_first_n_sub=1, #load 前？个被试的脑图，共12个被试
                                                         #load_specific_sub=["S1"], #如果需要某些特定被试的脑图作为训练数据，可以指定该参数，输入为list。如果不指定需要制定load_first_n_sub
                                                         load_first_n_story=2, #load 前？个故事的脑图，共11个故事
                                                         #load_specific_story=["souls","naked"] #如果需要某些特定故事的脑图作为训练数据，输入为list。可以指定该参数，如果不指定需要制定load_first_n_story
                                                         #test_story="howtodraw", 如果需要某一个特定故事作为测试集，可以指定该参数，如果不指定则每个故事的test_ratio比例作为测试集合
                                                         test_ratio=0.2,
                                                         load_fmri=True, #是否加载脑图)
                                                        )
                                                        
    # print(len(tr_story_ds), len(tr_story_ds.all_fmri))
    # print(len(te_story_ds), len(te_story_ds.all_fmri))
    tr_video_ds, te_video_ds = create_video_fmri_dataset(video_hdf_dir='/cw/liir_data/NoCsBack/cross_modal_brain/video_stimuli/',
                                                        fmri_dir='/cw/liir_data/NoCsBack/cross_modal_brain/video_fmri',
                                                        load_first_n_sub=1, #load 前？个被试的脑图，共12个被试
                                                        #load_specific_sub=["S1"], #如果需要某些特定被试的脑图作为训练数据，可以指定该参数，输入为list。如果不指定需要制定load_first_n_sub
                                                         )
