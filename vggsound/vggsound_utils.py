import csv
from path_settings import paths_config
from typing import Literal

def get_vid_label_lookup(split:Literal['train', 'test']):
    """ reuturns a lookup dict for vggsound videos and their label according to vggsound.csv """
    vggsound_dict = {}
    with open(paths_config.vggsound_path, "r") as vgg:
        reader = csv.reader(vgg)
        for row in reader:
            if split != row[3]: 
                continue

            vid_id = row[0]
            label = row[2]
            
            vggsound_dict[vid_id] = label

    return vggsound_dict


def get_label_category_lookup():
    """ reuturns a lookup dict for vggsound labels and their 'super category' according to stat.csv """
    vggsound_dict = {}
    with open(paths_config.stat_path, "r") as vgg:
        reader = csv.reader(vgg)
        for row in reader:
            label = row[0]
            category = row[2]
            
            vggsound_dict[label] = category

    return vggsound_dict



if __name__=="__main__":
    my_dict = get_vid_label_lookup('test')
    my_labes = get_label_category_lookup()

    video_ids = set(my_dict.keys())
    dict_labels = set(my_dict.values())
    labels = set(my_labes.keys())


    assert dict_labels == labels
    assert len(my_dict) == len(video_ids)
    assert len(my_labes) == len(labels)
