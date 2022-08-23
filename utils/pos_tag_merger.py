import pickle
import copy
from typing import List

from data_def import Sentence, Morp, NE, Word
from transformers import AutoTokenizer, ElectraTokenizer

### MAIN ##
if "__main__" == __name__:
    src_file_path = "../data/pkl/ne_mp_old_nikl.pkl"
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    results = { # tag: (word_item, morp_item_list) pair
        "JX": [],
        "EP": [],
        "EF": [],
        "EC": [],
        "ETM": [],
    }
    src_list: List[Sentence] = []
    with open(src_file_path, mode="rb") as load_pkl:
        src_list = pickle.load(load_pkl)
        print(f"LOAD SIZE: {len(src_list)}") # 371571

    target_pos_list = results.keys()
    end_point_pos = ["SF", "SP", "SS", "SE", "SO", "SW"]

    for src_idx, src_item in enumerate(src_list):
        if 0 == (src_idx % 1000):
            print(f"{src_idx} is Processing... {src_item.text}")

        new_morp_list = []
        for word_item_idx, word_item in enumerate(src_item.word_list):
            word_item_id = word_item.id

            extract_morp_item_list = [x for x in src_item.morp_list if x.word_id == word_item_id]
            concat_morp_item_form = [x.form for x in extract_morp_item_list]
            concat_morp_item_form = "".join(concat_morp_item_form)
            if word_item.form == concat_morp_item_form:
                new_morp_list.extend(extract_morp_item_list)
                continue
            else:
                # print(word_item)
                # print(extract_morp_item_list)
                # print(concat_morp_item_form)

                new_morp_item_list = []
                ignore_morp_id = []
                conv_word_form = copy.deepcopy(word_item.form)
                prev_pos = ""
                for morp_idx, morp_item in enumerate(extract_morp_item_list):
                    if morp_item.id in ignore_morp_id:
                        continue
                    if morp_item.form in conv_word_form:
                        if 0 >= len(conv_word_form.replace(morp_item.form, "")):
                            prev_pos = morp_item.label # 만나/VV + 아/EC = 만나
                            continue
                        new_morp_item_list.append(copy.deepcopy(morp_item))
                        conv_word_form = conv_word_form.replace(morp_item.form, "")
                    else:
                        end_idx = len(extract_morp_item_list)
                        extract_morp_label_list = [x for x in extract_morp_item_list]
                        for label_idx in range(morp_idx, len(extract_morp_label_list)):
                            if extract_morp_label_list[label_idx].label in end_point_pos:
                                end_idx = label_idx + 1
                                break
                        merge_label = [extract_morp_label_list[i].label for i in range(morp_idx, end_idx)]
                        if 0 < len(prev_pos):
                            merge_label.insert(0, prev_pos)
                            prev_pos = ""
                        if merge_label[-1] in end_point_pos:
                            front_merge_label = merge_label[:-1]
                            front_merge_label = "+".join(front_merge_label)
                            new_morp_item = Morp(id=morp_item.id, form=conv_word_form[:-1], label=front_merge_label,
                                                 word_id=morp_item.word_id, position=morp_item.position)
                            new_morp_item_list.append(copy.deepcopy(new_morp_item))
                            new_morp_item = Morp(id=morp_item.id+1, form=conv_word_form[-1], label=merge_label[-1],
                                                 word_id=morp_item.word_id, position=morp_item.position+1)
                            new_morp_item_list.append(copy.deepcopy(new_morp_item))
                        else:
                            merge_label = "+".join(merge_label)
                            new_morp_item = Morp(id=morp_item.id, form=conv_word_form, label=merge_label,
                                                 word_id=morp_item.word_id, position=morp_item.position)
                            new_morp_item_list.append(copy.deepcopy(new_morp_item))
                            # print(new_morp_item)
                        # add ignore
                        ignore_morp_id.extend([extract_morp_label_list[i].id for i in range(morp_idx, end_idx)])

                        continue
                new_morp_list.extend(new_morp_item_list)
        src_list[src_idx].morp_list = new_morp_list
        #print([x.form+"/"+x.label for x in src_list[src_idx].morp_list])
    # end, src_list loop

    # save
    with open("../data/pkl/merge_ne_mp_old_nikl.pkl", mode="wb") as save_file:
        pickle.dump(src_list, save_file)
        print(f"SAVE FILE SIZE: {len(src_list)}")