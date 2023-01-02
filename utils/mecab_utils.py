from typing import Tuple
import copy

#=======================================================================================
def convert_morp_connected_tokens(sent_lvl_pos: Tuple[str, str], src_text: str):
#=======================================================================================
    ret_conv_morp_tokens = []

    g_SYMBOL_TAGS = [
        "SF", "SE", # (마침표, 물음표, 느낌표), 줄임표
        "SSO", "SSC", # 여는 괄호, 닫는 괄호
        "SC", "SY", # 구분자, (붙임표, 기타 기호)
    ]

    # 어절별 글자 수 체크해서 띄워쓰기 적기
    split_text = src_text.split(" ")
    char_cnt_list = [len(st) for st in split_text]

    total_eojeol_morp = []
    use_check = [False for _ in range(len(sent_lvl_pos))]
    for char_cnt in char_cnt_list:
        eojeol_morp = []
        curr_char_cnt = 0
        for ej_idx, eojeol in enumerate(sent_lvl_pos):
            if char_cnt == curr_char_cnt:
                total_eojeol_morp.append(copy.deepcopy(eojeol_morp))
                eojeol_morp.clear()
                break
            if use_check[ej_idx]:
                continue
            eojeol_morp.append(eojeol)
            curr_char_cnt += len(eojeol[0])
            use_check[ej_idx] = True
        if 0 < len(eojeol_morp):
            total_eojeol_morp.append(eojeol_morp)

    for eojeol in total_eojeol_morp:
        for mp_idx, morp in enumerate(eojeol):  # morp (마케팅, NNG)
            if 0 == mp_idx or morp[1] in g_SYMBOL_TAGS:
                ret_conv_morp_tokens.append((morp[0], morp[1], False))
            else:
                if eojeol[mp_idx - 1][1] not in g_SYMBOL_TAGS:
                    conv_morp = (morp[0], morp[1], True)
                    ret_conv_morp_tokens.append(conv_morp)
                else:
                    ret_conv_morp_tokens.append((morp[0], morp[1], False))

    return ret_conv_morp_tokens


#=======================================================================================
def convert_character_pos_tokens(sent_lvl_pos: Tuple[str, str], src_text: str):
#=======================================================================================
    # 어절별 글자 수 체크해서 띄워쓰기 적기
    split_text = src_text.split(" ")
    char_cnt_list = [len(st) for st in split_text]

    total_eojeol_morp = []
    use_check = [False for _ in range(len(sent_lvl_pos))]
    for char_cnt in char_cnt_list:
        curr_char_cnt = 0
        for ej_idx, eojeol in enumerate(sent_lvl_pos):
            if char_cnt == curr_char_cnt:
                total_eojeol_morp.append([" ", "O"])
                break
            if use_check[ej_idx]:
                continue
            total_eojeol_morp.append([eojeol[0], eojeol[1].split("+")])
            curr_char_cnt += len(eojeol[0])
            use_check[ej_idx] = True

    return total_eojeol_morp