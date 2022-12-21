from typing import Tuple

#=======================================================================================
def convert_morp_connected_tokens(sent_lvl_pos: Tuple[str, str]):
#=======================================================================================
    ret_conv_morp_tokens = []

    g_SYMBOL_TAGS = [
        "SF", "SE", # (마침표, 물음표, 느낌표), 줄임표
        "SSO", "SSC", # 여는 괄호, 닫는 괄호
        "SC", "SY", # 구분자, (붙임표, 기타 기호)
    ]

    for eojeol in sent_lvl_pos:
        for mp_idx, morp in enumerate(eojeol): # morp (마케팅, NNG)
            if 0 == mp_idx or morp[1] in g_SYMBOL_TAGS:
                ret_conv_morp_tokens.append((morp[0], morp[1], False))
            else:
                if eojeol[mp_idx-1][1] not in g_SYMBOL_TAGS:
                    conv_morp = (morp[0], morp[1], True)
                    ret_conv_morp_tokens.append(conv_morp)
                else:
                    ret_conv_morp_tokens.append((morp[0], morp[1], False))

    return ret_conv_morp_tokens