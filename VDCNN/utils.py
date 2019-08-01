# coding: utf-8

import re


def split_to_jamo(string):
    # unicode of Korean. start: 44032, end: 55199
    BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

    # 초성 리스트 0 ~ 18 (19개)
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ',
                    'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
                    'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    # 중성 리스트. 00 ~ 20
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                     'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                     'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ',
                     'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
                     'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    def split(sequence):
        split_string = list(sequence)
        list_of_tokens = []
        for char in split_string:
            # 한글 여부 확인 후 분리
            if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', char) is not None:
                if ord(char) < BASE_CODE:
                    list_of_tokens.append(char)
                    continue

                # 초성
                char_code = ord(char) - BASE_CODE
                char1 = int(char_code / CHOSUNG)
                list_of_tokens.append(CHOSUNG_LIST[char1])

                # 중성
                char2 = int((char_code - (CHOSUNG - char1)) / JUNGSUNG)
                list_of_tokens.append(JUNGSUNG_LIST[char2])

                # 중성
                char3 = int((char_code - (CHOSUNG - char1)) - (JUNGSUNG * char2))
                list_of_tokens.append(JONGSUNG_LIST[char3])

            else:
                list_of_tokens.append(char)

        return list_of_tokens

    return split(string)
