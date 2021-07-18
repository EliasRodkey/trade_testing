flag = "cvpbPGS{arkg_gvzr_V'yy_gel_2_ebhaqf_bs_ebg13_GYpXOHqX}"
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
shifted_alphabet = 'NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm'

def shift_13(message:str="test")->str:
    # takes input string of LETTERS OF ENGLISH ALPHABET ONLY
    # and shifts to 13 letters later
    return_message = ""
    for letter in message:
        i = alphabet.find(letter)
        return_message = return_message + shifted_alphabet[i]
    return return_message

for message in flag.split("'")[1].split("_"):
    print(shift_13(message))
"picoCTF{next_time_I'll_try_2_rounds_of_rot13_TLcKBUdK}"

print(flag[2])