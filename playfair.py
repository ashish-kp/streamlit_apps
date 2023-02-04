import numpy as np
import re
import streamlit as st

class play_fair:
    def __init__(self, key, plaintext):
        self.key = key
        self.plaintext = plaintext
    def check_if_valid_key(self):
        set_key = set(self.key)
        # print(set_key, self.key)
        if len(set_key) != len(self.key):
            raise ValueError("Please enter a key with distinct alphabets.")
        else:
            return True
    def gen_key(self):
        key_mat = []
        if self.check_if_valid_key():
            key_mat[:] = self.key.upper()
            for i in range(65, 65 + 26):
                if chr(i) not in key_mat:
                    key_mat.append(str(chr(i)))
            for j in range(10):
                if str(j) not in key_mat:
                    key_mat.append(str(j))
            return np.array(key_mat).reshape(6, 6)
    def show_key(self):
        print(self.gen_key())
    def encrypt(self):
        ptext = re.sub('[\W_]+', '', self.plaintext.upper())
        # print(ptext, "plaintext")
        if len(ptext) % 2 != 0:
            ptext = ptext + 'X'
        key_mat = self.gen_key()
        encrypted_text = []
        for i in range(0, len(ptext), 2):
            x1, y1 = (np.where(key_mat == ptext[i]))
            x2, y2 = (np.where(key_mat == ptext[i + 1]))

            if ptext[i] != ptext[i + 1]:
                # print(ptext[i], x1, x2, ptext[i + 1], y1, y2)
                if int(y1) == int(y2):
                    encrypted_text.append(key_mat[(int(x1) + 1) % 6][int(y1)])
                    encrypted_text.append(key_mat[(int(x2) + 1) % 6][int(y2)])
                    # print(key_mat[(int(x1) + 1) % 6][int(y1)], key_mat[(int(x2) + 1) % 6][int(y2)], (int(x1) + 1) % 6, y1, (int(x2) + 1) % 6, y2, "1")
                elif int(x1) == int(x2):
                    encrypted_text.append(key_mat[int(x1)][(int(y1) + 1) % 6])
                    encrypted_text.append(key_mat[int(x2)][(int(y2) + 1) % 6])
                    # print(key_mat[int(x1)][(int(y1) + 1) % 6], key_mat[int(x2)][(int(y2) + 1) % 6], x1, (int(y1) + 1) % 6, x2, (int(y2) + 1) % 6, "2")
                else:
                    encrypted_text.append(key_mat[int(x1)][int(y2)]) 
                    encrypted_text.append(key_mat[int(x2)][int(y1)])
                    # print(key_mat[int(x1)][int(y2)], key_mat[int(x2)][int(y1)], x1, y2, y1, x2, "3")
            else:
                # print(ptext[i], x1, x2, ptext[i + 1], y1, y2)
                if ptext[i] == 'X' and ptext[i + 1] == 'X':
                    encrypted_text.append(key_mat[int(x1)][(int(y1) + 1) % 6])
                    encrypted_text.append(key_mat[int(x1)][(int(y1) + 1) % 6])
                else:
                    x2, y2 = (np.where(key_mat == 'X'))
                    encrypted_text.append(key_mat[int(x1)][int(y2)])
                    encrypted_text.append(key_mat[int(x2)][int(y1)]) 
                    encrypted_text.append(key_mat[int(x1)][int(y2)])
                    encrypted_text.append(key_mat[int(x2)][int(y1)]) 
        return "".join(encrypted_text)
    def decrypt(self):
        ciphertext = self.encrypt()
        key_mat = self.gen_key()
        decrypted_text = []
        for i in range(0, len(ciphertext), 2):
            x1, y1 = (np.where(key_mat == ciphertext[i]))
            x2, y2 = (np.where(key_mat == ciphertext[i + 1]))
            if ciphertext[i] != ciphertext[i + 1] and ciphertext[i] != 'X':
                if int(y1) == int(y2):
                    decrypted_text.append(key_mat[(int(x1) - 1) % 6][int(y1)])
                    if key_mat[(int(x2) - 1) % 6][int(y2)] != 'X':
                        decrypted_text.append(key_mat[(int(x2) - 1) % 6][int(y2)])
                elif int(x1) == int(x2):
                    decrypted_text.append(key_mat[int(x1)][(int(y1) - 1) % 6])
                    if key_mat[int(x2)][(int(y2) - 1) % 6] != 'X':
                        decrypted_text.append(key_mat[int(x2)][(int(y2) - 1) % 6])
                else:
                    decrypted_text.append(key_mat[int(x1)][int(y2)]) 
                    if key_mat[int(x2)][int(y1)] != 'X':
                        decrypted_text.append(key_mat[int(x2)][int(y1)])
            else:
                if ciphertext[i] == ciphertext[i + 1]:
                    decrypted_text.append('X')
                    decrypted_text.append('X')
                else:
                    if ciphertext[i + 1] == 'X':
                        decrypted_text.append(key_mat[int(x1)][int(y2)])
                        decrypted_text.append(key_mat[int(x2)][int(y1)]) 
        return "".join(decrypted_text)


                # if 

st.title("Play Fair Encoding") 
key = st.text_input("Enter a key (non-repetitive alphabets and numbers)")
key_clean = re.sub('[\W_]+', '', key.upper())
plaintext = st.text_input("Enter the plaintext")
my_algo = play_fair(key_clean, plaintext)
if key != '' and plaintext != '':
	dec_text = f"The text recieved after decryption is {my_algo.decrypt()}"
	st.text(dec_text)
	if st.button("Show the key and encrypted text"):
			key_mat = my_algo.gen_key()
			st.text("The key matrix is \n")
			st.text(str(key_mat))
			enc_text = f"The text sent for encryption is {my_algo.encrypt()}"
			st.text(enc_text)