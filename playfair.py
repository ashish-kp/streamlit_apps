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
        ptext0 = re.sub('[\W_]+', '', self.plaintext.upper())
        # print(ptext0, "plaintext")

        key_mat = self.gen_key()
        ptext1 = ptext0[0]
        for i in range(1, len(ptext0)):
            if ptext0[i] == ptext0[i - 1]:
                ptext1 = ptext1 + 'X' + ptext0[i]
            else:
                ptext1 = ptext1 + ptext0[i]
        if len(ptext1) % 2 != 0:
            ptext1 = ptext1 + 'X'
        ptext = ""
        count = 0
        print(ptext1)
        for i in range(0, len(ptext1), 2):
            x = (np.where(key_mat == ptext1[i]))
            y = (np.where(key_mat == ptext1[i + 1]))
            r1, c1, r2, c2 = x[0][0], x[1][0], y[0][0], y[1][0]
            if r1 == r2:
                ptext = ptext + key_mat[r1, (c1 + 1) % 6] + key_mat[r2, (c2 + 1) % 6]
            elif c1 == c2:
                ptext = ptext + key_mat[(r1 + 1) % 6, c1] + key_mat[(r2 + 1) % 6, c2]
            else:
                ptext = ptext + key_mat[r1, c2] + key_mat[r2, c1]
        return ptext

    def decrypt(self, user_given = None):
        if user_given:
            ciphertext = user_given.upper()
        else:
            ciphertext = self.encrypt()
        key_mat = self.gen_key()
        ptext1 = ciphertext
        ptext = ''
        for i in range(0, len(ptext1), 2):
            x = (np.where(key_mat == ptext1[i]))
            y = (np.where(key_mat == ptext1[i + 1]))
            r1, c1, r2, c2 = x[0][0], x[1][0], y[0][0], y[1][0]
            if r1 == r2:
                ptext = ptext + key_mat[r1, (c1 - 1) % 6] + key_mat[r2, (c2 - 1) % 6]
            elif c1 == c2:
                ptext = ptext + key_mat[(r1 - 1) % 6, c1] + key_mat[(r2 - 1) % 6, c2]
            else:
                ptext = ptext + key_mat[r1, c2] + key_mat[r2, c1]
        return ptext

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
		st.write(np.array(key_mat))
		enc_text = f"The text sent for decryption is {my_algo.encrypt()}"
		st.text(enc_text)
# 	if st.button("Check for your own ciphertext"):
# 		ciph = st.text_input("Enter a ciphertext made with the above key, anything else will result in an error.")
# 		if ciph != '':
# 			if st.button("Do it!"):
# 				st.write(my_algo.decrypt(re.sub('[\W_]+', '', ciph.upper())))
